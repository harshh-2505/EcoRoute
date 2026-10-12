import asyncio
import json

from fastapi.testclient import TestClient

from ecoroute.api import create_app
from ecoroute.providers import Provider, ProviderResult
from ecoroute.quality import check_answer, parse_verdict


class Scripted(Provider):
    """Returns a fixed answer per model id (or a default) and logs each call."""

    def __init__(self, name, answers=None, default="A solid, complete answer. " * 30):
        super().__init__()
        self.name, self.answers, self.default, self.calls = name, answers or {}, default, []

    def configured(self):
        return True

    async def complete(self, model, messages, max_tokens, options=None):
        self.calls.append(model.id)
        text = self.answers.get(model.id, self.default)
        if callable(text):
            text = text(messages)
        return ProviderResult(text, 30, max(1, len(text.split())), 2.0)


def live(service, **providers):
    service.settings.mode = "live"
    service.providers = providers


def test_check_answer_flags_clear_failures():
    cx = {"tier": 3, "task_type": "coding", "expected_output_tokens": 500}
    assert not check_answer(cx, "")["ok"]
    assert "cut off" in check_answer(cx, "def f(): pass", "length")["issues"][0]
    bad = check_answer(cx, "I'm not sure how to do that.")
    assert not bad["ok"] and any("refused" in i for i in bad["issues"]) and bad["confidence"] < 1
    assert check_answer(cx, "```python\ndef f():\n    return 1\n```\n" + "explained " * 60)["ok"]
    assert check_answer({"tier": 1}, "Tokyo.")["ok"]  # short is fine for trivial prompts


def test_parse_verdict():
    v = parse_verdict('Sure: {"score_a": 8, "score_b": 8.5, "verdict": "equal", "reason": "both fine"}')
    assert v == {"score_routed": 8.0, "score_baseline": 8.5, "verdict": "equal", "reason": "both fine"}
    assert parse_verdict("no json here") is None
    assert parse_verdict('{"score_a": 3, "score_b": 9}')["verdict"] == "baseline_better"


def test_cascade_escalates_failed_answer(service):
    weak = "groq-gpt-oss-20b"
    groq = Scripted("groq", {weak: "I'm not sure."})
    gem = Scripted("gemini")
    live(service, groq=groq, gemini=gem)
    out = asyncio.run(service.chat("Explain in detail how TCP congestion control works, step by step, "
                                   "comparing Reno and CUBIC.", policy="eco",
                                   constraints={"cascade": True, "allowed_providers": ["groq", "gemini"]},
                                   use_cache=False))
    esc = out.extra["escalation"]
    assert esc["from"] == weak and out.model_id != weak
    assert service.catalog.get(out.model_id).quality > service.catalog.get(weak).quality
    assert out.extra["quality_check"]["ok"]
    rows = {r["id"]: r for r in service.ledger.recent(10)}
    assert rows[esc["first_request_id"]]["check_ok"] == 0 and rows[out.request_id]["escalated"] == 1
    # both attempts are on the ledger, so the extra energy is counted honestly
    assert service.ledger.summary()["totals"]["n"] == 2


def test_cascade_off_by_default_and_skips_simulated(service):
    out = asyncio.run(service.chat("Write a Python function to merge two sorted lists.", constraints={"cascade": True}))
    assert "escalation" not in out.extra and "quality_check" not in out.extra


def test_feedback_escalate_and_penalties(service):
    client = TestClient(create_app(service))
    first = client.post("/v1/chat", json={"prompt": "Summarise the causes of the French Revolution."}).json()
    rid = first["request_id"]
    r = client.post("/v1/feedback", json={"request_id": rid, "rating": "down", "escalate": True}).json()
    again = r["escalated"]
    assert again["model_id"] != first["model_id"]
    assert again["decision"]["required_tier"] >= first["decision"]["required_tier"] + 1 or \
        again["decision"]["required_tier"] == 5
    assert client.post("/v1/feedback", json={"request_id": 99999, "rating": "up"}).status_code == 404
    assert client.post("/v1/feedback", json={"request_id": rid, "rating": "meh"}).status_code == 400

    # three thumbs-down on the same model start to lower its quality score in routing
    for _ in range(3):
        rid2 = client.post("/v1/chat", json={"prompt": "Summarise the causes of the French Revolution.",
                                             "use_cache": False}).json()["request_id"]
        client.post("/v1/feedback", json={"request_id": rid2, "rating": "down"})
    pen = service.feedback_penalties()
    assert pen.get(first["model_id"], 0) > 0
    d = client.post("/v1/route", json={"prompt": "Summarise the causes of the French Revolution."}).json()
    c = next(x for x in d["candidates"] if x["model_id"] == first["model_id"])
    assert c["scores"]["feedback_penalty"] == pen[first["model_id"]]
    q = client.get("/v1/quality").json()
    assert q["feedback"]["down"] == 4 and q["feedback"]["satisfaction"] == 0


def test_shadow_eval_grades_blind_and_logs_overhead(service, monkeypatch):
    def judge(messages):
        body = messages[-1]["content"]
        a = body.split("=== ANSWER A ===")[1].split("=== ANSWER B ===")[0]
        routed_is_a = "routed" in a
        return json.dumps({"score_a": 9 if routed_is_a else 7, "score_b": 7 if routed_is_a else 9,
                           "verdict": "equal", "reason": "fine"})

    service.settings.judge_model = "openai-gpt-5-mini"
    service.settings.shadow_rate = 1.0
    openai = Scripted("openai", {"openai-gpt-5": "baseline answer " * 20, "openai-gpt-5-mini": judge})
    groq = Scripted("groq", default="routed answer " * 20)
    live(service, openai=openai, groq=groq)
    client = TestClient(create_app(service))
    out = client.post("/v1/chat", json={"prompt": "What is the boiling point of water at sea level?",
                                        "constraints": {"allowed_providers": ["groq"]}}).json()
    assert out["model_id"].startswith("groq")
    q = client.get("/v1/quality").json()
    ev = q["evals"]
    assert ev["n"] == 1 and ev["parity_rate"] == 1.0
    assert q["recent_evals"][0]["score_routed"] == 9  # un-swapped correctly whichever side it was shown on
    assert ev["overhead_energy_wh"] > 0
    assert "openai-gpt-5" in openai.calls and "openai-gpt-5-mini" in openai.calls
    # overhead calls are not counted as routed traffic
    assert client.get("/v1/metrics/summary").json()["totals"]["n"] == 1
    # on-demand eval of the same request
    assert client.post(f"/v1/quality/eval/{out['request_id']}").json()["verdict"] == "equal"


def test_shadow_eval_skipped_in_simulate(service):
    service.settings.shadow_rate = 1.0
    client = TestClient(create_app(service))
    rid = client.post("/v1/chat", json={"prompt": "hi"}).json()["request_id"]
    assert client.get("/v1/quality").json()["evals"]["n"] == 0
    assert client.post(f"/v1/quality/eval/{rid}").status_code == 400
