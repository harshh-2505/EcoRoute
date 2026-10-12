import asyncio
import csv
import io

from fastapi.testclient import TestClient

from ecoroute.api import create_app
from ecoroute.ledger import canonical
from ecoroute.providers import Provider, ProviderResult


def client_for(service):
    return TestClient(create_app(service))


def test_canonical_merges_rewordings_but_not_different_questions():
    same = ["What's the capital of France?", "please tell me what is the capital of france",
            "Hey, what is the capital of France? Thanks!"]
    assert len({canonical(x) for x in same}) == 1
    assert canonical("What is 2+2?") != canonical("What is 2+3?")
    assert canonical("convert 3.5 km to miles") != canonical("convert 35 km to miles")
    assert canonical("Is it raining?") != canonical("Is it not raining?")
    assert canonical("x = a - b") != canonical("x = a + b")


def test_reworded_prompt_hits_cache(service):
    c = client_for(service)
    first = c.post("/v1/chat", json={"prompt": "What's the capital of Japan?"}).json()
    second = c.post("/v1/chat", json={"prompt": "please tell me what is the capital of japan"}).json()
    assert first["source"] == "simulated" and second["source"] == "cache"


class Recorder(Provider):
    def __init__(self, name):
        super().__init__()
        self.name, self.calls = name, []

    def configured(self):
        return True

    async def complete(self, model, messages, max_tokens, options=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return ProviderResult("short answer", 20, 3, 1.0)


def test_concise_mode_adds_instruction_and_caps_tokens(service):
    service.settings.mode = "live"
    prov = Recorder("groq")
    service.providers = {"groq": prov}
    out = asyncio.run(service.chat("Write a 600 word essay about wind power.", constraints={"concise": True},
                                   use_cache=False))
    sent = prov.calls[0]
    assert sent["messages"][0]["role"] == "system" and "concise" in sent["messages"][0]["content"]
    assert sent["max_tokens"] <= int(out.decision.complexity["expected_output_tokens"] * 1.2)
    assert out.extra["concise"] is True
    plain = asyncio.run(service.chat("Write a 600 word essay about wind power.", use_cache=False))
    assert prov.calls[1]["messages"][0]["role"] == "user" and prov.calls[1]["max_tokens"] > sent["max_tokens"]
    assert plain.extra["concise"] is False


def test_report_json_csv_html(service):
    c = client_for(service)
    c.post("/v1/chat", json={"prompt": "Explain how heat pumps work."}, headers={"X-EcoRoute-Team": "growth"})
    c.post("/v1/chat", json={"prompt": "Write a Python function to parse CSV files."})
    r = c.get("/v1/report").json()
    assert r["totals"]["n"] == 2 and r["saved"]["energy_wh"] > 0
    lo, hi, mid = r["range"]["low"]["energy_wh"], r["range"]["high"]["energy_wh"], r["totals"]["energy_wh"]
    assert lo < mid < hi
    assert {t["team"] for t in r["by_team"]} == {"growth", "unassigned"}
    assert "uncertainty" in r["methodology"]
    rows = list(csv.DictReader(io.StringIO(c.get("/v1/report?format=csv").text)))
    assert rows[-1]["model_id"] == "TOTAL" and int(rows[-1]["requests"]) == 2
    page = c.get("/v1/report?format=html")
    assert page.status_code == 200 and "sustainability report" in page.text and "Methodology" in page.text
    assert c.get("/v1/report?month=2001-01").json()["totals"]["n"] == 0
    assert c.get("/v1/report?month=oct").status_code == 400


def test_batch_waits_for_clean_grid_then_runs(service):
    c = client_for(service)
    job = c.post("/v1/batch", json={"deadline_hours": 12, "requests": [
        {"custom_id": "a", "prompt": "Summarise the history of solar power."},
        {"custom_id": "b", "prompt": "List three uses of hydrogen fuel."}]}).json()
    assert job["status"] == "queued" and job["n"] == 2
    # No live grid data and the deadline is far away: nothing runs yet.
    assert c.post("/v1/batch-run").json() == {"ran": 0, "waiting": 2}
    # Live data shows a clean grid in every region: the items run now.
    for r in service.catalog.regions.values():
        r.live, r.carbon_g_per_kwh = True, r.average_g_per_kwh * 0.5
    assert c.post("/v1/batch-run").json()["ran"] == 2
    done = c.get(f"/v1/batch/{job['id']}").json()
    assert done["status"] == "completed" and all(i["status"] == "done" and i["response"] for i in done["items"])
    assert {i["custom_id"] for i in done["items"]} == {"a", "b"}
    assert c.get("/v1/metrics/summary").json()["totals"]["n"] == 2


def test_batch_runs_at_deadline_and_cron_secret(service):
    service.settings.access_keys = {"k": "ops"}
    service.settings.admin_key = "adm"
    service.settings.cron_secret = "cron-1"
    c = client_for(service)
    h = {"Authorization": "Bearer k"}
    job = c.post("/v1/batch", json={"deadline_hours": 0.5, "requests": [{"prompt": "hi"}]}, headers=h).json()
    assert job["team"] == "ops"
    assert c.get("/v1/batch-run").status_code == 401
    assert c.get("/v1/batch-run", headers=h).status_code == 403
    assert c.get("/v1/batch-run", headers={"Authorization": "Bearer cron-1"}).json()["ran"] == 1
    assert c.get(f"/v1/batch/{job['id']}", headers=h).json()["status"] == "completed"


def test_batch_validation(service):
    c = client_for(service)
    assert c.post("/v1/batch", json={"requests": []}).status_code == 400
    assert c.post("/v1/batch", json={"requests": [{"custom_id": "x"}]}).status_code == 400
    assert c.get("/v1/batch/nope").status_code == 404
