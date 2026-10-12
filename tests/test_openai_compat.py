import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from ecoroute.api import create_app
from ecoroute.providers import AnthropicProvider, GeminiProvider, Provider, ProviderResult

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
                   "additionalProperties": False}}}]


class ToolProvider(Provider):
    """Answers with a tool call and records the options it was given."""

    def __init__(self, name):
        super().__init__()
        self.name, self.seen = name, []

    def configured(self):
        return True

    async def complete(self, model, messages, max_tokens, options=None):
        self.seen.append({"messages": messages, "options": options})
        call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Pune"}'}}
        return ProviderResult("", 40, 12, 3.0, tool_calls=[call], finish_reason="tool_calls")


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def _sse(text: str) -> list:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]


def test_streaming_returns_openai_chunks_and_logs(client):
    r = client.post("/v1/chat/completions", json={
        "model": "ecoroute/eco", "stream": True, "messages": [{"role": "user", "content": "Explain why the sky is blue."}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]")
    chunks = _sse(r.text)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert "Simulated response" in text
    last = chunks[-1]
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"]["completion_tokens"] > 0 and last["ecoroute"]["impact"]["energy_wh"] > 0
    assert all(c["model"] == last["model"] for c in chunks)
    assert client.get("/v1/metrics/summary").json()["totals"]["n"] == 1


def test_streaming_bad_model_is_http_error(client):
    r = client.post("/v1/chat/completions", json={"model": "nope", "stream": True,
                                                  "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 404


def test_tools_route_to_tool_capable_models_and_pass_through(service, client):
    service.settings.mode = "live"
    prov = ToolProvider("groq")
    service.providers = {"groq": prov}
    r = client.post("/v1/chat/completions", json={
        "model": "ecoroute/eco", "tools": TOOLS, "tool_choice": "auto", "temperature": 0.2,
        "messages": [{"role": "user", "content": "hi, weather in Pune?"}]}).json()
    msg = r["choices"][0]["message"]
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather" and msg["content"] is None
    assert prov.seen[0]["options"]["tools"] == TOOLS and prov.seen[0]["options"]["temperature"] == 0.2
    assert r["model"].startswith("groq-")


def test_tool_requests_need_tier_two(service):
    d = asyncio.run(service.decide("hi", options={"tools": TOOLS}))
    assert d.required_tier == 2
    tiny = next(c for c in d.candidates if c.model_id == "local-llama3.2-1b")
    assert not tiny.eligible


def test_tool_result_messages_are_accepted(service, client):
    service.settings.mode = "live"
    prov = ToolProvider("groq")
    service.providers = {"groq": prov}
    call = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Pune"}'}}
    r = client.post("/v1/chat/completions", json={"model": "ecoroute/auto", "tools": TOOLS, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "Weather in Pune?"}]},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "31C, sunny"}]})
    assert r.status_code == 200
    sent = prov.seen[0]["messages"]
    assert sent[0]["content"] == "Weather in Pune?" and sent[1]["tool_calls"] == [call]
    assert sent[2] == {"role": "tool", "content": "31C, sunny", "tool_call_id": "call_1"}


def test_responses_api_models_excluded_for_tools(service):
    d = asyncio.run(service.decide("Prove that sqrt(2) is irrational, rigorously.", options={"tools": TOOLS},
                                   constraints={"allowed_providers": ["openai"], "min_tier": 5}))
    pro = next(c for c in d.candidates if c.model_id == "openai-gpt-5-pro")
    assert not pro.eligible and "tool" in pro.reason
    assert d.model_id != "openai-gpt-5-pro"


def test_anthropic_translation():
    p = AnthropicProvider("k")
    call = {"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Pune"}'}}
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "31C"}]
    body = p._body(type("M", (), {"model": "claude-x"})(), msgs, 100,
                   {"tools": TOOLS, "tool_choice": "required", "stop": "END"})
    assert body["system"] == "be brief" and body["tool_choice"] == {"type": "any"}
    assert body["tools"][0]["input_schema"]["required"] == ["city"]
    assert body["messages"][1]["content"][0] == {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                                                 "input": {"city": "Pune"}}
    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert body["stop_sequences"] == ["END"]


def test_gemini_translation():
    p = GeminiProvider("k")
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Pune"}'}}
    body = p._body([{"role": "user", "content": "weather?"}, {"role": "assistant", "content": "", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": "c1", "content": "31C"}], 100, {"tools": TOOLS})
    decl = body["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "get_weather" and "additionalProperties" not in decl["parameters"]
    assert body["contents"][1]["parts"][0]["functionCall"] == {"name": "get_weather", "args": {"city": "Pune"}}
    assert body["contents"][2]["parts"][0]["functionResponse"]["name"] == "get_weather"


# ----------------------------------------------------------------------------- auth & teams
@pytest.fixture
def locked(service):
    service.settings.access_keys = {"key-a": "search", "key-b": "support", "key-free": None}
    service.settings.admin_key = "admin-1"
    return TestClient(create_app(service))


def test_auth_required_when_keys_configured(locked):
    body = {"prompt": "hi"}
    assert locked.post("/v1/chat", json=body).status_code == 401
    assert locked.post("/v1/chat", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert locked.post("/v1/chat", json=body, headers={"Authorization": "Bearer key-a"}).status_code == 200
    assert locked.get("/health").json()["auth"] is True  # health stays public
    assert locked.get("/v1/models").status_code == 200


def test_requests_are_attributed_to_teams_and_scoped(locked):
    for key, prompt in (("key-a", "hi"), ("key-a", "what is DNS?"), ("key-b", "thanks")):
        locked.post("/v1/chat", json={"prompt": prompt}, headers={"X-API-Key": key})
    locked.post("/v1/chat", json={"prompt": "hello"}, headers={"Authorization": "Bearer key-free",
                                                               "X-EcoRoute-Team": "growth"})
    admin = {"Authorization": "Bearer admin-1"}
    teams = {t["team"]: t["n"] for t in locked.get("/v1/metrics/summary", headers=admin).json()["by_team"]}
    assert teams == {"search": 2, "support": 1, "growth": 1}
    mine = locked.get("/v1/metrics/summary", headers={"Authorization": "Bearer key-a"}).json()
    assert mine["totals"]["n"] == 2  # a team key only sees its own team
    rows = locked.get("/v1/metrics/requests?team=support", headers=admin).json()
    assert [r["team"] for r in rows] == ["support"]


def test_admin_only_actions(locked):
    assert locked.delete("/v1/metrics", headers={"Authorization": "Bearer key-a"}).status_code == 403
    assert locked.put("/v1/budget", json={"monthly_budget_usd": 3}, headers={"Authorization": "Bearer key-a"}).status_code == 403
    assert locked.delete("/v1/metrics", headers={"Authorization": "Bearer admin-1"}).status_code == 200


def test_parse_access_keys():
    from ecoroute.config import parse_access_keys
    assert parse_access_keys(" a:team-1, b ,, c:") == {"a": "team-1", "b": None, "c": None}


def test_openai_provider_native_stream(service, monkeypatch):
    from ecoroute.providers import OpenAICompatibleProvider

    prov = OpenAICompatibleProvider("groq", "https://x", "k")
    lines = [json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
             json.dumps({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]}),
             json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}), "[DONE]"]
    seen = {}

    async def fake_sse(url, body, headers=None):
        seen["body"] = body
        for x in lines:
            yield x

    monkeypatch.setattr(prov, "_sse", fake_sse)

    async def run():
        return [ev async for ev in prov.stream(service.catalog.get("groq-gpt-oss-20b"), [{"role": "user", "content": "hi"}], 50)]

    evs = asyncio.run(run())
    assert [e.text for e in evs if not e.done] == ["Hel", "lo"]
    assert seen["body"]["stream"] and seen["body"]["stream_options"] == {"include_usage": True}
    res = evs[-1].result
    assert (res.text, res.input_tokens, res.output_tokens) == ("Hello", 5, 2)


def test_stream_falls_back_before_first_token(service):
    from ecoroute.providers import ProviderError

    class Broken(Provider):
        name = "groq"

        def configured(self):
            return True

        async def complete(self, model, messages, max_tokens, options=None):
            raise ProviderError("down")

    class Ok(ToolProvider):
        async def complete(self, model, messages, max_tokens, options=None):
            return ProviderResult("fine answer", 3, 2, 1.0)

    service.settings.mode = "live"
    service.providers = {"groq": Broken(), "gemini": Ok("gemini")}

    async def run():
        return [ev async for ev in service.stream_chat("hello there", use_cache=False)]

    evs = asyncio.run(run())
    done = evs[-1]["outcome"]
    assert done.fallback_used and done.model_id.startswith("gemini") and done.text == "fine answer"
