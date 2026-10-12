import pytest
from fastapi.testclient import TestClient

from ecoroute.api import create_app


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["mode"] == "simulate"


def test_route_does_not_log(client):
    r = client.post("/v1/route", json={"prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["complexity"]["tier"] == 1
    assert client.get("/v1/metrics/summary").json()["totals"]["n"] == 0


def test_chat_logs_and_caches(client):
    body = {"prompt": "Explain photosynthesis briefly.", "policy": "eco"}
    first = client.post("/v1/chat", json=body).json()
    second = client.post("/v1/chat", json=body).json()
    assert first["source"] == "simulated" and second["source"] == "cache"
    assert second["impact"]["energy_wh"] == 0
    s = client.get("/v1/metrics/summary").json()
    assert s["totals"]["n"] == 2 and s["totals"]["cache_hits"] == 1
    assert s["saved"]["energy_wh"] > 0


def test_openai_compatible_endpoint(client):
    r = client.post("/v1/chat/completions", json={
        "model": "ecoroute/eco", "messages": [{"role": "user", "content": "What is 2+2?"}]})
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "chat.completion"
    assert j["choices"][0]["message"]["content"]
    assert "impact" in j["ecoroute"]


def test_openai_endpoint_unknown_model(client):
    r = client.post("/v1/chat/completions", json={"model": "nope", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 404


def test_models_and_grid(client):
    m = client.get("/v1/models").json()
    assert all("energy" in x and x["energy"]["label"] in "ABCDEFG" for x in m["models"])
    assert client.get("/v1/grid").json()["regions"]


def test_budget_roundtrip(client):
    r = client.put("/v1/budget", json={"monthly_budget_usd": 5}).json()
    assert r["monthly_budget_usd"] == 5
    assert client.get("/v1/budget").json()["monthly_budget_usd"] == 5


def test_seed_timeseries_and_reset(client):
    assert client.post("/v1/demo/seed?n=10").status_code == 200
    assert client.get("/v1/metrics/summary").json()["totals"]["n"] == 10
    assert client.get("/v1/metrics/timeseries?days=30&bucket=day").json()
    assert len(client.get("/v1/metrics/requests?limit=5").json()) == 5
    client.delete("/v1/metrics")
    assert client.get("/v1/metrics/summary").json()["totals"]["n"] == 0


def test_dashboard_served(client):
    page = client.get("/")
    assert page.status_code == 200 and 'data-tab="quality"' in page.text and 'data-tab="embed"' in page.text
    assert client.get("/static/app.js").status_code == 200
    receipt = client.get("/static/receipt.js")
    assert receipt.status_code == 200 and "EcoReceipt" in receipt.text


def test_bad_request(client):
    assert client.post("/v1/chat", json={}).status_code == 400
    assert client.post("/v1/route", json={"prompt": "x", "policy": "turbo"}).status_code == 400
