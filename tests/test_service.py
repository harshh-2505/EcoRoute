import asyncio

from ecoroute.providers import Provider, ProviderError, ProviderResult


class FakeProvider(Provider):
    def __init__(self, name, fail=False):
        super().__init__()
        self.name, self.fail, self.calls = name, fail, 0

    def configured(self):
        return True

    async def complete(self, model, messages, max_tokens):
        self.calls += 1
        if self.fail:
            raise ProviderError("boom")
        return ProviderResult(f"answer from {model.id}", 12, 34, 5.0)


def test_live_mode_routes_only_to_configured_providers(service):
    service.settings.mode = "live"
    service.providers = {"groq": FakeProvider("groq")}
    out = asyncio.run(service.chat("What is the capital of France?", use_cache=False))
    assert out.model_id.startswith("groq-") and out.source == "live"
    assert out.impact["output_tokens"] == 34  # real usage is recorded, not the estimate


def test_fallback_on_provider_error(service):
    service.settings.mode = "live"
    bad, good = FakeProvider("groq", fail=True), FakeProvider("gemini")
    service.providers = {"groq": bad, "gemini": good}
    out = asyncio.run(service.chat("hello there", use_cache=False))
    assert out.fallback_used and out.model_id.startswith("gemini")
    assert out.errors and bad.calls >= 1


def test_auto_mode_falls_back_to_simulation(service):
    service.settings.mode = "auto"
    service.providers = {"groq": FakeProvider("groq")}
    service.providers["groq"].configured = lambda: False
    available, mode = asyncio.run(service.availability())
    assert mode == "simulate" and len(available) == len(service.catalog.models)


def test_budget_status(service):
    service.set_budget(usd=1.0)
    asyncio.run(service.chat("Write a 300 word story about a river.", use_cache=False))
    b = service.budget_status()
    assert b["spent_usd"] > 0 and 0 < b["cost_ratio"] < 1


def test_responses_api_models_use_responses_endpoint(service, monkeypatch):
    from ecoroute.providers import OpenAICompatibleProvider

    prov = OpenAICompatibleProvider("openai", "https://api.openai.com/v1", "k")
    seen = {}

    async def fake_post(url, json, headers=None):
        seen["url"], seen["body"] = url, json
        return {"output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
                "usage": {"input_tokens": 7, "output_tokens": 3}}

    monkeypatch.setattr(prov, "_post", fake_post)
    model = service.catalog.get("openai-gpt-5.5-pro")
    res = asyncio.run(prov.complete(model, [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}], 50))
    assert seen["url"].endswith("/responses") and seen["body"]["instructions"] == "s"
    assert (res.text, res.input_tokens, res.output_tokens) == ("hi", 7, 3)
