"""Provider adapters. Each turns a list of chat messages into text plus token usage."""
from __future__ import annotations

import asyncio
import random
import time
import zlib
from dataclasses import dataclass

import httpx

from ..catalog import ModelSpec
from ..complexity import estimate_tokens


@dataclass
class ProviderResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    simulated: bool = False


class ProviderError(RuntimeError):
    pass


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system") or None
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


class Provider:
    name = "base"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def configured(self) -> bool:
        return False

    async def available_models(self, models: list[ModelSpec]) -> set[str]:
        return {m.id for m in models if m.provider == self.name} if self.configured() else set()

    async def complete(self, model: ModelSpec, messages: list[dict], max_tokens: int) -> ProviderResult:
        raise NotImplementedError

    async def _post(self, url: str, json: dict, headers: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=json, headers=headers or {})
        if r.status_code >= 400:
            raise ProviderError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()


class OpenAICompatibleProvider(Provider):
    """OpenAI and any OpenAI-compatible endpoint (Groq, Together, OpenRouter, vLLM...)."""

    def __init__(self, name: str, base_url: str, api_key: str | None, timeout: float = 60.0,
                 max_tokens_field: str = "max_tokens"):
        super().__init__(timeout)
        self.name, self.base_url, self.api_key = name, base_url.rstrip("/"), api_key
        self.max_tokens_field = max_tokens_field

    def configured(self) -> bool:
        return bool(self.api_key)

    async def complete(self, model, messages, max_tokens):
        if model.api == "responses":
            return await self._responses(model, messages, max_tokens)
        t0 = time.perf_counter()
        data = await self._post(
            f"{self.base_url}/chat/completions",
            {"model": model.model, "messages": messages, self.max_tokens_field: max_tokens},
            {"Authorization": f"Bearer {self.api_key}"},
        )
        usage = data.get("usage") or {}
        text = (data["choices"][0]["message"].get("content") or "").strip()
        return ProviderResult(text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                              (time.perf_counter() - t0) * 1000)

    async def _responses(self, model, messages, max_tokens):
        """OpenAI Responses API, for models not served on /chat/completions (the *-pro models)."""
        t0 = time.perf_counter()
        system, rest = _split_system(messages)
        body = {"model": model.model, "input": rest, "max_output_tokens": max_tokens}
        if system:
            body["instructions"] = system
        data = await self._post(f"{self.base_url}/responses", body,
                                {"Authorization": f"Bearer {self.api_key}"})
        text = "".join(c.get("text", "") for item in data.get("output", []) if item.get("type") == "message"
                       for c in item.get("content", []) if c.get("type") == "output_text")
        u = data.get("usage") or {}
        return ProviderResult(text.strip(), u.get("input_tokens", 0), u.get("output_tokens", 0),
                              (time.perf_counter() - t0) * 1000)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None, timeout: float = 60.0):
        super().__init__(timeout)
        self.api_key = api_key

    def configured(self) -> bool:
        return bool(self.api_key)

    async def complete(self, model, messages, max_tokens):
        t0 = time.perf_counter()
        system, rest = _split_system(messages)
        body = {"model": model.model, "max_tokens": max_tokens, "messages": rest}
        if system:
            body["system"] = system
        data = await self._post("https://api.anthropic.com/v1/messages", body, {
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage") or {}
        return ProviderResult(text.strip(), u.get("input_tokens", 0), u.get("output_tokens", 0),
                              (time.perf_counter() - t0) * 1000)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str | None, timeout: float = 60.0):
        super().__init__(timeout)
        self.api_key = api_key

    def configured(self) -> bool:
        return bool(self.api_key)

    async def complete(self, model, messages, max_tokens):
        t0 = time.perf_counter()
        system, rest = _split_system(messages)
        body: dict = {
            "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]} for m in rest],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model.model}:generateContent"
        data = await self._post(url, body, {"x-goog-api-key": self.api_key})
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        u = data.get("usageMetadata") or {}
        out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        return ProviderResult("".join(p.get("text", "") for p in parts).strip(),
                              u.get("promptTokenCount", 0), out, (time.perf_counter() - t0) * 1000)


class OllamaProvider(Provider):
    """Local open-source models served by Ollama (https://ollama.com)."""
    name = "ollama"

    def __init__(self, host: str, timeout: float = 120.0):
        super().__init__(timeout)
        self.host = host.rstrip("/")
        self._cache: tuple[float, set[str]] | None = None

    def configured(self) -> bool:
        return True

    async def _installed(self) -> set[str]:
        now = time.time()
        if self._cache and now - self._cache[0] < 30:
            return self._cache[1]
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                r = await client.get(f"{self.host}/api/tags")
            names = {m["name"] for m in r.json().get("models", [])} if r.status_code == 200 else set()
        except (httpx.HTTPError, ValueError):
            names = set()
        self._cache = (now, names)
        return names

    async def available_models(self, models):
        installed = await self._installed()
        out = set()
        for m in models:
            if m.provider == self.name and (m.model in installed or f"{m.model}:latest" in installed):
                out.add(m.id)
        return out

    async def complete(self, model, messages, max_tokens):
        t0 = time.perf_counter()
        data = await self._post(f"{self.host}/api/chat", {
            "model": model.model, "messages": messages, "stream": False,
            "options": {"num_predict": max_tokens}})
        return ProviderResult((data.get("message") or {}).get("content", "").strip(),
                              data.get("prompt_eval_count", 0), data.get("eval_count", 0),
                              (time.perf_counter() - t0) * 1000)


class SimulatedProvider(Provider):
    """Offline stand-in for every model so the whole system can be demoed without API keys.

    `max_tokens` is interpreted as the expected length; output varies ±30% around it; latency follows
    the catalog's tokens/second. The response text explains that it is simulated.
    """
    name = "simulated"
    delay = True  # emulate latency (capped at 250 ms); disabled in benchmarks/tests

    def configured(self) -> bool:
        return True

    async def available_models(self, models):
        return {m.id for m in models}

    async def complete(self, model, messages, max_tokens):
        prompt = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        in_tok = sum(estimate_tokens(m.get("content", "")) for m in messages)
        rng = random.Random(zlib.crc32(f"{model.id}|{prompt}".encode()))
        out_tok = max(8, int(max_tokens * rng.uniform(0.7, 1.3)))
        latency = 150 + out_tok / max(model.tps, 1) * 1000
        if self.delay:
            await asyncio.sleep(min(latency / 1000, 0.25))  # keep demos snappy
        snippet = prompt[:120] + ("…" if len(prompt) > 120 else "")
        text = (f"[Simulated response from {model.display_name}]\n\n"
                f"EcoRoute is running in simulate mode, so no real model was called. In live mode "
                f"this prompt would be answered by {model.display_name} ({model.provider}).\n\n"
                f"Prompt: \"{snippet}\"\n\n"
                f"Simulated length: {out_tok} output tokens. Add an API key or start Ollama to get "
                f"real answers.")
        return ProviderResult(text, in_tok, out_tok, latency, simulated=True)


def build_providers(settings) -> dict[str, Provider]:
    k, t = settings.api_keys, settings.request_timeout_s
    return {
        "openai": OpenAICompatibleProvider("openai", "https://api.openai.com/v1", k.get("openai"), t,
                                           max_tokens_field="max_completion_tokens"),
        "groq": OpenAICompatibleProvider("groq", "https://api.groq.com/openai/v1", k.get("groq"), t),
        "mistral": OpenAICompatibleProvider("mistral", "https://api.mistral.ai/v1", k.get("mistral"), t),
        "deepseek": OpenAICompatibleProvider("deepseek", "https://api.deepseek.com", k.get("deepseek"), t),
        "xai": OpenAICompatibleProvider("xai", "https://api.x.ai/v1", k.get("xai"), t),
        "anthropic": AnthropicProvider(k.get("anthropic"), t),
        "gemini": GeminiProvider(k.get("gemini"), t),
        "ollama": OllamaProvider(settings.ollama_host, max(t, 120)),
    }
