"""Provider adapters. Each turns a list of chat messages into text (or tool calls) plus token usage.

Messages use the OpenAI chat shape: ``{"role", "content", "tool_calls"?, "tool_call_id"?, "name"?}``.
``options`` carries the OpenAI request fields EcoRoute passes through: ``tools``, ``tool_choice``,
``temperature``, ``top_p`` and ``stop``. Each adapter translates them to its provider's API.
"""
from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx

from ..catalog import ModelSpec
from ..complexity import estimate_tokens

# Providers whose chat APIs accept function/tool definitions.
TOOL_PROVIDERS = {"openai", "groq", "mistral", "deepseek", "xai", "anthropic", "gemini", "ollama"}


@dataclass
class ProviderResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    simulated: bool = False
    tool_calls: list[dict] | None = None
    finish_reason: str = "stop"  # stop | length | tool_calls


@dataclass
class StreamEvent:
    """One piece of a streamed answer: a text delta, or the final usage summary (``done``)."""
    text: str = ""
    done: bool = False
    result: ProviderResult | None = field(default=None)


class ProviderError(RuntimeError):
    pass


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system" and m.get("content")) or None
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


def _args(raw) -> dict:
    """Tool-call arguments arrive as a JSON string (OpenAI) but some APIs want an object."""
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {"value": v}
    except ValueError:
        return {"raw": raw}


def _tool_call(name: str, args, call_id: str | None = None) -> dict:
    return {"id": call_id or f"call_{uuid.uuid4().hex[:16]}", "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


def _chunks(text: str, size: int = 3):
    """Split text into small word groups for pseudo-streaming."""
    words = text.split(" ")
    for i in range(0, len(words), size):
        yield " ".join(words[i:i + size]) + (" " if i + size < len(words) else "")


class Provider:
    name = "base"

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout

    def configured(self) -> bool:
        return False

    async def available_models(self, models: list[ModelSpec]) -> set[str]:
        return {m.id for m in models if m.provider == self.name} if self.configured() else set()

    async def complete(self, model: ModelSpec, messages: list[dict], max_tokens: int,
                       options: dict | None = None) -> ProviderResult:
        raise NotImplementedError

    async def stream(self, model: ModelSpec, messages: list[dict], max_tokens: int,
                     options: dict | None = None) -> AsyncIterator[StreamEvent]:
        """Default: one blocking call, replayed as chunks. Providers with native streaming override this."""
        res = await self.complete(model, messages, max_tokens, options)
        for piece in _chunks(res.text):
            yield StreamEvent(text=piece)
        yield StreamEvent(done=True, result=res)

    async def _post(self, url: str, json: dict, headers: dict | None = None) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(url, json=json, headers=headers or {})
        if r.status_code >= 400:
            raise ProviderError(f"{self.name} HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    async def _sse(self, url: str, json: dict, headers: dict | None = None) -> AsyncIterator[str]:
        """POST and yield each ``data:`` payload of a server-sent-event stream (or each NDJSON line)."""
        async with httpx.AsyncClient(timeout=self.timeout) as client, \
                client.stream("POST", url, json=json, headers=headers or {}) as r:
            if r.status_code >= 400:
                body = (await r.aread()).decode(errors="replace")
                raise ProviderError(f"{self.name} HTTP {r.status_code}: {body[:300]}")
            async for line in r.aiter_lines():
                line = line.strip()
                if not line or line.startswith(("event:", ":")):
                    continue
                yield line[5:].strip() if line.startswith("data:") else line


class OpenAICompatibleProvider(Provider):
    """OpenAI and any OpenAI-compatible endpoint (Groq, Together, OpenRouter, vLLM...)."""

    def __init__(self, name: str, base_url: str, api_key: str | None, timeout: float = 60.0,
                 max_tokens_field: str = "max_tokens"):
        super().__init__(timeout)
        self.name, self.base_url, self.api_key = name, base_url.rstrip("/"), api_key
        self.max_tokens_field = max_tokens_field

    def configured(self) -> bool:
        return bool(self.api_key)

    def _body(self, model, messages, max_tokens, options):
        body = {"model": model.model, "messages": messages, self.max_tokens_field: max_tokens}
        for k in ("tools", "tool_choice", "temperature", "top_p", "stop"):
            if (options or {}).get(k) is not None:
                body[k] = options[k]
        return body

    async def complete(self, model, messages, max_tokens, options=None):
        if model.api == "responses":
            return await self._responses(model, messages, max_tokens)
        t0 = time.perf_counter()
        data = await self._post(f"{self.base_url}/chat/completions", self._body(model, messages, max_tokens, options),
                                {"Authorization": f"Bearer {self.api_key}"})
        usage = data.get("usage") or {}
        choice = data["choices"][0]
        msg = choice["message"]
        return ProviderResult((msg.get("content") or "").strip(), usage.get("prompt_tokens", 0),
                              usage.get("completion_tokens", 0), (time.perf_counter() - t0) * 1000,
                              tool_calls=msg.get("tool_calls") or None,
                              finish_reason=choice.get("finish_reason") or "stop")

    async def stream(self, model, messages, max_tokens, options=None):
        if model.api == "responses" or (options or {}).get("tools"):
            async for ev in super().stream(model, messages, max_tokens, options):
                yield ev
            return
        t0 = time.perf_counter()
        body = self._body(model, messages, max_tokens, options) | {
            "stream": True, "stream_options": {"include_usage": True}}
        text, usage, finish = [], {}, "stop"
        async for data in self._sse(f"{self.base_url}/chat/completions", body,
                                    {"Authorization": f"Bearer {self.api_key}"}):
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            for ch in chunk.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    text.append(piece)
                    yield StreamEvent(text=piece)
                finish = ch.get("finish_reason") or finish
        full = "".join(text)
        yield StreamEvent(done=True, result=ProviderResult(
            full.strip(), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            (time.perf_counter() - t0) * 1000, finish_reason=finish))

    async def _responses(self, model, messages, max_tokens):
        """OpenAI Responses API, for models not served on /chat/completions (the *-pro models)."""
        t0 = time.perf_counter()
        system, rest = _split_system(messages)
        body = {"model": model.model, "input": [{"role": m["role"], "content": m.get("content") or ""} for m in rest],
                "max_output_tokens": max_tokens}
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
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str | None, timeout: float = 60.0):
        super().__init__(timeout)
        self.api_key = api_key

    def configured(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _messages(rest: list[dict]) -> list[dict]:
        """OpenAI messages → Anthropic: tool calls become tool_use blocks, tool replies tool_result blocks."""
        out: list[dict] = []
        for m in rest:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""), "content": m.get("content") or ""}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)  # consecutive tool results share one user turn
                else:
                    out.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                            "input": _args(c["function"].get("arguments"))} for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def _body(self, model, messages, max_tokens, options):
        system, rest = _split_system(messages)
        o = options or {}
        body: dict = {"model": model.model, "max_tokens": max_tokens, "messages": self._messages(rest)}
        if system:
            body["system"] = system
        if o.get("temperature") is not None:
            body["temperature"] = min(1.0, o["temperature"])
        if o.get("top_p") is not None:
            body["top_p"] = o["top_p"]
        if o.get("stop"):
            body["stop_sequences"] = [o["stop"]] if isinstance(o["stop"], str) else o["stop"]
        if o.get("tools"):
            body["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                              "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                             for t in o["tools"] if t.get("type", "function") == "function"]
            tc = o.get("tool_choice")
            if tc == "required":
                body["tool_choice"] = {"type": "any"}
            elif tc == "none":
                body["tool_choice"] = {"type": "none"}
            elif isinstance(tc, dict) and tc.get("function"):
                body["tool_choice"] = {"type": "tool", "name": tc["function"]["name"]}
        return body

    def _headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    async def complete(self, model, messages, max_tokens, options=None):
        t0 = time.perf_counter()
        data = await self._post(self.URL, self._body(model, messages, max_tokens, options), self._headers())
        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [_tool_call(b["name"], b.get("input") or {}, b.get("id")) for b in blocks if b.get("type") == "tool_use"]
        u = data.get("usage") or {}
        stop = data.get("stop_reason")
        finish = "tool_calls" if stop == "tool_use" else "length" if stop == "max_tokens" else "stop"
        return ProviderResult(text.strip(), u.get("input_tokens", 0), u.get("output_tokens", 0),
                              (time.perf_counter() - t0) * 1000, tool_calls=calls or None, finish_reason=finish)

    async def stream(self, model, messages, max_tokens, options=None):
        if (options or {}).get("tools"):
            async for ev in super().stream(model, messages, max_tokens, options):
                yield ev
            return
        t0 = time.perf_counter()
        body = self._body(model, messages, max_tokens, options) | {"stream": True}
        text, in_tok, out_tok, finish = [], 0, 0, "stop"
        async for data in self._sse(self.URL, body, self._headers()):
            ev = json.loads(data)
            kind = ev.get("type")
            if kind == "message_start":
                in_tok = ((ev.get("message") or {}).get("usage") or {}).get("input_tokens", 0)
            elif kind == "content_block_delta" and (ev.get("delta") or {}).get("type") == "text_delta":
                piece = ev["delta"].get("text", "")
                text.append(piece)
                yield StreamEvent(text=piece)
            elif kind == "message_delta":
                out_tok = (ev.get("usage") or {}).get("output_tokens", out_tok)
                if (ev.get("delta") or {}).get("stop_reason") == "max_tokens":
                    finish = "length"
            elif kind == "error":
                raise ProviderError(f"anthropic stream error: {ev.get('error')}")
        yield StreamEvent(done=True, result=ProviderResult(
            "".join(text).strip(), in_tok, out_tok, (time.perf_counter() - t0) * 1000, finish_reason=finish))


def _gemini_schema(schema):
    """Gemini accepts an OpenAPI subset: drop JSON-Schema keys it rejects."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items()
                if k not in ("additionalProperties", "$schema", "$defs", "strict")}
    if isinstance(schema, list):
        return [_gemini_schema(x) for x in schema]
    return schema


class GeminiProvider(Provider):
    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str | None, timeout: float = 60.0):
        super().__init__(timeout)
        self.api_key = api_key

    def configured(self) -> bool:
        return bool(self.api_key)

    def _body(self, messages, max_tokens, options):
        system, rest = _split_system(messages)
        o = options or {}
        names = {c["id"]: c["function"]["name"] for m in rest for c in (m.get("tool_calls") or [])}
        contents = []
        for m in rest:
            if m["role"] == "tool":
                part = {"functionResponse": {"name": m.get("name") or names.get(m.get("tool_call_id"), "tool"),
                                             "response": {"content": m.get("content") or ""}}}
                contents.append({"role": "user", "parts": [part]})
            elif m["role"] == "assistant":
                parts = [{"text": m["content"]}] if m.get("content") else []
                parts += [{"functionCall": {"name": c["function"]["name"], "args": _args(c["function"].get("arguments"))}}
                          for c in m.get("tool_calls") or []]
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            else:
                contents.append({"role": "user", "parts": [{"text": m.get("content") or ""}]})
        gen: dict = {"maxOutputTokens": max_tokens}
        if o.get("temperature") is not None:
            gen["temperature"] = o["temperature"]
        if o.get("top_p") is not None:
            gen["topP"] = o["top_p"]
        if o.get("stop"):
            gen["stopSequences"] = [o["stop"]] if isinstance(o["stop"], str) else o["stop"]
        body: dict = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if o.get("tools"):
            body["tools"] = [{"functionDeclarations": [
                {"name": t["function"]["name"], "description": t["function"].get("description", ""),
                 **({"parameters": _gemini_schema(t["function"]["parameters"])} if t["function"].get("parameters") else {})}
                for t in o["tools"] if t.get("type", "function") == "function"]}]
            mode = {"required": "ANY", "none": "NONE"}.get(o.get("tool_choice"), "AUTO")
            body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
        return body

    @staticmethod
    def _usage(data) -> tuple[int, int]:
        u = data.get("usageMetadata") or {}
        return u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)

    async def complete(self, model, messages, max_tokens, options=None):
        t0 = time.perf_counter()
        data = await self._post(f"{self.BASE}/{model.model}:generateContent", self._body(messages, max_tokens, options),
                                {"x-goog-api-key": self.api_key})
        cand = (data.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        calls = [_tool_call(p["functionCall"]["name"], p["functionCall"].get("args") or {})
                 for p in parts if "functionCall" in p]
        in_tok, out_tok = self._usage(data)
        finish = "tool_calls" if calls else "length" if cand.get("finishReason") == "MAX_TOKENS" else "stop"
        return ProviderResult("".join(p.get("text", "") for p in parts if not p.get("thought")).strip(),
                              in_tok, out_tok, (time.perf_counter() - t0) * 1000,
                              tool_calls=calls or None, finish_reason=finish)

    async def stream(self, model, messages, max_tokens, options=None):
        if (options or {}).get("tools"):
            async for ev in super().stream(model, messages, max_tokens, options):
                yield ev
            return
        t0 = time.perf_counter()
        url = f"{self.BASE}/{model.model}:streamGenerateContent?alt=sse"
        text, in_tok, out_tok, finish = [], 0, 0, "stop"
        async for data in self._sse(url, self._body(messages, max_tokens, options), {"x-goog-api-key": self.api_key}):
            chunk = json.loads(data)
            cand = (chunk.get("candidates") or [{}])[0]
            for p in cand.get("content", {}).get("parts", []):
                if p.get("text") and not p.get("thought"):
                    text.append(p["text"])
                    yield StreamEvent(text=p["text"])
            if cand.get("finishReason") == "MAX_TOKENS":
                finish = "length"
            if chunk.get("usageMetadata"):
                in_tok, out_tok = self._usage(chunk)
        yield StreamEvent(done=True, result=ProviderResult(
            "".join(text).strip(), in_tok, out_tok, (time.perf_counter() - t0) * 1000, finish_reason=finish))


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

    def _body(self, model, messages, max_tokens, options, stream):
        o = options or {}
        msgs = []
        for m in messages:
            mm = {"role": m["role"], "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = [{"function": {"name": c["function"]["name"],
                                                  "arguments": _args(c["function"].get("arguments"))}}
                                    for c in m["tool_calls"]]
            msgs.append(mm)
        opts = {"num_predict": max_tokens}
        for k in ("temperature", "top_p", "stop"):
            if o.get(k) is not None:
                opts[k] = o[k]
        body = {"model": model.model, "messages": msgs, "stream": stream, "options": opts}
        if o.get("tools"):
            body["tools"] = o["tools"]
        return body

    async def complete(self, model, messages, max_tokens, options=None):
        t0 = time.perf_counter()
        data = await self._post(f"{self.host}/api/chat", self._body(model, messages, max_tokens, options, False))
        msg = data.get("message") or {}
        calls = [_tool_call(c["function"]["name"], c["function"].get("arguments") or {})
                 for c in msg.get("tool_calls") or []]
        return ProviderResult(msg.get("content", "").strip(), data.get("prompt_eval_count", 0),
                              data.get("eval_count", 0), (time.perf_counter() - t0) * 1000,
                              tool_calls=calls or None, finish_reason="tool_calls" if calls else "stop")

    async def stream(self, model, messages, max_tokens, options=None):
        if (options or {}).get("tools"):
            async for ev in super().stream(model, messages, max_tokens, options):
                yield ev
            return
        t0 = time.perf_counter()
        text, last = [], {}
        async for line in self._sse(f"{self.host}/api/chat", self._body(model, messages, max_tokens, options, True)):
            last = json.loads(line)
            piece = (last.get("message") or {}).get("content", "")
            if piece:
                text.append(piece)
                yield StreamEvent(text=piece)
        yield StreamEvent(done=True, result=ProviderResult(
            "".join(text).strip(), last.get("prompt_eval_count", 0), last.get("eval_count", 0),
            (time.perf_counter() - t0) * 1000, finish_reason="length" if last.get("done_reason") == "length" else "stop"))


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

    async def complete(self, model, messages, max_tokens, options=None):
        prompt = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
        in_tok = sum(estimate_tokens(m.get("content") or "") for m in messages)
        rng = random.Random(zlib.crc32(f"{model.id}|{prompt}".encode()))
        out_tok = max(8, int(max_tokens * rng.uniform(0.7, 1.3)))
        latency = 150 + out_tok / max(model.tps, 1) * 1000
        if self.delay:
            await asyncio.sleep(min(latency / 1000, 0.25))  # keep demos snappy
        snippet = prompt[:120] + ("…" if len(prompt) > 120 else "")
        tools = (options or {}).get("tools")
        text = (f"[Simulated response from {model.display_name}]\n\n"
                f"EcoRoute is running in simulate mode, so no real model was called. In live mode "
                f"this prompt would be answered by {model.display_name} ({model.provider})"
                + (f", with {len(tools)} tool(s) available to it" if tools else "") + ".\n\n"
                f"Prompt: \"{snippet}\"\n\n"
                f"Simulated length: {out_tok} output tokens. Add an API key or start Ollama to get "
                f"real answers.")
        return ProviderResult(text, in_tok, out_tok, latency, simulated=True)

    async def stream(self, model, messages, max_tokens, options=None):
        delay, self.delay = self.delay, False
        try:
            res = await self.complete(model, messages, max_tokens, options)
        finally:
            self.delay = delay
        pieces = list(_chunks(res.text))
        for piece in pieces:
            if self.delay:
                await asyncio.sleep(min(0.6 / len(pieces), 0.03))
            yield StreamEvent(text=piece)
        yield StreamEvent(done=True, result=res)


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
