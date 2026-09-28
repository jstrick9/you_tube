"""Free-tier LLM router with automatic failover.

Order (configurable): Gemini (Google AI Studio free key) → Groq (free key) →
OpenRouter ':free' models (free key) → Pollinations (no key, last resort).
Any provider without credentials is skipped. Every call asks for strict JSON
and is parsed defensively, so a flaky model never crashes the pipeline.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import time
from typing import Any

from .common import http

log = logging.getLogger("autotube.llm")


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> Any:
    text = text.strip()
    # strip ```json fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find the outermost {...} or [...]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            chunk = text[start:end + 1]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                # tolerate trailing commas
                chunk2 = re.sub(r",\s*([}\]])", r"\1", chunk)
                try:
                    return json.loads(chunk2)
                except json.JSONDecodeError:
                    continue
    repaired = _repair_truncated(text)
    if repaired is not None:
        return repaired
    raise LLMError(f"Could not parse JSON from model output: {text[:300]}")


def _repair_truncated(text: str):
    """Salvage JSON cut off mid-stream: drop the dangling partial element, close brackets."""
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        return None
    s = text[start:]
    for cut in range(len(s), max(len(s) - 4000, 0), -1):
        if s[cut - 1] not in "}]\"0123456789el":   # plausible value endings
            continue
        chunk = s[:cut]
        stack, in_str, esc = [], False, False
        for ch in chunk:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        if in_str:
            continue
        candidate = re.sub(r",\s*$", "", chunk) + "".join(reversed(stack))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# ── providers ─────────────────────────────────────────────────────────────────
def _gemini(model: str, system: str, user: str, temperature: float, images: list[bytes] | None = None) -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("no GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    parts: list[dict] = [{"text": user}]
    for img in images or []:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img).decode()}})
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
    }
    r = http().post(url, params={"key": key}, json=body, timeout=90)
    if r.status_code != 200:
        raise LLMError(f"gemini {model} HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError) as e:
        raise LLMError(f"gemini bad response: {str(data)[:200]}") from e


def _openai_compatible(base: str, key: str | None, model: str, system: str, user: str,
                       temperature: float, json_mode: bool = True, extra_headers: dict | None = None,
                       extra_body: dict | None = None, images: list[bytes] | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if extra_headers:
        headers.update(extra_headers)
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user if not images else
                     [{"type": "text", "text": user}] + [
                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(i).decode()}}
                         for i in images]}],
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if extra_body:
        body.update(extra_body)
    r = http().post(base, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise LLMError(f"{base} {model} HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        content = data["choices"][0]["message"].get("content") or ""
        if not content.strip():
            raise LLMError(f"empty content (finish={data['choices'][0].get('finish_reason')})")
        return content
    except (KeyError, IndexError) as e:
        raise LLMError(f"bad response: {str(data)[:200]}") from e


def _groq(model, system, user, temperature, images=None):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise LLMError("no GROQ_API_KEY")
    return _openai_compatible("https://api.groq.com/openai/v1/chat/completions", key, model, system, user, temperature,
                              images=images)


def _openrouter(model, system, user, temperature, images=None):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("no OPENROUTER_API_KEY")
    return _openai_compatible("https://openrouter.ai/api/v1/chat/completions", key, model, system, user,
                              temperature, extra_headers={"X-Title": "AutoTube"}, images=images)


def _pollinations(model, system, user, temperature):
    # Keyless community endpoint. An optional token raises limits.
    key = os.environ.get("POLLINATIONS_TOKEN")
    # Its anonymous tier is a reasoning model: keep reasoning short so tokens go to the answer.
    return _openai_compatible("https://text.pollinations.ai/openai", key, model, system, user, temperature,
                              json_mode=False, extra_body={"max_tokens": 16000, "reasoning_effort": "low",
                                                           "seed": random.randint(1, 10**9)})  # seed defeats response cache


PROVIDERS = {
    "gemini": (_gemini, "gemini_models"),
    "groq": (_groq, "groq_models"),
    "openrouter": (_openrouter, "openrouter_models"),
    "pollinations": (_pollinations, "pollinations_models"),
}


class LLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg.get("llm", {})
        self.order = self.cfg.get("providers", list(PROVIDERS))
        self.temperature = float(self.cfg.get("temperature", 0.8))
        self._dead: set[str] = set()   # provider/model combos that hard-failed this run
        self.last_used = ""
        keyed = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
        # "lite" = only the keyless fallback is available → callers send smaller prompts
        self.lite = not any(os.environ.get(keyed[p]) for p in self.order if p in keyed)

    def vision_available(self) -> bool:
        keyed = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
        return any(os.environ.get(keyed.get(p, "-")) and self._vision_models(p)
                   for p in self.cfg.get("vision_providers", ["gemini", "groq", "openrouter"]))

    def _vision_models(self, pname: str) -> list[str]:
        default = self.cfg.get("gemini_models", []) if pname == "gemini" else []
        return self.cfg.get(f"{pname}_vision_models", default)

    def vision_json(self, system: str, user: str, images: list[bytes], validate=None) -> Any:
        """Like json(), but sends JPEG images to vision-capable models only (never the keyless tier)."""
        order = [p for p in self.cfg.get("vision_providers", ["gemini", "groq", "openrouter"]) if p in PROVIDERS]
        return self.json(system, user, temperature=0.1, validate=validate, images=images, order=order,
                         models_of=self._vision_models)

    def json(self, system: str, user: str, temperature: float | None = None,
             validate=None, attempts_per_model: int = 2, images: list[bytes] | None = None,
             order: list[str] | None = None, models_of=None) -> Any:
        """Return parsed JSON. `validate(obj)` may raise to force a retry/failover."""
        temp = self.temperature if temperature is None else temperature
        errors = []
        for pname in (order or self.order):
            fn, models_key = PROVIDERS[pname]
            models = models_of(pname) if models_of else self.cfg.get(models_key, [])
            for model in models:
                tag = f"{pname}:{model}" + (":vision" if images else "")
                if tag in self._dead:
                    continue
                feedback = ""
                for attempt in range(attempts_per_model):
                    try:
                        prompt = user + (f"\n\nIMPORTANT — your previous answer was rejected: {feedback}. Fix this."
                                         if feedback else "")
                        raw = fn(model, system, prompt, temp, images) if images else fn(model, system, prompt, temp)
                        obj = _extract_json(raw)
                        if validate:
                            validate(obj)
                        self.last_used = tag
                        log.debug("LLM ok via %s", tag)
                        return obj
                    except LLMError as e:
                        msg = str(e)
                        errors.append(f"{tag}: {msg[:160]}")
                        if msg.startswith("no ") and "_API_KEY" in msg:
                            self._dead.add(tag)
                            break
                        if "HTTP 404" in msg or "HTTP 400" in msg or "HTTP 401" in msg or "HTTP 403" in msg:
                            self._dead.add(tag)
                            break
                        if "HTTP 429" in msg or "HTTP 402" in msg:
                            time.sleep(6 + 6 * attempt)
                        elif re.search(r"HTTP 50[0234]", msg):      # overloaded / transient
                            time.sleep(8 + 8 * attempt)
                        else:
                            time.sleep(1.5)
                    except Exception as e:  # validation or network
                        errors.append(f"{tag}: {type(e).__name__}: {str(e)[:160]}")
                        if isinstance(e, AssertionError):
                            feedback = str(e)
                        time.sleep(1)
        raise LLMError("All LLM providers failed:\n  " + "\n  ".join(errors[-12:]))
