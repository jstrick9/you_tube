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


def _retry_after(msg: str, default: float) -> float:
    """Honour a Retry-After the provider actually told us, when it is sane.

    Guessing 6 seconds against a provider asking for 2 wastes the run's time budget; guessing 6
    against one asking for 60 just fails again. Capped so a hostile or broken header cannot stall
    the whole run.
    """
    m = re.search(r'retry[-_ ]?after["\s:]+(\d+(?:\.\d+)?)', msg, re.I)
    if m:
        try:
            return max(1.0, min(45.0, float(m.group(1))))
        except ValueError:
            pass
    return default


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
                         models_of=self._vision_models, attempts_per_model=1)

    def _review_models(self, pname: str) -> list[str]:
        """Models used for fact-checking/review. Falls back to the normal list for that provider."""
        return self.cfg.get(f"{pname}_review_models", self.cfg.get(f"{pname}_models", []))

    def review_json(self, system: str, user: str, avoid: str | None = None, **kw) -> Any:
        """Review on an INDEPENDENT model: a writer grading its own draft tends to agree with itself.

        Tries `llm.review_providers` (a deliberately different order from the writer's) and skips the
        exact provider:model that produced the draft. If that leaves nothing reachable, it falls back
        to the normal order rather than failing — an imperfect review beats no review, and the caller
        is told via `last_used` whether the reviewer was actually independent.
        """
        order = [p for p in self.cfg.get("review_providers", []) if p in PROVIDERS] or list(self.order)
        try:
            return self.json(system, user, order=order, models_of=self._review_models, avoid=avoid, **kw)
        except LLMError as e:
            if not avoid:
                raise
            # Log WHY, not just that it happened. This message used to be bare, and the
            # accumulated per-model errors - the only record of which provider refused
            # and with what status - were discarded with the exception. Diagnosing a
            # lost reviewer then took a full live run per guess. Independence feeds the
            # provenance dossier, so a silent fallback is the one failure here that must
            # never be quiet.
            log.warning("no independent reviewer reachable, falling back to the full "
                        "provider list — reviewer attempts failed with: %s", str(e)[:600])
            return self.json(system, user, **kw)

    def json(self, system: str, user: str, temperature: float | None = None,
             validate=None, attempts_per_model: int = 2, images: list[bytes] | None = None,
             order: list[str] | None = None, models_of=None, avoid: str | None = None) -> Any:
        """Return parsed JSON. `validate(obj)` may raise to force a retry/failover.

        `avoid` is a provider:model tag to skip (used to keep the reviewer independent of the writer).
        """
        temp = self.temperature if temperature is None else temperature
        errors = []
        for pname in (order or self.order):
            fn, models_key = PROVIDERS[pname]
            models = models_of(pname) if models_of else self.cfg.get(models_key, [])
            for model in models:
                tag = f"{pname}:{model}" + (":vision" if images else "")
                if tag in self._dead or (avoid and tag == avoid):
                    continue
                feedback = ""
                for attempt in range(attempts_per_model):
                    last_try = attempt == attempts_per_model - 1
                    try:
                        prompt = user + (f"\n\nIMPORTANT — your previous answer was rejected: {feedback}. Fix this."
                                         if feedback else "")
                        raw = fn(model, system, prompt, temp, images) if images else fn(model, system, prompt, temp)
                        obj = _extract_json(raw)
                        if validate:
                            validate(obj)
                        self.last_used = tag
                        log.debug("LLM ok via %s", tag)
                        if avoid:            # a review call: say who graded it
                            log.info("  reviewed by %s (writer was %s)", tag, avoid)
                        return obj
                    except LLMError as e:
                        msg = str(e)
                        errors.append(f"{tag}: {msg[:160]}")
                        if msg.startswith("no ") and "_API_KEY" in msg:
                            self._dead.add(tag)
                            break

                        if "HTTP 404" in msg or "HTTP 400" in msg or "HTTP 401" in msg or "HTTP 403" in msg:
                            # Retiring a model for the rest of the run is right - a 400 will
                            # not fix itself - but it was silent, and a model skipped via
                            # `continue` appends no error. A provider could therefore die on
                            # its first call inside a request that then succeeded elsewhere,
                            # and every later call would skip it with no trace at all. That is
                            # how the reviewer was lost without a single Groq line in the log.
                            log.warning("retiring %s for this run: %s", tag, msg[:200])
                            self._dead.add(tag)
                            break
                        # Only back off if we are going to try this model again. Sleeping on the
                        # final attempt just delayed the failover: 12s per rate-limited model,
                        # across every model of every provider, out of a 120-minute run budget.
                        if not last_try:
                            if "HTTP 429" in msg or "HTTP 402" in msg:
                                time.sleep(_retry_after(msg, 6 + 6 * attempt))
                            elif re.search(r"HTTP 50[0234]", msg):      # overloaded / transient
                                time.sleep(8 + 8 * attempt)
                            else:
                                time.sleep(1.5)
                    except Exception as e:  # validation or network
                        errors.append(f"{tag}: {type(e).__name__}: {str(e)[:160]}")
                        if isinstance(e, AssertionError):
                            feedback = str(e)
                        if not last_try:
                            time.sleep(1)
        # An empty `errors` list is not "nothing went wrong" - it means every candidate
        # was skipped before being called, because it was already retired or was the
        # model being avoided. Reporting that distinctly is the difference between a
        # diagnosable failure and a dead end.
        if not errors:
            tried = [f"{p}:{m}" for p in (order or self.order)
                     for m in (models_of(p) if models_of else self.cfg.get(PROVIDERS[p][1], []))]
            skipped = [t for t in tried if t in self._dead]
            raise LLMError(
                "All LLM providers failed: no model was even attempted. "
                f"candidates={tried or '[none configured]'} "
                f"already-retired={skipped or '[]'} avoiding={avoid or '-'}")
        raise LLMError("All LLM providers failed:\n  " + "\n  ".join(errors[-12:]))
