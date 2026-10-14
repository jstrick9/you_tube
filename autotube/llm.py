"""Configurable LLM router with automatic, per-run provider/model failover.

Providers are tried in the order in config.yaml; missing keys, permanent HTTP errors,
and payment-required models are skipped for that run. Every call asks for strict JSON
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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from . import run_report
from .common import http
from .provider_health import (DEFAULT_402_COOLDOWN_SECONDS, DEFAULT_429_COOLDOWN_SECONDS,
                              ProviderCircuitBreakers)

log = logging.getLogger("autotube.llm")


def _retry_after(msg: str, default: float, retry_after_seconds: float | None = None) -> float:
    """Honour a Retry-After the provider actually told us, when it is sane.

    Guessing 6 seconds against a provider asking for 2 wastes the run's time budget; guessing 6
    against one asking for 60 just fails again. Capped so a hostile or broken header cannot stall
    the whole run.
    """
    if isinstance(retry_after_seconds, (int, float)) and retry_after_seconds >= 0:
        return max(1.0, min(45.0, float(retry_after_seconds)))
    m = re.search(r'retry[-_ ]?after["\s:]+(\d+(?:\.\d+)?)', msg, re.I)
    if m:
        try:
            return max(1.0, min(45.0, float(m.group(1))))
        except ValueError:
            pass
    return default


def advertised_retry(msg: str) -> float:
    """The provider's Retry-After as actually stated, uncapped. 0 if it said nothing.

    _retry_after clamps to 45s so a bad header cannot stall a run, which is right for
    deciding how long to sleep but wrong for deciding whether to wait at all. A model
    that says "retry after 1100" is telling us it is gone for the next twenty minutes;
    sleeping 45 seconds and asking again just burns the run's time budget.
    """
    m = re.search(r'retry[-_ ]?after[\"\s:]+(\d+(?:\.\d+)?)\s*(ms)?', msg, re.I)
    if not m:
        return 0.0
    try:
        secs = float(m.group(1))
    except ValueError:
        return 0.0
    return secs / 1000.0 if m.group(2) else secs


def _request_too_large(message: str) -> bool:
    """Recognize model input/context limits even when a provider labels them HTTP 429.

    These are model/request-size failures, not quota exhaustion: cooling down the whole
    provider strands healthy models and needlessly persists a rate-limit block.
    """
    text = str(message or "").lower()
    return any(marker in text for marker in (
        "request too large", "prompt too long", "input too long", "context length exceeded",
        "maximum context length", "too many tokens in the input", "input token limit",
    ))


def _retry_after_header(value: str | None) -> float | None:
    """Parse standard Retry-After delay-seconds or HTTP-date without guessing at resets."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
        if seconds > 100_000_000:  # tolerate providers that send an epoch timestamp in this header
            seconds -= datetime.now(timezone.utc).timestamp()
        return max(0.0, seconds)
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        pass
    # Providers such as Groq expose reset windows as "1m12.5s" or "250ms".
    compact = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?"
                           r"(?:(\d+(?:\.\d+)?)(ms|s))?", text, re.I)
    if compact and any(compact.groups()):
        hours, minutes, tail, unit = compact.groups()
        seconds = float(hours or 0) * 3600 + float(minutes or 0) * 60 + float(tail or 0)
        return seconds / 1000 if unit and unit.lower() == "ms" else seconds
    return None


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None,
                 retry_after_seconds: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


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
        # Same treatment as the OpenAI-compatible path: keep enough of the body to see
        # whether a 429 is a per-minute wait or a spent daily quota, and record it so the
        # run can tell "rate limited" apart from "broken".
        hint = r.headers.get("retry-after")
        raise LLMError(f"gemini {model} HTTP {r.status_code}"
                       + (f" (retry after {hint})" if hint else "") + f": {r.text[:600]}",
                       status_code=r.status_code, retry_after_seconds=_retry_after_header(hint))
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
    if r.status_code == 400 and json_mode and "json_validate" in r.text.lower():
        # Reasoning models (Groq's gpt-oss family) reject response_format=json_object:
        # they emit a reasoning preamble that cannot satisfy strict JSON mode, so the
        # server rejects the request outright. The prompt already demands JSON and the
        # caller already tolerates fenced output, so drop the constraint and ask again.
        # Without this the entire Groq provider 400s on every call, which is what left
        # the review stage with no model but the writer's own.
        body.pop("response_format", None)
        r = http().post(base, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        # 200 chars cut these messages off mid-sentence, immediately before the part that
        # matters: a 429 body says whether the limit is per-minute or per-day and when it
        # resets, and that is the difference between "wait 30 seconds" and "the quota is
        # gone until tomorrow, stop retrying". Keep the retry hint too.
        detail = r.text[:600]
        hint = r.headers.get("retry-after") or r.headers.get("x-ratelimit-reset-requests")
        retry_after = _retry_after_header(hint)
        raise LLMError(f"{base} {model} HTTP {r.status_code}"
                       + (f" (retry after {hint})" if hint else "") + f": {detail}",
                       status_code=r.status_code, retry_after_seconds=retry_after)
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
    # Bound the completion budget so small-context Groq models do not reserve a
    # provider-default maximum larger than the prompt can leave room for.
    return _openai_compatible("https://api.groq.com/openai/v1/chat/completions", key, model, system, user, temperature,
                              extra_body={"max_tokens": 2048}, images=images)


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


def _cerebras(model, system, user, temperature, images=None):
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        raise LLMError("no CEREBRAS_API_KEY")
    return _openai_compatible("https://api.cerebras.ai/v1/chat/completions", key, model, system, user,
                              temperature, images=images)


def _mistral(model, system, user, temperature, images=None):
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise LLMError("no MISTRAL_API_KEY")
    return _openai_compatible("https://api.mistral.ai/v1/chat/completions", key, model, system, user,
                              temperature, images=images)


def cloudflare_creds() -> tuple[str | None, str | None]:
    """Token and account id, accepting the common "Cloudfare" misspelling.

    Both spellings are honoured on purpose. The secrets were created as CLOUDFARE_*,
    the code read CLOUDFLARE_*, and the result was a provider that looked configured,
    reported nothing, and silently never ran - the exact failure mode that has cost
    this repo the most time. Accepting both is one line; a silent mismatch is a day.
    """
    key = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFARE_API_TOKEN")
    acct = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or os.environ.get("CLOUDFARE_ACCOUNT_ID")
    return key, acct


def _nvidia(model, system, user, temperature, images=None):
    """NVIDIA NIM. Free tier, no card, OpenAI-compatible, and it serves vision models."""
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise LLMError("no NVIDIA_API_KEY")
    return _openai_compatible("https://integrate.api.nvidia.com/v1/chat/completions", key, model, system,
                              user, temperature, images=images)


def _cloudflare(model, system, user, temperature, images=None):
    """Workers AI. 10,000 neurons/day free, no card, and it hard-blocks instead of billing."""
    key, acct = cloudflare_creds()
    if not (key and acct):
        raise LLMError("no CLOUDFLARE_API_TOKEN")
    return _openai_compatible(f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions",
                              key, model, system, user, temperature, images=images)


# Which env var gates each provider. Kept in one place because it was previously
# duplicated inline in two methods, and a provider added to one but not the other is
# invisible until something silently never gets tried.
def _provider_key(pname: str) -> str | None:
    """Is this provider actually usable? Cloudflare needs two values, not one."""
    if pname == "cloudflare":
        key, acct = cloudflare_creds()
        return key if (key and acct) else None
    env = KEYED_PROVIDERS.get(pname)
    return os.environ.get(env) if env else None


KEYED_PROVIDERS = {
    "gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY",
    "cerebras": "CEREBRAS_API_KEY", "mistral": "MISTRAL_API_KEY",
    "cloudflare": "CLOUDFLARE_API_TOKEN", "nvidia": "NVIDIA_API_KEY",
}

PROVIDERS = {
    "gemini": (_gemini, "gemini_models"),
    "groq": (_groq, "groq_models"),
    "openrouter": (_openrouter, "openrouter_models"),
    "cerebras": (_cerebras, "cerebras_models"),
    "mistral": (_mistral, "mistral_models"),
    "cloudflare": (_cloudflare, "cloudflare_models"),
    "nvidia": (_nvidia, "nvidia_models"),
    "pollinations": (_pollinations, "pollinations_models"),
}


class LLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg.get("llm", {})
        self.order = self.cfg.get("providers", list(PROVIDERS))
        self.temperature = float(self.cfg.get("temperature", 0.8))
        self._dead: set[str] = set()   # provider/model combos that hard-failed this run
        self._blocked_providers: set[str] = set()
        self.circuits = ProviderCircuitBreakers()
        self.last_used = ""
        keyed = KEYED_PROVIDERS
        # "lite" = only the keyless fallback is available → callers send smaller prompts
        self.lite = not any(_provider_key(p) for p in self.order if p in keyed)
        receipt = run_report.current_receipt()
        for provider, row in self.circuits.snapshot().items():
            if row.get("reason") == "rate_limited":
                note_rate_limited(f"{provider}:provider_cooldown")
            if receipt:
                receipt.note_provider_event(provider, row.get("status_code"), "cooldown_skip",
                                            row.get("blocked_until"))

    def _breaker(self) -> ProviderCircuitBreakers:
        if not hasattr(self, "circuits"):
            self.circuits = ProviderCircuitBreakers()
        return self.circuits

    def _provider_open(self, provider: str) -> bool:
        return provider in getattr(self, "_blocked_providers", set()) or self._breaker().is_open(provider)

    def _open_provider(self, provider: str, status: int | None, reason: str, seconds: float,
                       retry_after: float | None = None) -> None:
        if not hasattr(self, "_blocked_providers"):
            self._blocked_providers = set()
        self._blocked_providers.add(provider)
        until = self._breaker().open(provider, status, reason, seconds, retry_after)
        receipt = run_report.current_receipt()
        if receipt:
            receipt.note_provider_event(provider, status, "circuit_open", until, retry_after)
        log.warning("provider %s circuit open until %s (%s)", provider, until, reason)

    def has_available_provider(self) -> bool:
        """Whether another configured text provider can still be attempted this run."""
        dead = getattr(self, "_dead", set())
        for provider in self.order:
            if provider not in PROVIDERS or self._provider_open(provider):
                continue
            _, models_key = PROVIDERS[provider]
            models = self.cfg.get(models_key, []) or []
            if (provider == "pollinations" or _provider_key(provider)) and any(
                    f"{provider}:{model}" not in dead for model in models):
                return True
        return False

    def vision_available(self) -> bool:
        return any(_provider_key(p) and self._vision_models(p) and not self._provider_open(p)
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
        if not hasattr(self, "_dead"):
            self._dead = set()
        if not hasattr(self, "_blocked_providers"):
            self._blocked_providers = set()
        circuits = self._breaker()
        receipt = run_report.current_receipt()
        for pname in (order or self.order):
            if self._provider_open(pname):
                until = circuits.blocked_until(pname)
                errors.append(f"{pname}: provider circuit open until {until or 'end of this run'}")
                continue
            fn, models_key = PROVIDERS[pname]
            models = models_of(pname) if models_of else self.cfg.get(models_key, [])
            for model in models:
                if self._provider_open(pname):
                    break
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
                        circuits.note_success(pname)
                        self.last_used = tag
                        log.debug("LLM ok via %s", tag)
                        if avoid:
                            log.info("  reviewed by %s (writer was %s)", tag, avoid)
                        return obj
                    except LLMError as e:
                        msg = str(e)
                        errors.append(f"{tag}: {msg[:160]}")
                        status = e.status_code or (int(m.group(1)) if (m := re.search(r"HTTP\s+(\d{3})", msg)) else None)
                        retry_after = e.retry_after_seconds
                        if retry_after is None:
                            retry_after = advertised_retry(msg) or None
                        if msg.startswith("no ") and "_API_KEY" in msg:
                            self._dead.add(tag)
                            break
                        if _request_too_large(msg):
                            # A 429 can mean this specific model cannot fit the prompt,
                            # not that the provider quota is exhausted. Retire only this
                            # model and continue through configured fallbacks.
                            self._dead.add(tag)
                            if receipt:
                                receipt.note_provider_event(pname, status, "model_request_too_large")
                            log.warning("skipping %s after a model input/context limit; trying the next model", tag)
                            break
                        if status == 402:
                            self._dead.add(tag)
                            self._open_provider(pname, status, "billing_or_entitlement",
                                                max(DEFAULT_402_COOLDOWN_SECONDS, retry_after or 0), retry_after)
                            break
                        if status == 429:
                            note_rate_limited(f"{pname}:{model}")
                            # Never retry earlier than the provider's stated wait. If it
                            # exceeds our bounded in-run wait, open a persisted cooldown.
                            if retry_after is not None and retry_after > 45:
                                self._dead.add(tag)
                                self._open_provider(pname, status, "rate_limited",
                                                    retry_after, retry_after)
                                break
                            if not last_try:
                                time.sleep(_retry_after(msg, 6 + 6 * attempt, retry_after))
                                continue
                            self._dead.add(tag)
                            self._open_provider(pname, status, "rate_limited",
                                                max(DEFAULT_429_COOLDOWN_SECONDS, retry_after or 0), retry_after)
                            break
                        if status in (401, 403):
                            self._dead.add(tag)
                            self._blocked_providers.add(pname)
                            if receipt:
                                receipt.note_provider_event(pname, status, "provider_skipped_auth")
                            log.warning("skipping provider %s for this run after HTTP %d", pname, status)
                            break
                        if status == 404 or status == 400:
                            self._dead.add(tag)
                            log.warning("retiring %s for this run: %s", tag, msg[:200])
                            break
                        if status is not None and 500 <= status <= 599:
                            if receipt:
                                receipt.note_provider_event(pname, status, "transient_failure")
                            if retry_after is not None and retry_after > 45:
                                self._dead.add(tag)
                                self._open_provider(pname, status, "transient_retry_after",
                                                    retry_after, retry_after)
                                break
                            opened = circuits.note_5xx(pname, status)
                            if opened:
                                self._blocked_providers.add(pname)
                                if receipt:
                                    row = circuits._entries.get(pname, {})
                                    receipt.note_provider_event(pname, status, "circuit_open",
                                                                row.get("blocked_until"))
                                log.warning("skipping provider %s after repeated HTTP 5xx failures", pname)
                                break
                            if not last_try:
                                time.sleep(_retry_after(msg, 8 + 8 * attempt, retry_after))
                                continue
                            break
                        if not last_try:
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


# ── rate-limit tracking ──────────────────────────────────────────────────────
# Free-tier exhaustion is an expected operating condition for this channel, not a
# defect, but it was indistinguishable from a crash: both ended as a red run that
# produced nothing. Five such runs in one day is how a red X stops meaning anything.
_RATE_LIMITED: set[str] = set()


def note_rate_limited(tag: str) -> None:
    _RATE_LIMITED.add(tag)


def rate_limited() -> set[str]:
    """provider:model tags that returned 429 since the process started."""
    return set(_RATE_LIMITED)


# ── model catalogue ──────────────────────────────────────────────────────────
def available_models(provider: str) -> list[str]:
    """Model ids a provider will actually accept right now, or [] if unknown.

    Three separate outages in this repo have been a configured model id that simply
    does not exist - "verified on Groq Sep 2026" models returning 404 a month later,
    and then replacements that were themselves wrong. A model name is not something
    worth guessing at: every provider publishes the list.
    """
    import os

    from .common import get_json
    try:
        if provider == "groq":
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                return []
            d = get_json("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {key}"})
            return sorted(m["id"] for m in d.get("data", []))
        if provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                return []
            d = get_json("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"key": key, "pageSize": 200})
            # Being listed is not the same as being usable. gemini-2.5-flash appears in
            # the catalogue but 404s on generateContent, so doctor called it OK while
            # live runs retired it on first use - exactly the false reassurance this
            # check exists to prevent. Only count models that support the call we make.
            return sorted(m["name"].removeprefix("models/") for m in d.get("models", [])
                          if "generateContent" in (m.get("supportedGenerationMethods") or []))
        if provider == "openrouter":
            d = get_json("https://openrouter.ai/api/v1/models")
            return sorted(m["id"] for m in d.get("data", []))
        # The OpenAI-compatible providers all expose /models the same way.
        simple = {
            "cerebras": ("https://api.cerebras.ai/v1/models", "CEREBRAS_API_KEY"),
            "mistral": ("https://api.mistral.ai/v1/models", "MISTRAL_API_KEY"),
            "nvidia": ("https://integrate.api.nvidia.com/v1/models", "NVIDIA_API_KEY"),
        }
        if provider in simple:
            url, env = simple[provider]
            key = os.environ.get(env)
            if not key:
                return []
            d = get_json(url, headers={"Authorization": f"Bearer {key}"})
            return sorted(m["id"] for m in d.get("data", []))
        if provider == "cloudflare":
            key, acct = cloudflare_creds()
            if not (key and acct):
                return []
            # Workers AI publishes its catalogue at ai/models/search, not at the
            # OpenAI-compatible ai/v1/models path the chat endpoint lives under. The
            # first guess 404'd, and because the failure was swallowed it looked
            # identical to a missing key.
            d = get_json(f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/models/search",
                         headers={"Authorization": f"Bearer {key}"}, params={"per_page": 200})
            rows = d.get("result") or d.get("data") or []
            return sorted(str(m.get("name") or m.get("id")) for m in rows if (m.get("name") or m.get("id")))
    except Exception as e:  # noqa: BLE001 - diagnostics must never break the caller
        # Reporting these two as one line is how a wrong catalogue URL spent a round
        # trip looking like an absent key. The caller still gets [], but the reason is
        # no longer invisible.
        log.warning("%s catalogue unreachable: %s", provider, str(e)[:200])
        return []
    return []


VISION_HINTS = ("vision", "llava", "pixtral", "-vl", "scout", "maverick", "image-to-text", "gemini")


def vision_candidates(provider: str) -> list[str]:
    """Models in a provider's live catalogue that look vision-capable.

    A hint list rather than a guarantee - providers do not label this consistently -
    but it turns "what else could serve vision?" from a research question into one
    doctor run against the keys actually held.
    """
    return [m for m in available_models(provider) if any(h in m.lower() for h in VISION_HINTS)]


def probe_vision(cfg: dict, provider: str, model: str) -> tuple[bool, str]:
    """Actually call a vision model with a tiny image. Listed is not the same as callable.

    gemini-2.5-flash is in Google's catalogue and 404s; Cloudflare lists Meta's vision
    model and returns 403 on it. Both looked fine to a catalogue check. The only way to
    know a vision model works is to send it an image.
    """
    import base64
    import io

    from .common import http
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, format="JPEG")
        img = buf.getvalue()
    except Exception as e:  # noqa: BLE001
        return False, f"could not build probe image: {e}"
    fn = PROVIDERS[provider][0]
    try:
        out = fn(model, "Reply with JSON only.",
                 'What colour dominates this image? Answer {"colour": "..."}',
                 0.0, images=[img])
        return True, (out or "")[:60].replace("\n", " ")
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:140]


def audit_models(cfg: dict) -> dict:
    """Check every configured model id against what the provider actually offers."""
    llm = cfg.get("llm", {})
    report: dict = {}
    for provider in ("gemini", "groq", "openrouter", "cerebras", "mistral", "cloudflare", "nvidia"):
        catalogue = available_models(provider)
        if not catalogue:
            report[provider] = {"catalogue": None}
            continue
        rows = {}
        for suffix in ("_models", "_review_models", "_vision_models"):
            for m in llm.get(f"{provider}{suffix}", []) or []:
                rows[m] = m in catalogue
        report[provider] = {"catalogue": len(catalogue), "configured": rows,
                            "missing": sorted(k for k, v in rows.items() if not v),
                            "sample": catalogue[:120]}
    return report
