"""Persistent, conservative provider circuit breakers for quota and transient outages."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from . import common

log = logging.getLogger("autotube.provider_health")

STATE_NAME = "provider_cooldowns.json"
MAX_COOLDOWN_SECONDS = 24 * 60 * 60
DEFAULT_429_COOLDOWN_SECONDS = 20 * 60
DEFAULT_402_COOLDOWN_SECONDS = 24 * 60 * 60
DEFAULT_5XX_COOLDOWN_SECONDS = 10 * 60
TRANSIENT_FAILURE_WINDOW_SECONDS = 30 * 60


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class ProviderCircuitBreakers:
    """Persist temporary provider outages across scheduled runs; never persist credentials."""

    def __init__(self, clock=None):
        self._clock = clock or common.now_utc
        raw = common.read_json(STATE_NAME, {})
        raw = raw if isinstance(raw, dict) else {}
        self._entries = raw.get("providers", {}) if isinstance(raw.get("providers"), dict) else {}
        self._entries = {name: dict(entry) for name, entry in self._entries.items()
                         if isinstance(name, str) and isinstance(entry, dict)}
        raw_failures = raw.get("failures_5xx", {}) if isinstance(raw.get("failures_5xx"), dict) else {}
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        now = now.astimezone(timezone.utc)
        self._failures_5xx: dict[str, dict] = {}
        for name, row in raw_failures.items():
            if not isinstance(name, str) or not isinstance(row, dict):
                continue
            count = row.get("count")
            last_at = _parse_utc(row.get("last_at"))
            if (type(count) is int and 0 < count < 1000 and last_at
                    and 0 <= (now - last_at).total_seconds() <= TRANSIENT_FAILURE_WINDOW_SECONDS):
                self._failures_5xx[name] = {"count": count, "last_at": last_at.isoformat()}
        self.events: list[dict] = []

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    def _save(self) -> None:
        now = self._now()
        active = {}
        for provider, row in self._entries.items():
            until = _parse_utc(row.get("blocked_until"))
            if until and until > now:
                active[provider] = row
        recent_failures = {}
        for provider, row in self._failures_5xx.items():
            last_at = _parse_utc(row.get("last_at"))
            age = (now - last_at).total_seconds() if last_at else float("inf")
            if 0 <= age <= TRANSIENT_FAILURE_WINDOW_SECONDS:
                recent_failures[provider] = {"count": int(row["count"]), "last_at": last_at.isoformat()}
        try:
            common.write_json(STATE_NAME, {"schema_version": 1, "providers": active,
                                           "failures_5xx": recent_failures})
        except OSError as exc:
            # The in-memory circuit remains effective even if the persistence volume is read-only.
            log.warning("could not persist provider cooldown ledger (%s)", type(exc).__name__)

    def is_open(self, provider: str) -> bool:
        row = self._entries.get(provider)
        until = _parse_utc(row.get("blocked_until")) if isinstance(row, dict) else None
        if until is not None and until > self._now():
            return True
        if row is not None:
            self._entries.pop(provider, None)
            self._save()
        return False

    def blocked_until(self, provider: str) -> str | None:
        row = self._entries.get(provider)
        until = _parse_utc(row.get("blocked_until")) if isinstance(row, dict) else None
        return until.isoformat() if until and until > self._now() else None

    def open(self, provider: str, status_code: int | None, reason: str,
             cooldown_seconds: float, retry_after_seconds: float | None = None) -> str:
        if not isinstance(provider, str) or not provider.strip():
            return ""
        try:
            seconds = float(cooldown_seconds)
        except (TypeError, ValueError, OverflowError):
            seconds = DEFAULT_429_COOLDOWN_SECONDS
        seconds = min(MAX_COOLDOWN_SECONDS, max(60.0, seconds))
        now = self._now()
        until = now + timedelta(seconds=seconds)
        previous = self._entries.get(provider)
        prev_until = _parse_utc(previous.get("blocked_until")) if isinstance(previous, dict) else None
        if prev_until and prev_until > until:
            until = prev_until
        row = {
            "reason": str(reason)[:64],
            "status_code": status_code if type(status_code) is int else None,
            "recorded_at": now.isoformat(),
            "blocked_until": until.isoformat(),
        }
        if isinstance(retry_after_seconds, (int, float)) and retry_after_seconds > 0:
            row["retry_after_seconds"] = round(min(float(retry_after_seconds), MAX_COOLDOWN_SECONDS), 2)
        self._entries[provider] = row
        self.events.append({"provider": provider, "status_code": row["status_code"],
                            "action": "circuit_open", "reason": row["reason"],
                            "cooldown_until": row["blocked_until"],
                            "retry_after_seconds": row.get("retry_after_seconds")})
        self._save()
        return until.isoformat()

    def note_5xx(self, provider: str, status_code: int, threshold: int = 3) -> bool:
        """Count transient errors across models and nearby runs; open after repeated failures."""
        now = self._now()
        previous = self._failures_5xx.get(provider, {})
        last_at = _parse_utc(previous.get("last_at")) if isinstance(previous, dict) else None
        age = (now - last_at).total_seconds() if last_at else float("inf")
        count = int(previous.get("count", 0)) + 1 if 0 <= age <= TRANSIENT_FAILURE_WINDOW_SECONDS else 1
        self._failures_5xx[provider] = {"count": count, "last_at": now.isoformat()}
        self.events.append({"provider": provider, "status_code": status_code,
                            "action": "transient_failure", "consecutive_failures": count})
        if count < max(1, threshold):
            self._save()
            return False
        self.open(provider, status_code, "repeated_http_5xx", DEFAULT_5XX_COOLDOWN_SECONDS)
        return True

    def note_success(self, provider: str) -> None:
        if self._failures_5xx.pop(provider, None) is not None:
            self._save()

    def snapshot(self) -> dict[str, dict]:
        return {name: dict(row) for name, row in list(self._entries.items()) if self.is_open(name)}
