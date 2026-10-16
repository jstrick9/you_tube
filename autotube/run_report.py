"""Small, secret-free production receipts for diagnosing partial and zero-output runs."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path

from . import common

log = logging.getLogger("autotube.run_report")
SCHEMA_VERSION = 1
REPORTS_TO_KEEP = 30
DETAILS_TO_KEEP = 100
_ACTIVE: ContextVar[RunReceipt | None] = ContextVar("autotube_run_receipt", default=None)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _bounded_text(value, limit: int = 160) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def classify_failure(stage: str, detail: str = "", status_code: int | None = None) -> str:
    """Map errors to a small stable vocabulary; never copy provider response bodies into the receipt."""
    text = (detail or "").lower()
    if status_code is None:
        match = re.search(r"\bhttp\s+(\d{3})\b", text)
        status_code = int(match.group(1)) if match else None
    if status_code == 402 or "payment required" in text:
        return "billing_or_entitlement"
    if status_code == 429 or "rate limit" in text or "quota" in text:
        return "provider_rate_limited"
    if status_code is not None and 500 <= status_code <= 599:
        return "provider_transient_5xx"
    if status_code in (408, 425) or any(term in text for term in
                                         ("timed out", "timeout", "connection reset", "connectionerror")):
        return "provider_transient_network"
    if status_code in (401, 403):
        return "provider_auth_or_permission"
    if status_code in (400, 404):
        return "provider_model_or_request"
    if "visionunavailable" in text or "vision unavailable" in text:
        return "vision_unavailable"
    if "novisualmatch" in text or "no verified visual" in text:
        return "no_verified_visual_match"
    return {
        "trends": "trend_collection_failed",
        "topic_selection": "no_eligible_trending_topics",
        "script": "script_or_review_gate",
        "tts": "narration_duration_or_synthesis",
        "visuals": "visual_research_or_match",
        "render_qa": "render_or_final_qa",
        "upload": "youtube_upload",
    }.get(stage, "pipeline_error")


class RunReceipt:
    """Collect bounded operational facts without prompts, credentials, or full error bodies."""

    def __init__(self, run_id: str, target: int, *, upload: bool, dry_run: bool = False):
        self._started = _now()
        self._monotonic = time.monotonic()
        self._last_failure: dict | None = None
        workflow_id = os.environ.get("GITHUB_RUN_ID", "").strip()
        self._artifact_id = re.sub(r"[^A-Za-z0-9_.-]", "-", workflow_id or run_id)[:100]
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "run_id": _bounded_text(run_id, 80),
            "workflow_run_id": workflow_id or None,
            "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "started_at": _iso(self._started),
            "finished_at": None,
            "duration_seconds": None,
            "mode": "dry_run" if dry_run else ("upload" if upload else "render_only"),
            "target_count": max(0, int(target)),
            "target_count_known": True,
            "produced_count": 0,
            "scheduled_count": 0,
            "upload_failure_count": 0,
            "outcome": "running",
            "stage": "initialization",
            "stage_entries": {},
            "retry_budgets": {},
            "topic_attempts": [],
            "failures": [],
            "retry_events": [],
            "provider_events": [],
            "grounding_repairs": [],
        }
        self._written = False

    @property
    def current_stage(self) -> str:
        return str(self.data.get("stage") or "unknown")

    @property
    def last_failure(self) -> dict | None:
        return dict(self._last_failure) if self._last_failure else None

    def set_target(self, target: int) -> None:
        self.data["target_count"] = max(0, int(target))

    def set_stage(self, stage: str) -> None:
        stage = _bounded_text(stage, 48) or "unknown"
        self.data["stage"] = stage
        counts = self.data["stage_entries"]
        counts[stage] = int(counts.get(stage, 0)) + 1

    def set_retry_budgets(self, budgets: dict) -> None:
        safe = {}
        for key, value in (budgets or {}).items():
            if isinstance(key, str) and type(value) is int and 0 <= value <= 100:
                safe[key[:48]] = value
        self.data["retry_budgets"] = safe

    def set_topic(self, topic: dict, index: int) -> None:
        self._last_failure = None
        self._current_topic = {
            "index": max(0, int(index)),
            "topic": _bounded_text(topic.get("topic") if isinstance(topic, dict) else "", 100),
            "topic_key": _bounded_text(topic.get("topic_key") if isinstance(topic, dict) else "", 120),
        }

    def note_topic(self, outcome: str, stage: str | None = None, category: str | None = None) -> None:
        row = dict(getattr(self, "_current_topic", {}))
        row.update({"outcome": "accepted" if outcome == "accepted" else "rejected",
                    "stage": _bounded_text(stage or self.current_stage, 48)})
        if category:
            row["failure_category"] = _bounded_text(category, 64)
        self.data["topic_attempts"].append(row)
        self.data["topic_attempts"] = self.data["topic_attempts"][-DETAILS_TO_KEEP:]

    def note_failure(self, stage: str, category: str, *, exception_type: str | None = None,
                     topic: str | None = None, attempt: int | None = None) -> None:
        row = {"at": _iso(), "stage": _bounded_text(stage, 48),
               "category": _bounded_text(category, 64)}
        if exception_type:
            row["exception_type"] = _bounded_text(exception_type, 80)
        if topic:
            row["topic"] = _bounded_text(topic, 100)
        if type(attempt) is int and attempt > 0:
            row["attempt"] = attempt
        self.data["failures"].append(row)
        self.data["failures"] = self.data["failures"][-DETAILS_TO_KEEP:]
        self._last_failure = {"stage": row["stage"], "category": row["category"]}

    def note_retry(self, stage: str, attempt: int, limit: int, action: str) -> None:
        if type(attempt) is not int or type(limit) is not int or not 1 <= attempt < limit:
            return
        self.data["retry_events"].append({"stage": _bounded_text(stage, 48), "attempt": attempt,
                                          "limit": limit, "action": _bounded_text(action, 80)})
        self.data["retry_events"] = self.data["retry_events"][-DETAILS_TO_KEEP:]

    def note_grounding_repair(self, topic: str, source_title: str, original_angle: str,
                              repaired_angle: str, source_quote: str) -> None:
        """Record a bounded source-backed correction, not the prompt or full model response."""
        safe_quote = re.sub(
            r"(?i)\b(api[_-]?key|token|client[_-]?secret)\b\s*[:=]\s*[^\s,;]+",
            r"\1=[REDACTED]", str(source_quote or ""))
        safe_quote = re.sub(
            r"\b(?:gh[pousr]_|github_pat_|AIza)[A-Za-z0-9_-]{12,}\b", "[REDACTED]", safe_quote)
        self.data["grounding_repairs"].append({
            "topic": _bounded_text(topic, 100),
            "source_title": _bounded_text(source_title, 120),
            "original_angle": _bounded_text(original_angle, 240),
            "repaired_angle": _bounded_text(repaired_angle, 240),
            "source_quote": _bounded_text(safe_quote, 280),
            "at": _iso(),
        })
        self.data["grounding_repairs"] = self.data["grounding_repairs"][-DETAILS_TO_KEEP:]

    def note_provider_event(self, provider: str, status_code: int | None, action: str,
                            cooldown_until: str | None = None, retry_after_seconds: float | None = None) -> None:
        row = {"provider": _bounded_text(provider, 40), "status_code": status_code,
               "action": _bounded_text(action, 48), "at": _iso()}
        if cooldown_until:
            row["cooldown_until"] = _bounded_text(cooldown_until, 40)
        if isinstance(retry_after_seconds, (int, float)) and retry_after_seconds >= 0:
            row["retry_after_seconds"] = round(min(float(retry_after_seconds), 86400), 2)
        self.data["provider_events"].append(row)
        self.data["provider_events"] = self.data["provider_events"][-DETAILS_TO_KEEP:]

    def finish(self, produced: int, target: int, scheduled: int = 0, upload_failures: int = 0) -> None:
        produced = max(0, int(produced))
        target = max(0, int(target))
        scheduled = max(0, int(scheduled))
        upload_failures = max(0, int(upload_failures))
        self.data.update({"produced_count": produced, "target_count": target,
                          "scheduled_count": scheduled, "upload_failure_count": upload_failures})
        if self.data.get("outcome") == "failed":
            return
        if self.data["mode"] == "dry_run":
            outcome = "dry_run_complete" if produced >= target else ("no_output" if produced == 0 else "partial")
        elif produced == 0:
            outcome = "no_output"
        elif produced < target or upload_failures:
            outcome = "partial"
        else:
            outcome = "complete"
        self.data["outcome"] = outcome

    def fail(self, exc: BaseException) -> None:
        stage = self.current_stage
        category = classify_failure(stage, str(exc))
        self.note_failure(stage, category, exception_type=type(exc).__name__)
        self.data["outcome"] = "failed"

    def persist(self) -> Path:
        if self._written:
            return common.OUTPUT_DIR / f"run-report-{self._artifact_id}.json"
        self._written = True
        finished = _now()
        self.data["finished_at"] = _iso(finished)
        self.data["duration_seconds"] = round(max(0.0, time.monotonic() - self._monotonic), 3)
        if self.data["outcome"] == "running":
            self.finish(self.data["produced_count"], self.data["target_count"],
                        self.data["scheduled_count"], self.data["upload_failure_count"])
        path = common.OUTPUT_DIR / f"run-report-{self._artifact_id}.json"
        try:
            common.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            log.warning("could not write production receipt artifact (%s)", type(exc).__name__)
        try:
            reports = common.read_json("run_reports.json", [])
            reports = reports if isinstance(reports, list) else []
            reports.append(self.data)
            common.write_json("run_reports.json", reports[-REPORTS_TO_KEEP:])
        except OSError as exc:
            log.warning("could not persist production receipt to state (%s)", type(exc).__name__)
        return path


def activate(receipt: RunReceipt) -> Token:
    return _ACTIVE.set(receipt)


def deactivate(token: Token) -> None:
    _ACTIVE.reset(token)


def current_receipt() -> RunReceipt | None:
    return _ACTIVE.get()
