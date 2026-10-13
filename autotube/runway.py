"""Capped, fail-soft Runway Dev text-to-video adapter.

This is the Runway Dev route only (including its hosted Veo model IDs); it is not
Google's direct Gemini API. The adapter is deliberately dormant until an API key and
account-specific rights confirmations are supplied. Calls are bounded by a persisted
UTC-day credit ledger, and no automatic rerolls are made.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .common import STATE_DIR, ffmpeg_bin, http

log = logging.getLogger("autotube.runway")

API_BASE = "https://api.dev.runwayml.com"
API_VERSION = "2024-11-06"
TERMS_URL = "https://runway.com/terms-of-use"
USAGE_RIGHTS_URL = "https://help.runwayml.com/hc/en-us/articles/18927776141715-Usage-rights"
COMMERCIAL_GUIDANCE_URL = "https://dev.runwayml.com/learn/commercial-use"

# Reference rates and request schema are captured from Runway Dev docs, but remain
# unverified for this account/partner route. Separate dated config confirmations gate
# every paid call. Provider estimates are checked against the pre-reserved credit cap.
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "gen4.5": {
        "credits_per_second": 12,
        "default_duration": 5,
        "default_ratio": "720:1280",
        "allowed_durations": set(range(2, 11)),
        "allowed_ratios": {"720:1280", "1280:720"},
        "minimum_output": (720, 1280),
        "rights_key": None,
    },
    "veo3.1_fast": {
        "credits_per_second": 10,
        "default_duration": 8,
        "default_ratio": "1080:1920",
        "allowed_durations": {4, 6, 8},
        "allowed_ratios": {"720:1280", "1280:720", "1080:1920", "1920:1080"},
        "minimum_output": (1080, 1920),
        # Runway-hosted Veo is a partner-model route and needs a separate confirmation.
        "rights_key": "hosted_veo_terms_confirmed",
    },
}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _credit_amount(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(amount) or amount < 0 or not amount.is_integer():
        return None
    return int(amount)


def _task_id(value) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) else None


def probe_video(path: Path) -> tuple[int, int, float]:
    """Read dimensions and duration using ffprobe, with an ffmpeg stderr fallback."""
    probe = shutil.which("ffprobe")
    if probe:
        result = subprocess.run(
            [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,duration:format=duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout or "{}")
            streams = data.get("streams") or []
            if streams:
                width, height = int(streams[0]["width"]), int(streams[0]["height"])
                duration = float(streams[0].get("duration") or (data.get("format") or {}).get("duration") or 0)
                if width > 0 and height > 0 and duration > 0:
                    return width, height, duration
    result = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(path)],
                            capture_output=True, text=True, timeout=30)
    output = result.stderr or ""
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", output)
    size_match = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", output, re.S)
    if not duration_match or not size_match:
        raise ValueError("generated output has no decodable video stream")
    duration = (int(duration_match[1]) * 3600 + int(duration_match[2]) * 60
                + float(duration_match[3]))
    width, height = int(size_match[1]), int(size_match[2])
    if width <= 0 or height <= 0 or duration <= 0:
        raise ValueError("generated output has invalid dimensions or duration")
    return width, height, duration


class RunwayDev:
    """Small API client with model/rights allowlists and a persistent credit ceiling."""

    def __init__(self, cfg: dict, *, session=None, ledger_path: Path | None = None,
                 clock=None, sleep=None):
        self.cfg = cfg if isinstance(cfg, dict) else {}
        media_cfg = self.cfg.get("media") if isinstance(self.cfg.get("media"), dict) else {}
        self.settings = media_cfg.get("runway_dev") if isinstance(media_cfg.get("runway_dev"), dict) else {}
        self.key = (os.environ.get("RUNWAYML_API_SECRET") or "").strip()
        self.session = session or http()
        self.ledger_path = Path(ledger_path or (STATE_DIR / "runway_usage.json"))
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.sleep = sleep or time.sleep
        self.max_video_mb = float(media_cfg.get("max_video_mb", 90))

    def _review_confirmed(self, flag_key: str, date_key: str) -> bool:
        if self.settings.get(flag_key) is not True:
            return False
        reviewed_on = self.settings.get(date_key)
        if not isinstance(reviewed_on, str):
            return False
        try:
            reviewed = date.fromisoformat(reviewed_on)
        except ValueError:
            return False
        return reviewed.isoformat() == reviewed_on and reviewed <= _as_utc(self.clock()).date()

    def _requirements_ready(self) -> bool:
        return (self.settings.get("enabled") is True
                and self._review_confirmed("account_terms_confirmed", "account_terms_reviewed_on")
                and self._review_confirmed("api_schema_confirmed", "api_schema_reviewed_on")
                and self._review_confirmed("pricing_confirmed", "pricing_reviewed_on"))

    def readiness(self) -> tuple[bool, str]:
        """Report local configuration only; never makes a paid or network API call."""
        if self.settings.get("enabled") is not True:
            return False, "disabled in config"
        if not self.key:
            return False, "RUNWAYML_API_SECRET is not set"
        if not self._review_confirmed("account_terms_confirmed", "account_terms_reviewed_on"):
            return False, "Runway account terms confirmation/date are not set or valid"
        if not self._review_confirmed("api_schema_confirmed", "api_schema_reviewed_on"):
            return False, "Runway API schema confirmation/date are not set or valid"
        if not self._review_confirmed("pricing_confirmed", "pricing_reviewed_on"):
            return False, "Runway pricing confirmation/date are not set or valid"
        models = self._active_models()
        if not models:
            rows = self.settings.get("models")
            rows = rows if isinstance(rows, list) else []
            if any(isinstance(row, dict) and row.get("id") == "veo3.1_fast" for row in rows):
                return False, "hosted Veo terms confirmation/date are not set or valid (Gen-4.5 may still be enabled)"
            return False, "no enabled, rights-confirmed model is configured"
        return True, "ready: " + ", ".join(row["id"] for row in models)

    def _active_models(self) -> list[dict]:
        if not self._requirements_ready() or not self.key:
            return []
        rows = self.settings.get("models") or []
        active = []
        for raw in rows:
            if not isinstance(raw, dict) or raw.get("enabled", True) is not True:
                continue
            model = raw.get("id")
            spec = MODEL_SPECS.get(model)
            if not spec:
                log.warning("Runway model %r is not in the reviewed allowlist; skipping", model)
                continue
            rights_key = spec.get("rights_key")
            if rights_key:
                if not self._review_confirmed(rights_key, "hosted_veo_terms_reviewed_on"):
                    log.info("Runway %s is dormant: hosted partner-model terms are unconfirmed", model)
                    continue
            try:
                duration = int(raw.get("duration", spec["default_duration"]))
            except (TypeError, ValueError, OverflowError):
                log.warning("Runway %s has an invalid duration; skipping", model)
                continue
            ratio = str(raw.get("ratio", spec["default_ratio"]))
            if duration not in spec["allowed_durations"] or ratio not in spec["allowed_ratios"]:
                log.warning("Runway %s duration/ratio is outside the reviewed API schema; skipping", model)
                continue
            if model == "veo3.1_fast" and raw.get("audio", False) is not False:
                log.warning("Runway Veo Fast must use audio=false; refusing an audio-generating route")
                continue
            expected_credits = duration * spec["credits_per_second"]
            reserve_credits = _credit_amount(raw.get("max_credits", expected_credits))
            if reserve_credits is None or reserve_credits < expected_credits:
                log.warning("Runway %s credit reserve is invalid or below its current estimate; skipping", model)
                continue
            active.append({**raw, "id": model, "duration": duration, "ratio": ratio,
                           "expected_credits": expected_credits, "reserve_credits": reserve_credits,
                           "spec": spec})
        return active

    def _payload(self, model_cfg: dict, prompt: str) -> dict:
        model = model_cfg["id"]
        payload = {
            "model": model,
            "promptText": str(prompt or "").strip()[:1000],
            "ratio": model_cfg["ratio"],
            "duration": model_cfg["duration"],
        }
        if model == "veo3.1_fast":
            # Native audio is not used by this channel; this also selects Runway's
            # separately documented lower no-audio rate.
            payload["audio"] = False
        if not payload["promptText"]:
            raise ValueError("Runway prompt must not be empty")
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.key}",
            "X-Runway-Version": API_VERSION,
            "Content-Type": "application/json",
        }

    def _read_ledger(self) -> dict:
        if not self.ledger_path.exists():
            return {"schema": 1, "days": {}}
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {"_invalid": "ledger unreadable"}
        if not isinstance(data, dict):
            return {"_invalid": "ledger schema malformed"}
        days = data.get("days")
        if data.get("schema") != 1 or not isinstance(days, dict):
            return {"_invalid": "ledger schema malformed"}
        for day in days.values():
            if (not isinstance(day, dict) or _credit_amount(day.get("reserved_credits")) is None
                    or not isinstance(day.get("tasks"), list)
                    or any(not isinstance(task, dict) for task in day["tasks"])):
                return {"_invalid": "ledger day malformed"}
        return data

    def _write_ledger(self, data: dict) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.ledger_path)

    def _reserve(self, model: str, credits: int, prompt: str) -> tuple[str, int] | None:
        credits = _credit_amount(credits)
        if credits is None or credits <= 0:
            log.error("Runway credit reservation amount is invalid; refusing paid generation")
            return None
        today = _as_utc(self.clock()).date().isoformat()
        data = self._read_ledger()
        if data.get("_invalid"):
            log.error("Runway credit ledger is invalid; refusing paid generation (%s)", data["_invalid"])
            return None
        days = data["days"]
        existing_day = days.get(today)
        if existing_day is None:
            day = {"reserved_credits": 0, "tasks": []}
        elif isinstance(existing_day, dict):
            day = existing_day
        else:
            log.error("Runway ledger day %s is malformed; refusing paid generation", today)
            return None
        cap = _credit_amount(self.settings.get("daily_credit_budget"))
        used = _credit_amount(day.get("reserved_credits"))
        tasks = day.get("tasks")
        if cap is None or cap <= 0 or used is None or not isinstance(tasks, list):
            log.error("Runway budget or daily ledger row is malformed; refusing paid generation")
            return None
        if used + credits > cap:
            log.info("Runway daily credit cap reached (%d/%d credits); using licensed-media fallback", used, cap)
            return None
        task_index = len(tasks)
        tasks.append({
            "model": model,
            "status": "reserved",
            "credits_reserved": credits,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "reserved_at": _as_utc(self.clock()).isoformat(),
        })
        day["reserved_credits"] = used + credits
        day["tasks"] = tasks
        days[today] = day
        # Keep a month of compact cost-accounting history; per-video output provenance
        # remains in the sidecar even after this operational window rolls off.
        for date_key in sorted(days)[:-30]:
            days.pop(date_key, None)
        data.update({"schema": 1, "days": days})
        self._write_ledger(data)
        return today, task_index

    def _update_reservation(self, ticket: tuple[str, int], **values) -> None:
        date_key, index = ticket
        data = self._read_ledger()
        days = data.get("days") if isinstance(data.get("days"), dict) else {}
        day = days.get(date_key) if isinstance(days.get(date_key), dict) else {}
        tasks = day.get("tasks") if isinstance(day.get("tasks"), list) else []
        if 0 <= index < len(tasks) and isinstance(tasks[index], dict):
            task = tasks[index]
            actual = _credit_amount(values.get("actual_credits"))
            reserved = _credit_amount(task.get("credits_reserved")) or 0
            previous_overrun = _credit_amount(task.get("budget_overrun_credits")) or 0
            if actual is not None and actual > reserved + previous_overrun:
                overrun = actual - reserved
                adjustment = overrun - previous_overrun
                day_total = _credit_amount(day.get("reserved_credits")) or 0
                day["reserved_credits"] = day_total + adjustment
                values["budget_overrun_credits"] = overrun
                log.warning("Runway task actual cost exceeded its reserve by %d credits; daily spend guard updated",
                            adjustment)
            task.update(values)
            day["tasks"] = tasks
            days[date_key] = day
            data.update({"schema": 1, "days": days})
            self._write_ledger(data)

    def _post_task(self, payload: dict) -> tuple[dict, int | None, bool]:
        response = self.session.post(
            f"{API_BASE}/v1/text_to_video", headers=self._headers(), json=payload, timeout=30)
        if not 200 <= int(response.status_code) < 300:
            raise RuntimeError(f"Runway create task returned HTTP {response.status_code}")
        data = response.json()
        if not isinstance(data, dict) or not _task_id(data.get("id")):
            raise RuntimeError("Runway create response did not include a safe task id")
        estimated_cost = data.get("estimatedCost")
        if estimated_cost is None:
            estimate, estimate_valid = None, True
        elif isinstance(estimated_cost, dict) and "credits" in estimated_cost:
            estimate = _credit_amount(estimated_cost["credits"])
            estimate_valid = estimate is not None
        else:
            estimate, estimate_valid = None, False
        return data, estimate, estimate_valid

    def _get_task(self, task_id: str) -> dict:
        response = self.session.get(
            f"{API_BASE}/v1/tasks/{task_id}", headers=self._headers(), timeout=30)
        if not 200 <= int(response.status_code) < 300:
            raise RuntimeError(f"Runway task poll returned HTTP {response.status_code}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Runway task response was not an object")
        return data

    def _cancel_task(self, task_id: str) -> None:
        try:
            response = self.session.delete(
                f"{API_BASE}/v1/tasks/{task_id}", headers=self._headers(), timeout=20)
            if not 200 <= int(response.status_code) < 300:
                log.info("Runway cancel request returned HTTP %s", response.status_code)
        except Exception as exc:  # noqa: BLE001 — best effort only
            log.info("Runway cancel request failed (%s)", type(exc).__name__)

    def _wait(self, task_id: str) -> dict:
        timeout = max(1, int(self.settings.get("task_timeout_seconds", 300)))
        interval = max(1, int(self.settings.get("poll_interval_seconds", 5)))
        deadline = time.monotonic() + timeout
        while True:
            data = self._get_task(task_id)
            status = str(data.get("status") or "").upper()
            if status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
                return data
            if time.monotonic() >= deadline:
                self._cancel_task(task_id)
                raise TimeoutError("Runway task exceeded the configured wait limit")
            self.sleep(min(interval, max(0.0, deadline - time.monotonic())))

    def _download_output(self, url: str, path: Path) -> None:
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Runway task output is not a secure HTTPS URL")
        response = self.session.get(url, timeout=(20, 90), stream=True)
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            response.raise_for_status()
            length = int(response.headers.get("content-length") or 0)
            limit = self.max_video_mb * 1_000_000
            if length > limit:
                raise ValueError("Runway output exceeds configured video-size limit")
            received = 0
            with tmp.open("wb") as stream:
                for chunk in response.iter_content(1 << 16):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > limit:
                        raise ValueError("Runway output exceeds configured video-size limit")
                    stream.write(chunk)
            if received == 0:
                raise ValueError("Runway output was empty")
            tmp.replace(path)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            tmp.unlink(missing_ok=True)

    def _candidate_from_task(self, model_cfg: dict, prompt: str, task: dict,
                             estimate: int | None, reserved: int) -> dict:
        task_id = _task_id(task.get("id"))
        outputs = task.get("output")
        if not task_id or not isinstance(outputs, list) or not outputs:
            raise ValueError("successful Runway task has no output URL")
        output_url = outputs[0]
        safe_model = re.sub(r"[^a-z0-9]+", "_", model_cfg["id"].lower()).strip("_")
        work = Path(self._work_dir)
        work.mkdir(parents=True, exist_ok=True)
        video_path = work / f"runway_{safe_model}_{task_id}.mp4"
        poster_path = work / f"runway_{safe_model}_{task_id}_poster.jpg"
        self._download_output(output_url, video_path)
        width, height, duration = probe_video(video_path)
        expected_w, expected_h = model_cfg["spec"]["minimum_output"]
        target_w, target_h = (int(x) for x in model_cfg["ratio"].split(":"))
        actual_ratio, target_ratio = width / height, target_w / target_h
        if (width < expected_w or height < expected_h or height <= width
                or abs(actual_ratio - target_ratio) > 0.015):
            raise ValueError(f"Runway output dimensions {width}x{height} do not satisfy {model_cfg['ratio']}")
        requested = float(model_cfg["duration"])
        if duration < max(3.0, requested * 0.75) or duration > requested + 0.75:
            raise ValueError(f"Runway output duration {duration:.2f}s is outside the configured range")
        mid = duration / 2
        result = subprocess.run(
            [ffmpeg_bin(), "-loglevel", "error", "-y", "-ss", f"{mid:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(poster_path)], capture_output=True, text=True, timeout=45)
        if result.returncode != 0 or not poster_path.exists():
            raise ValueError("Runway output could not produce a poster frame")
        with Image.open(poster_path) as poster:
            thumbnail = poster.convert("RGB").copy()
        actual_cost = _credit_amount((task.get("cost") or {}).get("credits"))
        generated_at = _as_utc(self.clock()).isoformat()
        settings = self.settings
        hosted_veo = model_cfg["id"] == "veo3.1_fast"
        return {
            "kind": "video",
            # A stable local task URI avoids persisting Runway's expiring signed output URL.
            "url": f"runway://task/{task_id}",
            "_local_path": str(video_path),
            "thumb": str(poster_path),
            "_poster": poster_path,
            "_img": thumbnail,
            "t_ref": mid,
            "duration": duration,
            "w": width,
            "h": height,
            "title": f"AI-generated illustration ({model_cfg['id']})",
            "license": "AI-generated; Runway account terms confirmed for this route",
            "author": "Runway Dev generated media",
            "source": f"Runway Dev API ({model_cfg['id']})",
            "page": "https://docs.dev.runwayml.com/api/",
            "synthetic": True,
            "provider": "Runway Dev API",
            "model": model_cfg["id"],
            "api_version": API_VERSION,
            "task_id": task_id,
            "prompt": prompt,
            "generated_at": generated_at,
            "ratio": model_cfg["ratio"],
            "audio": False,
            "requested_duration_seconds": model_cfg["duration"],
            "actual_duration_seconds": round(duration, 3),
            "output_width": width,
            "output_height": height,
            "credits_reserved": reserved,
            "estimated_cost_credits": estimate,
            "actual_cost_credits": int(actual_cost) if actual_cost is not None else None,
            "estimated_cost_usd": round((estimate if estimate is not None else reserved) / 100, 2),
            "file_sha256": _sha256_file(video_path),
            "rights_confirmation": {
                "runway_account_terms_confirmed": settings.get("account_terms_confirmed") is True,
                "runway_account_terms_reviewed_on": settings.get("account_terms_reviewed_on"),
                "hosted_veo_terms_confirmed": settings.get("hosted_veo_terms_confirmed") is True if hosted_veo else None,
                "hosted_veo_terms_reviewed_on": settings.get("hosted_veo_terms_reviewed_on") if hosted_veo else None,
                "terms_url": TERMS_URL,
                "usage_rights_url": USAGE_RIGHTS_URL,
                "commercial_guidance_url": COMMERCIAL_GUIDANCE_URL,
            },
        }

    def _generate_one(self, model_cfg: dict, prompt: str) -> dict | None:
        model = model_cfg["id"]
        reserved = model_cfg.get("reserve_credits", model_cfg["expected_credits"])
        ticket = self._reserve(model, reserved, prompt)
        if ticket is None:
            return None
        task_id = None
        estimate = None
        try:
            payload = self._payload(model_cfg, prompt)
            task_response, estimate, estimate_valid = self._post_task(payload)
            task_id = task_response["id"]
            self._update_reservation(ticket, task_id=task_id, status="submitted",
                                     estimated_credits=estimate)
            if not estimate_valid:
                self._cancel_task(task_id)
                self._update_reservation(ticket, status="cancelled_invalid_cost_estimate",
                                         cost_estimate_invalid=True)
                log.warning("Runway %s returned an invalid cost estimate; task cancelled", model)
                return None
            if estimate is not None and estimate > reserved:
                self._cancel_task(task_id)
                self._update_reservation(ticket, status="cancelled_cost_estimate_over_cap",
                                         cost_estimate_exceeded=True)
                log.warning("Runway %s estimate exceeded the pre-reserved cap; task cancelled", model)
                return None
            task = self._wait(task_id)
            actual = _credit_amount((task.get("cost") or {}).get("credits"))
            self._update_reservation(ticket, status=str(task.get("status") or "unknown").lower(),
                                     actual_credits=actual)
            if str(task.get("status") or "").upper() != "SUCCEEDED":
                log.info("Runway %s task did not succeed (%s); licensed fallback remains active",
                         model, task.get("failureCode") or task.get("status") or "unknown")
                return None
            candidate = self._candidate_from_task(model_cfg, prompt, task, estimate, reserved)
            self._update_reservation(ticket, status="output_validated",
                                     file_sha256=candidate["file_sha256"],
                                     actual_credits=candidate.get("actual_cost_credits"))
            return candidate
        except Exception as exc:  # noqa: BLE001 — generation is supplemental; keep licensed paths usable
            info = {"status": "failed", "error_type": type(exc).__name__}
            if task_id:
                info["task_id"] = task_id
            if estimate is not None:
                info["estimated_credits"] = estimate
            self._update_reservation(ticket, **info)
            log.warning("Runway %s generation unavailable (%s); using licensed-media fallback",
                        model, type(exc).__name__)
            return None

    def generate_candidates(self, prompt: str, work: Path) -> list[dict]:
        """Make at most one attempt per configured model; never retries or waives QA."""
        self._work_dir = Path(work)
        ready, why = self.readiness()
        if not ready:
            log.info("Runway Dev supplemental visuals dormant: %s", why)
            return []
        candidates = []
        for model_cfg in self._active_models():
            candidate = self._generate_one(model_cfg, prompt)
            if candidate:
                candidates.append(candidate)
        return candidates
