"""Best-effort recovery for delayed or dropped GitHub Actions production schedules.

All recovery dispatches use the existing top-up workflow, which is idempotent against
that day's target. These helpers deliberately avoid changing content, quality gates,
provider order, or the upload decision.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time, timedelta, timezone

API_VERSION = "2022-11-28"
DAILY_WORKFLOW = "daily.yml"


def _utc(value) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _hours(field: str) -> list[int] | None:
    """Expand the simple hour lists/ranges used by this repo's cron schedules."""
    result: set[int] = set()
    try:
        for part in field.split(","):
            base, slash, step_text = part.partition("/")
            step = int(step_text) if slash else 1
            if step <= 0:
                return None
            if base == "*":
                result.update(range(0, 24, step))
            elif "-" in base:
                first_text, last_text = base.split("-", 1)
                first, last = int(first_text), int(last_text)
                if not 0 <= first <= last <= 23:
                    return None
                result.update(range(first, last + 1, step))
            else:
                hour = int(base)
                if not 0 <= hour <= 23:
                    return None
                result.add(hour)
    except (TypeError, ValueError):
        return None
    return sorted(result)


def scheduled_slot_utc(schedule: str, now: datetime | None = None) -> datetime | None:
    """Return the most recent UTC occurrence for this repo's minute/hour cron shapes."""
    fields = (schedule or "").split()
    if len(fields) != 5 or fields[2:] != ["*", "*", "*"]:
        return None
    try:
        minute = int(fields[0])
    except (TypeError, ValueError):
        return None
    if not 0 <= minute <= 59:
        return None
    hours = _hours(fields[1])
    if not hours:
        return None
    reference = _utc(now or datetime.now(timezone.utc))
    if reference is None:
        return None
    candidates = []
    today = reference.date()
    for offset in range(3):
        day = today - timedelta(days=offset)
        candidates.extend(datetime.combine(day, time(hour=h, minute=minute), timezone.utc)
                          for h in hours)
    due = [slot for slot in candidates if slot <= reference]
    return max(due) if due else None


def should_skip_delayed_schedule(schedule: str, runs: list[dict], now: datetime | None = None,
                                 current_run_id=None) -> tuple[bool, str]:
    """Do not replay a delayed cron if a recovery/production run already covered its slot."""
    reference = _utc(now or datetime.now(timezone.utc))
    expected = scheduled_slot_utc(schedule, reference)
    if reference is None or expected is None:
        return False, "cron slot could not be resolved; proceeding with normal production"
    own_id = str(current_run_id) if current_run_id is not None else None
    for run in runs if isinstance(runs, list) else []:
        if not isinstance(run, dict) or (own_id is not None and str(run.get("id")) == own_id):
            continue
        if run.get("status") in {"queued", "in_progress", "waiting", "pending", "requested"}:
            return True, "another daily production run is already active"
        started = _utc(run.get("run_started_at") or run.get("created_at"))
        if started is not None and started >= expected:
            number = run.get("run_number", "?")
            return True, f"daily run #{number} already started after this scheduled slot"
    return False, "no later production run has covered this slot"


def should_dispatch_top_up(needed: int, runs: list[dict], now: datetime | None = None,
                          cooldown_minutes: int = 90) -> tuple[bool, str]:
    """Dispatch only when work remains and no daily run is active or very recent."""
    if type(needed) is not int or needed <= 0:
        return False, "daily target is already satisfied"
    reference = _utc(now or datetime.now(timezone.utc))
    if reference is None:
        return False, "current time is invalid; refusing duplicate dispatch"
    active = {"queued", "in_progress", "waiting", "pending", "requested"}
    valid_runs = [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []
    if any(r.get("status") in active for r in valid_runs):
        return False, "a daily production run is already active"
    starts = [_utc(r.get("run_started_at") or r.get("created_at")) for r in valid_runs]
    starts = [s for s in starts if s is not None]
    if starts and reference - max(starts) < timedelta(minutes=max(0, int(cooldown_minutes))):
        return False, f"a daily production attempt started within the last {cooldown_minutes} minutes"
    return True, f"{needed} video(s) remain and no recent daily attempt is active"


def list_daily_runs(repository: str, token: str, api_url: str = "https://api.github.com") -> list[dict]:
    """Fetch recent runs for the production workflow; errors are raised for fail-closed recovery."""
    repository = (repository or "").strip()
    token = (token or "").strip()
    if not repository or "/" not in repository or not token:
        raise ValueError("repository and workflow token are required")
    base = (api_url or "https://api.github.com").rstrip("/")
    repo = urllib.parse.quote(repository, safe="/")
    url = f"{base}/repos/{repo}/actions/workflows/{DAILY_WORKFLOW}/runs?branch=main&per_page=50"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "AutoTube-Schedule-Watchdog",
        "X-GitHub-Api-Version": API_VERSION,
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"GitHub Actions API returned HTTP {exc.code}") from None
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GitHub Actions API request failed: {type(exc).__name__}") from None
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        raise RuntimeError("GitHub Actions API returned an invalid run list")
    return [run for run in runs if isinstance(run, dict)]


def dispatch_top_up(repository: str, token: str, api_url: str = "https://api.github.com") -> None:
    """Start the daily workflow on main with its safe top-up guard enabled."""
    repository = (repository or "").strip()
    token = (token or "").strip()
    if not repository or "/" not in repository or not token:
        raise ValueError("repository and workflow token are required")
    base = (api_url or "https://api.github.com").rstrip("/")
    repo = urllib.parse.quote(repository, safe="/")
    url = f"{base}/repos/{repo}/actions/workflows/{DAILY_WORKFLOW}/dispatches"
    body = json.dumps({"ref": "main", "inputs": {"top_up": "true"}}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "AutoTube-Schedule-Watchdog",
        "X-GitHub-Api-Version": API_VERSION,
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            if response.status != 204:
                raise RuntimeError(f"workflow_dispatch returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"workflow_dispatch returned HTTP {exc.code}") from None
    except (OSError, TimeoutError) as exc:
        raise RuntimeError(f"workflow_dispatch request failed: {type(exc).__name__}") from None
