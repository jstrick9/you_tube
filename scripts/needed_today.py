"""Lightweight pre-check for scheduled top-up runs (stdlib + PyYAML only, runs before the heavy install).

Prints how many of today's videos (channel timezone) are still missing and writes `needed=N` to
$GITHUB_OUTPUT, so backup/catch-up runs that have nothing to do finish in seconds instead of
spending Actions minutes installing PyTorch. Mirrors autotube.pipeline.remaining_today().
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from autotube.target import daily_target  # noqa: E402  (after sys.path is set)
from autotube.watchdog import list_daily_runs, should_skip_delayed_schedule  # noqa: E402
cfg = yaml.safe_load((root / "config.yaml").read_text())
tz = ZoneInfo(cfg["channel"].get("timezone", "UTC"))
today = datetime.now(timezone.utc).astimezone(tz).date()
if "AUTOTUBE_UPLOAD_CONFIGURED" in os.environ:
    credentials_present = os.environ.get("AUTOTUBE_UPLOAD_CONFIGURED", "").lower() == "true"
else:
    credentials_present = all(os.environ.get(k) for k in
                              ("YT_REFRESH_TOKEN", "YT_CLIENT_ID", "YT_CLIENT_SECRET"))
uploading = cfg["upload"]["enabled"] and credentials_present
ok = {"scheduled"} if uploading else {"scheduled", "rendered"}
hist_path = root / "state" / "history.json"
hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
done = sum(1 for h in hist if h.get("status") in ok and h.get("created_at")
           and datetime.fromisoformat(h["created_at"]).astimezone(tz).date() == today)
needed = max(0, daily_target(cfg) - done)
skip_reason = ""
# A scheduled event may sit in GitHub's queue for an hour or more. If a safe recovery
# run already started after this cron's intended slot, do not replay the delayed slot.
# If the API is unavailable, fail open so normal production is not silently skipped.
if (needed > 0 and os.environ.get("GITHUB_EVENT_NAME") == "schedule"
        and os.environ.get("GITHUB_WORKFLOW") == "AutoTube daily"
        and os.environ.get("GITHUB_TOKEN")):
    try:
        event_path = Path(os.environ.get("GITHUB_EVENT_PATH", ""))
        event = json.loads(event_path.read_text()) if event_path.is_file() else {}
        runs = list_daily_runs(os.environ.get("GITHUB_REPOSITORY", ""),
                               os.environ["GITHUB_TOKEN"],
                               os.environ.get("GITHUB_API_URL", "https://api.github.com"))
        skip, skip_reason = should_skip_delayed_schedule(
            event.get("schedule", ""), runs, datetime.now(timezone.utc),
            current_run_id=os.environ.get("GITHUB_RUN_ID"))
        if skip:
            needed = 0
    except Exception as exc:  # noqa: BLE001 — API trouble must not suppress a scheduled production attempt
        print(f"recent-run guard unavailable ({type(exc).__name__}); proceeding with the scheduled top-up")
        skip_reason = ""
suffix = f"; skipping delayed replay: {skip_reason}" if skip_reason else ""
print(f"{today}: {done} done, {needed} still needed{suffix}")
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"needed={needed}\n")
sys.exit(0)
