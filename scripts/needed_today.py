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
cfg = yaml.safe_load((root / "config.yaml").read_text())
tz = ZoneInfo(cfg["channel"].get("timezone", "UTC"))
today = datetime.now(timezone.utc).astimezone(tz).date()
uploading = cfg["upload"]["enabled"] and all(os.environ.get(k) for k in
                                              ("YT_REFRESH_TOKEN", "YT_CLIENT_ID", "YT_CLIENT_SECRET"))
ok = {"scheduled"} if uploading else {"scheduled", "rendered"}
hist_path = root / "state" / "history.json"
hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
done = sum(1 for h in hist if h.get("status") in ok and h.get("created_at")
           and datetime.fromisoformat(h["created_at"]).astimezone(tz).date() == today)
needed = max(0, int(cfg["schedule"]["videos_per_day"]) - done)
print(f"{today}: {done} done, {needed} still needed")
if os.environ.get("GITHUB_OUTPUT"):
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"needed={needed}\n")
sys.exit(0)
