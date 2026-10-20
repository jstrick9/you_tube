"""Privatize a YouTube upload, clear its scheduled publish time, and reconcile channel history."""
from __future__ import annotations

import argparse
import json
import re

from autotube import common, youtube


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="11-character YouTube video ID")
    args = parser.parse_args()
    video_id = args.video_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        parser.error("video_id must be an 11-character YouTube ID")

    observed = youtube.unschedule(video_id)
    history = common.read_json("history.json", [])
    matches = [row for row in history if isinstance(row, dict) and row.get("video_id") == video_id]
    if matches:
        row = matches[-1]
        row["status"] = "withdrawn"
        row["withdrawn_at"] = common.now_utc().isoformat()
        row["withdrawal_reason"] = "content_quality_correction"
        row["youtube_status"] = {key: observed.get(key) for key in
                                 ("video_id", "privacy_status", "upload_status", "publish_at")}
        common.write_json("history.json", history)
    print(json.dumps({
        "video_id": video_id,
        "youtube_status": {key: observed.get(key) for key in
                           ("video_id", "privacy_status", "upload_status", "publish_at")},
        "history_updated": bool(matches),
        "status": "withdrawn" if matches else "unscheduled",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
