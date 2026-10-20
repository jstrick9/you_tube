"""Read a YouTube video's current status and report whether this token can update it."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import youtube


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id", help="11-character YouTube video ID")
    args = parser.parse_args()
    video_id = args.video_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        parser.error("video_id must be an 11-character YouTube ID")

    scopes = youtube.granted_scopes()
    status = youtube.video_status(video_id)
    update_authorized = bool(scopes & {
        youtube.DELETE_SCOPE,
        "https://www.googleapis.com/auth/youtube",
    })
    result = {
        "video_id": video_id,
        "exists": status is not None,
        "youtube_status": ({key: status.get(key) for key in
                            ("video_id", "title", "privacy_status", "upload_status", "publish_at")}
                           if status else None),
        "update_authorized": update_authorized,
        "scope_check": {
            "youtube_force_ssl": youtube.DELETE_SCOPE in scopes,
            "youtube_full": "https://www.googleapis.com/auth/youtube" in scopes,
        },
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
