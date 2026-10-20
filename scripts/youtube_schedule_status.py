"""Read the authenticated channel's actual scheduled uploads and optionally inspect one video ID."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import youtube


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def scheduled_uploads() -> list[dict]:
    """Read all owned uploads through the channel's uploads playlist, then check actual status."""
    yt = youtube.service()
    channels = yt.channels().list(part="contentDetails", mine=True, maxResults=1).execute().get("items", [])
    if not channels:
        raise RuntimeError("YouTube did not return an owned channel")
    uploads_id = (((channels[0].get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads"))
    if not uploads_id:
        raise RuntimeError("YouTube did not return the channel's uploads playlist")

    ids: list[str] = []
    page_token = None
    while True:
        response = yt.playlistItems().list(
            part="contentDetails", playlistId=uploads_id, maxResults=50, pageToken=page_token,
        ).execute()
        ids.extend(str((row.get("contentDetails") or {}).get("videoId") or "")
                   for row in response.get("items", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    rows = []
    for start in range(0, len(ids), 50):
        batch = [video_id for video_id in ids[start:start + 50] if video_id]
        if not batch:
            continue
        response = yt.videos().list(part="snippet,status", id=",".join(batch), maxResults=len(batch)).execute()
        for item in response.get("items", []):
            status = item.get("status") or {}
            publish_at = status.get("publishAt")
            if not publish_at or status.get("privacyStatus") != "private":
                continue
            rows.append({
                "video_id": item.get("id"),
                "title": (item.get("snippet") or {}).get("title", ""),
                "privacy_status": status.get("privacyStatus"),
                "upload_status": status.get("uploadStatus", "unknown"),
                "publish_at": publish_at,
            })
    return sorted(rows, key=lambda row: (_timestamp(row.get("publish_at")) or datetime.max.replace(tzinfo=timezone.utc)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", help="Optionally read back a specific video, including private/public state")
    args = parser.parse_args()

    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    target = youtube.video_status(args.video_id) if args.video_id else None
    scheduled = scheduled_uploads()
    now = datetime.now(timezone.utc)
    future = [row for row in scheduled if (_timestamp(row.get("publish_at")) or now) > now]
    result = {
        "checked_at": checked_at,
        "target_video": target,
        "scheduled_private_uploads": scheduled,
        "next_scheduled_video": future[0] if future else None,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
