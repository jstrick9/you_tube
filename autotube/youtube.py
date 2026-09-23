"""YouTube Data API v3 client: OAuth refresh-token auth, resumable upload,
scheduled publishing, thumbnails, and analytics reads.

Auth is fully headless: a one-time local consent (scripts/get_refresh_token.py)
yields a refresh token stored as a GitHub secret. No browser is ever needed again.

Quota notes (2026): videos.insert draws from its own "Video Uploads" bucket
(100 calls/day by default); everything else draws from the 10,000-unit pool
(videos.list = 1 unit, thumbnails.set = 50). This system uses ~3 uploads and
<200 units per day — far below the limits.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("autotube.youtube")

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def credentials():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    cid, secret, refresh = (os.environ.get(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"))
    if not all((cid, secret, refresh)):
        raise RuntimeError("Missing YT_CLIENT_ID / YT_CLIENT_SECRET / YT_REFRESH_TOKEN")
    # scopes=None → reuse exactly what was granted at consent time (works whether the token
    # came from scripts/get_refresh_token.py or the browser-based OAuth Playground).
    creds = Credentials(None, refresh_token=refresh.strip(), client_id=cid.strip(), client_secret=secret.strip(),
                        token_uri="https://oauth2.googleapis.com/token", scopes=None)
    creds.refresh(Request())
    return creds


def service(name: str = "youtube", version: str = "v3"):
    from googleapiclient.discovery import build
    return build(name, version, credentials=credentials(), cache_discovery=False)


def upload(video: Path, meta: dict, cfg: dict, publish_at: datetime | None, thumb: Path | None = None) -> str:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    yt = service()
    status = {
        "selfDeclaredMadeForKids": bool(cfg["compliance"]["made_for_kids"]),
        "containsSyntheticMedia": bool(meta.get("contains_synthetic_media",
                                                cfg["compliance"]["contains_synthetic_media"])),
        "embeddable": True,
        "license": "youtube",
    }
    if publish_at:
        status["privacyStatus"] = "private"          # required for scheduled publishing
        status["publishAt"] = publish_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    else:
        status["privacyStatus"] = cfg["upload"].get("default_privacy", "public")
    body = {
        "snippet": {
            "title": meta["title"][:100],
            "description": meta["description"][:4900],
            "tags": meta.get("tags", [])[:30],
            "categoryId": str(cfg["upload"].get("category_id", "27")),
            "defaultLanguage": cfg["channel"].get("language", "en"),
            "defaultAudioLanguage": cfg["channel"].get("language", "en"),
        },
        "status": status,
    }
    media = MediaFileUpload(str(video), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media,
                             notifySubscribers=bool(cfg["schedule"].get("notify_subscribers", True)))
    resp, retries = None, 0
    while resp is None:
        try:
            _, resp = req.next_chunk()
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retries < 6:
                retries += 1
                time.sleep(2 ** retries)
                continue
            raise
    vid = resp["id"]
    log.info("uploaded https://youtube.com/shorts/%s (publishAt=%s)", vid, status.get("publishAt", "now"))
    # Custom thumbnails need a verified channel (phone verification). Shorts mostly use a
    # frame from the video anyway, so failure here is non-fatal.
    if thumb and thumb.exists():
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb), mimetype="image/jpeg")).execute()
        except Exception as e:  # noqa: BLE001
            log.info("thumbnail not set (%s) — fine for Shorts", str(e)[:120])
    return vid


def video_stats(ids: list[str]) -> dict[str, dict]:
    """views/likes/comments via Data API (1 unit per 50 videos)."""
    if not ids:
        return {}
    yt = service()
    out = {}
    for i in range(0, len(ids), 50):
        r = yt.videos().list(part="statistics,status,snippet", id=",".join(ids[i:i + 50])).execute()
        for it in r.get("items", []):
            s = it.get("statistics", {})
            out[it["id"]] = {"views": int(s.get("viewCount", 0)), "likes": int(s.get("likeCount", 0)),
                             "comments": int(s.get("commentCount", 0)),
                             "privacy": it["status"].get("privacyStatus"),
                             "upload_status": it["status"].get("uploadStatus"),
                             "rejection": it["status"].get("rejectionReason")}
    return out


def retention(ids: list[str], start: str, end: str) -> dict[str, dict]:
    """averageViewPercentage + averageViewDuration via YouTube Analytics API (free)."""
    if not ids:
        return {}
    try:
        ya = service("youtubeAnalytics", "v2")
        out = {}
        for i in range(0, len(ids), 200):
            r = ya.reports().query(ids="channel==MINE", startDate=start, endDate=end,
                                   metrics="views,averageViewPercentage,averageViewDuration",
                                   dimensions="video", filters="video==" + ",".join(ids[i:i + 200]),
                                   maxResults=200).execute()
            for row in r.get("rows", []):
                out[row[0]] = {"a_views": row[1], "avg_view_pct": row[2], "avg_view_dur": row[3]}
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("analytics API unavailable: %s", str(e)[:200])
        return {}
