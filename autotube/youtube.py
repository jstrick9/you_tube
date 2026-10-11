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

# Deleting or re-privatising an existing video needs a broader scope than uploading a
# new one: youtube.upload is write-only for fresh uploads and cannot touch anything
# already on the channel. Listed here so a re-consent mints a token that can prune; an
# existing token keeps working unchanged for upload-only operation.
DELETE_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"

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


def video_status(video_id: str) -> dict | None:
    """Read the current YouTube upload state without mutating the video."""
    video_id = str(video_id or "").strip()
    if not video_id:
        raise ValueError("video_id must not be empty")
    items = service().videos().list(part="snippet,status", id=video_id, maxResults=1).execute().get("items", [])
    if not items:
        return None
    item = items[0]
    status = item.get("status") or {}
    return {
        "video_id": item.get("id", video_id),
        "title": (item.get("snippet") or {}).get("title", ""),
        "privacy_status": status.get("privacyStatus", "unknown"),
        "upload_status": status.get("uploadStatus", "unknown"),
        "publish_at": status.get("publishAt"),
        "rejection_reason": status.get("rejectionReason"),
    }


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
    """Per-video Shorts analytics for the requested date window.

    ``engagedViews`` is the monetization-relevant Shorts count; ``views`` now also counts starts
    and replays. ``averageViewPercentage`` is average watch percentage, not the share of people who
    finished. Callers must name those metrics accurately rather than infer a completion rate from
    the last point of an audience-retention curve.
    """
    if not ids:
        return {}
    try:
        ya = service("youtubeAnalytics", "v2")
        out = {}
        for i in range(0, len(ids), 200):
            r = ya.reports().query(ids="channel==MINE", startDate=start, endDate=end,
                                   metrics="views,engagedViews,averageViewPercentage,averageViewDuration,"
                                           "subscribersGained,subscribersLost,shares",
                                   dimensions="video", filters="video==" + ",".join(ids[i:i + 200]),
                                   maxResults=200).execute()
            for row in r.get("rows", []):
                out[row[0]] = {"a_views": row[1], "engaged_views_90d": row[2],
                               "avg_view_pct": row[3], "avg_view_dur": row[4],
                               "subs_gained": row[5], "subs_lost": row[6], "shares": row[7]}
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("analytics API unavailable: %s", str(e)[:200])
        return {}


def retention_curves(ids: list[str], start: str, end: str) -> dict[str, dict]:
    """Per-video audienceWatchRatio curve across elapsedVideoTimeRatio.

    The curve helps locate where watch activity drops, but its values are *ratios of segment watches
    to total video views*, not literal percentages of unique viewers remaining. Rewatches can push a
    point above 1.0. We therefore store it as a relative watch-ratio curve and never label the final
    point as a completion rate.

    The elapsedVideoTimeRatio dimension only accepts ONE video per query, so this costs one Analytics
    call per video. Callers pass only videos that don't already have a curve, so a normal day is a
    handful of calls.

    Returns per video: {audience_watch_ratio_curve, watch_ratio_at_15pct, end_watch_ratio, curve_points}.
    """
    if not ids:
        return {}
    out: dict[str, dict] = {}
    try:
        ya = service("youtubeAnalytics", "v2")
    except Exception as e:  # noqa: BLE001
        log.warning("analytics API unavailable for curves: %s", str(e)[:160])
        return {}
    for vid in ids:
        try:
            r = ya.reports().query(ids="channel==MINE", startDate=start, endDate=end,
                                   metrics="audienceWatchRatio", dimensions="elapsedVideoTimeRatio",
                                   filters=f"video=={vid}", sort="elapsedVideoTimeRatio").execute()
            rows = [(float(a), float(b)) for a, b in r.get("rows", []) if b is not None]
            if len(rows) < 5:
                continue
            out[vid] = summarize_curve(rows)
        except Exception as e:  # noqa: BLE001
            log.debug("retention curve for %s unavailable: %s", vid, str(e)[:120])
    if out:
        log.info("retention curves: %d videos", len(out))
    return out


def summarize_curve(rows: list[tuple[float, float]]) -> dict:
    """Condense audienceWatchRatio samples without turning them into viewer percentages.

    audienceWatchRatio counts views of a segment relative to all video views. A replay can make a
    point exceed 1.0; the final point is therefore not a literal share of viewers who completed.
    """
    rows = sorted(rows)

    def at(x: float) -> float:
        prev = rows[0]
        for t, v in rows:
            if t >= x:
                if t == prev[0]:
                    return v
                f = (x - prev[0]) / (t - prev[0])
                return prev[1] + f * (v - prev[1])
            prev = (t, v)
        return rows[-1][1]

    return {
        "watch_ratio_at_15pct": round(at(0.15), 4),
        "end_watch_ratio": round(rows[-1][1], 4),
        "curve_points": len(rows),
        # 11-point resample, stored under the metric's actual name for transparent reporting.
        "audience_watch_ratio_curve": [round(at(i / 10), 4) for i in range(11)],
    }


def can_delete() -> bool:
    """Does the stored refresh token actually carry delete permission?

    Checked rather than assumed: a token minted for youtube.upload will fail a delete
    with a 403 that reads like a permissions bug in our code. Better to say plainly
    that the scope is missing.
    """
    try:
        c = credentials()
    except Exception:  # noqa: BLE001
        return False
    granted = set(getattr(c, "scopes", None) or [])
    return bool(granted & {DELETE_SCOPE, "https://www.googleapis.com/auth/youtube"})


def delete(video_ids: list[str], dry_run: bool = True) -> dict:
    """Delete videos from the channel. Defaults to dry run - this is irreversible.

    Returns a per-id result rather than raising on the first failure, so one bad id
    does not strand the rest half-done.
    """
    import logging

    log = logging.getLogger("autotube.youtube")
    out: dict[str, str] = {}
    if dry_run:
        for v in video_ids:
            out[v] = "dry-run"
        log.info("prune dry run: %d video(s) would be deleted", len(video_ids))
        return out
    if not can_delete():
        raise RuntimeError(
            "the stored YouTube token cannot delete: it holds youtube.upload only. "
            f"Re-authorise including {DELETE_SCOPE} and refresh YT_REFRESH_TOKEN.")
    svc = service()
    for v in video_ids:
        try:
            svc.videos().delete(id=v).execute()
            out[v] = "deleted"
            log.info("deleted %s", v)
        except Exception as e:  # noqa: BLE001
            out[v] = f"failed: {str(e)[:120]}"
            log.error("delete %s failed: %s", v, e)
    return out
