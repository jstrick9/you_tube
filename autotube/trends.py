"""Trend intelligence: collect, normalize, filter and rank trending topics.

Free, keyless sources (all verified working):
  • Google Trends daily RSS (per region)           – what people are searching
  • Wikipedia top pageviews (yesterday)             – what people are reading about
  • Reddit r/todayilearned + r/science top (RSS)    – fact-shaped curiosity signals
  • Hacker News front page (Algolia API)            – tech trends
  • YouTube viral-Shorts outliers (channel OAuth; search.list)  – fact Shorts that just took off
  • Wikipedia spikes (page views vs. the page's own 30-day normal) – sudden curiosity, not perennial pages
  • On this day (Wikimedia feed)                    – same/next-day anniversaries, round numbers boosted
  • YouTube mostPopular chart (API key or channel OAuth, 1 quota unit) – what's winning on YT

Each candidate gets a normalized score in [0,1] per source; scores from multiple
sources are merged (cross-source agreement = stronger trend). Candidates are then
filtered for safety (blocked topics, living-person biographies, tragedies) and
freshness (not produced in the last N days).
"""
from __future__ import annotations

import logging
import math
import os
import re
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, quote, urlsplit

import feedparser

from .common import get_json, http, now_utc, read_json

log = logging.getLogger("autotube.trends")

WIKI_SKIP_PREFIX = ("Special:", "Main_Page", "Wikipedia:", "File:", "Portal:", "Help:", "Talk:",
                    "Template:", "Category:", "User:", "Deaths_in", "List_of", "Cleopatra_(disambiguation)")
WIKI_SKIP_EXACT = {"-", "Main_Page", "Undefined", "XXX", "Search"}


def _norm_rank(i: int, n: int) -> float:
    return round(1.0 - (i / max(n, 1)) * 0.85, 4)


def _parse_traffic(value: str | int | None) -> int:
    """Parse Google Trends RSS approximate traffic, including 25K+/1.2M+ suffixes."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value)) if math.isfinite(value) and value.is_integer() else 0
    match = re.fullmatch(r"\s*((?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?)\s*([KMB]?)\s*\+?\s*",
                         str(value or "").upper())
    if not match:
        return 0
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return 0
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[match.group(2)]
    return max(0, int(number * multiplier))


def _provider_count(value) -> int | None:
    """Parse a non-negative API count; Boolean values and coercible garbage are not counts."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        try:
            return int(value.strip())
        except (ValueError, OverflowError):
            return None
    return None


def google_trends(geo: str) -> list[dict]:
    out = []
    try:
        feed = feedparser.parse(http().get(f"https://trends.google.com/trending/rss?geo={geo}", timeout=20).content)
        n = len(feed.entries)
        for i, e in enumerate(feed.entries):
            t = _parse_traffic(e.get("ht_approx_traffic", "0"))
            news = []
            for k in ("ht_news_item_title",):
                if e.get(k):
                    news.append(e.get(k))
            out.append({
                "topic": e.title.strip(), "source": "google_trends",
                "score": min(1.0, 0.4 + math.log10(t + 10) / 7) * _norm_rank(i, n) ** 0.3,
                "context": news, "traffic": t, "source_rank": i + 1,
                "source_url": e.get("ht_news_item_url") or e.get("link") or "",
                "published_at": e.get("published") or e.get("updated") or "",
            })
    except Exception as e:  # noqa: BLE001
        log.warning("google trends failed: %s", e)
    return out


def wikipedia_top(lang: str = "en", limit: int = 60) -> list[dict]:
    out = []
    for back in (1, 2):
        d = now_utc() - timedelta(days=back)
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang}.wikipedia/all-access/"
               f"{d:%Y}/{d:%m}/{d:%d}")
        try:
            data = get_json(url)
            arts = data["items"][0]["articles"]
            arts = [a for a in arts if not a["article"].startswith(WIKI_SKIP_PREFIX)
                    and a["article"] not in WIKI_SKIP_EXACT][:limit]
            for i, a in enumerate(arts):
                out.append({"topic": a["article"].replace("_", " "), "source": "wikipedia",
                            "score": _norm_rank(i, len(arts)), "wiki_title": a["article"],
                            "source_url": f"https://{lang}.wikipedia.org/wiki/{quote(a['article'].replace(' ', '_'))}",
                            "source_date": d.date().isoformat(), "source_rank": i + 1,
                            "views": a["views"]})
            break
        except Exception as e:  # noqa: BLE001
            log.warning("wikipedia top (%s) failed: %s", d.date(), e)
    return out


def reddit_rss(sub: str, limit: int = 25) -> list[dict]:
    out = []
    try:
        r = http().get(f"https://www.reddit.com/r/{sub}/top/.rss?t=day&limit={limit}",
                       headers={"User-Agent": "Mozilla/5.0 (AutoTube trend reader)"}, timeout=20)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        n = len(feed.entries)
        if n == 0:
            log.info("reddit r/%s RSS returned no entries (content-type=%s, parse_warning=%s)", sub,
                     str(r.headers.get("Content-Type", ""))[:80], bool(getattr(feed, "bozo", False)))
        for i, e in enumerate(feed.entries):
            title = re.sub(r"^(TIL( that)?|TIL:)\s*", "", e.title, flags=re.I).strip()
            out.append({"topic": title, "source": f"reddit_{sub}", "score": _norm_rank(i, n) * 0.9,
                        "context": [e.title], "source_url": e.get("link") or "",
                        "published_at": e.get("published") or e.get("updated") or "",
                        "source_rank": i + 1, "is_fact": True})
    except Exception as e:  # noqa: BLE001
        log.warning("reddit %s failed: %s", sub, e)
    return out


def hackernews(limit: int = 20) -> list[dict]:
    out = []
    try:
        data = get_json("https://hn.algolia.com/api/v1/search", params={"tags": "front_page", "hitsPerPage": limit})
        hits = sorted(data.get("hits", []), key=lambda h: -(h.get("points") or 0))
        for i, h in enumerate(hits):
            item_url = h.get("url") or (f"https://news.ycombinator.com/item?id={h.get('objectID')}"
                                         if h.get("objectID") else "")
            out.append({"topic": h["title"], "source": "hackernews", "score": _norm_rank(i, len(hits)) * 0.7,
                        "context": [item_url] if item_url else [], "source_url": item_url,
                        "points": h.get("points"), "source_rank": i + 1,
                        "published_at": h.get("created_at") or "", "category_hint": "technology"})
    except Exception as e:  # noqa: BLE001
        log.warning("hackernews failed: %s", e)
    return out


def _yt_oauth_ready() -> bool:
    return all(os.environ.get(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"))


def youtube_chart(region: str, limit: int = 30) -> list[dict]:
    """videos.list chart=mostPopular — costs 1 quota unit (API key, or the channel's OAuth token)."""
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key and not _yt_oauth_ready():
        return []
    out = []
    try:
        params = {"part": "snippet,statistics", "chart": "mostPopular", "regionCode": region, "maxResults": limit}
        if key:
            data = get_json("https://www.googleapis.com/youtube/v3/videos", params={**params, "key": key})
        else:
            from . import youtube
            data = youtube.service().videos().list(**params).execute()
        items = data.get("items", [])
        for i, it in enumerate(items):
            sn = it["snippet"]
            video_id = it.get("id") or ""
            raw_views = (it.get("statistics") or {}).get("viewCount")
            views = _provider_count(raw_views)
            out.append({"topic": sn["title"], "source": "youtube_chart", "score": _norm_rank(i, len(items)) * 0.8,
                        "context": sn.get("tags", [])[:8], "yt_category": sn.get("categoryId"),
                        "source_url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
                        "video_id": video_id, "view_count": views if views is not None else 0,
                        "published_at": sn.get("publishedAt") or "", "channel_title": sn.get("channelTitle", ""),
                        "source_rank": i + 1})
    except Exception as e:  # noqa: BLE001
        log.warning("youtube chart failed: %s", e)
    return out


def _clean_title(t: str) -> str:
    t = re.sub(r"#\S+", "", t)                                   # hashtags
    t = re.sub(r"[^\w\s'’\-:,.!?$%&()/]", "", t)                # emoji / symbols
    return re.sub(r"\s+", " ", t).strip(" -|:")


def _parse_ts(ts: str):
    from datetime import datetime
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def youtube_duration_seconds(value: str | None) -> int | None:
    """Parse the ISO-8601 duration returned by videos.list(contentDetails)."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", value.strip())
    if not match or not any(match.groups()):
        return None
    try:
        hours, minutes, seconds = match.groups()
        total = int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds or 0)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(total) or total <= 0:
        return None
    return int(round(total))


# ── lanes & format DNA ────────────────────────────────────────────────────────
HOOK_PATTERNS = [
    ("question",      r"^\s*(why|how|what|who|when|where|which|can|did|do|does|is|are|was|could|should)\b|\?"),
    ("number_first",  r"^\s*(?:top\s+)?\d|^\s*\d+\s|\b\d+\s+(things|facts|reasons|ways|times)\b"),
    ("disbelief",     r"\b(actually|no one|nobody|never|wait|insane|crazy|wild|unbelievable|won'?t believe|shocking|turns out)\b"),
    ("you_statement", r"\b(you|your|you'?re|yourself)\b"),
]


def hook_pattern(title: str) -> str:
    """Classify a title into one of the channel's hook_styles.

    Deliberately ordered: a title can satisfy several patterns, and the earlier ones are the more
    specific claim about its shape. Anything unmatched is a plain assertion — a bold_claim.
    """
    t = (title or "").strip().lower()
    for name, pat in HOOK_PATTERNS:
        if re.search(pat, t):
            return name
    return "bold_claim"


def lane_queries(cfg: dict) -> list[tuple[str, str]]:
    """(lane, query) pairs for this run: every lane represented, rotating within each lane.

    Two guarantees that a single flat list cannot give at once. Every lane is searched on every
    run, because a lane that goes unsearched looks exactly like a lane with nothing trending in it
    and would quietly stop feeding the channel. And every query still gets its turn over a few
    days, because the cap usually allows fewer queries than the lanes hold.

    Rotating a flattened list satisfies only the second: the day's window can land entirely inside
    two lanes and skip the rest.
    """
    t = cfg.get("trends", {}) or {}
    lanes = t.get("lanes") or {}
    if not lanes:
        pairs = [("", q) for q in (t.get("youtube_outlier_queries") or [])]
        cap = int(t.get("max_queries_per_run", 0) or len(pairs))
        return pairs[:cap] if cap else pairs

    names = [n for n, spec in lanes.items() if (spec.get("queries") or [])]
    if not names:
        return []
    total = sum(len(lanes[n]["queries"]) for n in names)
    cap = int(t.get("max_queries_per_run", 0) or total)
    cap = max(len(names), min(cap, total))        # never starve a lane entirely

    day = now_utc().timetuple().tm_yday
    # round-robin the budget across lanes, so a cap smaller than the query count still touches each
    take = {n: 0 for n in names}
    for i in range(cap):
        n = names[i % len(names)]
        if take[n] < len(lanes[n]["queries"]):
            take[n] += 1
    out: list[tuple[str, str]] = []
    for n in names:
        qs = lanes[n]["queries"]
        off = day % len(qs)
        out += [(n, qs[(off + i) % len(qs)]) for i in range(take[n])]
    return out


def lane_of(cfg: dict, category: str) -> str:
    for name, spec in (cfg.get("trends", {}).get("lanes") or {}).items():
        if category in (spec.get("categories") or []):
            return name
    return ""


def _pulse_summary(rows: list[dict]) -> dict:
    """Aggregate metadata patterns only; never retain an example title or creator identity."""
    hook_velocity: dict[str, float] = {}
    hook_counts: dict[str, int] = {}
    duration_velocity: dict[str, float] = {}
    duration_counts: dict[str, int] = {}
    total_hook_velocity = 0.0
    total_duration_velocity = 0.0
    valid_patterns = {name for name, _ in HOOK_PATTERNS} | {"bold_claim"}
    for row in rows:
        if not isinstance(row, dict):
            continue
        pattern = row.get("hook")
        if not isinstance(pattern, str) or pattern not in valid_patterns:
            title = row.get("topic")
            pattern = hook_pattern(title if isinstance(title, str) else "")
        velocity = row.get("vph", 0)
        try:
            velocity = float(velocity) if not isinstance(velocity, bool) else 0.0
        except (TypeError, ValueError, OverflowError):
            velocity = 0.0
        if not math.isfinite(velocity) or velocity < 0:
            velocity = 0.0
        hook_counts[pattern] = hook_counts.get(pattern, 0) + 1
        hook_velocity[pattern] = hook_velocity.get(pattern, 0.0) + velocity
        total_hook_velocity += velocity

        seconds = row.get("duration_seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            continue
        try:
            seconds = float(seconds)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(seconds) or seconds <= 0:
            continue
        if seconds <= 20:
            band = "le_20s"
        elif seconds <= 40:
            band = "21_40s"
        elif seconds <= 60:
            band = "41_60s"
        else:
            band = "over_60s"
        duration_counts[band] = duration_counts.get(band, 0) + 1
        duration_velocity[band] = duration_velocity.get(band, 0.0) + velocity
        total_duration_velocity += velocity

    hook_shares = ({key: round(value / total_hook_velocity, 4) for key, value in
                    sorted(hook_velocity.items(), key=lambda item: -item[1])}
                   if total_hook_velocity > 0 else {})
    duration_shares = ({key: round(value / total_duration_velocity, 4) for key, value in
                        sorted(duration_velocity.items(), key=lambda item: -item[1])}
                       if total_duration_velocity > 0 else {})
    return {
        "sample": len(rows),
        "hook_style": hook_shares,
        # This is a title-text classifier, not the spoken opening of the video.
        "title_hook_style": dict(hook_shares),
        "videos": hook_counts,
        "duration_sample": sum(duration_counts.values()),
        "duration_bands": duration_shares,
    }


def format_pulse(rows: list[dict]) -> dict:
    """Summarize broad format signals from viral Shorts without collecting their execution.

    The official video metadata used here supports title-pattern labels and clip duration only.
    Beat counts, spoken hooks and caption density are not available from that metadata, so they
    are explicitly left unmeasured. All shares are weighted by observed views/hour (an association,
    not evidence that a format caused a breakout) and are stored as anonymous aggregates, never as
    title, description, script, transcript, footage or creator examples.
    """
    valid_rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    summary = _pulse_summary(valid_rows)
    by_lane: dict[str, dict] = {}
    lane_names = sorted({row.get("lane") for row in valid_rows
                         if isinstance(row.get("lane"), str) and row["lane"]})
    for lane in lane_names:
        by_lane[lane] = _pulse_summary([row for row in valid_rows if row.get("lane") == lane])
    return {
        "schema_version": 2,
        "updated": now_utc().isoformat(timespec="seconds"),
        **summary,
        "by_lane": by_lane,
        "aggregate_basis": "YouTube title-pattern labels and official video duration metadata; views/hour weighted",
        "not_measured": ["spoken_hook", "beat_count", "caption_density"],
    }


def _rank_youtube_outlier_rows(rows: list[dict]) -> list[dict]:
    """Turn measured YouTube rows into bounded selector evidence without creator executions."""
    rows.sort(key=lambda r: -(math.log10(r["vph"] + 1) + 0.6 * math.log10(r["ratio"] + 1)))
    n = len(rows)
    out = []
    for i, row in enumerate(rows[:40]):
        subscriber_multiple = row["views"] / row["subs"] if row["subs"] > 0 else 0.0
        out.append({
            "topic": row["topic"], "source": "youtube_outliers", "is_fact": True,
            "lane": row["lane"], "hook": row["hook"],
            "score": round(min(1.0, 0.55 + 0.45 * _norm_rank(i, n)), 4),
            "context": [f"viral Short now: {row['views']:,} views in {row['hours']:.0f}h"
                        + (f", {subscriber_multiple:.0f}x its channel's subscribers" if row["subs"] else "")],
            "source_url": row["source_url"], "video_id": row["video_id"],
            "published_at": row["published_at"], "channel_title": row["channel_title"],
            "evidence": {"views": row["views"], "hours": round(row["hours"], 1), "vph": round(row["vph"]),
                         "breakout": round(subscriber_multiple, 1), "video_id": row["video_id"],
                         "video_url": row["source_url"], "published_at": row["published_at"],
                         "channel_title": row["channel_title"], "search_query": row["query"],
                         "duration_seconds": row["duration_seconds"], "subscriber_count": row["subs"],
                         "channel_id": row.get("channel_id", "")},
        })
    return out


def _refresh_cached_youtube_outliers(yt, cached: list[dict], cfg: dict,
                                     pairs: list[tuple[str, str]], min_views: int,
                                     max_outlier_hours: float) -> list[dict]:
    """Re-fetch cached video metrics before treating yesterday's search as today's trend evidence.

    Search.list is the quota-expensive discovery call (100 units/query), so its result IDs are cached for
    the UTC day. videos.list and channels.list are only one unit per 50 IDs each; they refresh view count,
    publication age, channel size and duration on every run. A stale/removed/private video is dropped. If
    the video metrics cannot be refreshed, the caller fails closed rather than reusing stale view/hour data.
    """
    valid_pairs = set(pairs)
    by_id: dict[str, dict] = {}
    for row in cached:
        if not isinstance(row, dict) or row.get("source") != "youtube_outliers":
            continue
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        video_id = str(row.get("video_id") or evidence.get("video_id") or "").strip()
        url = str(row.get("source_url") or evidence.get("video_url") or "").strip()
        lane = row.get("lane") or evidence.get("lane")
        query = evidence.get("search_query")
        if (not video_id or not _youtube_video_url_matches(url, video_id)
                or not isinstance(lane, str) or not isinstance(query, str)
                or (lane, query) not in valid_pairs):
            continue
        by_id.setdefault(video_id, {"lane": lane, "query": query})
    if not by_id:
        return []

    videos: dict[str, dict] = {}
    ids = list(by_id)
    for i in range(0, len(ids), 50):
        page = yt.videos().list(part="snippet,statistics,contentDetails",
                                id=",".join(ids[i:i + 50])).execute()
        for video in page.get("items", []):
            if isinstance(video, dict) and video.get("id") in by_id:
                videos[video["id"]] = video

    channel_ids = sorted({str((video.get("snippet") or {}).get("channelId") or "").strip()
                          for video in videos.values()
                          if str((video.get("snippet") or {}).get("channelId") or "").strip()})
    subscribers: dict[str, int] = {}
    try:
        for i in range(0, len(channel_ids), 50):
            page = yt.channels().list(part="statistics", id=",".join(channel_ids[i:i + 50])).execute()
            for channel in page.get("items", []):
                if not isinstance(channel, dict) or not channel.get("id"):
                    continue
                count = _provider_count((channel.get("statistics") or {}).get("subscriberCount"))
                if count is not None:
                    subscribers[channel["id"]] = count
    except Exception as e:  # noqa: BLE001
        # Current views and age are the qualification evidence; subscriber counts only refine the rank.
        log.warning("youtube cached outlier subscriber refresh failed: %s", str(e)[:160])

    own = str(cfg.get("channel", {}).get("name", "")).strip().casefold()
    now = now_utc()  # take the age measurement after both API calls, immediately before building evidence
    rows = []
    for video_id, meta in by_id.items():
        video = videos.get(video_id)
        if not video:
            continue
        snippet = video.get("snippet") if isinstance(video.get("snippet"), dict) else {}
        stats = video.get("statistics") if isinstance(video.get("statistics"), dict) else {}
        views = _provider_count(stats.get("viewCount"))
        title = snippet.get("title")
        channel_title = str(snippet.get("channelTitle") or "").strip()
        if (views is None or views < min_views
                or (own and channel_title.casefold() == own)
                or not isinstance(title, str)):
            continue
        published_at = snippet.get("publishedAt")
        try:
            published = _parse_ts(published_at)
            if published.tzinfo is None:
                continue
            age_hours = (now - published.astimezone(timezone.utc)).total_seconds() / 3600
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(age_hours) or not 0 < age_hours <= max_outlier_hours:
            continue
        topic = _clean_title(title)
        if len(topic) < 8:
            continue
        channel_id = str(snippet.get("channelId") or "").strip()
        # Do not describe a cached subscriber ratio as current if channels.list failed or hid it.
        # View velocity and publication age remain the live qualification metrics.
        subscriber_count = subscribers.get(channel_id, 0)
        hours = max(1.0, age_hours)
        vph = views / hours
        ratio = views / max(subscriber_count, 1000)
        rows.append({"topic": topic, "vph": vph, "ratio": ratio, "views": views, "hours": hours,
                     "subs": subscriber_count, "duration_seconds": youtube_duration_seconds(
                         (video.get("contentDetails") or {}).get("duration")),
                     "video_id": video_id, "source_url": f"https://www.youtube.com/watch?v={video_id}",
                     "published_at": published_at, "channel_title": channel_title,
                     "channel_id": channel_id, "query": meta["query"], "lane": meta["lane"],
                     "hook": hook_pattern(topic)})
    return rows


def youtube_outliers(cfg: dict) -> list[dict]:
    """Find current viral Shorts in approved lanes while keeping expensive discovery within quota.

    Search.list (100 units/query) discovers recent outliers once per UTC day. On later runs that day,
    cached video IDs are revalidated with fresh YouTube view, publication, and channel statistics before
    their evidence can reach the selector. No creator title, description, script, caption, footage, audio
    or identity is used as a format example; only aggregate lane-level format metrics are retained.
    """
    t = cfg.get("trends", {})
    pairs = lane_queries(cfg)
    if not pairs or not _yt_oauth_ready():
        return []
    try:
        min_views = int(t.get("outlier_min_views", 30000))
        max_outlier_hours = float(t.get("outlier_window_hours", 96))
    except (TypeError, ValueError, OverflowError):
        log.warning("invalid YouTube outlier thresholds; skipping source")
        return []
    if min_views <= 0 or not math.isfinite(max_outlier_hours) or max_outlier_hours <= 0:
        log.warning("invalid YouTube outlier thresholds; skipping source")
        return []

    from .common import write_json
    today = now_utc().date().isoformat()
    cache = read_json("trend_cache.json", {})
    if not isinstance(cache, dict):
        cache = {}
    if cache.get("yt_outliers_date") == today and isinstance(cache.get("yt_outliers"), list):
        try:
            from . import youtube
            rows = _refresh_cached_youtube_outliers(
                youtube.service(), cache["yt_outliers"], cfg, pairs, min_views, max_outlier_hours)
            out = _rank_youtube_outlier_rows(rows)
            log.info("youtube outliers: %d from cached search, live-revalidated (%d cached IDs)",
                     len(out), len(cache["yt_outliers"]))
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("youtube cached outlier refresh failed; ignoring stale metrics: %s", str(e)[:200])
            return []

    region = cfg["channel"].get("region", "US")
    lang = cfg["channel"].get("language", "en")
    since = now_utc() - timedelta(hours=max_outlier_hours)
    try:
        from . import youtube
        yt = youtube.service()
        found: dict[str, tuple[str, str]] = {}        # videoId -> (query, lane)
        for lane, query in pairs:
            try:
                response = yt.search().list(
                    part="id", q=query, type="video", videoDuration="short", order="viewCount",
                    publishedAfter=since.strftime("%Y-%m-%dT%H:%M:%SZ"), regionCode=region,
                    relevanceLanguage=lang, safeSearch="strict", maxResults=25).execute()
                for item in response.get("items", []):
                    video_id = ((item.get("id") or {}).get("videoId")
                                if isinstance(item, dict) else None)
                    if video_id:
                        found.setdefault(video_id, (query, lane))
            except Exception as e:  # noqa: BLE001
                log.warning("youtube outlier search %r failed: %s", query, str(e)[:120])
        ids = list(found)
        videos = []
        for i in range(0, len(ids), 50):
            page = yt.videos().list(part="snippet,statistics,contentDetails",
                                    id=",".join(ids[i:i + 50])).execute()
            videos.extend(page.get("items", []))
        channel_ids = sorted({str((video.get("snippet") or {}).get("channelId") or "").strip()
                              for video in videos
                              if str((video.get("snippet") or {}).get("channelId") or "").strip()})
        subscribers: dict[str, int] = {}
        for i in range(0, len(channel_ids), 50):
            page = yt.channels().list(part="statistics", id=",".join(channel_ids[i:i + 50])).execute()
            for channel in page.get("items", []):
                count = _provider_count((channel.get("statistics") or {}).get("subscriberCount"))
                if count is not None:
                    subscribers[channel["id"]] = count
        own = str(cfg.get("channel", {}).get("name", "")).strip().casefold()
        now = now_utc()
        rows = []
        for video in videos:
            snippet = video.get("snippet") if isinstance(video.get("snippet"), dict) else {}
            stats = video.get("statistics") if isinstance(video.get("statistics"), dict) else {}
            views = _provider_count(stats.get("viewCount"))
            channel_title = str(snippet.get("channelTitle") or "").strip()
            if (views is None or views < min_views
                    or (own and channel_title.casefold() == own)):
                continue
            published_at = snippet.get("publishedAt")
            try:
                published = _parse_ts(published_at)
                if published.tzinfo is None:
                    continue
                age_hours = (now - published.astimezone(timezone.utc)).total_seconds() / 3600
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(age_hours) or not 0 < age_hours <= max_outlier_hours:
                continue
            topic_raw = snippet.get("title")
            if not isinstance(topic_raw, str):
                continue
            topic = _clean_title(topic_raw)
            if len(topic) < 8:
                continue
            video_id = str(video.get("id") or "").strip()
            if not video_id:
                continue
            channel_id = str(snippet.get("channelId") or "").strip()
            subscriber_count = subscribers.get(channel_id, 0)
            hours = max(1.0, age_hours)
            vph = views / hours
            ratio = views / max(subscriber_count, 1000)
            query, lane = found.get(video_id, ("", ""))
            rows.append({"topic": topic, "vph": vph, "ratio": ratio, "views": views, "hours": hours,
                         "subs": subscriber_count, "duration_seconds": youtube_duration_seconds(
                             (video.get("contentDetails") or {}).get("duration")),
                         "video_id": video_id, "source_url": f"https://www.youtube.com/watch?v={video_id}",
                         "published_at": published_at, "channel_title": channel_title, "channel_id": channel_id,
                         "query": query, "lane": lane, "hook": hook_pattern(topic)})

        out = _rank_youtube_outlier_rows(rows)
        by_lane: dict[str, int] = {}
        for row in out:
            by_lane[row.get("lane") or "-"] = by_lane.get(row.get("lane") or "-", 0) + 1
        log.info("youtube outliers: %d viral Shorts from %d searches across %d lanes (%s)",
                 len(out), len(pairs), len(by_lane),
                 ", ".join(f"{key}:{value}" for key, value in sorted(by_lane.items())))
        cache.update({"yt_outliers_date": today, "yt_outliers": out})
        write_json("trend_cache.json", cache)
        if t.get("format_pulse", True) and rows:
            pulse = format_pulse(rows)
            write_json("format_pulse.json", pulse)
            log.info("format pulse (%d viral Shorts): %s", pulse["sample"],
                     ", ".join(f"{key} {value:.0%}" for key, value in pulse["hook_style"].items()))
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("youtube outliers failed: %s", str(e)[:200])
        return []


def wikipedia_spikes(items: list[dict], lang: str = "en", top: int = 40) -> None:
    """Re-score Wikipedia top pages by how far above THEIR OWN 30-day normal they are (in place).

    A page that is always popular (e.g. a perennial reference page) is not a trend; one read 5x more than usual is.
    """
    from concurrent.futures import ThreadPoolExecutor
    from statistics import median
    end = now_utc() - timedelta(days=1)
    start = end - timedelta(days=30)

    def spike(it):
        title = it.get("wiki_title") or it["topic"].replace(" ", "_")
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/user/"
               f"{quote(title, safe='')}/daily/{start:%Y%m%d}/{end:%Y%m%d}")
        try:
            views = [x["views"] for x in get_json(url, retries=1)["items"]]
            if len(views) < 8:
                return it, None
            base = median(views[:-3]) or 1
            return it, max(views[-2:]) / base
        except Exception:  # noqa: BLE001
            return it, None

    wiki = [i for i in items if i["source"] == "wikipedia"][:top]
    with ThreadPoolExecutor(4) as ex:
        for it, ratio in ex.map(spike, wiki):
            if ratio is None:
                continue
            it["spike"] = round(ratio, 1)
            # 1x normal → score x0.45 (perennial), 3x → ~x0.8, 8x+ → x1.0 (and a small bonus)
            mult = 0.45 + 0.55 * min(1.0, math.log2(max(ratio, 1.0)) / 3)
            it["score"] = round(min(1.0, it["score"] * mult + (0.1 if ratio >= 8 else 0)), 4)
            it.setdefault("context", []).append(f"Wikipedia: {it.get('views', 0):,} views/day, {ratio:.1f}x its normal")


SOURCE_ARMS = ["youtube_outliers", "google_trends", "youtube_chart", "reddit", "hackernews", "wikipedia",
               "on_this_day", "evergreen"]

def source_family(source: str) -> str:
    """Independent trend platform, so repeats within one feed cannot masquerade as consensus."""
    if str(source).startswith("reddit_"):
        return "reddit"
    if source in {"youtube_outliers", "youtube_chart"}:
        return "youtube"
    if source in {"wikipedia", "on_this_day"}:
        return "wikimedia"
    return str(source or "")


def _signal_record(row: dict, captured_at: str) -> dict:
    """Compact, serializable evidence that can be audited after the temporary run is gone."""
    evidence = dict(row.get("evidence") or {}) if isinstance(row.get("evidence"), dict) else {}
    for key in ("traffic", "views", "view_count", "spike", "source_rank", "source_date", "points",
                "video_id", "published_at", "channel_title", "yt_category", "lane"):
        if row.get(key) not in (None, ""):
            evidence.setdefault(key, row[key])
    context = [str(x)[:280] for x in (row.get("context") or [])[:3] if x]
    return {"source": str(row.get("source") or ""),
            "source_family": source_family(str(row.get("source") or "")),
            "topic": str(row.get("topic") or "")[:200],
            "score": row.get("score"), "source_url": str(row.get("source_url") or ""),
            "captured_at": captured_at, "context": context, "evidence": evidence}


def _signal_is_recent(signal: dict, max_age_hours: float, now: datetime | None = None) -> bool:
    """Require a recent, auditable collection timestamp; source names alone are never evidence."""
    raw = signal.get("captured_at")
    if not raw:
        return False
    try:
        captured = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=timezone.utc)
        age = (now or now_utc()) - captured.astimezone(timezone.utc)
        return timedelta(minutes=-5) <= age <= timedelta(hours=max_age_hours)
    except (TypeError, ValueError, OverflowError):
        return False


def _evidence_number(value, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _trend_threshold(cfg: dict, key: str, default: float, *, integer: bool = False) -> float | int | None:
    """Read a positive numeric threshold without treating Boolean or malformed config as evidence."""
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        return None
    number = _evidence_number(raw, float("nan"))
    if not math.isfinite(number) or number <= 0 or (integer and not number.is_integer()):
        return None
    return int(number) if integer else number


def _valid_http_source_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname) \
            and not parsed.username and not parsed.password
    except (TypeError, ValueError):
        return False


def _youtube_video_url_matches(source_url: str, video_id: str) -> bool:
    """A YouTube trend signal must cite the same real video ID recorded in its evidence."""
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        return False
    try:
        parsed = urlsplit(source_url)
        if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
            return False
        host = parsed.hostname.lower().rstrip(".")
        parts = [part for part in parsed.path.split("/") if part]
        if host in {"youtu.be", "www.youtu.be"}:
            cited_id = parts[0] if len(parts) == 1 else ""
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if parts == ["watch"]:
                video_ids = parse_qs(parsed.query, keep_blank_values=True).get("v", [])
                cited_id = video_ids[0] if len(video_ids) == 1 else ""
            elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                cited_id = parts[1]
            else:
                cited_id = ""
        else:
            return False
        return cited_id == video_id
    except (TypeError, ValueError):
        return False


def has_current_trend_evidence(candidate: dict, min_wikipedia_spike: float = 3.0,
                               trend_cfg: dict | None = None) -> bool:
    """Require measured, recent, source-linked trend evidence.

    A measured YouTube outlier or a sufficiently strong Google Trends result can qualify on its own.
    Reddit, Hacker News, YouTube-chart and Wikipedia-spike signals are discovery/corroboration
    evidence only: at least two *independent source families* must agree. Multiple subreddits still
    count as one family; a Wikipedia listing, anniversary, evergreen seed, or source label alone never
    proves a topic is trending. Every accepted signal must retain a URL and recent capture timestamp.
    """
    if not isinstance(candidate, dict):
        return False
    if trend_cfg is not None and not isinstance(trend_cfg, dict):
        return False
    tcfg = trend_cfg or {}
    wiki_threshold = _evidence_number(min_wikipedia_spike, float("nan"))
    min_views = _trend_threshold(tcfg, "outlier_min_views", 30_000, integer=True)
    max_outlier_hours = _trend_threshold(tcfg, "outlier_window_hours", 72)
    min_google_traffic = _trend_threshold(tcfg, "min_google_trend_traffic", 1_000, integer=True)
    min_hn_points = _trend_threshold(tcfg, "min_hackernews_points", 10, integer=True)
    max_capture_age = _trend_threshold(tcfg, "trend_evidence_max_age_hours", 36)
    min_families = _trend_threshold(tcfg, "min_independent_trend_families", 2, integer=True)
    if (not math.isfinite(wiki_threshold) or wiki_threshold <= 0 or min_views is None
            or max_outlier_hours is None or min_google_traffic is None or min_hn_points is None
            or max_capture_age is None or min_families is None or min_families < 2):
        return False

    signals = candidate.get("signals")
    if not isinstance(signals, list):
        return False
    now = now_utc()
    corroborating_families: set[str] = set()
    for signal in signals:
        if not isinstance(signal, dict) or not _signal_is_recent(signal, max_capture_age, now):
            continue
        source = str(signal.get("source") or "")
        evidence = signal.get("evidence") if isinstance(signal.get("evidence"), dict) else {}
        source_url = str(signal.get("source_url") or evidence.get("video_url") or "").strip()
        if not _valid_http_source_url(source_url):
            continue

        if source == "youtube_outliers":
            views = _evidence_number(evidence.get("views") or evidence.get("view_count"), float("nan"))
            hours = _evidence_number(evidence.get("hours"), float("nan"))
            vph = _evidence_number(evidence.get("vph"), float("nan"))
            expected_vph = views / hours if math.isfinite(views) and math.isfinite(hours) and hours > 0 else float("nan")
            measured_rate_agrees = (math.isfinite(expected_vph) and math.isfinite(vph) and vph > 0
                                    and abs(vph - expected_vph) <= max(1.0, expected_vph * 0.1))
            raw_published = evidence.get("published_at") or signal.get("published_at")
            try:
                published = datetime.fromisoformat(str(raw_published).replace("Z", "+00:00"))
                if published.tzinfo is None:
                    raise ValueError("publication timestamp has no timezone")
                actual_hours = (now - published.astimezone(timezone.utc)).total_seconds() / 3600
                measured_time_agrees = (math.isfinite(actual_hours) and 0 < actual_hours <= max_outlier_hours
                                        and abs(hours - max(1.0, actual_hours)) <= 0.11)
            except (TypeError, ValueError, OverflowError):
                measured_time_agrees = False
            evidence_id = str(evidence.get("video_id") or "").strip()
            signal_id = str(signal.get("video_id") or "").strip()
            video_id = evidence_id or signal_id
            ids_agree = not (evidence_id and signal_id) or evidence_id == signal_id
            if (ids_agree and _youtube_video_url_matches(source_url, video_id) and measured_rate_agrees
                    and measured_time_agrees and views >= min_views and 0 < hours <= max_outlier_hours):
                return True
            continue

        if source == "google_trends":
            traffic = _evidence_number(evidence.get("traffic") or signal.get("traffic"))
            if traffic >= min_google_traffic:
                return True
            if traffic > 0:
                corroborating_families.add("google_trends")
            continue

        if source == "wikipedia":
            spike = _evidence_number(evidence.get("spike") or signal.get("spike"))
            if spike >= wiki_threshold:
                corroborating_families.add("wikimedia")
            continue

        if source == "youtube_chart":
            views = _evidence_number(evidence.get("view_count") or evidence.get("views"))
            video_id = str(evidence.get("video_id") or signal.get("video_id") or "").strip()
            if views > 0 and _youtube_video_url_matches(source_url, video_id):
                corroborating_families.add("youtube")
            continue

        if source == "hackernews":
            points = _evidence_number(evidence.get("points") or signal.get("points"))
            if points >= min_hn_points:
                corroborating_families.add("hackernews")
            continue

        if source.startswith("reddit_"):
            corroborating_families.add("reddit")

    return len(corroborating_families) >= min_families


def primary_source(sources: list[str]) -> str:
    """The strongest trend source behind a topic (used as a learned 'source' arm)."""
    norm = ["reddit" if x.startswith("reddit") else x for x in sources or []]
    for s in SOURCE_ARMS:
        if s in norm:
            return s
    return norm[0] if norm else "evergreen"


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def on_this_day(lang: str = "en") -> list[dict]:
    """Anniversaries for today and tomorrow (videos publish same/next day). Round anniversaries score higher."""
    out = []
    now = now_utc()
    for d in (now, now + timedelta(days=1)):
        try:
            data = get_json(f"https://api.wikimedia.org/feed/v1/wikipedia/{lang}/onthisday/selected/{d:%m}/{d:%d}",
                            retries=2)
        except Exception as e:  # noqa: BLE001
            log.warning("on this day (%s) failed: %s", d.date(), e)
            continue
        for ev in data.get("selected", [])[:25]:
            try:
                year = int(ev.get("year"))
            except (TypeError, ValueError):
                continue
            ago = d.year - year
            pages = ev.get("pages") or []
            if ago <= 0 or not pages:
                continue
            title = pages[0].get("titles", {}).get("canonical") or pages[0].get("title", "")
            round_n = ago % 100 == 0 or ago % 50 == 0 or ago % 25 == 0
            out.append({"topic": pages[0].get("titles", {}).get("normalized", title.replace("_", " ")),
                        "source": "on_this_day", "wiki_title": title,
                        "score": 0.62 if round_n else (0.5 if ago % 10 == 0 else 0.4),
                        "context": [f"{'TODAY' if d is now else 'TOMORROW'} is the {_ordinal(ago)} anniversary ({d:%b} {d.day}, "
                                    f"{year}): {ev.get('text', '')[:140]}"]})
    return out


# ── filtering ─────────────────────────────────────────────────────────────────
def _blocked(text: str, blocked: list[str]) -> str | None:
    low = f" {text.lower()} "
    for b in blocked:
        if re.search(rf"\b{re.escape(b.lower())}\b", low):
            return b
    return None


def is_living_person(summary: dict) -> bool:
    """Heuristic from Wikipedia REST summary description, e.g. 'American actor (born 1980)'."""
    desc = (summary.get("description") or "").lower()
    extract = (summary.get("extract") or "")[:300].lower()
    if re.search(r"\(born\s+\d{4}\)", desc) or re.search(r"\bborn\s+(\d{1,2}\s+\w+\s+)?\d{4}\)", desc):
        return True
    if re.search(r"\(born [^)]*\d{4}\)", extract) and "died" not in extract[:200]:
        return True
    if any(w in desc for w in ("footballer", "actor", "actress", "singer", "rapper", "politician",
                               "businessman", "businesswoman", "influencer", "youtuber", "wrestler",
                               "basketball player", "tennis player", "golfer", "quarterback", "presenter")):
        if not re.search(r"\d{4}\s*[–-]\s*\d{4}", desc):
            return True
    return False


def recent_topics(days: int) -> set[str]:
    hist = read_json("history.json", [])
    cutoff = now_utc() - timedelta(days=days)
    keys = set()
    for h in hist:
        try:
            from datetime import datetime
            if datetime.fromisoformat(h["created_at"]) >= cutoff:
                keys.add(h.get("topic_key", "").lower())
                if h.get("wiki_title"):
                    keys.add(h["wiki_title"].lower().replace(" ", "_"))
        except Exception:  # noqa: BLE001
            continue
    return keys


def collect(cfg: dict) -> list[dict]:
    region = cfg["channel"].get("region", "US")
    lang = cfg["channel"].get("language", "en")
    blocked = cfg["compliance"].get("blocked_topics", [])
    tcfg = cfg.get("trends", {})
    raw: list[dict] = []
    raw += youtube_outliers(cfg)
    raw += google_trends(region)
    wiki = wikipedia_top(lang)
    if tcfg.get("wikipedia_spike_check", 40):
        wikipedia_spikes(wiki, lang, int(tcfg.get("wikipedia_spike_check", 40)))
    raw += wiki
    if tcfg.get("on_this_day", True):
        raw += on_this_day(lang)
    reddit_subs = tcfg.get("reddit_subreddits", ["todayilearned", "science", "space"])
    if not isinstance(reddit_subs, list):
        reddit_subs = ["todayilearned", "science", "space"]
    seen_subs: set[str] = set()
    for subreddit in reddit_subs:
        if not isinstance(subreddit, str):
            continue
        subreddit = subreddit.strip()
        key = subreddit.casefold()
        if not re.fullmatch(r"[a-z0-9_]{2,21}", key) or key in seen_subs:
            continue
        seen_subs.add(key)
        limit = 25 if key == "todayilearned" else 15 if key in {"science", "space"} else 10
        raw += reddit_rss(subreddit, limit)
    raw += hackernews()
    raw += youtube_chart(region)
    source_counts: dict[str, int] = {}
    for row in raw:
        source = str(row.get("source") or "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    source_detail = ", ".join(f"{source}:{count}" for source, count in sorted(source_counts.items()))
    log.info("collected %d raw trend signals from %d sources (%s)",
             len(raw), len(source_counts), source_detail or "no source rows")

    # Merge exact-title duplicates, but only award a cross-source bonus for an independent
    # platform family. Two Reddit feeds or two YouTube feeds are multiple observations, not
    # independent-platform confirmation. Keep each measured signal for downstream review.
    merged: dict[str, dict] = {}
    captured_at = now_utc().isoformat(timespec="seconds")
    for r in raw:
        key = re.sub(r"[^a-z0-9 ]", "", r["topic"].lower()).strip()
        if not key or len(key) < 3:
            continue
        signal = _signal_record(r, captured_at)
        if key in merged:
            m = merged[key]
            family = signal["source_family"]
            if family not in m["source_families"]:
                m["score"] = min(1.0, max(m["score"], r["score"]) + 0.15)
                m["source_families"].append(family)
            else:
                m["score"] = max(m["score"], r["score"])
            if r["source"] not in m["sources"]:
                m["sources"].append(r["source"])
            context = m.setdefault("context", [])
            for item in r.get("context") or []:
                if item and item not in context:
                    context.append(item)
            signature = (signal["source"], signal["source_url"], tuple(signal["context"]))
            if signature not in {(x["source"], x["source_url"], tuple(x["context"])) for x in m["signals"]}:
                m["signals"].append(signal)
        else:
            family = signal["source_family"]
            merged[key] = {**r, "sources": [r["source"]], "source_families": [family],
                           "signals": [signal], "topic_key": key}

    # Safety triage (autotube/safety.py). At the candidate stage we only drop HARD-blocked subjects;
    # merely sensitive words ("war", "crash", "trial", "flood") are kept and judged later, in context,
    # once we have the actual Wikipedia article to judge. Dropping them here is what starved the funnel.
    from . import safety
    out, hard, flagged = [], 0, 0
    for c in merged.values():
        text = " ".join([c["topic"]] + [str(x) for x in c.get("context") or []])
        verdict, term = safety.screen(text, cfg)
        if verdict == "block":
            c["rejected"] = f"blocked:{term}"
            hard += 1
            continue
        if verdict == "review":
            c["sensitive"] = term            # judged in context at grounding time
            flagged += 1
        out.append(c)
    out.sort(key=lambda x: -x["score"])
    min_wiki_spike = float(tcfg.get("min_wikipedia_spike", 3.0))
    eligible = [c for c in out if has_current_trend_evidence(c, min_wiki_spike, tcfg)]
    log.info("%d candidates after merge + safety triage (%d hard-blocked, %d kept for context review); "
             "%d have a qualifying current trend signal (%d excluded)",
             len(out), hard, flagged, len(eligible), len(out) - len(eligible))
    return eligible


def wiki_summary(title: str, lang: str = "en") -> dict | None:
    try:
        return get_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}")
    except Exception:  # noqa: BLE001
        return None
