"""Trend intelligence: collect, normalize, filter and rank trending topics.

Free, keyless sources (all verified working):
  • Google Trends daily RSS (per region)           – what people are searching
  • Wikipedia top pageviews (yesterday)             – what people are reading about
  • Reddit r/todayilearned + r/science top (RSS)    – fact-shaped curiosity signals
  • Hacker News front page (Algolia API)            – tech trends
  • Optional: YouTube mostPopular chart (API key, 1 quota unit) – what's winning on YT

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
from datetime import timedelta
from urllib.parse import quote

import feedparser

from .common import get_json, http, now_utc, read_json

log = logging.getLogger("autotube.trends")

WIKI_SKIP_PREFIX = ("Special:", "Main_Page", "Wikipedia:", "File:", "Portal:", "Help:", "Talk:",
                    "Template:", "Category:", "User:", "Deaths_in", "List_of", "Cleopatra_(disambiguation)")
WIKI_SKIP_EXACT = {"-", "Main_Page", "Undefined", "XXX", "Search"}


def _norm_rank(i: int, n: int) -> float:
    return round(1.0 - (i / max(n, 1)) * 0.85, 4)


def google_trends(geo: str) -> list[dict]:
    out = []
    try:
        feed = feedparser.parse(http().get(f"https://trends.google.com/trending/rss?geo={geo}", timeout=20).content)
        n = len(feed.entries)
        for i, e in enumerate(feed.entries):
            traffic = e.get("ht_approx_traffic", "0").replace("+", "").replace(",", "")
            try:
                t = int(traffic)
            except ValueError:
                t = 0
            news = []
            for k in ("ht_news_item_title",):
                if e.get(k):
                    news.append(e.get(k))
            out.append({
                "topic": e.title.strip(), "source": "google_trends",
                "score": min(1.0, 0.4 + math.log10(t + 10) / 7) * _norm_rank(i, n) ** 0.3,
                "context": news, "traffic": t,
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
        feed = feedparser.parse(r.content)
        n = len(feed.entries)
        for i, e in enumerate(feed.entries):
            title = re.sub(r"^(TIL( that)?|TIL:)\s*", "", e.title, flags=re.I).strip()
            out.append({"topic": title, "source": f"reddit_{sub}", "score": _norm_rank(i, n) * 0.9,
                        "context": [e.title], "is_fact": True})
    except Exception as e:  # noqa: BLE001
        log.warning("reddit %s failed: %s", sub, e)
    return out


def hackernews(limit: int = 20) -> list[dict]:
    out = []
    try:
        data = get_json("https://hn.algolia.com/api/v1/search", params={"tags": "front_page", "hitsPerPage": limit})
        hits = sorted(data.get("hits", []), key=lambda h: -(h.get("points") or 0))
        for i, h in enumerate(hits):
            out.append({"topic": h["title"], "source": "hackernews", "score": _norm_rank(i, len(hits)) * 0.7,
                        "context": [h.get("url") or ""], "category_hint": "technology"})
    except Exception as e:  # noqa: BLE001
        log.warning("hackernews failed: %s", e)
    return out


def youtube_chart(region: str, limit: int = 30) -> list[dict]:
    """videos.list chart=mostPopular — costs 1 quota unit. Optional."""
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return []
    out = []
    try:
        data = get_json("https://www.googleapis.com/youtube/v3/videos", params={
            "part": "snippet,statistics", "chart": "mostPopular", "regionCode": region,
            "maxResults": limit, "key": key})
        items = data.get("items", [])
        for i, it in enumerate(items):
            sn = it["snippet"]
            out.append({"topic": sn["title"], "source": "youtube_chart", "score": _norm_rank(i, len(items)) * 0.8,
                        "context": sn.get("tags", [])[:8], "yt_category": sn.get("categoryId")})
    except Exception as e:  # noqa: BLE001
        log.warning("youtube chart failed: %s", e)
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
    raw: list[dict] = []
    raw += google_trends(region)
    raw += wikipedia_top(lang)
    raw += reddit_rss("todayilearned")
    raw += reddit_rss("science", 15)
    raw += reddit_rss("space", 10)
    raw += hackernews()
    raw += youtube_chart(region)
    log.info("collected %d raw trend signals from %d sources", len(raw), len({r['source'] for r in raw}))

    # merge duplicates across sources (cross-source agreement boosts score)
    merged: dict[str, dict] = {}
    for r in raw:
        key = re.sub(r"[^a-z0-9 ]", "", r["topic"].lower()).strip()
        if not key or len(key) < 3:
            continue
        if key in merged:
            m = merged[key]
            m["score"] = min(1.0, max(m["score"], r["score"]) + 0.15)
            m["sources"].append(r["source"])
            m["context"] = (m.get("context") or []) + (r.get("context") or [])
        else:
            merged[key] = {**r, "sources": [r["source"]], "topic_key": key}

    out = []
    for c in merged.values():
        b = _blocked(" ".join([c["topic"]] + [str(x) for x in c.get("context") or []]), blocked)
        if b:
            c["rejected"] = f"blocked:{b}"
            continue
        out.append(c)
    out.sort(key=lambda x: -x["score"])
    log.info("%d candidates after merge + safety filter", len(out))
    return out


def wiki_summary(title: str, lang: str = "en") -> dict | None:
    try:
        return get_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}")
    except Exception:  # noqa: BLE001
        return None
