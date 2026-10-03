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
            out.append({"topic": sn["title"], "source": "youtube_chart", "score": _norm_rank(i, len(items)) * 0.8,
                        "context": sn.get("tags", [])[:8], "yt_category": sn.get("categoryId")})
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


def format_pulse(rows: list[dict]) -> dict:
    """Which hook shapes are actually winning on YouTube right now.

    The reward loop can only learn from our own uploads, which is a handful of videos a week — far
    too slow to notice that, say, number-first hooks are hot this month. These rows are other
    people's viral Shorts, so they are a live read on format demand that costs us nothing and
    arrives thousands of videos at a time. Weighted by views per hour rather than by video count,
    so one breakout counts for more than ten mild performers.
    """
    tally: dict[str, dict] = {}
    for r in rows:
        pat = hook_pattern(r.get("topic", ""))
        d = tally.setdefault(pat, {"videos": 0, "vph": 0.0})
        d["videos"] += 1
        d["vph"] += float(r.get("vph") or 0)
    total = sum(d["vph"] for d in tally.values()) or 1.0
    return {
        "updated": now_utc().isoformat(timespec="seconds"),
        "sample": len(rows),
        "hook_style": {k: round(v["vph"] / total, 4) for k, v in sorted(
            tally.items(), key=lambda kv: -kv[1]["vph"])},
        "videos": {k: v["videos"] for k, v in tally.items()},
    }


def youtube_outliers(cfg: dict) -> list[dict]:
    """Fact/explainer Shorts that are going viral RIGHT NOW (proven demand for the topic).

    search.list (100 units/query) for recent short videos ordered by views, then videos.list + channels.list
    (1 unit per 50) to compute views/hour and views relative to the channel's size. A Short with far more views
    than its channel has subscribers is an "outlier": the topic itself is what's spreading. Cached once per day.
    Only the TOPIC is used — our video is an original, sourced take, never a copy.
    """
    t = cfg.get("trends", {})
    pairs = lane_queries(cfg)
    if not pairs or not _yt_oauth_ready():
        return []
    from .common import write_json
    today = now_utc().date().isoformat()
    cache = read_json("trend_cache.json", {})
    if cache.get("yt_outliers_date") == today and isinstance(cache.get("yt_outliers"), list):
        log.info("youtube outliers: %d (cached today)", len(cache["yt_outliers"]))
        return cache["yt_outliers"]
    region = cfg["channel"].get("region", "US")
    lang = cfg["channel"].get("language", "en")
    min_views = int(t.get("outlier_min_views", 30000))
    since = now_utc() - timedelta(hours=float(t.get("outlier_window_hours", 96)))
    out: list[dict] = []
    try:
        from . import youtube
        yt = youtube.service()
        found: dict[str, tuple[str, str]] = {}        # videoId -> (query, lane)
        for lane, q in pairs:
            try:
                r = yt.search().list(part="id", q=q, type="video", videoDuration="short", order="viewCount",
                                     publishedAfter=since.strftime("%Y-%m-%dT%H:%M:%SZ"), regionCode=region,
                                     relevanceLanguage=lang, safeSearch="strict", maxResults=25).execute()
                for it in r.get("items", []):
                    found.setdefault(it["id"]["videoId"], (q, lane))
            except Exception as e:  # noqa: BLE001
                log.warning("youtube outlier search %r failed: %s", q, str(e)[:120])
        ids = list(found)
        vids = []
        for i in range(0, len(ids), 50):
            vids += yt.videos().list(part="snippet,statistics", id=",".join(ids[i:i + 50])).execute().get("items", [])
        ch_ids = list({v["snippet"]["channelId"] for v in vids})
        subs: dict[str, int] = {}
        for i in range(0, len(ch_ids), 50):
            for c in yt.channels().list(part="statistics", id=",".join(ch_ids[i:i + 50])).execute().get("items", []):
                subs[c["id"]] = int(c["statistics"].get("subscriberCount") or 0)
        own = cfg["channel"].get("name", "").lower()
        now = now_utc()
        rows = []
        for v in vids:
            sn, st = v["snippet"], v.get("statistics", {})
            views = int(st.get("viewCount") or 0)
            if views < min_views or sn.get("channelTitle", "").lower() == own:
                continue
            hours = max(1.0, (now - _parse_ts(sn["publishedAt"])).total_seconds() / 3600)
            s_count = subs.get(sn["channelId"], 0)
            vph = views / hours
            ratio = views / max(s_count, 1000)
            topic = _clean_title(sn["title"])
            if len(topic) < 8:
                continue
            rows.append({"topic": topic, "vph": vph, "ratio": ratio, "views": views, "hours": hours, "subs": s_count,
                         "desc": _clean_title(sn.get("description", ""))[:140],
                         "query": found.get(v["id"], ("", ""))[0], "lane": found.get(v["id"], ("", ""))[1],
                         "hook": hook_pattern(topic)})
        # rank: speed (views/hour) and breakout (views vs. channel size) both matter
        rows.sort(key=lambda r: -(math.log10(r["vph"] + 1) + 0.6 * math.log10(r["ratio"] + 1)))
        n = len(rows)
        for i, r in enumerate(rows[:40]):
            out.append({
                "topic": r["topic"], "source": "youtube_outliers", "is_fact": True,
                "lane": r["lane"], "hook": r["hook"],
                "score": round(min(1.0, 0.55 + 0.45 * _norm_rank(i, n)), 4),
                "context": [f"viral Short now: {r['views']:,} views in {r['hours']:.0f}h"
                            + (f", {r['ratio']:.0f}x its channel's subscribers" if r["subs"] else ""), r["desc"]],
                "evidence": {"views": r["views"], "hours": round(r["hours"], 1), "vph": round(r["vph"]),
                             "breakout": round(r["ratio"], 1)},
            })
        by_lane: dict[str, int] = {}
        for o in out:
            by_lane[o.get("lane") or "-"] = by_lane.get(o.get("lane") or "-", 0) + 1
        log.info("youtube outliers: %d viral Shorts from %d searches across %d lanes (%s)",
                 len(out), len(pairs), len(by_lane),
                 ", ".join(f"{k}:{v}" for k, v in sorted(by_lane.items())))
        cache.update({"yt_outliers_date": today, "yt_outliers": out})
        write_json("trend_cache.json", cache)
        if t.get("format_pulse", True) and rows:
            pulse = format_pulse(rows)
            write_json("format_pulse.json", pulse)
            log.info("format pulse (%d viral Shorts): %s", pulse["sample"],
                     ", ".join(f"{k} {v:.0%}" for k, v in pulse["hook_style"].items()))
    except Exception as e:  # noqa: BLE001
        log.warning("youtube outliers failed: %s", str(e)[:200])
    return out


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


SOURCE_ARMS = ["youtube_outliers", "wikipedia", "google_trends", "reddit", "hackernews", "on_this_day",
               "youtube_chart", "evergreen"]


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
    log.info("%d candidates after merge + safety triage (%d hard-blocked, %d kept for context review)",
             len(out), hard, flagged)
    return out


def wiki_summary(title: str, lang: str = "en") -> dict | None:
    try:
        return get_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title.replace(' ', '_'), safe='')}")
    except Exception:  # noqa: BLE001
        return None
