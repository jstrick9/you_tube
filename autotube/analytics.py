"""Feedback loop: measure published videos → reward → update strategy bandit.

Reward for a video = 0.6 * percentile(views/hour) + 0.4 * percentile(avg view %),
computed against the channel's own recent videos (so it adapts as the channel grows
and is robust to the overall view level). Also tracks health signals: videos that
got rejected/blocked are penalized hard and alert the operator.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from . import youtube
from .common import now_utc, read_json, write_json
from .strategy import Strategy

log = logging.getLogger("autotube.analytics")


def _pct_rank(values: list[float], x: float) -> float:
    if not values:
        return 0.5
    below = sum(v < x for v in values)
    equal = sum(v == x for v in values)
    return (below + 0.5 * equal) / len(values)


def run(cfg: dict) -> dict:
    hist = read_json("history.json", [])
    uploaded = [h for h in hist if h.get("video_id")]
    if not uploaded:
        log.info("no uploaded videos yet — nothing to learn")
        return {"evaluated": 0}
    ids = [h["video_id"] for h in uploaded][-300:]
    stats = youtube.video_stats(ids)
    start = min(datetime.fromisoformat(h["created_at"]) for h in uploaded).date().isoformat()
    ret = youtube.retention(ids, start, now_utc().date().isoformat())

    now = now_utc()
    if ids and not stats:
        # Nothing came back at all → treat as an API/auth problem, not mass deletion; change nothing.
        log.warning("no statistics returned for %d videos — skipping this analytics pass", len(ids))
        return {"evaluated": 0}
    checked = set(ids)
    for h in uploaded:
        if h["video_id"] not in checked:
            continue
        s = stats.get(h["video_id"])
        if not s:
            # Video deleted/unavailable: YouTube API policy III.E.4 → stop keeping its API data.
            h["status"] = "missing"
            h.pop("metrics", None)
            continue
        pub = datetime.fromisoformat(h.get("publish_at") or h["created_at"])
        hours = max(1.0, (now - pub).total_seconds() / 3600)
        h.setdefault("metrics", {}).update(s)
        h["metrics"].update(ret.get(h["video_id"], {}))
        h["metrics"]["hours_live"] = round(hours, 1)
        h["metrics"]["vph"] = round(s["views"] / hours, 3)
        h["metrics"]["checked_at"] = now.isoformat()
        if s.get("rejection") or s.get("upload_status") in ("rejected", "failed"):
            h["status"] = "rejected"

    # YouTube API policy III.E.4: stored statistics must be re-verified at least every 30 days.
    # Anything not re-checked within 30 days (e.g. very old videos outside the refresh window) is dropped.
    for h in uploaded:
        m = h.get("metrics")
        if m and (now - datetime.fromisoformat(m.get("checked_at") or h["created_at"])).days >= 30:
            h.pop("metrics", None)

    # learn from matured, not-yet-scored videos
    min_h = cfg["analytics"]["evaluate_after_hours"]
    w_ret = cfg["analytics"]["retention_weight"]
    matured = [h for h in uploaded if h.get("metrics", {}).get("hours_live", 0) >= min_h]
    vph_pool = [h["metrics"]["vph"] for h in matured]
    ret_pool = [h["metrics"].get("avg_view_pct", 0) for h in matured if h["metrics"].get("avg_view_pct")]
    strat = Strategy(cfg)
    learned = 0
    for h in matured:
        if h.get("reward") is not None:
            continue
        if h.get("status") == "rejected":
            r = 0.0
        else:
            r_v = _pct_rank(vph_pool, h["metrics"]["vph"])
            av = h["metrics"].get("avg_view_pct")
            r_r = _pct_rank(ret_pool, av) if av else r_v
            r = (1 - w_ret) * r_v + w_ret * r_r
        h["reward"] = round(r, 4)
        strat.update(h["choice"], r)
        learned += 1
    strat.save()
    write_json("history.json", hist)

    # rolling summary for the dashboard
    last7 = [h for h in uploaded if datetime.fromisoformat(h["created_at"]) > now - timedelta(days=7)]
    summary = {
        "updated": now.isoformat(), "videos_total": len(uploaded), "videos_7d": len(last7),
        "views_7d": sum(h.get("metrics", {}).get("views", 0) for h in last7),
        "learned_this_run": learned, "strategy": strat.report(),
        "top_videos": sorted(
            [{"title": h["title"], "id": h["video_id"], "views": h.get("metrics", {}).get("views", 0),
              "avg_view_pct": h.get("metrics", {}).get("avg_view_pct")} for h in uploaded],
            key=lambda x: -x["views"])[:10],
        "rejected": [h["video_id"] for h in uploaded if h.get("status") == "rejected"],
    }
    write_json("analytics_summary.json", summary)
    log.info("analytics: %d videos tracked, %d newly learned", len(uploaded), learned)
    return summary
