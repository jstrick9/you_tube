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


def _spread(values: list[float]) -> float:
    """Coefficient of variation — how much real signal a pool carries."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return (var ** 0.5) / mean


def _score(values: list[float], x: float, target: float, min_n: int, min_spread: float) -> float:
    """Blend an ABSOLUTE score against a fixed target with the PERCENTILE rank within the channel.

    Percentile rank is scale-free and adapts as a channel grows, which is why the original design used
    it — but it is degenerate on a flat distribution. With every video landing in a narrow band (as
    ours did: 20 videos, all between 460 and 1,280 views) percentile ranking turns pure noise into
    confident-looking rewards and the bandit learns nothing.

    So: score against an absolute target until the pool is both large enough and varied enough to rank
    meaningfully, then fade into percentile ranking. The weight moves smoothly, so there's no cliff.
    """
    absolute = max(0.0, min(1.0, x / target)) if target > 0 else 0.5
    if len(values) < min_n or _spread(values) < min_spread:
        return absolute
    w = min(1.0, (len(values) - min_n) / max(min_n, 1))     # fade in over the next min_n videos
    return (1 - w) * absolute + w * _pct_rank(values, x)


MAX_SANE_VIEW_PCT = 400.0   # Shorts loop, so >100% is real; 3474% on 22 views is an API artifact


def clean_view_pct(v) -> float | None:
    """averageViewPercentage, with nonsense filtered out.

    Shorts replay, so values above 100% are legitimate and are a *good* sign. But the API returns wild
    values for very low view counts (we saw 3474% on a 22-view video — a reported 1,146 s average view
    duration on a 32 s video). Those are artifacts, and left in the pool they drag every other video's
    percentile down.
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v <= 0 or v > MAX_SANE_VIEW_PCT else v


def learnable(h: dict, acfg: dict) -> bool:
    """Is there enough traffic on this video for its numbers to mean anything?

    A video with 22 views has a retention figure built from a handful of sessions. Feeding that to the
    bandit as if it were equal evidence to a 1,200-view video is how a learner teaches itself nonsense.
    """
    m = h.get("metrics") or {}
    return int(m.get("views") or 0) >= int(acfg.get("min_views_to_learn", 50))


def compute_reward(m: dict, pools: dict, acfg: dict) -> tuple[float, dict]:
    """Reward in [0,1] for one matured video, plus the per-component breakdown (kept for the dashboard).

    Weighted toward RETENTION, not views. Views per hour is a downstream signal dominated by how much
    traffic the algorithm chose to send; retention is the upstream signal we actually control, and it
    is what decides whether a Short escapes its seed audience in the first place.
    """
    min_n = int(acfg.get("rank_min_videos", 40))
    min_spread = float(acfg.get("rank_min_spread", 0.35))
    w_ret = float(acfg.get("w_retention", 0.65))
    w_reach = float(acfg.get("w_reach", 0.25))
    w_eng = float(acfg.get("w_engagement", 0.10))

    # ── retention: completion first, then average view %, each against an absolute Shorts target ──
    target_pct = float(acfg.get("target_avg_view_pct", 75.0))
    target_comp = float(acfg.get("target_completion", 0.55))
    avg_pct = clean_view_pct(m.get("avg_view_pct"))
    completion = m.get("completion")
    if completion is not None:
        r_ret = _score(pools["completion"], completion, target_comp, min_n, min_spread)
        if avg_pct is not None:
            r_ret = 0.5 * r_ret + 0.5 * _score(pools["avg_pct"], avg_pct, target_pct, min_n, min_spread)
    elif avg_pct is not None:
        r_ret = _score(pools["avg_pct"], avg_pct, target_pct, min_n, min_spread)
    else:
        r_ret = None

    # ── reach ──
    r_reach = _score(pools["vph"], m.get("vph", 0.0), float(acfg.get("target_vph", 40.0)), min_n, min_spread)

    # ── engagement: comments are worth more than likes; both are strong "send it to a friend" proxies ──
    views = max(1, int(m.get("views") or 0))
    eng = ((m.get("likes") or 0) + 3 * (m.get("comments") or 0)) / views
    r_eng = max(0.0, min(1.0, eng / float(acfg.get("target_engagement", 0.06))))

    if r_ret is None:                 # no retention data yet → redistribute its weight onto reach
        w_reach, w_ret = w_reach + w_ret, 0.0
        r_ret = 0.0
    total = w_ret + w_reach + w_eng or 1.0
    reward = (w_ret * r_ret + w_reach * r_reach + w_eng * r_eng) / total
    return round(max(0.0, min(1.0, reward)), 4), {
        "retention": round(r_ret, 3), "reach": round(r_reach, 3), "engagement": round(r_eng, 3),
        "mode": "absolute" if len(pools["vph"]) < min_n or _spread(pools["vph"]) < min_spread else "blended",
    }


def _median(v: list[float]) -> float | None:
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def retention_diagnosis(matured: list[dict], acfg: dict) -> dict:
    """Turn the channel's retention curves into one actionable sentence.

    On Shorts there are only three places to lose people, and each has a different fix:
      hook   (first ~15%)  → the first line isn't earning the next 3 seconds
      middle                → the body sags; lines aren't each adding something new
      ending                → the payoff/loop isn't pulling viewers through or into a replay
    Logged every pass and written to analytics_summary.json so the operator can see the real problem
    instead of inferring it from a view count.
    """
    curves = [h["metrics"]["curve"] for h in matured
              if isinstance(h.get("metrics", {}).get("curve"), list) and len(h["metrics"]["curve"]) == 11]
    if len(curves) < 3:
        return {"videos": len(curves), "verdict": ""}
    avg = [sum(c[i] for c in curves) / len(curves) for i in range(11)]
    hook = _median([h["metrics"].get("hook_retention") for h in matured
                    if h.get("metrics", {}).get("hook_retention") is not None]) or avg[1]
    completion = _median([h["metrics"].get("completion") for h in matured
                          if h.get("metrics", {}).get("completion") is not None]) or avg[-1]
    swipes = [h["metrics"].get("swipe_point") for h in matured if h.get("metrics", {}).get("swipe_point")]
    target_hook = float(acfg.get("target_hook_retention", 0.80))
    target_comp = float(acfg.get("target_completion", 0.55))
    # biggest single drop between adjacent deciles, ignoring the inevitable step at the very start
    drops = [(avg[i] - avg[i + 1], i) for i in range(1, 10)]
    worst_drop, worst_i = max(drops) if drops else (0.0, 0)

    if hook < target_hook:
        verdict = (f"HOOK is the leak: only {hook:.0%} are still there at 15% (target {target_hook:.0%}). "
                   f"Open with the single most surprising specific fact; cut all setup.")
        where = "hook"
    elif completion < target_comp:
        verdict = (f"ENDING is the leak: {completion:.0%} reach the end (target {target_comp:.0%}). "
                   f"Tighten the back half and make the last line loop into the first.")
        where = "ending"
    elif worst_drop > 0.12:
        verdict = (f"MIDDLE sags: biggest drop is {worst_drop:.0%} between {worst_i * 10}% and "
                   f"{(worst_i + 1) * 10}%. That body line isn't adding anything new.")
        where = "middle"
    else:
        verdict = (f"Curve is healthy: {hook:.0%} past the hook, {completion:.0%} complete.")
        where = "ok"
    return {
        "videos": len(curves), "where": where, "verdict": verdict,
        "avg_curve": [round(x, 3) for x in avg],
        "median_hook_retention": round(hook, 3), "median_completion": round(completion, 3),
        "median_swipe_point": round(_median(swipes), 3) if swipes else None,
        "worst_drop": {"size": round(worst_drop, 3), "from_pct": worst_i * 10, "to_pct": (worst_i + 1) * 10},
    }


def run(cfg: dict) -> dict:
    hist = read_json("history.json", [])
    uploaded = [h for h in hist if h.get("video_id")]
    if not uploaded:
        log.info("no uploaded videos yet — nothing to learn")
        return {"evaluated": 0}
    ids = [h["video_id"] for h in uploaded][-300:]
    stats = youtube.video_stats(ids)
    start = min(datetime.fromisoformat(h["created_at"]) for h in uploaded).date().isoformat()
    today = now_utc().date().isoformat()
    ret = youtube.retention(ids, start, today)
    # Retention CURVES: one Analytics call per video, so only fetch for videos that have matured but
    # don't have a curve yet (plus a periodic refresh of the newest few, whose curves are still settling).
    min_h = cfg["analytics"]["evaluate_after_hours"]
    by_id = {h["video_id"]: h for h in uploaded}
    want_curve = [h["video_id"] for h in uploaded
                  if h.get("metrics", {}).get("hours_live", 0) >= min_h
                  and not h.get("metrics", {}).get("curve")][-40:]
    curves = youtube.retention_curves(want_curve, start, today) if want_curve else {}

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
            if h.get("status") != "withdrawn":
                h["status"] = "missing"
            h.pop("metrics", None)
            continue
        pub = datetime.fromisoformat(h.get("publish_at") or h["created_at"])
        hours = max(1.0, (now - pub).total_seconds() / 3600)
        h.setdefault("metrics", {}).update(s)
        h["metrics"].update(ret.get(h["video_id"], {}))
        h["metrics"].update(curves.get(h["video_id"], {}))
        if h["metrics"].get("avg_view_pct") is not None and clean_view_pct(h["metrics"]["avg_view_pct"]) is None:
            log.info("discarding implausible avg_view_pct %.0f%% on %s (%d views)",
                     h["metrics"]["avg_view_pct"], h["video_id"], h["metrics"].get("views", 0))
            h["metrics"]["avg_view_pct_raw"] = h["metrics"].pop("avg_view_pct")
        h["metrics"]["hours_live"] = round(hours, 1)
        h["metrics"]["vph"] = round(s["views"] / hours, 3)
        h["metrics"]["checked_at"] = now.isoformat()
        if s.get("rejection") or s.get("upload_status") in ("rejected", "failed"):
            h["status"] = "rejected"
        elif s.get("privacy") == "private" and pub < now - timedelta(hours=1) and h.get("status") == "scheduled":
            # its publish time has passed but it's still private → the owner pulled it; don't learn from it
            h["status"] = "withdrawn"

    # YouTube API policy III.E.4: stored statistics must be re-verified at least every 30 days.
    # Anything not re-checked within 30 days (e.g. very old videos outside the refresh window) is dropped.
    for h in uploaded:
        m = h.get("metrics")
        if m and (now - datetime.fromisoformat(m.get("checked_at") or h["created_at"])).days >= 30:
            h.pop("metrics", None)

    # ── learn from matured, not-yet-scored videos ────────────────────────────────────────────────
    acfg = cfg["analytics"]
    matured = [h for h in uploaded if h.get("metrics", {}).get("hours_live", 0) >= min_h
               and h.get("status") not in ("withdrawn", "missing")]
    # pools are built only from videos with enough traffic to be meaningful
    pool_src = [h for h in matured if learnable(h, acfg)]
    pools = {
        "vph": [h["metrics"]["vph"] for h in pool_src],
        "avg_pct": [p for h in pool_src if (p := clean_view_pct(h["metrics"].get("avg_view_pct"))) is not None],
        "completion": [h["metrics"]["completion"] for h in pool_src if h["metrics"].get("completion") is not None],
    }
    strat = Strategy(cfg)
    strat.age(now)                       # time-based forgetting, applied once per pass (see strategy.py)
    learned = 0
    for h in matured:
        if h.get("reward") is not None or h.get("status") == "withdrawn":
            continue
        if h.get("status") != "rejected" and not learnable(h, acfg):
            continue                      # too little traffic to be evidence; re-checked on a later pass
        if h.get("status") == "rejected":
            r, parts = 0.0, {"retention": 0, "reach": 0, "engagement": 0, "mode": "rejected"}
        else:
            r, parts = compute_reward(h["metrics"], pools, acfg)
        h["reward"] = r
        h["reward_parts"] = parts
        strat.update(h["choice"], r)
        learned += 1
    strat.save()

    # ── where are we losing people? ──────────────────────────────────────────────────────────────
    diag = retention_diagnosis(pool_src, acfg)
    if diag.get("verdict"):
        log.info("retention: %s", diag["verdict"])
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
        "retention": diag,
    }
    write_json("analytics_summary.json", summary)
    log.info("analytics: %d videos tracked, %d newly learned", len(uploaded), learned)
    return summary
