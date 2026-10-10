"""Feedback loop: measure published videos → reward → update strategy bandit.

The reward uses Shorts engaged views per hour and averageViewPercentage when available, with a
clearly marked public-view fallback for Analytics API gaps. These are optimization inputs, not claims
about an official YouTube distribution gate. Rejected/blocked videos are penalized and surfaced.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

from . import originality, youtube
from .common import now_utc, read_json, write_json
from .strategy import PRIOR, Strategy

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


MAX_SANE_VIEW_PCT = 400.0   # Repeated watching can exceed 100%; 3474% on 22 views was an API artifact


def clean_view_pct(v) -> float | None:
    """Validate ``averageViewPercentage`` while retaining plausible repeat-watching values.

    Do not assume automatic Shorts looping produces values above 100%; YouTube's metric definition
    excludes looping-clip traffic. Repeated playback can still produce a value above 100%. The API has
    also returned wild values at very low view counts (including 3474% on a 22-view video), so values
    above this generous sanity ceiling are excluded from learning.
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v <= 0 or v > MAX_SANE_VIEW_PCT else v


def learnable(h: dict, acfg: dict) -> bool:
    """Is there enough engaged-view traffic for these Shorts metrics to mean anything?"""
    m = h.get("metrics") or {}
    n = m.get("engaged_views_90d")
    if n is None:  # migration / Analytics API outage; retain a clearly weaker public-view fallback
        n = m.get("views") or 0
    return int(n) >= int(acfg.get("min_views_to_learn", 50))


def compute_reward(m: dict, pools: dict, acfg: dict) -> tuple[float, dict]:
    """Reward in [0,1] for one matured video, plus the per-component breakdown (kept for the dashboard).

    Weighted toward average watch percentage and engaged views. Public view counts are retained for
    reporting, but are not substituted for engaged Shorts views where the Analytics API provides them.
    The relative weights are internal optimization choices, not descriptions of YouTube's ranking system.
    """
    min_n = int(acfg.get("rank_min_videos", 40))
    min_spread = float(acfg.get("rank_min_spread", 0.35))
    w_ret = float(acfg.get("w_retention", 0.65))
    w_reach = float(acfg.get("w_reach", 0.25))
    w_eng = float(acfg.get("w_engagement", 0.10))

    # ── watch time: use the documented averageViewPercentage metric ──
    # audienceWatchRatio's final curve point is not a completion share; it is a segment-watch ratio
    # and can exceed 1.0 when viewers replay a segment. Keep that curve for shape diagnosis only.
    target_pct = float(acfg.get("target_avg_view_pct", 70.0))
    avg_pct = clean_view_pct(m.get("avg_view_pct"))
    r_ret = (_score(pools["avg_pct"], avg_pct, target_pct, min_n, min_spread)
             if avg_pct is not None else None)

    # ── engaged-view velocity; do not mix public-view fallback into this score pool ──
    has_engaged_rate = m.get("vph_basis") == "engaged_views"
    target_vph = float(acfg.get("target_engaged_vph", acfg.get("target_vph", 40.0)))
    r_reach = (_score(pools["vph"], m.get("vph", 0.0), target_vph, min_n, min_spread)
               if has_engaged_rate else None)

    # ── subscriber conversion: an independent YPP requirement ──
    # Normalize the 90-day subscriber change by engaged views for that same reporting window when
    # available; retain raw views only as an explicitly weaker fallback.
    net_subs = (m.get("subs_gained") or 0) - (m.get("subs_lost") or 0)
    subs_denominator = m.get("engaged_views_90d")
    if subs_denominator is None:
        subs_denominator = m.get("views") or 0
    subs_per_1k = net_subs / max(1, int(subs_denominator)) * 1000.0
    r_subs = max(0.0, min(1.0, subs_per_1k / float(acfg.get("target_subs_per_1k", 2.0))))

    # ── engagement: internal like/comment-rate proxy, not a YouTube ranking claim ──
    views = max(1, int(m.get("views") or 0))
    eng = ((m.get("likes") or 0) + 3 * (m.get("comments") or 0)) / views
    r_eng = max(0.0, min(1.0, eng / float(acfg.get("target_engagement", 0.06))))

    if r_ret is None:                 # no average-view data yet; redistribute its weight to an available signal
        if r_reach is not None:
            w_reach += w_ret
        else:
            w_eng += w_ret
        w_ret = 0.0
        r_ret = 0.0
    if r_reach is None:               # public-view fallback is not comparable to engaged-view velocity
        w_reach, r_reach = 0.0, 0.0
    w_subs = float(acfg.get("w_subs", 0.20))
    total = w_ret + w_reach + w_eng + w_subs or 1.0
    reward = (w_ret * r_ret + w_reach * r_reach + w_eng * r_eng + w_subs * r_subs) / total

    # ── internal breakout preference ─────────────────────────────────────────────────────────────
    # The other components saturate at their configured targets. This optional log-scaled term gives
    # the local optimizer a graded preference for strong engaged-view velocity above that target; it
    # is an experimentation choice, not a claim about YouTube recommendations or channel growth.
    r_break = 0.0
    tgt_vph = target_vph
    mult = max(1.5, float(acfg.get("breakout_multiple", 5.0)))
    if has_engaged_rate and tgt_vph > 0 and (m.get("vph") or 0) > tgt_vph:
        over = (m["vph"] / tgt_vph - 1.0) / (mult - 1.0)
        r_break = max(0.0, min(1.0, math.log1p(over * (math.e - 1))))
    w_break = float(acfg.get("w_breakout", 0.15)) if has_engaged_rate else 0.0
    reward = (1.0 - w_break) * reward + w_break * r_break

    return round(max(0.0, min(1.0, reward)), 4), {
        "retention": round(r_ret, 3), "reach": round(r_reach, 3), "engagement": round(r_eng, 3),
        "subs": round(r_subs, 3), "subs_per_1k": round(subs_per_1k, 2), "breakout": round(r_break, 3),
        "mode": "absolute" if len(pools["vph"]) < min_n or _spread(pools["vph"]) < min_spread else "blended",
    }


def _median(v: list[float]) -> float | None:
    v = sorted(x for x in v if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _subs_progress(uploaded: list[dict], now) -> dict:
    """Per-video subscriber conversion from recent uploads; not the channel's subscriber total."""
    recent = [h for h in uploaded
              if datetime.fromisoformat(h["created_at"]) > now - timedelta(days=90)
              and h.get("metrics")]
    net = sum((h["metrics"].get("subs_gained") or 0) - (h["metrics"].get("subs_lost") or 0)
              for h in recent)
    measured = [h for h in recent if h["metrics"].get("engaged_views_90d") is not None]
    if measured and len(measured) == len(recent):
        views = sum(int(h["metrics"].get("engaged_views_90d") or 0) for h in measured)
        basis = "engaged_views"
    else:
        views = sum(int(h["metrics"].get("views") or 0) for h in recent)
        basis = "public_views_fallback"
    per_1k = round(net / max(1, views) * 1000.0, 2)
    out = {"net_90d": net, "views_90d": views, "per_1k_views": per_1k, "videos": len(recent),
           "denominator_basis": basis, "videos_with_engaged_views": len(measured),
           "complete_engaged_view_coverage": len(measured) == len(recent)}
    if net > 0:
        # how long at this pace to each YPP gate
        out["days_to_500"] = round(500 / (net / 90.0), 1)
        out["days_to_1000"] = round(1000 / (net / 90.0), 1)
    else:
        out["days_to_500"] = out["days_to_1000"] = None
    return out


def average_view_target(matured: list[dict], acfg: dict) -> dict:
    """Measure the channel against its *internal* average-view target.

    YouTube's Analytics API does not expose a simple percentage of unique viewers who finished a
    Short. ``averageViewPercentage`` is the closest directly queryable watch-through proxy: average
    percentage watched per playback. The 70% threshold is this channel's target, not a published
    YouTube distribution rule. ``audienceWatchRatio`` is a segment-watch ratio and must not be used
    as a completion rate.
    """
    gate = float(acfg.get("average_view_gate_pct", 70.0))
    values = [pct for h in matured
              if (pct := clean_view_pct(h.get("metrics", {}).get("avg_view_pct"))) is not None]
    if not values:
        return {"n": 0, "gate_pct": gate, "metric": "average_view_percentage",
                "verdict": "no average view percentage data yet"}
    passing = [v for v in values if v >= gate]
    med = _median(values) or 0.0
    rate = len(passing) / len(values)
    if len(passing) == len(values):
        verdict = (f"ALL {len(values)}/{len(values)} measured videos clear the internal {gate:.0f}% "
                   f"average-view target (median {med:.0f}%). This is not a viewer-completion rate.")
    elif passing:
        verdict = (f"{len(passing)}/{len(values)} measured videos clear the internal {gate:.0f}% "
                   f"average-view target (median {med:.0f}%). This is average watch percentage, "
                   f"not the share of viewers who reached the end.")
    else:
        verdict = (f"0/{len(values)} measured videos clear the internal {gate:.0f}% average-view target "
                   f"(median {med:.0f}%). This is average watch percentage, not the share of viewers "
                   f"who reached the end; use the curve only to locate relative drop-offs.")
    return {"n": len(values), "gate_pct": gate, "metric": "average_view_percentage",
            "passing": len(passing), "pass_rate": round(rate, 3),
            "median_avg_view_pct": round(med, 2), "verdict": verdict}


# Backward-compatible function name only; this is not a YouTube distribution gate.
distribution_gate = average_view_target


def retention_diagnosis(matured: list[dict], acfg: dict) -> dict:
    """Report average-view performance and locate the largest *relative* curve drop.

    AudienceWatchRatio is not a unique-viewer survival curve: it is segment watches divided by total
    video views and may exceed 1.0 when viewers replay parts. The curve is useful for comparing its
    own shape, but its points are never described as percentages of viewers who remain or complete.
    """
    curves = []
    for h in matured:
        m = h.get("metrics", {})
        curve = m.get("audience_watch_ratio_curve") or m.get("curve")
        if isinstance(curve, list) and len(curve) == 11:
            curves.append(curve)
    pcts = [pct for h in matured
            if (pct := clean_view_pct(h.get("metrics", {}).get("avg_view_pct"))) is not None]
    median_pct = _median(pcts)
    gate = float(acfg.get("average_view_gate_pct", 70.0))
    if len(curves) < 3:
        verdict = ""
        if median_pct is not None:
            verdict = (f"Median average view percentage is {median_pct:.0f}% "
                       f"(internal target {gate:.0f}%); not enough retention curves to locate a drop.")
        return {"videos": len(curves), "where": "unknown", "verdict": verdict,
                "median_avg_view_pct": round(median_pct, 2) if median_pct is not None else None}

    avg = [sum(c[i] for c in curves) / len(curves) for i in range(11)]
    baseline = max(abs(avg[0]), 1e-9)
    # Express each decline as a share of the opening watch-ratio level. This keeps a curve that starts
    # at 1.3 from being misread as 130% of viewers and makes the shape comparable across videos.
    drops = [((avg[i] - avg[i + 1]) / baseline, i) for i in range(10)]
    worst_drop, worst_i = max(drops, key=lambda x: x[0]) if drops else (0.0, 0)
    worst_drop = max(0.0, worst_drop)
    pct_text = (f" Median average view percentage is {median_pct:.0f}% against the internal "
                f"{gate:.0f}% target." if median_pct is not None else "")

    if worst_drop > 0.12:
        where = "hook" if worst_i < 2 else "ending" if worst_i >= 8 else "middle"
        zone = "opening" if where == "hook" else "ending" if where == "ending" else "middle"
        verdict = (f"Largest relative watch-ratio drop: {worst_drop:.0%} of the opening level between "
                   f"{worst_i * 10}% and {(worst_i + 1) * 10}% of runtime ({zone}). "
                   f"This is a ratio-curve change, not a percentage of viewers.{pct_text}")
    elif median_pct is not None and median_pct < gate:
        where = "watch_time"
        verdict = (f"Average view percentage is below the internal {gate:.0f}% target; available "
                   f"watch-ratio curves show no single abrupt decile drop.{pct_text}")
    else:
        where = "ok"
        verdict = ("No single abrupt relative watch-ratio drop is visible in the available curves."
                   + pct_text)
    return {
        "videos": len(curves), "where": where, "verdict": verdict,
        "avg_curve": [round(x, 4) for x in avg],
        "median_avg_view_pct": round(median_pct, 2) if median_pct is not None else None,
        "worst_drop": {"size_of_opening_ratio": round(worst_drop, 4),
                       "from_pct": worst_i * 10, "to_pct": (worst_i + 1) * 10},
    }



REWARD_SCHEMA_VERSION = 2


def migrate_legacy_retention_metrics(history: list[dict]) -> int:
    """Rename fields produced by the old, incorrect interpretation of audienceWatchRatio."""
    changed = 0
    moves = {
        "completion": "end_watch_ratio",
        "hook_retention": "watch_ratio_at_15pct",
        "curve": "audience_watch_ratio_curve",
        "swipe_point": "first_below_half_watch_ratio_at",
    }
    for h in history:
        m = h.get("metrics")
        if not isinstance(m, dict):
            continue
        for old, new in moves.items():
            if old in m:
                m.setdefault(new, m[old])
                del m[old]
                changed += 1
    return changed


def _shorts_progress(uploaded: list[dict]) -> dict:
    """Trailing-window engaged Shorts views for YPP, with coverage shown rather than guessed."""
    public = [h for h in uploaded
              if h.get("status") not in ("withdrawn", "missing", "rejected")
              and (h.get("metrics") or {}).get("privacy") == "public"]
    measured = [h for h in public if (h.get("metrics") or {}).get("engaged_views_90d") is not None]
    total = sum(int(h["metrics"].get("engaged_views_90d") or 0) for h in measured)
    coverage = len(measured) / len(public) if public else 1.0
    return {
        "metric": "engaged_views",
        "engaged_views_90d": total,
        "expanded_ypp_threshold": 3_000_000,
        "current_full_ypp_threshold": 10_000_000,
        "feb_2027_full_ypp_threshold": 20_000_000,
        "expanded_threshold_progress": round(total / 3_000_000, 6),
        "current_full_threshold_progress": round(total / 10_000_000, 6),
        "feb_2027_full_threshold_progress": round(total / 20_000_000, 6),
        "public_videos": len(public),
        "videos_with_metric": len(measured),
        "coverage": round(coverage, 3),
        "complete": len(measured) == len(public),
        "note": "Partial if coverage is below 100%; only YouTube Analytics engagedViews count toward this proxy.",
    }


def _reset_reward_model(strat: Strategy, history: list[dict]) -> bool:
    """Rebuild old rewards once; the former version mislabeled end-segment ratios as completion."""
    version = int(strat.state.get("reward_schema_version", 0) or 0)
    if version >= REWARD_SCHEMA_VERSION:
        return False
    had_old_state = bool(strat.state.get("updates")) or any(h.get("reward") is not None for h in history)
    if had_old_state:
        for h in history:
            h.pop("reward", None)
            h.pop("reward_parts", None)
            h.pop("reward_version", None)
        arms = {dim: {name: {"a": PRIOR[0], "b": PRIOR[1], "n": 0} for name in names}
                for dim, names in strat.options().items()}
        strat.state = {"arms": arms, "updates": 0, "reward_schema_version": REWARD_SCHEMA_VERSION}
        log.warning("rebuilt strategy rewards: old retention input was mislabeled audienceWatchRatio")
    else:
        strat.state["reward_schema_version"] = REWARD_SCHEMA_VERSION
    return had_old_state

def run(cfg: dict) -> dict:
    hist = read_json("history.json", [])
    uploaded = [h for h in hist if h.get("video_id")]
    if not uploaded:
        log.info("no uploaded videos yet — nothing to learn")
        return {"evaluated": 0}
    ids = [h["video_id"] for h in uploaded][-int(cfg["analytics"].get("refresh_videos", 300)):]
    stats = youtube.video_stats(ids)
    now = now_utc()
    first_upload = min(datetime.fromisoformat(h["created_at"]).date() for h in uploaded)
    report_start = max(first_upload, (now.date() - timedelta(days=89))).isoformat()
    today = now.date().isoformat()
    # Query a trailing 90-calendar-day window: Shorts YPP eligibility is based on engaged views in a rolling
    # 90-day period, and watch quality should describe recent viewers rather than lifetime averages.
    ret = youtube.retention(ids, report_start, today)
    migrate_legacy_retention_metrics(hist)
    # Retention curves cost one Analytics call per video; fetch only matured videos without one.
    min_h = cfg["analytics"]["evaluate_after_hours"]
    want_curve = [h["video_id"] for h in uploaded
                  if h.get("metrics", {}).get("hours_live", 0) >= min_h
                  and not h.get("metrics", {}).get("audience_watch_ratio_curve")][-40:]
    curves = youtube.retention_curves(want_curve, report_start, today) if want_curve else {}
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
        metrics = h.setdefault("metrics", {})
        metrics.update(s)
        # These are windowed Analytics API metrics. Remove the previous window before applying the
        # current response so an API outage/omitted row cannot masquerade as fresh 90-day data.
        for key in ("a_views", "engaged_views_90d", "avg_view_pct", "avg_view_pct_raw",
                    "avg_view_dur", "subs_gained", "subs_lost", "shares"):
            metrics.pop(key, None)
        metrics.update(ret.get(h["video_id"], {}))
        metrics["engaged_views_window"] = {"start": report_start, "end": today}
        h["metrics"].update(curves.get(h["video_id"], {}))
        if h["metrics"].get("avg_view_pct") is not None and clean_view_pct(h["metrics"]["avg_view_pct"]) is None:
            log.info("discarding implausible avg_view_pct %.0f%% on %s (%d views)",
                     h["metrics"]["avg_view_pct"], h["video_id"], h["metrics"].get("views", 0))
            h["metrics"]["avg_view_pct_raw"] = h["metrics"].pop("avg_view_pct")
        h["metrics"]["hours_live"] = round(hours, 1)
        engaged = h["metrics"].get("engaged_views_90d")
        if engaged is not None:
            window_hours = min(hours, 90.0 * 24.0)
            h["metrics"]["vph"] = round(float(engaged) / max(window_hours, 1.0), 3)
            h["metrics"]["vph_basis"] = "engaged_views"
        else:
            h["metrics"]["vph"] = round(s["views"] / hours, 3)
            h["metrics"]["vph_basis"] = "public_views_fallback"
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
        "vph": [h["metrics"]["vph"] for h in pool_src
                if h["metrics"].get("vph_basis") == "engaged_views"],
        "avg_pct": [p for h in pool_src if (p := clean_view_pct(h["metrics"].get("avg_view_pct"))) is not None],
    }
    strat = Strategy(cfg)
    _reset_reward_model(strat, uploaded)
    strat.age(now)                       # time-based forgetting, applied once per pass (see strategy.py)
    learned = 0
    for h in matured:
        if h.get("reward") is not None and h.get("reward_version") == REWARD_SCHEMA_VERSION:
            continue
        if h.get("status") == "withdrawn":
            continue
        if h.get("status") != "rejected" and not learnable(h, acfg):
            continue                      # too little traffic to be evidence; re-checked on a later pass
        if h.get("status") == "rejected":
            r, parts = 0.0, {"retention": 0, "reach": 0, "engagement": 0, "mode": "rejected"}
        else:
            r, parts = compute_reward(h["metrics"], pools, acfg)
        h["reward"] = r
        h["reward_parts"] = parts
        h["reward_version"] = REWARD_SCHEMA_VERSION
        strat.update(h["choice"], r)
        learned += 1
    strat.state["reward_schema_version"] = REWARD_SCHEMA_VERSION
    strat.save()

    # ── where are we losing people? ──────────────────────────────────────────────────────────────
    diag = retention_diagnosis(pool_src, acfg)
    if diag.get("verdict"):
        log.info("retention: %s", diag["verdict"])
    # Keep the configured average-view goal separate from the relative curve-shape diagnosis.
    gate = average_view_target(matured, acfg)
    if gate.get("verdict"):
        log.info("average-view target: %s", gate["verdict"])
    write_json("history.json", hist[-int(cfg["analytics"].get("history_keep", 2000)):])

    # rolling summary for the dashboard
    last7 = [h for h in uploaded if datetime.fromisoformat(h["created_at"]) > now - timedelta(days=7)]
    summary = {
        "updated": now.isoformat(), "videos_total": len(uploaded), "videos_7d": len(last7),
        # Cumulative public views on videos uploaded in the last 7 days, not views earned during that period.
        "views_7d": sum(h.get("metrics", {}).get("views", 0) for h in last7),
        "learned_this_run": learned, "strategy": strat.report(),
        "top_videos": sorted(
            [{"title": h["title"], "id": h["video_id"], "views": h.get("metrics", {}).get("views", 0),
              "avg_view_pct": h.get("metrics", {}).get("avg_view_pct")} for h in uploaded],
            key=lambda x: -x["views"])[:10],
        "rejected": [h["video_id"] for h in uploaded if h.get("status") == "rejected"],
        "retention": diag,
        "average_view_target": gate,
        # This is an approximate per-video subscriber-conversion trajectory, not the channel's current
        # subscriber total or an eligibility decision. The YPP progress proxy above separately sums engagedViews.
        "subscribers": _subs_progress(uploaded, now),
        "shorts_monetization": _shorts_progress(uploaded),
        # What a policy reviewer sees looking at the channel rather than at one video.
        # The per-draft similarity check is pairwise and cannot see aggregate drift:
        # every episode can sit under the limit while all of them are one template.
        "originality": originality.channel_audit(hist, cfg),
    }
    write_json("analytics_summary.json", summary)
    log.info("analytics: %d videos tracked, %d newly learned", len(uploaded), learned)
    return summary
