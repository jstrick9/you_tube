"""Generate a self-contained HTML dashboard from state/ (history, strategy, analytics).

    python scripts/dashboard.py   → dashboard.html
"""
import html
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autotube.analytics import average_view_watchlist  # noqa: E402
from autotube.common import load_config, read_json  # noqa: E402
from autotube.strategy import Strategy  # noqa: E402


def _number_label(value, suffix: str = "") -> str:
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return "–"
    if not math.isfinite(number) or number < 0:
        return "–"
    return f"{number:,.1f}{suffix}"


def _count_label(value) -> str:
    try:
        number = int(value)
    except (OverflowError, TypeError, ValueError):
        return "–"
    return f"{number:,}" if number >= 0 else "–"


def _watchlist_card(watchlist: dict, target_gate: float) -> str:
    gate = _number_label(watchlist.get("gate_pct", target_gate)).removesuffix(".0")
    try:
        evaluated = max(0, int(watchlist.get("evaluated") or 0))
        below = max(0, int(watchlist.get("below_target") or 0))
    except (OverflowError, TypeError, ValueError):
        evaluated = below = 0
    videos = watchlist.get("videos") if isinstance(watchlist.get("videos"), list) else []
    if not evaluated:
        summary = "No mature videos with valid average-view data yet."
    elif not below:
        summary = f"No videos below {gate}% among {evaluated} measured videos."
    else:
        summary = f"{below} of {evaluated} measured videos are below {gate}%."

    rows = []
    for video in videos:
        if not isinstance(video, dict):
            continue
        title = html.escape(str(video.get("title") or "Untitled"))
        video_id = str(video.get("id") or "")
        if video_id:
            url = html.escape(f"https://www.youtube.com/watch?v={quote(video_id, safe='')}", quote=True)
            title = f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{title}</a>"
        topic = html.escape(str(video.get("topic") or "–"))
        category = html.escape(str(video.get("category") or "–"))
        video_format = html.escape(str(video.get("format") or "–"))
        hook_style = html.escape(str(video.get("hook_style") or "–"))
        pct = _number_label(video.get("avg_view_pct"), "%")
        duration = _number_label(video.get("duration_seconds"), "s")
        views = _count_label(video.get("views"))
        engaged = _count_label(video.get("engaged_views_90d"))
        rows.append(
            f"<tr><td>{title}</td><td class=topic>{topic}</td><td><b>{pct}</b></td><td>{duration}</td>"
            f"<td>{category}</td><td>{video_format}</td><td>{hook_style}</td><td>{views}</td><td>{engaged}</td></tr>")
    listed = len(rows)
    if not rows:
        message = "No measured videos are currently below target." if evaluated else "No measured data yet."
        rows = [f"<tr><td colspan=9>{html.escape(message)}</td></tr>"]
    shown_note = (f"Showing the {listed} lowest of {below} under-target videos. "
                  if below > listed else "Sorted from lowest average-view percentage upward. ")
    return (f"<div class=card><h3>Videos below the {gate}% average-view target</h3>"
            f"<b>{html.escape(summary)}</b>"
            f"<em>{html.escape(shown_note)}Average percentage watched per playback—not viewer completion. "
            f"Public views and engaged views are separate context counts; low counts can make per-video values noisy; "
            f"— means unavailable.</em>"
            f"<div class=table-wrap><table class=watchlist><tr><th>Video</th><th>Topic</th><th>Avg view %</th>"
            f"<th>Runtime</th><th>Category</th><th>Format</th><th>Hook</th><th>Public views</th>"
            f"<th>Engaged views (90d)</th></tr>{''.join(rows)}</table></div></div>")


def bar(v: float) -> str:
    return f'<div class="b"><i style="width:{v*100:.0f}%"></i></div>'


def main():
    cfg = load_config()
    hist = read_json("history.json", [])
    summ = read_json("analytics_summary.json", {})
    rep = Strategy(cfg).report()
    rows = "".join(
        f"<tr><td>{html.escape(h.get('created_at', '')[:16])}</td><td>{html.escape(h.get('title', ''))}</td>"
        f"<td>{html.escape(str(h.get('choice', {}).get('format')))}</td><td>{h.get('review_score')}</td>"
        f"<td>{h.get('metrics', {}).get('views', '–')}</td><td>{h.get('reward', '–')}</td>"
        f"<td>{html.escape(str(h.get('status')))}</td></tr>" for h in reversed(hist[-60:]))
    strat = "".join(
        f"<div class=card><h3>{dim} <em>{html.escape(d['status'])}</em></h3>" + "".join(
            f"<div class=r><span>{html.escape(a['arm'])}</span>{bar(a['mean'])}<em>{a['mean']:.2f} · n={a['n']}</em></div>"
            for a in d["arms"]) + "</div>" for dim, d in rep.items())
    ret = summ.get("retention") or {}
    curve = ret.get("avg_curve") or []
    target = summ.get("average_view_target") or {}
    target_n = int(target.get("n") or 0)
    target_gate = float(target.get("gate_pct") or 70.0)
    if target_n:
        target_summary = (f"{int(target.get('passing') or 0)}/{target_n} measured videos at or above "
                          f"{target_gate:.0f}% · median {float(target.get('median_avg_view_pct') or 0):.1f}%")
    else:
        target_summary = "No averageViewPercentage data yet"
    target_card = (f"<div class=card><h3>Internal average-view target</h3>"
                   f"<b>{html.escape(target_summary)}</b>"
                   f"<em>Average percentage watched per playback; not viewer completion or a YouTube distribution rule.</em></div>")
    watchlist = summ.get("average_view_watchlist")
    if not isinstance(watchlist, dict):
        # Older analytics snapshots predate the watchlist field; derive it locally from the same
        # mature-video cohort so the dashboard updates immediately without waiting for another API run.
        min_hours = float(cfg.get("analytics", {}).get("evaluate_after_hours", 48))
        matured = []
        for video in hist:
            if not isinstance(video, dict) or video.get("status") in ("withdrawn", "missing"):
                continue
            metrics = video.get("metrics") or {}
            if not isinstance(metrics, dict):
                continue
            try:
                hours_live = float(metrics.get("hours_live") or 0)
            except (OverflowError, TypeError, ValueError):
                continue
            if hours_live >= min_hours:
                matured.append(video)
        watchlist = average_view_watchlist(matured, cfg.get("analytics", {}))
    watchlist_card = _watchlist_card(watchlist, target_gate)
    monetization = summ.get("shorts_monetization") or {}
    engaged_90d = int(monetization.get("engaged_views_90d") or 0)
    coverage = float(monetization.get("coverage") or 0)
    coverage_text = (f"{coverage:.0%} per-video coverage" if "coverage" in monetization
                     else "coverage not yet available")
    avg_view = ret.get("median_avg_view_pct")
    spark = ""
    if curve:
        pts = " ".join(f"{i * 30},{60 - min(1.2, v) * 50:.0f}" for i, v in enumerate(curve))
        avg_text = f"median average view {avg_view:.1f}%" if avg_view is not None else "average view n/a"
        spark = (f"<div class=card><h3>Relative audienceWatchRatio curve (not completion)</h3>"
                 f"<svg width=320 height=70><polyline fill=none stroke='#ffd400' stroke-width=2 points='{pts}'/></svg>"
                 f"<div style='color:#999;font-size:12px'>{html.escape(avg_text)} · curve is a segment-watch ratio</div></div>")
    verdict = (f"<div style='background:#1a1d24;border-left:3px solid #ffd400;padding:12px;margin:16px 0'>"
               f"<b>Average-view / curve note:</b> {html.escape(ret.get('verdict', 'not enough data yet'))}</div>")
    segment = summ.get("retention_segments") or {}
    segment_coverage = float(segment.get("video_coverage") or 0)
    segment_rows_reported = int(segment.get("segment_metric_rows_reported") or 0)
    segment_rows_expected = int(segment.get("segment_rows_in_curve") or 0)
    segment_row_coverage = float(segment.get("row_coverage") or 0)
    segment_row_coverage_text = (f"rows with all three metrics {segment_rows_reported}/{segment_rows_expected} "
                                  f"({segment_row_coverage:.0%})" if segment_rows_expected else "all-metric row coverage n/a")
    curve_refresh = summ.get("retention_curve_refresh") or {}
    refresh_attempted = int(curve_refresh.get("attempted_videos") or 0)
    if refresh_attempted:
        refresh_text = (f"Latest query pass: {int(curve_refresh.get('usable_reports_returned') or 0)}/"
                        f"{refresh_attempted} usable reports; "
                        f"{int(curve_refresh.get('reports_with_segment_metrics') or 0)} had event metrics.")
    else:
        refresh_text = "No retention-curve queries were due in the latest pass."
    if curve_refresh.get("note"):
        refresh_text += " " + str(curve_refresh["note"])
    segment_bins = segment.get("bins") if isinstance(segment.get("bins"), list) else []
    segment_rows = "".join(
        f"<tr><td>{html.escape(str(row.get('from_pct', '–')))}–{html.escape(str(row.get('to_pct', '–')))}%</td>"
        f"<td>{html.escape(str(row.get('started_watching', '–')))}</td>"
        f"<td>{html.escape(str(row.get('stopped_watching', '–')))}</td>"
        f"<td>{html.escape(str(row.get('total_segment_impressions', '–')))}</td></tr>"
        for row in segment_bins if isinstance(row, dict))
    segment_table = (f"<details><summary>Playback-segment counts by runtime</summary>"
                     f"<div style='max-height:300px;overflow:auto'><table><tr><th>Runtime</th><th>Started</th>"
                     f"<th>Stopped</th><th>In-video segment views</th></tr>{segment_rows}</table></div></details>"
                     if segment_rows else "<p>No fresh segment-event data yet.</p>")
    segment_meaning = segment.get("meaning") if isinstance(segment.get("meaning"), dict) else {}
    segment_definitions = "".join(
        f"<li><b>{label}:</b> {html.escape(str(segment_meaning.get(key)))}</li>"
        for key, label in (("started_watching", "startedWatching"),
                           ("stopped_watching", "stoppedWatching"),
                           ("total_segment_impressions", "totalSegmentImpressions"))
        if segment_meaning.get(key))
    definitions_details = (f"<details><summary>Metric meanings</summary><ul>{segment_definitions}</ul></details>"
                          if segment_definitions else "")
    segment_card = (f"<div class=card><h3>In-video retention segments</h3>"
                    f"<b>{int(segment.get('videos_with_segment_data') or 0)}/"
                    f"{int(segment.get('matured_videos_eligible') or 0)} mature videos covered ({segment_coverage:.0%})</b>"
                    f"<em>Pooled playback events in 5% runtime bins, not unique-viewer percentages or feed swipes; "
                    f"{html.escape(segment_row_coverage_text)}.</em>"
                    f"<em>{html.escape(refresh_text)}</em>"
                    f"{segment_table}{definitions_details}</div>")
    feed = summ.get("feed_discovery") or {}
    feed_note = feed.get("note") or "Studio feed-exposure metrics are unavailable through this API integration."
    feed_card = (f"<div class=card><h3>Shorts-feed exposure</h3><b>Unavailable via Analytics API</b>"
                 f"<em>{html.escape(str(feed_note))}</em></div>")
    experiment = summ.get("turn_experiment") or {}
    variant_rows = []
    for variant, stats in (experiment.get("by_variant") or {}).items():
        score = stats.get("median_turn_strength")
        avg_pct = stats.get("median_average_view_percentage")
        variant_rows.append(
            f"<li><b>{html.escape(str(variant))}</b>: assigned {int(stats.get('assigned_videos') or 0)}, "
            f"turn score n={int(stats.get('turn_score_n') or 0)} / median {score if score is not None else 'n/a'}, "
            f"average-view n={int(stats.get('retention_n') or 0)} / median {avg_pct if avg_pct is not None else 'n/a'}%, "
            f"at internal {target_gate:.0f}% target: {int(stats.get('over_internal_average_view_target') or 0)}/"
            f"{int(stats.get('retention_n') or 0)}</li>")
    relation = (experiment.get("all_variants") or {}).get("turn_strength_vs_average_view_percentage") or {}
    rho = relation.get("spearman_r")
    relation_text = (f"turn-score vs average-view Spearman r={rho} (n={relation.get('n_pairs', 0)}; descriptive only)"
                     if rho is not None else f"turn-score/average-view relationship not estimable yet (n={relation.get('n_pairs', 0)})")
    turn_card = (f"<div class=card><h3>Randomized beat-two study</h3>"
                 f"<em>{html.escape(str(experiment.get('status') or 'collecting'))} · no treatment is auto-selected</em>"
                 f"<ul>{''.join(variant_rows) or '<li>No assigned videos yet.</li>'}</ul>"
                 f"<em>{html.escape(relation_text)}. Turn score is optional telemetry, never an approval gate.</em></div>")
    diagnostic_cards = f"<div class=grid>{segment_card}{feed_card}{turn_card}</div>"
    monetization_card = (f"<div>Engaged Shorts views (90d)<b>{engaged_90d:,}</b>"
                         f"<em>{html.escape(coverage_text)} · YPP progress estimate, not Studio eligibility</em></div>")
    page = f"""<!doctype html><meta charset=utf-8><title>AutoTube dashboard</title>
<style>body{{font:14px system-ui;background:#0f1115;color:#e6e6e6;margin:24px}}h1{{margin:0 0 4px}}
.k{{display:flex;gap:12px;margin:16px 0}}.k div{{background:#1a1d24;padding:14px 18px;border-radius:10px}}
.k b{{display:block;font-size:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card{{background:#1a1d24;border-radius:10px;padding:12px}}.r{{display:grid;grid-template-columns:130px 1fr 90px;gap:8px;align-items:center;margin:4px 0}}
.b{{background:#2a2f3a;height:8px;border-radius:4px}}.b i{{display:block;height:100%;background:#ffd400;border-radius:4px}}
em{{color:#999;font-size:12px}}table{{width:100%;border-collapse:collapse;margin-top:16px}}td,th{{padding:6px;border-bottom:1px solid #262a33;text-align:left}}
.table-wrap{{overflow-x:auto}}.watchlist{{min-width:1050px}}.watchlist .topic{{max-width:280px;white-space:normal}}
a{{color:#ffd400}}</style>
<h1>📈 AutoTube — {html.escape(cfg['channel']['name'])}</h1><div style=color:#999>updated {html.escape(summ.get('updated', 'n/a')[:16])}</div>
<div class=k><div>Videos total<b>{len([h for h in hist if h.get('video_id')])}</b></div><div>Last 7d<b>{summ.get('videos_7d', 0)}</b></div>
<div>Views on uploads (last 7d)<b>{summ.get('views_7d', 0)}</b></div>{monetization_card}<div>Model updates<b>{Strategy(cfg).state.get('updates', 0)}</b></div></div>
{target_card}
{watchlist_card}
{verdict}
<h2>Retention and early-turn diagnostics</h2>{diagnostic_cards}
<h2>What the system has learned</h2><div class=grid>{spark}{strat}</div>
<h2>Recent videos</h2><table><tr><th>When</th><th>Title</th><th>Format</th><th>QA</th><th>Public views</th><th>Reward</th><th>Status</th></tr>{rows}</table>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print("wrote dashboard.html")


if __name__ == "__main__":
    main()
