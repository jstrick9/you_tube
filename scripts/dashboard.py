"""Generate a self-contained HTML dashboard from state/ (history, strategy, analytics).

    python scripts/dashboard.py   → dashboard.html
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autotube.common import load_config, read_json  # noqa: E402
from autotube.strategy import Strategy  # noqa: E402


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
    monetization_card = (f"<div>Engaged Shorts views (90d)<b>{engaged_90d:,}</b>"
                         f"<em>{html.escape(coverage_text)} · YPP progress estimate, not Studio eligibility</em></div>")
    page = f"""<!doctype html><meta charset=utf-8><title>AutoTube dashboard</title>
<style>body{{font:14px system-ui;background:#0f1115;color:#e6e6e6;margin:24px}}h1{{margin:0 0 4px}}
.k{{display:flex;gap:12px;margin:16px 0}}.k div{{background:#1a1d24;padding:14px 18px;border-radius:10px}}
.k b{{display:block;font-size:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card{{background:#1a1d24;border-radius:10px;padding:12px}}.r{{display:grid;grid-template-columns:130px 1fr 90px;gap:8px;align-items:center;margin:4px 0}}
.b{{background:#2a2f3a;height:8px;border-radius:4px}}.b i{{display:block;height:100%;background:#ffd400;border-radius:4px}}
em{{color:#999;font-size:12px}}table{{width:100%;border-collapse:collapse;margin-top:16px}}td,th{{padding:6px;border-bottom:1px solid #262a33;text-align:left}}</style>
<h1>📈 AutoTube — {html.escape(cfg['channel']['name'])}</h1><div style=color:#999>updated {html.escape(summ.get('updated', 'n/a')[:16])}</div>
<div class=k><div>Videos total<b>{len([h for h in hist if h.get('video_id')])}</b></div><div>Last 7d<b>{summ.get('videos_7d', 0)}</b></div>
<div>Views on uploads (last 7d)<b>{summ.get('views_7d', 0)}</b></div>{monetization_card}<div>Model updates<b>{Strategy(cfg).state.get('updates', 0)}</b></div></div>
{target_card}
{verdict}
<h2>What the system has learned</h2><div class=grid>{spark}{strat}</div>
<h2>Recent videos</h2><table><tr><th>When</th><th>Title</th><th>Format</th><th>QA</th><th>Public views</th><th>Reward</th><th>Status</th></tr>{rows}</table>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print("wrote dashboard.html")


if __name__ == "__main__":
    main()
