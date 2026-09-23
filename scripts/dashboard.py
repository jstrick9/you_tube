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
        f"<div class=card><h3>{dim}</h3>" + "".join(
            f"<div class=r><span>{html.escape(a['arm'])}</span>{bar(a['mean'])}<em>{a['mean']:.2f} · n={a['n']}</em></div>"
            for a in arms) + "</div>" for dim, arms in rep.items())
    page = f"""<!doctype html><meta charset=utf-8><title>AutoTube dashboard</title>
<style>body{{font:14px system-ui;background:#0f1115;color:#e6e6e6;margin:24px}}h1{{margin:0 0 4px}}
.k{{display:flex;gap:12px;margin:16px 0}}.k div{{background:#1a1d24;padding:14px 18px;border-radius:10px}}
.k b{{display:block;font-size:24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.card{{background:#1a1d24;border-radius:10px;padding:12px}}.r{{display:grid;grid-template-columns:130px 1fr 90px;gap:8px;align-items:center;margin:4px 0}}
.b{{background:#2a2f3a;height:8px;border-radius:4px}}.b i{{display:block;height:100%;background:#ffd400;border-radius:4px}}
em{{color:#999;font-size:12px}}table{{width:100%;border-collapse:collapse;margin-top:16px}}td,th{{padding:6px;border-bottom:1px solid #262a33;text-align:left}}</style>
<h1>📈 AutoTube — {html.escape(cfg['channel']['name'])}</h1><div style=color:#999>updated {html.escape(summ.get('updated', 'n/a')[:16])}</div>
<div class=k><div>Videos total<b>{len([h for h in hist if h.get('video_id')])}</b></div><div>Last 7d<b>{summ.get('videos_7d', 0)}</b></div>
<div>Views 7d<b>{summ.get('views_7d', 0)}</b></div><div>Model updates<b>{Strategy(cfg).state.get('updates', 0)}</b></div></div>
<h2>What the system has learned</h2><div class=grid>{strat}</div>
<h2>Recent videos</h2><table><tr><th>When</th><th>Title</th><th>Format</th><th>QA</th><th>Views</th><th>Reward</th><th>Status</th></tr>{rows}</table>"""
    (ROOT / "dashboard.html").write_text(page, encoding="utf-8")
    print("wrote dashboard.html")


if __name__ == "__main__":
    main()
