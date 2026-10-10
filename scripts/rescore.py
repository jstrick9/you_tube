"""One-time utility: rebuild the bandit from the current analytics reward schema.

The former pipeline mislabeled the final ``audienceWatchRatio`` point as viewer completion. This
utility recomputes rewards from ``averageViewPercentage`` and engaged-view velocity when available;
curve telemetry is never scored as completion. The normal analytics pass now performs this migration
automatically, so use this script only when explicitly rebuilding a local state snapshot. Run:

    python scripts/rescore.py            # show what would change
    python scripts/rescore.py --apply    # write state/
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube.analytics import (REWARD_SCHEMA_VERSION, clean_view_pct, compute_reward, learnable,
                                migrate_legacy_retention_metrics, retention_diagnosis)  # noqa: E402
from autotube.common import load_config, read_json, write_json  # noqa: E402
from autotube.strategy import PRIOR, Strategy  # noqa: E402


def main(apply: bool = False) -> None:
    cfg = load_config()
    acfg = cfg["analytics"]
    hist = read_json("history.json", [])
    migrate_legacy_retention_metrics(hist)
    min_h = acfg["evaluate_after_hours"]

    matured = [h for h in hist if h.get("video_id")
               and h.get("metrics", {}).get("hours_live", 0) >= min_h
               and h.get("status") not in ("withdrawn", "missing")]
    pool_src = [h for h in matured if learnable(h, acfg)]
    pools = {
        "vph": [h["metrics"]["vph"] for h in pool_src
                if h["metrics"].get("vph_basis") == "engaged_views"],
        "avg_pct": [p for h in pool_src if (p := clean_view_pct(h["metrics"].get("avg_view_pct"))) is not None],
    }

    strat = Strategy(cfg)
    for arms in strat.state["arms"].values():          # wipe the posteriors trained on the bad signal
        for v in arms.values():
            v["a"], v["b"], v["n"] = PRIOR[0], PRIOR[1], 0
    strat.state["updates"] = 0
    strat.state.pop("aged_at", None)

    print(f"{'title':<44} {'views':>6} {'avg%':>6} {'old':>7} {'new':>7}")
    print("-" * 76)
    n = 0
    for h in hist:
        old = h.pop("reward", None)
        h.pop("reward_parts", None)
        if h not in matured:
            continue
        if not learnable(h, acfg) and h.get("status") != "rejected":
            print(f"{h['title'][:44]:<44} {h['metrics'].get('views', 0):>6} "
                  f"{'':>6} {str(old):>7} {'skip':>7}  (too few views to be evidence)")
            continue
        r, parts = (0.0, {"mode": "rejected"}) if h.get("status") == "rejected" else compute_reward(
            h["metrics"], pools, acfg)
        h["reward"], h["reward_parts"] = r, parts
        h["reward_version"] = REWARD_SCHEMA_VERSION
        strat.update(h["choice"], r)
        n += 1
        print(f"{h['title'][:44]:<44} {h['metrics'].get('views', 0):>6} "
              f"{round(clean_view_pct(h['metrics'].get('avg_view_pct')) or 0, 1):>6} {str(old):>7} {r:>7.3f}")

    diag = retention_diagnosis(pool_src, acfg)
    print(f"\nrescored {n} videos")
    print("retention:", diag.get("verdict") or "no curves yet — the next analytics pass fetches them")
    for dim, d in strat.report().items():
        top = ", ".join(f"{a['arm']}={a['mean']:.2f}(n={a['n']})" for a in d["arms"][:3])
        print(f"  {dim:<12} {d['status']:<18} {top}")

    strat.state["reward_schema_version"] = REWARD_SCHEMA_VERSION
    if apply:
        write_json("history.json", hist)
        strat.save()
        print("\n✓ state/ updated")
    else:
        print("\n(dry run — re-run with --apply to write state/)")


if __name__ == "__main__":
    main("--apply" in sys.argv)
