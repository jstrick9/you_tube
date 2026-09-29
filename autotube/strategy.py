"""Self-optimizing content strategy (multi-armed bandit, Thompson sampling).

Every video is a combination of "arms" across several dimensions:
    category · format · hook_style · voice · slot
After a video has been live for N hours, analytics.py converts its performance
into a reward in [0,1] (percentile vs. the channel's own recent videos, blending
views/hour and average-view-percentage). Each arm used gets a fractional Beta
update. Future choices are sampled from the posteriors → the channel naturally
drifts toward what works for THIS audience, while still exploring.

A slow decay (forgetting factor) lets the model adapt when tastes shift.
"""
from __future__ import annotations

import logging
import random

from .common import read_json, write_json

log = logging.getLogger("autotube.strategy")

DIMENSIONS = ("category", "format", "hook_style", "voice", "source")
DECAY = 0.97          # applied on every update → recent results matter more
PRIOR = (1.0, 1.0)    # Beta(1,1)


class Strategy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.explore = float(cfg.get("analytics", {}).get("exploration", 0.15))
        self.state = read_json("strategy.json", {"arms": {}, "updates": 0})
        arms = self.state.setdefault("arms", {})
        options = self.options()
        for dim, opts in options.items():
            d = arms.setdefault(dim, {})
            for o in opts:
                d.setdefault(o, {"a": PRIOR[0], "b": PRIOR[1], "n": 0})

    def options(self) -> dict[str, list[str]]:
        c = self.cfg
        return {
            "category": list(c["channel"]["categories"]),
            "format": list(c["content"]["formats"]),
            "hook_style": list(c["content"]["hook_styles"]),
            "voice": list(c["video"]["voices"]),
            "source": ["youtube_outliers", "wikipedia", "google_trends", "reddit", "hackernews", "on_this_day",
                       "youtube_chart", "evergreen"],
        }

    # ── sampling ──────────────────────────────────────────────────────────────
    def sample(self, dim: str, exclude: set[str] | None = None) -> str:
        arms = {k: v for k, v in self.state["arms"][dim].items()
                if k in self.options()[dim] and k not in (exclude or set())}
        if not arms:
            arms = self.state["arms"][dim]
        if random.random() < self.explore:
            return random.choice(list(arms))
        draws = {k: random.betavariate(max(v["a"], 0.05), max(v["b"], 0.05)) for k, v in arms.items()}
        return max(draws, key=draws.get)

    def source_weight(self, source: str) -> float:
        """Posterior mean for a trend source (which kind of trend actually gets OUR videos views)."""
        v = self.state["arms"].get("source", {}).get(source)
        return v["a"] / (v["a"] + v["b"]) if v else 0.5

    def category_weight(self, category: str) -> float:
        """Posterior mean for a category, used to re-rank trending topics."""
        v = self.state["arms"]["category"].get(category)
        if not v:
            return 0.5
        return v["a"] / (v["a"] + v["b"])

    def plan(self, n: int) -> list[dict]:
        """Pick n distinct combos. Avoid identical format/voice back-to-back for variety."""
        plans, used_formats = [], set()
        for _ in range(n):
            fmt = self.sample("format", exclude=used_formats if len(used_formats) < len(self.options()["format"]) - 1 else None)
            used_formats.add(fmt)
            plans.append({
                "format": fmt,
                "hook_style": self.sample("hook_style"),
                "voice": self.sample("voice"),
            })
        return plans

    # ── learning ──────────────────────────────────────────────────────────────
    def update(self, choice: dict, reward: float) -> None:
        reward = max(0.0, min(1.0, reward))
        for dim in DIMENSIONS:
            arm = choice.get(dim)
            if not arm or arm not in self.state["arms"].get(dim, {}):
                continue
            for v in self.state["arms"][dim].values():   # forgetting
                v["a"] = PRIOR[0] + (v["a"] - PRIOR[0]) * DECAY
                v["b"] = PRIOR[1] + (v["b"] - PRIOR[1]) * DECAY
            a = self.state["arms"][dim][arm]
            a["a"] += reward
            a["b"] += 1.0 - reward
            a["n"] += 1
        self.state["updates"] = self.state.get("updates", 0) + 1

    def save(self) -> None:
        write_json("strategy.json", self.state)

    def report(self) -> dict:
        out = {}
        for dim, arms in self.state["arms"].items():
            out[dim] = sorted(
                [{"arm": k, "mean": round(v["a"] / (v["a"] + v["b"]), 3), "n": v["n"]} for k, v in arms.items()],
                key=lambda x: -x["mean"])
        return out
