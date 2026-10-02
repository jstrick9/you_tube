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

from datetime import datetime

from .common import now_utc, read_json, write_json

log = logging.getLogger("autotube.strategy")

DIMENSIONS = ("category", "format", "hook_style", "voice", "source")
HALF_LIFE_DAYS = 45.0   # evidence decays with TIME, not with upload count (see Strategy.age)
PRIOR = (1.0, 1.0)      # Beta(1,1)


class Strategy:
    # class-level defaults so tests (and any caller) can build a bare Strategy via __new__
    frozen: set[str] = set()
    min_obs: int = 0
    explore: float = 0.15

    def __init__(self, cfg: dict):
        self.cfg = cfg
        acfg = cfg.get("analytics", {}) or {}
        self.explore = float(acfg.get("exploration", 0.15))
        # Dimensions we deliberately do NOT learn. With 15 formats x 5 hooks x 5 voices x 12 categories
        # there are thousands of cells and only a few videos a day; spreading the evidence across all of
        # them means no dimension ever separates. Frozen dimensions rotate deterministically for variety
        # and spend zero statistical power, so category and format converge several times faster.
        self.frozen = set(acfg.get("frozen_dimensions") or [])
        self.min_obs = int(acfg.get("min_observations", 8))
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
            "format": self.formats_in_season(),
            "hook_style": list(c["content"]["hook_styles"]),
            "voice": list(c["video"]["voices"]),
            "source": ["youtube_outliers", "wikipedia", "google_trends", "reddit", "hackernews", "on_this_day",
                       "youtube_chart", "evergreen"],
        }

    def formats_in_season(self, month: int | None = None) -> list[str]:
        """Configured formats minus seasonal ones that are out of season (e.g. creepy_true outside October)."""
        month = month or now_utc().month
        seasonal = self.cfg["content"].get("seasonal_formats") or {}
        return [f for f in self.cfg["content"]["formats"]
                if f not in seasonal or month in [int(m) for m in seasonal[f]]]

    def seasonal_now(self) -> list[str]:
        seasonal = self.cfg["content"].get("seasonal_formats") or {}
        return [f for f in self.formats_in_season() if f in seasonal]

    def fit(self, plan: dict, topic: dict, used: set[str] | None = None) -> dict:
        """Make the plan's format one the topic can honestly support (the topic selector lists them in
        topic['formats']). In season, a seasonal format (creepy_true in October) wins once per run if it fits."""
        allowed = self.formats_in_season()
        fits = [f for f in (topic.get("formats") or []) if f in allowed]
        plan = dict(plan)
        used = used if used is not None else set()
        for f in self.seasonal_now():
            if f in fits and f not in used:
                plan["format"] = f
                return plan
        fresh = [f for f in fits if f not in used]            # don't make the same format twice in one run
        pool = fresh or fits
        if pool and (plan.get("format") not in pool):
            plan["format"] = self.sample("format", exclude=set(allowed) - set(pool))
        return plan

    # ── sampling ──────────────────────────────────────────────────────────────
    def observations(self, dim: str) -> int:
        return sum(v.get("n", 0) for v in self.state["arms"].get(dim, {}).values())

    def sample(self, dim: str, exclude: set[str] | None = None) -> str:
        opts = self.options()[dim]
        arms = {k: v for k, v in self.state["arms"][dim].items()
                if k in opts and k not in (exclude or set())}
        if not arms:
            arms = {k: v for k, v in self.state["arms"][dim].items() if k in opts} or self.state["arms"][dim]
        # A frozen dimension, or one with too little evidence to tell its arms apart, is chosen at
        # random rather than by a posterior built from noise. Thompson sampling on 3 observations is
        # not exploration, it is superstition.
        if dim in self.frozen or self.observations(dim) < self.min_obs or random.random() < self.explore:
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
        plans, used_formats, used_voices = [], set(), set()
        for _ in range(n):
            fmt = self.sample("format", exclude=used_formats if len(used_formats) < len(self.options()["format"]) - 1 else None)
            used_formats.add(fmt)
            plans.append({
                "format": fmt,
                "hook_style": self.sample("hook_style"),
                "voice": self.cfg.get("persona", {}).get("voice") or self.sample(
                    "voice", exclude=used_voices if len(used_voices) < len(self.options()["voice"]) - 1 else None),
            })
            used_voices.add(plans[-1]["voice"])
        return plans

    # ── learning ──────────────────────────────────────────────────────────────
    def age(self, now=None) -> None:
        """Forget with the calendar, not with the upload counter.

        The old code multiplied every arm by 0.97 on EVERY update. At 3 videos/day that is 0.97^90 by
        the end of a month — early evidence had evaporated long before enough of it accumulated to
        separate 15 formats. Decay is now a 45-day half-life applied once per analytics pass, so the
        model still adapts when tastes shift but can actually accumulate evidence in the meantime.
        """
        now = now or now_utc()
        last = self.state.get("aged_at")
        self.state["aged_at"] = now.isoformat()
        if not last:
            return
        try:
            days = (now - datetime.fromisoformat(last)).total_seconds() / 86400.0
        except (TypeError, ValueError):
            return
        if days <= 0:
            return
        factor = 0.5 ** (days / HALF_LIFE_DAYS)
        for arms in self.state["arms"].values():
            for v in arms.values():
                v["a"] = PRIOR[0] + (v["a"] - PRIOR[0]) * factor
                v["b"] = PRIOR[1] + (v["b"] - PRIOR[1]) * factor

    def update(self, choice: dict, reward: float) -> None:
        reward = max(0.0, min(1.0, reward))
        for dim in DIMENSIONS:
            if dim in self.frozen:
                continue                 # not learned → don't pretend we measured it
            arm = choice.get(dim)
            if not arm or arm not in self.state["arms"].get(dim, {}):
                continue
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
            n = sum(v.get("n", 0) for v in arms.values())
            status = ("frozen" if dim in self.frozen
                      else "learning" if n >= self.min_obs else f"exploring ({n}/{self.min_obs})")
            out[dim] = {"status": status, "observations": n, "arms": sorted(
                [{"arm": k, "mean": round(v["a"] / (v["a"] + v["b"]), 3), "n": v["n"]} for k, v in arms.items()],
                key=lambda x: -x["mean"])}
        return out
