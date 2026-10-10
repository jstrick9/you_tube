"""Optimising for the tail, and earning comments.

Two findings from the channel's own telemetry:
  * 22 measured videos, max/median view ratio 1.36 - nothing has ever broken out, and the reward
    function could not have noticed if it had, because every component clamps at 1.0.
  * 233 likes and five comments total.
"""
import pytest
import yaml
from pathlib import Path
from autotube.analytics import compute_reward

ROOT = Path(__file__).resolve().parents[1]
POOLS = {"vph": [], "avg_pct": []}
ACFG = {"target_engaged_vph": 40.0, "w_breakout": 0.15, "breakout_multiple": 5.0}


def reward(vph, **kw):
    m = {"vph": vph, "vph_basis": "engaged_views", "engaged_views_90d": 1000,
         "views": 1000, "likes": 10, "comments": 0, "avg_view_pct": 50.8}
    m.update(kw)
    return compute_reward(m, POOLS, ACFG)


def test_reward_no_longer_saturates_the_moment_it_hits_target():
    """A 3x video and a 1x video scoring the same is why the bandit selects for 'reliably average'."""
    at_target = reward(40.0)[0]
    three_x = reward(120.0)[0]
    assert three_x > at_target + 0.05, f"{three_x} vs {at_target}: breakouts are still invisible"


def test_reward_is_monotonic_in_reach():
    rs = [reward(v)[0] for v in (5, 20, 40, 80, 120, 200)]
    assert rs == sorted(rs), rs
    assert len(set(rs)) == len(rs), f"ties mean lost signal: {rs}"


def test_breakout_component_is_zero_at_or_below_target_and_full_at_the_multiple():
    assert reward(40.0)[1]["breakout"] == 0.0
    assert reward(10.0)[1]["breakout"] == 0.0
    assert reward(40.0 * 5.0)[1]["breakout"] == pytest.approx(1.0, abs=0.01)


def test_breakout_is_log_scaled_so_early_growth_matters_most():
    """1x->2x must move the needle more than 4x->5x; that is where the learning signal lives."""
    b = {k: reward(40.0 * k)[1]["breakout"] for k in (1, 2, 4, 5)}
    assert (b[2] - b[1]) > (b[5] - b[4]) > 0


def test_reward_stays_in_range_and_degrades_gracefully():
    for v in (0.0, 1e9):
        r, parts = reward(v)
        assert 0.0 <= r <= 1.0 and 0.0 <= parts["breakout"] <= 1.0
    r, _ = compute_reward({"views": 0}, POOLS, ACFG)      # no vph at all
    assert 0.0 <= r <= 1.0


def test_breakout_can_be_switched_off_without_changing_anything_else():
    off = compute_reward({"vph": 120.0, "vph_basis": "engaged_views", "engaged_views_90d": 1000,
                          "views": 1000, "likes": 10, "comments": 0, "avg_view_pct": 50.8},
                         POOLS, {**ACFG, "w_breakout": 0.0})
    on_at_zero = compute_reward({"vph": 10.0, "vph_basis": "engaged_views", "engaged_views_90d": 1000,
                                 "views": 1000, "likes": 10, "comments": 0, "avg_view_pct": 50.8},
                                POOLS, {**ACFG, "w_breakout": 0.0})
    assert off[1]["breakout"] > 0 and 0 <= off[0] <= 1 and 0 <= on_at_zero[0] <= 1


# ── comment devices ──────────────────────────────────────────────────────────
def test_comment_device_is_a_learned_dimension_with_a_do_nothing_arm():
    from autotube.strategy import DIMENSIONS
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    devices = cfg["content"]["comment_devices"]
    assert "comment_device" in DIMENSIONS
    assert "none" in devices, "there must be a control arm, or the bandit cannot learn that devices hurt"
    assert len(devices) >= 3


def test_every_configured_device_has_written_guidance():
    from autotube.scriptwriter import COMMENT_GUIDE
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    for d in cfg["content"]["comment_devices"]:
        assert d in COMMENT_GUIDE, f"{d} is selectable but the prompt says nothing about it"


def test_the_planner_assigns_a_device_to_every_video():
    from autotube.strategy import Strategy
    from autotube.common import load_config
    plans = Strategy(load_config()).plan(12)
    assert all(p.get("comment_device") for p in plans)


def test_devices_do_not_smuggle_in_a_call_to_action():
    """check_closer bans CTAs; a device that requires one would deadlock generation."""
    from autotube.scriptwriter import COMMENT_GUIDE
    from autotube.gates import BANNED_CLOSERS
    import re
    for name, text in COMMENT_GUIDE.items():
        low = text.lower()
        for pat in BANNED_CLOSERS:
            if re.search(pat, low):
                # only acceptable inside an explicit prohibition
                assert "never" in low or "do not" in low or "don't" in low, \
                    f"device {name!r} tells the model to write a banned closer: {pat!r}"


@pytest.mark.parametrize("closer", [
    "the pipe or the tunnel, which would you have taken",
    "and that makes it the dumbest decision in aviation history",
    "and nobody has ever worked out why it stopped",
    "almost nobody gets that part right",
])
def test_device_shaped_endings_survive_the_closer_gate(closer):
    from autotube.gates import check_closer
    assert check_closer(closer) == [], closer


@pytest.mark.parametrize("closer", ["comment below and let me know", "thanks for watching, see you next time"])
def test_real_ctas_are_still_blocked(closer):
    from autotube.gates import check_closer
    assert check_closer(closer) != []


# ── subscriber conversion: the YPP gate views cannot substitute for ─────────
SUBCFG = {"target_engaged_vph": 40.0, "w_retention": 0.50, "w_reach": 0.20, "w_subs": 0.20,
          "w_engagement": 0.10, "target_subs_per_1k": 2.0, "w_breakout": 0.15}


def _r(subs_gained=0, subs_lost=0, views=1000):
    m = {"vph": 20, "vph_basis": "engaged_views", "engaged_views_90d": views,
         "views": views, "likes": 10, "comments": 0, "avg_view_pct": 50.8,
         "subs_gained": subs_gained, "subs_lost": subs_lost}
    return compute_reward(m, POOLS, SUBCFG)


def test_subscriber_conversion_changes_the_reward():
    """1,000 subscribers is required regardless of view count; the optimiser must be able to see it."""
    assert _r(subs_gained=2)[0] > _r(subs_gained=0)[0] + 0.1


def test_subscriber_component_is_monotonic_and_capped():
    vals = [_r(subs_gained=n)[1]["subs"] for n in (0, 1, 2, 10)]
    assert vals == sorted(vals) and vals[-1] == 1.0 and vals[0] == 0.0


def test_lost_subscribers_are_netted_off():
    assert _r(subs_gained=5, subs_lost=5)[1]["subs"] == 0.0
    assert _r(subs_gained=5, subs_lost=10)[1]["subs"] == 0.0        # never negative


def test_conversion_is_a_rate_not_a_count():
    """A video with 10x the views and 10x the subs converted equally well."""
    a = _r(subs_gained=2, views=1000)[1]["subs"]
    b = _r(subs_gained=20, views=10000)[1]["subs"]
    assert a == b


def test_missing_subscriber_data_does_not_crash_or_punish_silently():
    r, parts = compute_reward({"vph": 20, "views": 1000, "avg_view_pct": 50}, POOLS, SUBCFG)
    assert 0.0 <= r <= 1.0 and parts["subs"] == 0.0


def test_analytics_actually_requests_the_metric():
    """The reward can only see subscribers if the API call asks for them."""
    from autotube import youtube
    src = Path(youtube.__file__).read_text()
    assert "subscribersGained" in src and "subscribersLost" in src


def test_weights_still_sum_to_one():
    a = yaml.safe_load((ROOT / "config.yaml").read_text())["analytics"]
    total = a["w_retention"] + a["w_reach"] + a["w_subs"] + a["w_engagement"]
    assert abs(total - 1.0) < 1e-9, f"weights sum to {total}"
