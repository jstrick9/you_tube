"""The learning loop: reward shaping, data hygiene, retention diagnosis, bandit forgetting."""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube.analytics import (clean_view_pct, compute_reward, learnable,  # noqa: E402
                                migrate_legacy_retention_metrics, retention_diagnosis,
                                _reset_reward_model, _shorts_progress, _score, _spread)
from autotube.common import now_utc  # noqa: E402
from autotube.strategy import PRIOR, Strategy  # noqa: E402
from autotube.youtube import summarize_curve  # noqa: E402

ACFG = {
    "w_retention": 0.65, "w_reach": 0.25, "w_engagement": 0.10,
    "target_avg_view_pct": 70.0, "target_engaged_vph": 40.0, "target_engagement": 0.06,
    "rank_min_videos": 40, "rank_min_spread": 0.35, "min_views_to_learn": 50,
}


def _m(views=1000, pct=None, vph=20.0, likes=20, comments=2, completion=None,
       engaged=1000, basis="engaged_views", subs_gained=0, subs_lost=0):
    d = {"views": views, "engaged_views_90d": engaged, "vph": vph, "vph_basis": basis,
         "likes": likes, "comments": comments, "subs_gained": subs_gained, "subs_lost": subs_lost}
    if pct is not None:
        d["avg_view_pct"] = pct
    if completion is not None:
        d["completion"] = completion       # legacy field must not enter the retention reward
    return d


# ── the bug this whole change exists to fix ───────────────────────────────────
def test_reward_tracks_retention_on_a_flat_view_distribution():
    """The old percentile reward was degenerate when every video gets a similar view count.

    Real data: 13 videos between 324 and 1,279 views. The old reward correlated -0.18 with average
    view percentage; the worst-retaining video (41%) scored 0.93 and the best (80%) scored 0.50.
    The new reward must be monotonic in retention when reach is roughly constant.
    """
    pool = {"vph": [20.0] * 13, "avg_pct": [30, 41, 42, 45, 48, 48, 50, 52, 52, 55, 59, 60, 80]}
    rewards = [compute_reward(_m(pct=p), pool, ACFG)[0] for p in (30, 45, 60, 80)]
    assert rewards == sorted(rewards), f"reward must rise with retention, got {rewards}"
    assert rewards[-1] - rewards[0] > 0.25, "reward must actually discriminate, not just order"


def test_absolute_mode_until_the_pool_has_spread():
    flat = [20.0] * 50                       # 50 videos but no variance → ranking is meaningless
    assert _spread(flat) < 0.01
    assert _score(flat, 20.0, target=40.0, min_n=40, min_spread=0.35) == 0.5   # 20/40, absolute
    varied = [v * 1.0 for v in range(1, 81)]
    assert _spread(varied) > 0.35
    s = _score(varied, 80.0, target=40.0, min_n=40, min_spread=0.35)
    assert s > 0.9, "a top performer in a varied pool should score high via the percentile blend"


def test_retention_is_weighted_above_reach():
    pool = {"vph": [20.0] * 5, "avg_pct": [50] * 5}
    great_retention_low_reach = compute_reward(_m(pct=90, vph=5), pool, ACFG)[0]
    poor_retention_high_reach = compute_reward(_m(pct=20, vph=80), pool, ACFG)[0]
    assert great_retention_low_reach > poor_retention_high_reach


def test_legacy_completion_value_is_not_used_as_retention():
    pool = {"vph": [20.0] * 5, "avg_pct": []}
    high, high_parts = compute_reward(_m(pct=None, completion=0.99), pool, ACFG)
    low, low_parts = compute_reward(_m(pct=None, completion=0.01), pool, ACFG)
    assert high == low and high_parts["retention"] == low_parts["retention"] == 0.0


def test_missing_retention_redistributes_its_weight():
    pool = {"vph": [20.0] * 5, "avg_pct": []}
    r, parts = compute_reward(_m(pct=None, vph=40), pool, ACFG)
    assert 0.0 <= r <= 1.0 and parts["retention"] == 0
    assert r > 0.5, "with no retention data, a strong reach number should still score well"


# ── data hygiene ──────────────────────────────────────────────────────────────
def test_implausible_view_percentage_is_discarded():
    assert clean_view_pct(80.3) == 80.3
    assert clean_view_pct(135.0) == 135.0        # plausible repeated playback; not assumed to be automatic looping
    assert clean_view_pct(3474.28) is None       # observed artifact on a 22-view video
    assert clean_view_pct(0) is None and clean_view_pct(None) is None and clean_view_pct("x") is None


def test_low_traffic_videos_are_not_evidence():
    assert not learnable({"metrics": {"views": 10000, "engaged_views_90d": 22}}, ACFG)
    assert learnable({"metrics": {"views": 22, "engaged_views_90d": 500}}, ACFG)
    assert learnable({"metrics": {"views": 500}}, ACFG)  # labeled public-view fallback for API gaps


def test_legacy_completion_curve_is_renamed_not_kept_as_a_viewer_percentage():
    history = [{"metrics": {"completion": 0.42, "hook_retention": 0.8,
                             "curve": [1.2] * 11, "swipe_point": 0.5}}]
    assert migrate_legacy_retention_metrics(history) == 4
    metrics = history[0]["metrics"]
    assert "completion" not in metrics and "hook_retention" not in metrics and "curve" not in metrics
    assert metrics["end_watch_ratio"] == 0.42
    assert metrics["watch_ratio_at_15pct"] == 0.8
    assert metrics["audience_watch_ratio_curve"] == [1.2] * 11


def test_ypp_progress_uses_engaged_views_and_exposes_incomplete_coverage():
    uploaded = [
        {"status": "public", "metrics": {"privacy": "public", "engaged_views_90d": 125}},
        {"status": "public", "metrics": {"privacy": "public", "views": 10000}},
        {"status": "withdrawn", "metrics": {"privacy": "public", "engaged_views_90d": 900}},
    ]
    progress = _shorts_progress(uploaded)
    assert progress["engaged_views_90d"] == 125
    assert progress["videos_with_metric"] == 1 and progress["public_videos"] == 2
    assert not progress["complete"] and progress["coverage"] == 0.5


def test_retention_query_maps_engaged_views_and_average_view_percentage(monkeypatch):
    from autotube import youtube

    requests = []

    class Request:
        def execute(self):
            return {"rows": [["video-1", 1200, 850, 72.5, 14, 3, 1, 8]]}

    class Reports:
        def query(self, **kwargs):
            requests.append(kwargs)
            return Request()

    class Analytics:
        def reports(self):
            return Reports()

    monkeypatch.setattr(youtube, "service", lambda *args: Analytics())
    out = youtube.retention(["video-1"], "2026-07-12", "2026-10-09")
    assert out["video-1"]["engaged_views_90d"] == 850
    assert out["video-1"]["avg_view_pct"] == 72.5
    assert "engagedViews" in requests[0]["metrics"]
    assert requests[0]["dimensions"] == "video"


def test_public_views_fallback_is_not_mixed_into_engaged_velocity_reward():
    pool = {"vph": [20.0, 60.0, 100.0], "avg_pct": [70.0] * 3}
    low = compute_reward(_m(pct=70, vph=1, engaged=None, basis="public_views_fallback"), pool, ACFG)[0]
    high = compute_reward(_m(pct=70, vph=1000, engaged=None, basis="public_views_fallback"), pool, ACFG)[0]
    assert low == high, "raw public-view velocity must not be scored against engaged-view targets"


def test_old_rewards_are_rebuilt_once_after_retention_metric_correction(monkeypatch):
    old_state = {"arms": {}, "updates": 9, "reward_schema_version": 0}
    monkeypatch.setattr("autotube.strategy.read_json", lambda *args, **kwargs: old_state)
    from autotube.common import load_config
    strat = Strategy(load_config())
    history = [{"reward": 0.9, "reward_parts": {"retention": 0.99}, "choice": {"format": "backstory"}}]
    assert _reset_reward_model(strat, history)
    assert "reward" not in history[0] and "reward_parts" not in history[0]
    assert strat.state["reward_schema_version"] == 2 and strat.state["updates"] == 0
    assert not _reset_reward_model(strat, history), "the correction should not reset the bandit repeatedly"


# ── retention curves ──────────────────────────────────────────────────────────
def test_summarize_curve_keeps_relative_watch_ratio_names():
    rows = [(i / 20, 1.0 if i / 20 < 0.1 else 0.4) for i in range(21)]   # sharp relative drop after the opening
    s = summarize_curve(rows)
    assert s["watch_ratio_at_15pct"] < 0.5 and s["end_watch_ratio"] == 0.4
    assert len(s["audience_watch_ratio_curve"]) == 11
    assert "completion" not in s


def test_diagnosis_names_hook_middle_or_ending():
    def vids(curve):
        return [{"metrics": {"audience_watch_ratio_curve": curve, "avg_view_pct": 62.0}}
                for _ in range(4)]
    hook_leak = retention_diagnosis(vids([1.0, 0.5, .48, .46, .44, .42, .40, .38, .36, .34, .32]), ACFG)
    assert hook_leak["where"] == "hook"
    end_leak = retention_diagnosis(vids([1.0, .95, .92, .90, .88, .85, .80, .70, .60, .55, .20]), ACFG)
    assert end_leak["where"] == "ending"
    assert "not a percentage of viewers" in end_leak["verdict"]
    healthy = retention_diagnosis(vids([1.0, .95, .92, .90, .88, .86, .84, .82, .80, .78, .76]), ACFG)
    assert healthy["where"] == "watch_time"    # average-view target is below 70; ratio curve has no abrupt leak
    assert retention_diagnosis([], ACFG)["verdict"] == ""


# ── bandit ────────────────────────────────────────────────────────────────────
def _cfg(**over):
    c = {
        "channel": {"categories": ["history", "science"]},
        "content": {"formats": ["facts3", "backstory"], "hook_styles": ["question", "bold_claim"],
                    "seasonal_formats": {}},
        "video": {"voices": ["v1", "v2"]},
        "analytics": {"exploration": 0.0, "frozen_dimensions": ["voice"], "min_observations": 2},
    }
    c["analytics"].update(over)
    return c


def test_frozen_dimensions_are_not_learned(tmp_path, monkeypatch):
    monkeypatch.setattr("autotube.strategy.read_json", lambda *a, **k: {"arms": {}, "updates": 0})
    s = Strategy(_cfg())
    s.update({"category": "history", "voice": "v1", "format": "facts3"}, 1.0)
    assert s.state["arms"]["category"]["history"]["n"] == 1
    assert s.state["arms"]["voice"]["v1"]["n"] == 0, "frozen dim must not consume evidence"
    assert s.report()["voice"]["status"] == "frozen"


def test_forgetting_is_time_based_not_update_based(monkeypatch):
    monkeypatch.setattr("autotube.strategy.read_json", lambda *a, **k: {"arms": {}, "updates": 0})
    s = Strategy(_cfg())
    for _ in range(30):                       # a month and a half of uploads at 3/day
        s.update({"category": "history"}, 1.0)
    # the old code decayed on every update (0.97^30); evidence must survive instead
    assert s.state["arms"]["category"]["history"]["a"] > 25, "30 wins must still read as 30 wins"
    s.age(now_utc())                          # first call only stamps the clock
    before = s.state["arms"]["category"]["history"]["a"]
    s.age(now_utc() + timedelta(days=45))     # one half-life later
    after = s.state["arms"]["category"]["history"]["a"]
    assert abs((after - PRIOR[0]) - (before - PRIOR[0]) * 0.5) < 0.01


def test_sampling_is_random_until_there_is_evidence(monkeypatch):
    monkeypatch.setattr("autotube.strategy.read_json", lambda *a, **k: {"arms": {}, "updates": 0})
    s = Strategy(_cfg(min_observations=5))
    s.state["arms"]["category"]["history"].update({"a": 9.0, "b": 1.0, "n": 1})   # 1 lucky video
    picks = {s.sample("category") for _ in range(60)}
    assert picks == {"history", "science"}, "must not chase a posterior built on one observation"


# --- Cold start -------------------------------------------------------------
# Live state showed every observation sitting on format names that had since been
# replaced in config, leaving nine of eleven live formats with n=0 and a posterior
# ordering that was pure noise. Untested arms must be measured before trusted.

def _fresh(monkeypatch, **over):
    monkeypatch.setattr("autotube.strategy.read_json", lambda *a, **k: {"arms": {}, "updates": 0})
    return Strategy(_cfg(**over))


def test_untested_arms_are_sampled_before_any_posterior_is_trusted(monkeypatch):
    s = _fresh(monkeypatch, min_observations=0, cold_start_min=3)
    # history looks spectacular on a single video; science has never run at all.
    s.state["arms"]["category"]["history"].update({"a": 99.0, "b": 1.0, "n": 1})
    picks = [s.sample("category") for _ in range(300)]
    assert picks.count("science") > 0, "an arm with no evidence must still get measured"
    assert picks.count("science") > picks.count("history"), \
        "the least-tested arm should be the likeliest, not merely possible"


def test_cold_start_stops_once_every_arm_has_evidence(monkeypatch):
    s = _fresh(monkeypatch, min_observations=0, cold_start_min=3)
    for arm in ("history", "science"):
        s.state["arms"]["category"][arm]["n"] = 3
    s.state["arms"]["category"]["history"].update({"a": 99.0, "b": 1.0})
    picks = [s.sample("category") for _ in range(200)]
    assert picks.count("history") > 190, "past cold start the posterior must take over"


def test_cold_start_is_not_deterministic(monkeypatch):
    """Always picking the single least-tested arm would make the schedule a fixed
    rotation - a predictable template is what the repetition heuristics look for."""
    s = _fresh(monkeypatch, min_observations=0, cold_start_min=3)
    s.state["arms"]["category"]["history"]["n"] = 2   # 1 short of the floor
    s.state["arms"]["category"]["science"]["n"] = 0
    picks = {s.sample("category") for _ in range(200)}
    assert picks == {"history", "science"}, "both under-tested arms must remain reachable"


def test_cold_start_respects_frozen_dimensions(monkeypatch):
    s = _fresh(monkeypatch, min_observations=0, cold_start_min=3)
    assert "voice" in s.frozen
    assert {s.sample("voice") for _ in range(100)} <= {"v1", "v2"}


def test_cold_start_can_be_disabled(monkeypatch):
    s = _fresh(monkeypatch, min_observations=0, cold_start_min=0)
    s.state["arms"]["category"]["history"].update({"a": 99.0, "b": 1.0, "n": 1})
    picks = [s.sample("category") for _ in range(200)]
    assert picks.count("history") > 150, "cold_start_min=0 must restore pure Thompson behaviour"


def test_every_live_format_is_measured_within_a_reasonable_run(monkeypatch):
    """The practical claim: cold start converts 'eventually' into 'within days'."""
    s = _fresh(monkeypatch, min_observations=0, cold_start_min=3)
    seen = {}
    for _ in range(40):
        f = s.sample("format")
        seen[f] = seen.get(f, 0) + 1
        s.state["arms"]["format"][f]["n"] = seen[f]
    assert set(seen) == {"facts3", "backstory"}
    assert all(v >= 3 for v in seen.values()), "every format should clear the floor"
