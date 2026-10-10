"""Internal duration and average-view goals; these tests do not assert a YouTube distribution rule."""
import pytest
import yaml
from pathlib import Path
from autotube.analytics import average_view_target
from autotube.scriptwriter import grounding_floor

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


# ── internal duration target ─────────────────────────────────────────────────
def test_duration_band_is_the_current_internal_experiment():
    """A configuration check only; duration is not asserted to be a platform threshold."""
    assert cfg()["video"]["target_seconds"] == [15, 20]
    assert cfg()["analytics"]["average_view_gate_pct"] == 70.0


# ── grounding scales with script length ──────────────────────────────────────
@pytest.mark.parametrize("n,expected", [(4, 2), (5, 3), (6, 3), (7, 3)])
def test_grounding_requirement_scales_with_segment_count(n, expected):
    assert grounding_floor(n, cfg()["content"]) == expected


def test_shortening_the_script_did_not_secretly_tighten_grounding():
    """A flat 3 was written for 5-6 segment scripts; at 4 segments it would be 3-of-4."""
    c = cfg()["content"]
    assert grounding_floor(6, c) == grounding_floor(5, c) == 3, "longer scripts must be unaffected"
    assert grounding_floor(4, c) < 3, "4-segment scripts must not inherit the 5-6 segment requirement"


def test_grounding_floor_never_returns_something_impossible():
    c = cfg()["content"]
    for n in range(1, 12):
        assert 0 < grounding_floor(n, c) <= max(1, n)


# ── internal average-view target ─────────────────────────────────────────────
def mk(*pcts):
    return [{"metrics": {"avg_view_pct": pct}} for pct in pcts]


def test_gate_reports_internal_average_view_target_not_viewer_completion():
    r = average_view_target(mk(42.0, 55.0, 69.0, 30.0), {"average_view_gate_pct": 70.0})
    assert r["passing"] == 0 and r["pass_rate"] == 0.0
    assert r["metric"] == "average_view_percentage"
    assert "0/4" in r["verdict"] and "internal 70%" in r["verdict"]
    assert "not the share of viewers" in r["verdict"]


def test_gate_recognises_all_videos_meeting_internal_average_view_target():
    r = average_view_target(mk(75.0, 82.0, 71.0, 90.0), {"average_view_gate_pct": 70.0})
    assert r["pass_rate"] == 1.0 and r["passing"] == 4
    assert "ALL 4/4" in r["verdict"] and "not a viewer-completion rate" in r["verdict"]


def test_gate_distinguishes_partial_from_total_failure():
    partial = average_view_target(mk(75.0, 20.0, 30.0, 10.0), {"average_view_gate_pct": 70.0})
    assert partial["passing"] == 1 and "1/4" in partial["verdict"]
    assert "ALL" not in partial["verdict"]


def test_gate_is_not_fooled_by_a_good_median_with_no_passes():
    r = average_view_target(mk(69.0, 68.0, 69.0), {"average_view_gate_pct": 70.0})
    assert r["passing"] == 0 and r["median_avg_view_pct"] == pytest.approx(69.0, abs=0.01)


def test_gate_handles_no_data_without_crashing_and_ignores_legacy_completion():
    assert average_view_target([], {})["n"] == 0
    assert average_view_target([{"metrics": {}}], {})["n"] == 0
    assert average_view_target([{"metrics": {"completion": 0.99}}], {})["n"] == 0


def test_gate_threshold_is_configurable_in_percentage_points():
    vids = mk(50.0, 60.0, 80.0)
    assert average_view_target(vids, {"average_view_gate_pct": 70.0})["passing"] == 1
    assert average_view_target(vids, {"average_view_gate_pct": 55.0})["passing"] == 2


# ── loop verdict (judged by the reviewer, not by a regex) ────────────────────
def test_loop_is_judged_semantically_not_lexically():
    """A deterministic version was tried and removed: it rejected clean loops.

    "so the ice keeps bleeding" loops perfectly into "a waterfall that runs blood red" and shares
    not one token with it. The note in gates.py records why; this test stops anyone re-adding it.
    """
    from autotube import gates
    src = Path(gates.__file__).read_text()
    assert "def check_loop" not in src, \
        "a lexical loop gate is back; it rejects semantically clean loops and burns generation retries"
    assert "semantic judgement" in src, "the reasoning note was removed"


def test_reviewer_is_asked_for_a_loop_verdict():
    from autotube import scriptwriter
    src = Path(scriptwriter.__file__).read_text()
    assert '"loops": true|false' in src
    assert "require_loop" in src


def test_loop_is_recorded_but_not_enforced_by_default():
    """A brand-new criterion must not start abandoning topics before its error rate is known."""
    assert cfg()["content"].get("require_loop") is False
