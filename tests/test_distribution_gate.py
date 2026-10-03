"""The 15-20s band and the completion gate that decides whether a Short gets distributed."""
import pytest
import yaml
from pathlib import Path
from autotube.analytics import distribution_gate
from autotube.scriptwriter import grounding_floor

ROOT = Path(__file__).resolve().parents[1]


def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


# ── the band ─────────────────────────────────────────────────────────────────
def test_band_avoids_the_25_to_40_second_dead_zone():
    """15-20s (single concept) and 45-58s (story) perform; 25-40s underperforms both."""
    lo, hi = cfg()["video"]["target_seconds"]
    assert hi <= 25 or lo >= 45, f"band {lo}-{hi}s overlaps the dead zone"


def test_band_is_inside_the_loop_sweet_spot():
    """15-25s is where looping actually drives replays, which is how avg view % passes 100%."""
    lo, hi = cfg()["video"]["target_seconds"]
    assert 15 <= lo and hi <= 25


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


# ── the distribution gate ────────────────────────────────────────────────────
def mk(*completions):
    return [{"metrics": {"completion": c}} for c in completions]


def test_gate_reports_total_failure_unambiguously():
    """Our real data: 12 videos, none above 70%. This must not read as 'needs improvement'."""
    r = distribution_gate(mk(0.16, 0.12, 0.49, 0.20, 0.09), {"completion_gate": 0.70})
    assert r["passing"] == 0 and r["pass_rate"] == 0.0
    assert "NO video clears" in r["verdict"]


def test_gate_recognises_a_healthy_channel():
    r = distribution_gate(mk(0.75, 0.82, 0.71, 0.90), {"completion_gate": 0.70})
    assert r["pass_rate"] == 1.0 and "healthy" in r["verdict"]


def test_gate_distinguishes_partial_from_total_failure():
    partial = distribution_gate(mk(0.75, 0.2, 0.3, 0.1), {"completion_gate": 0.70})
    assert partial["passing"] == 1 and "only 1/4" in partial["verdict"]
    assert "NO video" not in partial["verdict"]


def test_gate_is_not_fooled_by_a_good_median_with_no_passes():
    """Reward can look fine on percentile rank while every video sits under the gate."""
    r = distribution_gate(mk(0.69, 0.68, 0.69), {"completion_gate": 0.70})
    assert r["passing"] == 0 and r["median_completion"] == pytest.approx(0.69, abs=0.01)


def test_gate_handles_no_data_without_crashing():
    assert distribution_gate([], {})["n"] == 0
    assert distribution_gate([{"metrics": {}}], {})["n"] == 0


def test_gate_threshold_is_configurable():
    vids = mk(0.5, 0.6, 0.8)
    assert distribution_gate(vids, {"completion_gate": 0.70})["passing"] == 1
    assert distribution_gate(vids, {"completion_gate": 0.55})["passing"] == 2
