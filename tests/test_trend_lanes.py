"""Lane-based trend expansion and the live format pulse (AUDIT 3.2)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import trends  # noqa: E402
from autotube.common import load_config  # noqa: E402
from autotube.strategy import Strategy  # noqa: E402

CFG = load_config()


# ── lanes ─────────────────────────────────────────────────────────────────────
def test_every_lane_is_searched_even_when_the_quota_cap_bites():
    """Taking the first N queries would starve the last lanes; interleaving must not."""
    cfg = {"trends": {"lanes": {
        "a": {"queries": ["a1", "a2", "a3", "a4"]},
        "b": {"queries": ["b1", "b2", "b3"]},
        "c": {"queries": ["c1", "c2"]},
    }, "max_queries_per_run": 4}}
    lanes = {lane for lane, _ in trends.lane_queries(cfg)}
    assert lanes == {"a", "b", "c"}, f"a capped run skipped a lane: {lanes}"


def test_cap_is_respected_and_rotates_over_days():
    cfg = {"trends": {"lanes": {"a": {"queries": [f"q{i}" for i in range(10)]}},
                      "max_queries_per_run": 3}}
    picked = trends.lane_queries(cfg)
    assert len(picked) == 3
    # over a year of offsets every query must get searched at least once
    seen = set()
    for day in range(1, 366):
        off = day % 10
        seen |= {f"q{(off + i) % 10}" for i in range(3)}
    assert seen == {f"q{i}" for i in range(10)}


def test_no_cap_returns_everything_and_empty_config_is_safe():
    cfg = {"trends": {"lanes": {"a": {"queries": ["x", "y"]}}}}
    assert len(trends.lane_queries(cfg)) == 2
    assert trends.lane_queries({}) == []
    assert trends.lane_queries({"trends": {"lanes": {}}}) == []


def test_legacy_flat_query_list_still_works():
    cfg = {"trends": {"youtube_outlier_queries": ["facts", "did you know"]}}
    assert trends.lane_queries(cfg) == [("", "facts"), ("", "did you know")]


def test_shipped_config_covers_all_lanes_within_quota():
    pairs = trends.lane_queries(CFG)
    assert len(pairs) == 14                                  # 14 x 100 units = 1,400 of 10,000/day
    assert {lane for lane, _ in pairs} == set(CFG["trends"]["lanes"])


def test_categories_map_back_to_a_lane():
    assert trends.lane_of(CFG, "space") == "science_nature"
    assert trends.lane_of(CFG, "history") == "history_mystery"
    assert trends.lane_of(CFG, "technology") == "tech_ai"
    assert trends.lane_of(CFG, "nonsense") == ""


def test_every_channel_category_belongs_to_some_lane():
    orphans = [c for c in CFG["channel"]["categories"] if not trends.lane_of(CFG, c)]
    assert not orphans, f"categories in no lane will never be trend-fed: {orphans}"


# ── hook classification ───────────────────────────────────────────────────────
def test_hook_patterns():
    cases = {
        "Why does the ocean glow?": "question",
        "How it actually works": "question",
        "5 things you never knew": "number_first",
        "Top 10 deepest places": "number_first",
        "You are breathing wrong": "you_statement",
        "Turns out the Moon is rusting": "disbelief",
        "Nobody noticed this for 200 years": "disbelief",
        "The Roman army used vinegar": "bold_claim",
    }
    for title, want in cases.items():
        assert trends.hook_pattern(title) == want, f"{title!r} → {trends.hook_pattern(title)}, want {want}"


def test_hook_pattern_never_crashes_on_junk():
    for junk in ["", "   ", "???", "123", "🐙"]:
        assert trends.hook_pattern(junk) in CFG["content"]["hook_styles"]


def test_every_hook_pattern_is_a_real_configured_hook_style():
    """A pulse key the strategy has no arm for would silently do nothing."""
    produced = {name for name, _ in trends.HOOK_PATTERNS} | {"bold_claim"}
    assert produced <= set(CFG["content"]["hook_styles"]), produced - set(CFG["content"]["hook_styles"])


# ── format pulse ──────────────────────────────────────────────────────────────
def test_pulse_weights_by_velocity_not_by_video_count():
    """Thirty mild performers must not outrank two genuine breakouts."""
    rows = ([{"topic": f"The thing number {i}", "vph": 100} for i in range(30)]
            + [{"topic": f"{i} things you never knew", "vph": 9000} for i in range(2)])
    pulse = trends.format_pulse(rows)
    assert pulse["videos"]["bold_claim"] == 30 and pulse["videos"]["number_first"] == 2
    assert pulse["hook_style"]["number_first"] > pulse["hook_style"]["bold_claim"]
    assert abs(sum(pulse["hook_style"].values()) - 1.0) < 1e-6


def test_pulse_is_empty_safe():
    p = trends.format_pulse([])
    assert p["sample"] == 0 and p["hook_style"] == {}


def test_pulse_tolerates_missing_vph():
    p = trends.format_pulse([{"topic": "Why is this"}, {"topic": "3 facts"}])
    assert p["sample"] == 2


# ── pulse → strategy ──────────────────────────────────────────────────────────
def test_pulse_biases_a_frozen_dimension_without_starving_any_option():
    s = Strategy.__new__(Strategy)
    keys = list(CFG["content"]["hook_styles"])
    s.read = None
    import autotube.strategy as st

    orig = st.read_json
    st.read_json = lambda *a, **k: {"hook_style": {"number_first": 0.9, "question": 0.1}}
    try:
        w = s.pulse_weights("hook_style", keys)
    finally:
        st.read_json = orig
    assert w and len(w) == len(keys)
    assert all(x > 0 for x in w), "no hook style may be weighted to zero"
    assert w[keys.index("number_first")] == max(w)


def test_no_pulse_file_means_unweighted_choice():
    s = Strategy.__new__(Strategy)
    import autotube.strategy as st
    orig = st.read_json
    st.read_json = lambda *a, **k: {}
    try:
        assert s.pulse_weights("hook_style", ["a", "b"]) is None
        assert s.pulse_weights("hook_style", []) is None
    finally:
        st.read_json = orig
