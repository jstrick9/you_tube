"""Recurring numbered series: the subscriber-conversion mechanism.

Views and subscribers are separate YPP gates. A standalone fact video is optimised
for the first and structurally bad at the second, because nothing in it promises a
specific next thing. These tests pin the properties that make numbering trustworthy -
if the numbers drift, duplicate or lie, the series is worse than no series at all.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import series  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

TCFG = {"content": {"series": {
    "unsolved": {"tag": "CASE", "categories": ["history", "mysteries"]},
    "field": {"tag": "FIELD", "categories": ["space", "nature"]},
}}}


def setup_function():
    series.reset()


# ── assignment ──────────────────────────────────────────────────────────────
def test_category_maps_to_its_series():
    assert series.assign("mysteries", TCFG) == "unsolved"
    assert series.assign("space", TCFG) == "field"


def test_unmapped_category_publishes_unbranded_rather_than_failing():
    """Series membership must never be able to block a video."""
    assert series.assign("sports", TCFG) is None
    assert series.assign(None, TCFG) is None
    assert series.assign("space", {"content": {}}) is None
    assert series.assign("space", {}) is None


# ── numbering ───────────────────────────────────────────────────────────────
def test_first_episode_is_one():
    assert series.episode_number("unsolved", []) == 1


def test_numbering_continues_from_history():
    hist = [{"series": "unsolved", "episode": 1}, {"series": "unsolved", "episode": 2}]
    assert series.episode_number("unsolved", hist) == 3


def test_series_are_numbered_independently():
    hist = [{"series": "unsolved", "episode": 9}]
    assert series.episode_number("field", hist) == 1


def test_numbering_survives_a_pruned_history():
    """len()+1 would reissue a number already live on the channel; max+1 cannot."""
    hist = [{"series": "unsolved", "episode": 47}]        # 1-46 pruned away
    assert series.episode_number("unsolved", hist) == 48


def test_numbering_recovers_from_titles_when_the_field_is_missing():
    hist = [{"series": "unsolved", "title": "CASE #012 — the vanishing #shorts"}]
    assert series.episode_number("unsolved", hist) == 13


def test_three_videos_in_one_run_get_three_different_numbers():
    """History is only written at the end of a run, so without the in-flight registry
    all three videos read the same file and all three claim the same number."""
    hist = [{"series": "unsolved", "episode": 5}]
    got = [series.episode_number("unsolved", hist) for _ in range(3)]
    assert got == [6, 7, 8]


def test_reset_clears_the_in_flight_registry_between_runs():
    hist = [{"series": "unsolved", "episode": 5}]
    series.episode_number("unsolved", hist)
    series.reset()
    assert series.episode_number("unsolved", hist) == 6


def test_other_series_entries_do_not_bump_the_count():
    hist = [{"series": "field", "episode": 80}, {"series": None, "episode": 90},
            {"series": "unsolved", "episode": 3}]
    assert series.episode_number("unsolved", hist) == 4


# ── titles ──────────────────────────────────────────────────────────────────
def test_title_is_decorated_with_a_zero_padded_label():
    out = series.decorate_title("the lighthouse keepers vanished", "unsolved", 7, TCFG)
    assert out == "CASE #007 — the lighthouse keepers vanished"


def test_label_is_dropped_rather_than_truncating_the_hook():
    """The hook earns the view; the label only earns the follow. If one must go, it
    is the label - a clipped hook costs a view today."""
    long = "x" * 88
    assert series.decorate_title(long, "unsolved", 7, TCFG, limit=90) == long


def test_unknown_series_or_zero_episode_leaves_the_title_alone():
    assert series.decorate_title("abc", None, 1, TCFG) == "abc"
    assert series.decorate_title("abc", "nope", 1, TCFG) == "abc"
    assert series.decorate_title("abc", "unsolved", 0, TCFG) == "abc"


# ── shipped config ──────────────────────────────────────────────────────────
def test_shipped_series_cover_every_channel_category():
    orphans = [c for c in CFG["channel"]["categories"] if not series.assign(c, CFG)]
    assert not orphans, f"these categories would publish unbranded: {orphans}"


def test_shipped_series_do_not_overlap():
    seen = {}
    for key, spec in CFG["content"]["series"].items():
        for cat in spec["categories"]:
            assert cat not in seen, f"{cat} is claimed by both {seen.get(cat)} and {key}"
            seen[cat] = key


def test_shipped_tags_are_short_enough_to_be_worth_the_characters():
    for key, spec in CFG["content"]["series"].items():
        assert 0 < len(spec["tag"]) <= 10, f"{key} tag eats too much of the title"
