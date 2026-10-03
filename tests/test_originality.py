"""Anti-bulk-production defences.

Enforcement against faceless channels is channel-wide and retroactive, so these checks
have to run before publication - by the time a strike lands the back catalogue is
already the evidence. Two triggers are in scope: scripts that repeat each other, and
scripts with no narrator voice at all.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import originality as o  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def sc(*lines, asides=()):
    return {"segments": [{"text": t, "aside": (asides[i] if i < len(asides) else "")}
                         for i, t in enumerate(lines)]}


BLOOD = sc("Antarctica has a waterfall that runs blood red",
           "For a century nobody could explain the colour",
           "It is iron in ancient brine oxidising the instant it meets air",
           "So the ice keeps bleeding and nobody can stop it",
           asides=("", "", "Nature, please.", ""))
REWRITE = sc("Antarctica has a waterfall that runs blood red",
             "For a hundred years nobody could explain the colour",
             "It is iron in ancient brine oxidising the moment it meets the air",
             "So the ice keeps bleeding and no one can stop it",
             asides=("", "", "Nature, again.", ""))
LIGHTHOUSE = sc("A lighthouse crew of three vanished in 1900",
                "The door was bolted and the lamps were trimmed",
                "Two coats were gone one still hung by the door",
                "The sea took them and left the paperwork",
                asides=("", "", "Tidy of them.", ""))


def sk(script):
    return o.sketch(o.narration_of(script))


# ── similarity ──────────────────────────────────────────────────────────────
def test_a_script_is_identical_to_itself():
    assert o.similarity(sk(BLOOD), sk(BLOOD)) == 1.0


def test_a_rewrite_of_the_same_episode_scores_far_above_the_limit():
    sim = o.similarity(sk(BLOOD), sk(REWRITE))
    assert sim > CFG["content"]["max_script_similarity"], f"rewrite only scored {sim:.2f}"


def test_a_different_episode_in_the_same_series_scores_far_below_the_limit():
    """The gap between these two numbers is the whole value of the check - if a real
    episode scored near the limit the gate would reject good work and burn retries."""
    sim = o.similarity(sk(BLOOD), sk(LIGHTHOUSE))
    assert sim < CFG["content"]["max_script_similarity"] / 2, f"false-positive risk: {sim:.2f}"


def test_empty_sketches_are_not_similar_to_anything():
    assert o.similarity([], sk(BLOOD)) == 0.0
    assert o.similarity(sk(BLOOD), []) == 0.0
    assert o.similarity([], []) == 0.0


def test_sketch_is_bounded_so_history_cannot_grow_without_bound():
    big = sc(" ".join(f"word{i}" for i in range(4000)))
    assert len(o.sketch(o.narration_of(big))) == o.SKETCH_SIZE


def test_sketch_is_stable_across_runs():
    assert o.sketch("the lighthouse keepers vanished without trace") == \
           o.sketch("the lighthouse keepers vanished without trace")


def test_similarity_is_symmetric():
    a, b = sk(BLOOD), sk(REWRITE)
    assert o.similarity(a, b) == o.similarity(b, a)


def test_length_difference_alone_does_not_read_as_difference():
    """A long script that fully contains a short one is a duplicate, not a new episode.

    Jaccard necessarily discounts containment - 4 shared phrases out of a 12-phrase
    union is 0.33, not 1.0 - so the claim worth pinning is only that padding a reused
    script with new sentences does not get it under the limit."""
    short = sc("the lighthouse crew of three vanished in nineteen hundred")
    long = sc("the lighthouse crew of three vanished in nineteen hundred",
              "investigators found the door bolted and the lamps trimmed and the table set")
    assert o.similarity(sk(short), sk(long)) > CFG["content"]["max_script_similarity"]


def test_stop_words_alone_never_make_two_scripts_look_related():
    a = sc("the and of to in on at by for with from as it is")
    b = sc("the and of to in on at by for with from as it is")
    c = sc("penguins commandeered an abandoned research station in the antarctic")
    assert o.similarity(sk(a), sk(c)) == 0.0


# ── commentary ──────────────────────────────────────────────────────────────
def test_a_script_with_no_aside_has_no_voice_of_its_own():
    assert o.commentary_ratio(sc("one two three four five six")) == 0.0


def test_one_real_aside_satisfies_the_shipped_requirement():
    for script in (LIGHTHOUSE, BLOOD):
        assert not any("no narrator voice" in i for i in o.check(script, CFG, []))


def test_the_requirement_is_a_count_not_a_percentage():
    """A ratio floor rejected a script sitting exactly on it (two aside words in fifty
    is 0.04, one ULP low in binary) and made the bar depend on how wordy the rest of
    the script was. One short aside must pass regardless of episode length."""
    cfg = {"content": {"min_asides": 1}}
    terse = sc("a b c d", asides=("Rude.",))
    wordy = sc(" ".join(["word"] * 300), asides=("Rude.",))
    assert o.check(terse, cfg, []) == []
    assert o.check(wordy, cfg, []) == []


def test_a_blank_aside_does_not_count_as_a_voice():
    cfg = {"content": {"min_asides": 1}}
    assert any("no narrator voice" in i for i in o.check(sc("a b c", asides=("   ",)), cfg, []))


def test_commentary_ratio_is_a_share_not_a_count():
    tight = sc("a b c d", asides=("Rude.",))
    padded = sc(" ".join(["word"] * 200), asides=("Rude.",))
    assert o.commentary_ratio(tight) > o.commentary_ratio(padded)


def test_commentary_ratio_handles_an_empty_script():
    assert o.commentary_ratio({"segments": []}) == 0.0
    assert o.commentary_ratio({}) == 0.0


# ── the check itself ────────────────────────────────────────────────────────
def test_check_flags_a_near_duplicate_against_history():
    hist = [{"title": "CASE #001 — the blood falls", "sketch": sk(BLOOD)}]
    issues = o.check(REWRITE, CFG, hist)
    assert any("too close to a published episode" in i for i in issues)
    assert "CASE #001" in " ".join(issues), "the issue must name what it collided with"


def test_check_passes_a_genuinely_new_episode():
    hist = [{"title": "CASE #001", "sketch": sk(BLOOD)}]
    assert o.check(LIGHTHOUSE, CFG, hist) == []


def test_check_flags_a_script_with_no_commentary():
    plain = sc("Antarctica has a waterfall that runs blood red",
               "It is iron in ancient brine oxidising the instant it meets air")
    assert any("no narrator voice" in i for i in o.check(plain, CFG, []))


def test_commentary_ratio_is_still_measured_even_though_it_is_not_a_gate():
    assert o.commentary_ratio(LIGHTHOUSE) > 0


def test_check_is_safe_on_an_empty_or_absent_history():
    assert o.check(LIGHTHOUSE, CFG, []) == []
    assert o.check(LIGHTHOUSE, CFG, None) == []


def test_check_ignores_history_entries_predating_the_sketch_field():
    """Every existing entry in state/history.json has no sketch; they must not crash
    the gate, and they must not be read as zero-similarity evidence of safety either."""
    hist = [{"title": "old"}, {"title": "older", "sketch": []}]
    assert o.check(REWRITE, CFG, hist) == []


def test_both_defences_can_be_switched_off():
    off = {"content": {"max_script_similarity": 1.0, "min_asides": 0}}
    hist = [{"title": "x", "sketch": sk(BLOOD)}]
    assert o.check(REWRITE, off, hist) == []


def test_only_the_configured_window_of_recent_episodes_is_compared():
    hist = [{"title": "ancient", "sketch": sk(BLOOD)}] + \
           [{"title": f"e{i}", "sketch": sk(sc(f"unrelated episode number {i} about nothing"))}
            for i in range(30)]
    cfg = {"content": {"max_script_similarity": 0.25, "similarity_window": 5}}
    assert o.check(REWRITE, cfg, hist) == [], "an episode 30 uploads ago is outside the window"


def test_shipped_config_actually_enables_both_defences():
    c = CFG["content"]
    assert c["max_script_similarity"] < 1.0
    assert c["min_asides"] >= 1


# ── channel-level audit ─────────────────────────────────────────────────────
# The per-draft check is pairwise and cannot see aggregate drift: every episode can sit
# under the similarity limit while all of them are one template. Reviewers are
# documented as assessing the body of work, so something has to look at it that way.

def ep(text, fmt="creepy_true", cat="mysteries", commentary=0.05):
    return {"sketch": o.sketch(text), "commentary_ratio": commentary,
            "choice": {"format": fmt, "category": cat}}


# These have to be genuinely different prose, not one sentence with the number changed.
# The first draft of this fixture was the latter, and the audit correctly flagged it as
# repetitive - which is the behaviour under test, so the fixture was the bug.
VARIED_TEXTS = [
    "three lighthouse keepers vanished behind a bolted door in nineteen hundred",
    "an iron rich brine oxidises the moment antarctic air touches it",
    "voyager one carries a gold record nobody alive will ever collect",
    "roman concrete heals its own cracks when seawater floods them",
    "a tardigrade survived ten days of hard vacuum and came back fine",
    "octopuses taste what they touch because their arms are covered in receptors",
]
VARIED = [ep(t, fmt=f) for t, f in zip(VARIED_TEXTS, [
    "creepy_true", "guess_reveal", "scale_shock", "what_if", "plot_twist", "myth_buster"])]


def test_a_varied_back_catalogue_reads_as_a_show():
    a = o.channel_audit(VARIED, CFG)
    assert a["verdict"] == "looks like a show", a["flags"]


def test_one_format_dominating_is_flagged_as_a_template():
    same = [ep(t) for t in VARIED_TEXTS]
    a = o.channel_audit(same, CFG)
    assert any("one template" in f for f in a["flags"])
    assert a["most_common_format"]["share"] == 1.0


def test_rewriting_the_same_script_is_caught_in_aggregate():
    dupes = [ep("the lighthouse keepers vanished without trace in nineteen hundred", fmt=f)
             for f in ["creepy_true", "guess_reveal", "scale_shock", "what_if", "plot_twist"]]
    a = o.channel_audit(dupes, CFG)
    assert a["pairs_over_similarity_limit"] > 0
    assert a["mean_pairwise_similarity"] > CFG["content"]["max_script_similarity"]


def test_a_catalogue_without_commentary_is_flagged():
    mute = [ep(t, fmt=f, commentary=0) for t, f in zip(VARIED_TEXTS, [
        "creepy_true", "guess_reveal", "scale_shock", "what_if", "plot_twist"])]
    assert any("narrator commentary" in f for f in o.channel_audit(mute, CFG)["flags"])


def test_audit_says_so_rather_than_guessing_on_a_new_channel():
    a = o.channel_audit([], CFG)
    assert a["episodes_examined"] == 0 and "not enough" in a["verdict"]
    assert "mean_pairwise_similarity" not in a, "must not imply a measurement it did not make"


def test_audit_ignores_episodes_predating_the_sketch_field():
    """Every entry currently in state/history.json has no sketch. They must not be
    silently counted as evidence that the catalogue is varied."""
    assert o.channel_audit([{"title": "old"}] * 10, CFG)["episodes_examined"] == 0


def test_audit_only_looks_at_the_configured_window():
    cfg = {"content": {"max_script_similarity": 0.25, "audit_window": 3}}
    assert o.channel_audit(VARIED, cfg)["episodes_examined"] == 3


def test_a_single_format_is_not_flagged_before_there_is_enough_evidence():
    """Three episodes of one format is a new channel, not a content farm."""
    few = [ep(t) for t in VARIED_TEXTS[:3]]
    assert not any("one template" in f for f in o.channel_audit(few, CFG)["flags"])
