"""The beat-two gate (AUDIT follow-up: the 3-5 second retention cliff).

Measured on the channel's first 12 videos with retention curves: 112% of audience at the 10%
mark, 71.5% at the 20% mark. On a 26-second Short that is second 3 to second 5 - after the hook
has done its job. The line in that slot is segment 2, and the model writes it as exposition.
"""
import pytest
from autotube.gates import check_turn, run_all

ESCALATION = [
    "Which meant the entire city had to be moved, brick by brick.",
    "And it gets worse: the water underneath was still liquid.",
    "Nobody noticed for thirty years.",
    "Except the machine had never been switched on.",
    "So the crew kept digging, and found a second door.",
    "That mistake cost the company four hundred million dollars.",
    "Then the readings started coming back from inside the sealed room.",
    "He did it twice more before anyone checked the paperwork.",
    "Every single one of them gave the same impossible answer.",
    "The lake has never frozen, not once, in fifteen million years.",
]

EXPOSITION = [
    "Born in 1887, he studied engineering in Vienna.",
    "In 1923, the lake was first surveyed by a Russian team.",
    "The story of the vault begins much earlier.",
    "It was built in 1890 by a local merchant.",
    "According to historians, the practice was common.",
    "Located in southern Chile, the lake is unusually deep.",
    "The history of the technique goes back centuries.",
    "For centuries, sailors avoided the strait entirely.",
    "Throughout history, people have tried to explain it.",
    "It dates back to the fourteenth century.",
    "To understand why, you need some background.",
    "The concept of a sealed ecosystem was not new.",
]


@pytest.mark.parametrize("line", ESCALATION)
def test_real_escalation_is_not_rejected(line):
    """False positives are expensive: a rejected draft burns a retry and can abandon the topic."""
    assert check_turn(line) == [], f"good turn rejected: {line!r}"


@pytest.mark.parametrize("line", EXPOSITION)
def test_background_exposition_is_caught(line):
    issues = check_turn(line)
    assert issues, f"exposition slipped through: {line!r}"
    assert "escalate" in issues[0].lower()


def test_the_gate_explains_what_to_write_instead():
    """A rejection message is a retry prompt; 'invalid' would just waste the retry."""
    msg = check_turn("Born in 1887, he studied engineering.")[0]
    assert "ESCALATE" in msg and "definition" in msg


def test_which_is_actually_fragment_is_rejected():
    msg = check_turn("Which is actually the Manicouagan Reservoir's outer rim")
    assert msg and "ESCALATE" in msg[0]


def test_turn_gate_is_wired_into_run_all():
    def spoken(s):
        return s["text"]

    script = {"title": "A perfectly fine title about a lake",
              "segments": [{"text": "This lake has not frozen in fifteen million years."},
                           {"text": "Born in 1887, the surveyor never saw it himself."},
                           {"text": "The water sits four kilometres below solid ice."},
                           {"text": "and it is still liquid down there today"}]}
    issues = run_all(script, spoken, {"content": {}})
    assert any("segment 2" in i for i in issues), issues


def test_short_scripts_do_not_trip_the_gate():
    def spoken(s):
        return s["text"]
    # a 2-segment script has no middle; the turn gate must not fire on the closer
    script = {"title": "Short one", "segments": [{"text": "x"}, {"text": "Born in 1887."}]}
    assert not any("segment 2" in i for i in run_all(script, spoken, {"content": {}}))


def test_empty_and_degenerate_input():
    assert check_turn("") == []
    assert check_turn(None) == []
    assert check_turn('"In 1923, it happened.') != []      # leading punctuation must not evade the gate
