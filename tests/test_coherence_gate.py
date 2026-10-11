"""Behavioral regression: reviewer coherence must affect script approval."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube.common import load_config  # noqa: E402
from autotube.scriptwriter import ScriptWriter  # noqa: E402


SOURCE = {
    "title": "Hunger stone",
    "url": "https://en.wikipedia.org/wiki/Hunger_stone",
    "text": (
        "A hunger stone is a hydrological landmark in Central Europe. Hunger stones were embedded into a river "
        "to commemorate droughts and warn future generations. One stone in the Elbe reads 'If you see me, then "
        "weep'. The oldest legible marks date from 1616. People carved the stones during severe droughts when "
        "the water level was low."
    ),
}
SCRIPT = {
    "title": "The river stones that say weep",
    "segments": [
        {"text": "The river itself warns you when the drought returns",
         "evidence": "Hunger stones were embedded into a river to commemorate droughts"},
        {"text": "That inscription tells river travellers to weep when the stone appears",
         "aside": "Cheerful bunch.",
         "evidence": "One stone in the Elbe reads 'If you see me, then weep'"},
        {"text": "Severe droughts expose these warnings when the riverbed falls into view",
         "evidence": "People carved the stones during severe droughts when the water level was low"},
        {"text": "The oldest legible marks on the stones date from 1616",
         "evidence": "The oldest legible marks date from 1616", "reveal": True},
        {"text": "so seeing one means the drought has returned",
         "evidence": "People carved the stones during severe droughts when the water level was low"},
    ],
}


class ReviewLLM:
    lite = False
    last_used = "fake:review"

    def __init__(self, coherence=8):
        self.coherence = coherence

    def json(self, system, user, **kwargs):
        result = {
            "score": 9,
            "hook_strength": 9,
            "entertainment": 8,
            "coherence": self.coherence,
            "factual_errors": [],
            "misleading_title": False,
            "advertiser_friendly": True,
            "policy_concerns": [],
            "loops": True,
            "value_add": "A clear explanation of the fact's cause and consequence.",
            "fixes": [],
        }
        if kwargs.get("validate"):
            kwargs["validate"](result)
        return result


def _writer(coherence):
    cfg = copy.deepcopy(load_config())
    cfg["content"]["min_coherence"] = 6
    writer = ScriptWriter(cfg, ReviewLLM(coherence), object())
    writer._history = []
    return writer


def test_low_coherence_fails_even_when_every_other_review_gate_passes():
    writer = _writer(5)
    ok, review = writer.check(copy.deepcopy(SCRIPT), SOURCE)
    assert not ok
    assert review["coherence"] == 5
    assert any("list of facts, not a story" in issue for issue in review["issues"])
    assert writer._history == [], "a rejected script must not be registered as approved"


def test_coherence_at_the_configured_threshold_can_pass():
    writer = _writer(6)
    ok, review = writer.check(copy.deepcopy(SCRIPT), SOURCE)
    assert ok, review
    assert review["coherence"] == 6
    assert len(writer._history) == 1


def test_missing_coherence_score_fails_closed():
    writer = _writer(0)
    writer.llm.coherence = None

    def without_score(system, user, **kwargs):
        result = {"score": 9, "hook_strength": 9, "entertainment": 8,
                  "factual_errors": [], "misleading_title": False,
                  "advertiser_friendly": True, "policy_concerns": [], "loops": True,
                  "value_add": "A clear explanation of the fact's cause and consequence.", "fixes": []}
        return result  # simulate an adapter that ignores its validation callback

    writer.llm.json = without_score
    ok, review = writer.check(copy.deepcopy(SCRIPT), SOURCE)
    assert not ok
    assert review["unavailable"]
    assert any("coherence must be a finite number" in issue for issue in review["review_schema_issues"])
    assert any("incomplete or malformed" in issue for issue in review["issues"])
