"""A generated script must tell the specific story that is trending, not just name the subject."""
from __future__ import annotations

import copy
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, scriptwriter  # noqa: E402


class Reviewer:
    last_used = "review-model"

    def __init__(self, alignment: float, fixes=None):
        self.alignment = alignment
        self.fixes = fixes or []
        self.prompt = ""
        self.calls = 0

    def json(self, system, user, temperature=None, validate=None, **kwargs):
        self.calls += 1
        self.prompt = user
        result = {
            "score": 9,
            "hook_strength": 9,
            "entertainment": 8,
            "coherence": 8,
            "loops": True,
            "factual_errors": [],
            "misleading_title": False,
            "advertiser_friendly": True,
            "policy_concerns": [],
            "value_add": "A clear explanation of the specific display trick.",
            "fixes": self.fixes,
            "trend_alignment": self.alignment,
            "trend_alignment_reason": "It does not explain the specific trick that made the topic trend.",
        }
        if validate:
            validate(result)
        return result


def _writer(monkeypatch, alignment=5, fixes=None):
    cfg = copy.deepcopy(common.load_config())
    cfg["content"]["require_grounding"] = False
    monkeypatch.setattr(scriptwriter.gates, "run_all", lambda *a, **k: [])
    monkeypatch.setattr(scriptwriter.originality, "check", lambda *a, **k: [])
    monkeypatch.setattr(scriptwriter, "unsupported_numbers", lambda *a, **k: [])
    monkeypatch.setattr(scriptwriter.safety, "check_script", lambda *a, **k: [])
    writer = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    writer.cfg = cfg
    writer.llm = Reviewer(alignment, fixes)
    writer.src_chars = 4000
    writer._history = []
    return writer


def _script():
    return {
        "title": "The DS faked 3D",
        "_writer_model": "writer-model",
        "segments": [
            {"text": "The Nintendo DS looked three-dimensional without 3D hardware."},
            {"text": "Its weak processor could not render the effect in real time."},
            {"text": "Developers faked depth with two flat images."},
            {"text": "That trick made the screen seem to move."},
        ],
    }


def _source():
    return {"title": "Nintendo DS", "text": "The Nintendo DS was a handheld console. " * 80}


def _topic(angle="developers faked 3D because the DS could not render depth"):
    return {
        "topic": "Nintendo DS could not render 3D, so developers faked depth",
        "angle": angle,
        "why_trending": "A recent viral Short demonstrates the fake 3D display trick.",
        "sources": ["youtube_outliers"],
        "context": ["viral Short: 2,000,000 views in 20 hours"],
        "signals": [{
            "source": "youtube_outliers",
            "context": ["viral Short: 2,000,000 views in 20 hours"],
            "evidence": {"views": 2_000_000, "hours": 20, "vph": 100_000},
        }],
    }


def test_independent_review_enforces_specific_trend_angle(monkeypatch):
    writer = _writer(monkeypatch, alignment=5)
    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert review["trend_alignment"] == 5
    assert any("specific current-trend angle" in issue for issue in review["issues"])
    assert "Selected specific angle" in writer.llm.prompt
    assert "2,000,000 views in 20 hours" in writer.llm.prompt


def test_strongly_aligned_script_can_pass_the_trend_gate(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    approved, review = writer.check(_script(), _source(), _topic())

    assert approved
    assert review["trend_alignment"] == 9


def test_configured_review_thresholds_drive_prompt_and_all_score_gates(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    writer.cfg["content"].update({
        "min_hook_score": 8,
        "min_entertainment": 9,
        "min_coherence": 9,
        "min_trend_alignment": 10,
    })
    writer.cfg["compliance"]["quality_gate_min_score"] = 10

    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert "below 10 = do not publish" in writer.llm.prompt
    assert "If entertainment < 9" in writer.llm.prompt
    assert "If hook_strength < 8" in writer.llm.prompt
    assert "Need at least 9/10." in writer.llm.prompt
    assert "Need at least 10/10." in writer.llm.prompt
    assert any("entertaining enough" in issue and "need 9" in issue for issue in review["issues"])
    assert any("not a story" in issue and "need 9" in issue for issue in review["issues"])
    assert any("current-trend angle" in issue and "need 10" in issue for issue in review["issues"])
    assert any("overall review score too low" in issue and "need 10" in issue for issue in review["issues"])
    assert writer._history == []


@pytest.mark.parametrize(("section", "key", "bad_value"), [
    ("content", "min_hook_score", True),
    ("content", "min_entertainment", float("nan")),
    ("content", "min_coherence", 11),
    ("content", "min_trend_alignment", "8"),
    ("compliance", "quality_gate_min_score", float("inf")),
])
def test_invalid_review_thresholds_fail_closed_before_provider_call(monkeypatch, section, key, bad_value):
    writer = _writer(monkeypatch, alignment=9)
    writer.cfg[section][key] = bad_value

    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert writer.llm.calls == 0
    assert any("review score thresholds must be finite" in issue for issue in review["issues"])
    assert writer._history == []


def test_unresolved_reviewer_fix_blocks_approval_even_with_high_scores(monkeypatch):
    writer = _writer(monkeypatch, alignment=9, fixes=["Rewrite the final line so it loops back to the hook."])
    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert any("independent reviewer still requests a rewrite" in issue for issue in review["issues"])
    assert review["fixes"] == ["Rewrite the final line so it loops back to the hook."]


def test_non_actionable_no_changes_text_does_not_block(monkeypatch):
    writer = _writer(monkeypatch, alignment=9, fixes=["No changes needed."])
    approved, review = writer.check(_script(), _source(), _topic())

    assert approved
    assert review["fixes"] == []


def test_missing_reviewer_fixes_array_fails_closed(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    writer.llm.json = lambda *a, **k: {
        "score": 9, "hook_strength": 9, "entertainment": 8, "coherence": 8,
        "loops": True, "factual_errors": [], "misleading_title": False,
        "advertiser_friendly": True, "policy_concerns": [],
        "value_add": "A clear explanation of the specific display trick.", "trend_alignment": 9,
        "trend_alignment_reason": "The selected specific angle is delivered.",
    }
    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert review["fixes"] == ["reviewer omitted the required fixes list"]
    assert any("fixes must be an array of strings" in issue for issue in review["review_schema_issues"])


@pytest.mark.parametrize("defect", [
    "missing_hook_strength", "missing_entertainment", "missing_advertiser_verdict",
    "missing_value_add", "empty_value_add", "truthy_boolean", "boolean_score", "nan_score", "out_of_range_score",
])
def test_malformed_reviewer_decisions_fail_closed(monkeypatch, defect):
    writer = _writer(monkeypatch, alignment=9)
    valid = {
        "score": 9, "hook_strength": 9, "entertainment": 8, "coherence": 8, "loops": True,
        "factual_errors": [], "misleading_title": False, "advertiser_friendly": True,
        "policy_concerns": [], "value_add": "A clear explanation of the specific display trick.",
        "fixes": [], "trend_alignment": 9,
        "trend_alignment_reason": "The specific selected angle is delivered.",
    }
    if defect == "missing_hook_strength":
        valid.pop("hook_strength")
    elif defect == "missing_entertainment":
        valid.pop("entertainment")
    elif defect == "missing_advertiser_verdict":
        valid.pop("advertiser_friendly")
    elif defect == "missing_value_add":
        valid.pop("value_add")
    elif defect == "empty_value_add":
        valid["value_add"] = "   "
    elif defect == "truthy_boolean":
        valid["advertiser_friendly"] = "true"
    elif defect == "boolean_score":
        valid["score"] = True
    elif defect == "nan_score":
        valid["score"] = float("nan")
    elif defect == "out_of_range_score":
        valid["score"] = 11
    writer.llm.json = lambda *a, **k: valid  # an adapter that ignores the validator is still checked here

    approved, review = writer.check(_script(), _source(), _topic())
    assert not approved
    assert review["unavailable"]
    assert review["review_schema_issues"]
    assert any("incomplete or malformed" in issue for issue in review["issues"])
    assert writer._history == []


def test_any_reviewer_issue_blocks_even_when_all_scores_are_high(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    writer.llm.json = lambda *a, **k: {
        "score": 10, "hook_strength": 10, "entertainment": 10, "coherence": 10, "loops": True,
        "factual_errors": [], "misleading_title": False, "advertiser_friendly": True,
        "policy_concerns": [], "value_add": "A clear explanation of the specific display trick.",
        "fixes": [], "issues": ["The final line still needs a rewrite."],
        "trend_alignment": 10, "trend_alignment_reason": "The selected specific angle is explicit.",
    }
    approved, review = writer.check(_script(), _source(), _topic())
    assert not approved
    assert "The final line still needs a rewrite." in review["issues"]
    assert writer._history == []


def test_missing_reviewer_identity_cannot_satisfy_required_independent_review(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    writer.cfg["compliance"]["require_independent_review"] = True
    writer.llm.last_used = ""

    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert review["unavailable"]
    assert review["independent"] is False
    assert any("independent reviewer identity" in issue for issue in review["issues"])
    assert writer._history == []


def test_malformed_independent_review_toggle_fails_closed_before_provider_call(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    writer.cfg["compliance"]["require_independent_review"] = 0

    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert writer.llm.calls == 0
    assert any("independent-review requirement must be a Boolean" in issue for issue in review["issues"])


def test_missing_writer_identity_cannot_satisfy_required_independent_review(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    writer.cfg["compliance"]["require_independent_review"] = True
    script = _script()
    script.pop("_writer_model")

    approved, review = writer.check(script, _source(), _topic())

    assert not approved
    assert review["unavailable"]
    assert review["independent"] is False
    assert any("independent reviewer identity" in issue for issue in review["issues"])


def test_malformed_grounding_toggle_cannot_disable_the_evidence_gate(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    writer.cfg["content"]["require_grounding"] = 0

    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert writer.llm.calls == 0
    assert any("require_grounding configuration must be a Boolean" in issue for issue in review["issues"])


def test_unrecognized_reviewer_fields_cannot_override_orchestration_metadata(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    script = _script()
    script["_writer_model"] = writer.llm.last_used
    writer.llm.json = lambda *a, **k: {
        "score": 9, "hook_strength": 9, "entertainment": 8, "coherence": 8, "loops": True,
        "factual_errors": [], "misleading_title": False, "advertiser_friendly": True,
        "policy_concerns": [], "value_add": "A clear explanation of the specific display trick.",
        "fixes": [], "trend_alignment": 9,
        "trend_alignment_reason": "The selected specific angle is delivered.",
        "reviewer": "forged-reviewer", "independent": True, "unavailable": True,
        "programmatic_ok": False,
    }

    approved, review = writer.check(script, _source(), _topic())

    assert approved
    assert review["reviewer"] == "review-model"
    assert review["independent"] is False
    assert "unavailable" not in review
    assert review["programmatic_ok"] is True


def test_unavailable_reviewer_never_passes_even_if_legacy_toggle_is_disabled(monkeypatch):
    from autotube.llm import LLMError

    writer = _writer(monkeypatch, alignment=9)
    writer.cfg["compliance"]["require_llm_review"] = False

    class Unavailable:
        def json(self, *a, **k):
            raise LLMError("all providers down")

    writer.llm = Unavailable()
    approved, review = writer.check(_script(), _source(), _topic())
    assert not approved and review["unavailable"]
    assert any("fact-check unavailable" in issue for issue in review["issues"])


def test_optional_loop_is_not_an_implicit_publish_gate(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    result = {
        "score": 9, "hook_strength": 9, "entertainment": 8, "coherence": 8,
        "loops": False, "factual_errors": [], "misleading_title": False,
        "advertiser_friendly": True, "policy_concerns": [],
        "value_add": "The source explains how developers simulated depth.", "fixes": [],
        "trend_alignment": 9, "trend_alignment_reason": "The selected specific angle is delivered.",
    }

    def no_loop_review(system, user, **kwargs):
        writer.llm.prompt = user
        return result

    writer.llm.json = no_loop_review
    approved, review = writer.check(_script(), _source(), _topic())
    assert approved
    assert review["loops"] is False
    assert "A loop is optional" in writer.llm.prompt


def test_required_loop_issue_itself_vetoes_an_otherwise_clean_review(monkeypatch):
    writer = _writer(monkeypatch, alignment=9)
    writer.cfg["content"]["require_loop"] = True
    monkeypatch.setattr(scriptwriter, "reviewer_schema_issues", lambda *a, **k: [])
    writer.llm.json = lambda *a, **k: {
        "score": 10, "hook_strength": 10, "entertainment": 10, "coherence": 10,
        "loops": False, "factual_errors": [], "misleading_title": False,
        "advertiser_friendly": True, "policy_concerns": [],
        "value_add": "The source explains the strange choice behind the story.", "fixes": [],
        "trend_alignment": 10, "trend_alignment_reason": "The specific trend angle is delivered.",
    }
    approved, review = writer.check(_script(), _source(), _topic())
    assert not approved
    assert any("last line does not flow back" in issue for issue in review["issues"])


def test_open_reviewer_fixes_block_even_if_legacy_toggle_is_disabled(monkeypatch):
    writer = _writer(monkeypatch, alignment=9, fixes=["Rewrite the payoff to complete the hook."])
    writer.cfg["content"]["reject_open_review_fixes"] = False
    approved, review = writer.check(_script(), _source(), _topic())
    assert not approved
    assert any("independent reviewer still requests a rewrite" in issue for issue in review["issues"])


def test_missing_trend_angle_fails_before_spending_review_call(monkeypatch):
    writer = _writer(monkeypatch, alignment=10)
    approved, review = writer.check(_script(), _source(), _topic(angle=""))

    assert not approved
    assert writer.llm.calls == 0
    assert any("no specific current-trend angle" in issue for issue in review["issues"])


def test_production_fails_closed_before_grounding_when_angle_is_missing(monkeypatch):
    writer = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    monkeypatch.setattr(scriptwriter, "ground", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("grounding must not run without a specific angle")))

    assert writer.produce({"topic": "A currently trending topic", "angle": ""}, {}) is None
