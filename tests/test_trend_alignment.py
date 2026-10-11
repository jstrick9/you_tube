"""A generated script must tell the specific story that is trending, not just name the subject."""
from __future__ import annotations

import copy
import sys
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
        "advertiser_friendly": True, "policy_concerns": [], "trend_alignment": 9,
    }
    approved, review = writer.check(_script(), _source(), _topic())

    assert not approved
    assert review["fixes"] == ["reviewer omitted the required fixes list"]
    assert any("omitted the required fixes list" in issue for issue in review["issues"])


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
