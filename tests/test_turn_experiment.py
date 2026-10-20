"""Randomized early-turn treatment, non-blocking telemetry, and timing provenance."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, pipeline, scriptwriter  # noqa: E402
from autotube.strategy import DIMENSIONS, Strategy, TURN_EXPERIMENT_ID, TURN_VARIANTS  # noqa: E402


def test_strategy_assigns_balanced_randomized_turn_treatments_per_run(monkeypatch):
    cfg = common.load_config()
    monkeypatch.setattr("autotube.strategy.read_json", lambda *a, **k: {"arms": {}, "updates": 0})
    plans = Strategy(cfg).plan(4)

    assert {plan["turn_variant"] for plan in plans} == set(TURN_VARIANTS)
    assert [plan["turn_variant"] for plan in plans].count(TURN_VARIANTS[0]) == 2
    assert [plan["turn_variant"] for plan in plans].count(TURN_VARIANTS[1]) == 2
    assert {plan["turn_experiment_id"] for plan in plans} == {TURN_EXPERIMENT_ID}
    assert "turn_variant" not in DIMENSIONS, "the randomized experiment must not be learned by the bandit"


def test_malformed_or_disabled_experiment_config_falls_back_without_blocking(monkeypatch):
    cfg = common.load_config()
    cfg = copy.deepcopy(cfg)
    cfg["analytics"]["turn_experiment"].update({"enabled": True, "variants": ["unknown"]})
    monkeypatch.setattr("autotube.strategy.read_json", lambda *a, **k: {"arms": {}, "updates": 0})

    plan = Strategy(cfg).plan(1)[0]
    assert plan["turn_variant"] == "standard_escalation"
    assert plan["turn_experiment_id"] is None


def test_writer_prompt_renders_assigned_treatment_without_changing_facts(monkeypatch):
    cfg = common.load_config()

    class CaptureThenStop:
        lite = False
        prompt = ""

        def json(self, system, user, **kwargs):
            self.prompt = user
            raise RuntimeError("prompt captured")

    writer = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    writer.cfg = cfg
    writer.llm = CaptureThenStop()
    writer.src_chars = 4000
    topic = {
        "topic": "A rising space mystery",
        "script_subject": "space observation",
        "angle": "the observation has a source-backed contradiction",
        "category": "science_nature",
        "why_trending": "A current science Short is accelerating.",
        "sources": ["youtube_outliers"],
    }
    plan = {
        "format": "facts3", "hook_style": "bold_claim", "voice": "en-US-AndrewNeural",
        "comment_device": "none", "turn_experiment_id": TURN_EXPERIMENT_ID,
        "turn_variant": "consequence_first",
    }
    source = {"title": "Space Beacon", "text": "Space Beacon revealed the observation. " * 30}

    with pytest.raises(RuntimeError, match="prompt captured"):
        writer.write(topic, plan, source)

    assert "CURRENT TREND SUBJECT (deliver its specific live-trend angle; do not copy the raw headline): space observation" in writer.llm.prompt
    assert "RESEARCH SOURCE SUBJECT (facts must come from this article; the trend angle still determines the story): Space Beacon" in writer.llm.prompt
    assert "RANDOMIZED EARLY-TURN TREATMENT (consequence_first)" in writer.llm.prompt
    assert "first clause" in writer.llm.prompt and "source-backed consequence" in writer.llm.prompt
    assert "Keep the script original; do not reuse another upload's topic" in writer.llm.prompt
    assert "6-9 words" in writer.llm.prompt
    assert "spoken question is" in writer.llm.prompt and "optional" in writer.llm.prompt
    assert "segment 2 (THE TURN)" in writer.llm.prompt
    assert "open a question that only the end" not in writer.llm.prompt


def test_passing_score_guardrails_require_an_independent_programmatic_pass():
    cfg = common.load_config()
    review = {
        "programmatic_ok": True, "independent": True, "score": 8, "hook_strength": 8,
        "entertainment": 7, "coherence": 5, "naturalness": 9, "trend_alignment": 9,
    }
    scores = scriptwriter._passing_review_scores(review, cfg)
    assert set(scores) == {"score", "hook_strength", "entertainment", "naturalness", "trend_alignment"}
    review["independent"] = False
    assert scriptwriter._passing_review_scores(review, cfg) == {}
    review["independent"] = True
    review["programmatic_ok"] = False
    assert scriptwriter._passing_review_scores(review, cfg) == {}


def test_review_feedback_gets_targeted_story_and_ordering_corrections():
    cfg = common.load_config()

    class CaptureThenStop:
        lite = False
        prompt = ""

        def json(self, system, user, **kwargs):
            self.prompt = user
            raise RuntimeError("prompt captured")

    writer = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    writer.cfg = cfg
    writer.llm = CaptureThenStop()
    writer.src_chars = 4000
    topic = {
        "topic": "The Giant Stone Sphere",
        "script_subject": "Stone Sphere",
        "angle": "the spheres were carved before their discovery",
        "category": "science_nature",
        "_draft_feedback": (
            "Rewrite Segment 2 to follow the 'consequence_first' treatment; this is a list of facts, not a story; "
            "not entertaining enough; final payoff is 11 words, it must be 6-9 factual words; "
            "independent reviewer still requests a rewrite: Enhance narrative tension in the middle segments."),
        "_rewrite_attempt": 3,
        "_prior_draft": {"segments": [
            {"text": "The stone spheres hide their makers.", "aside": "The rock has secrets."},
            {"text": "Some were carved before their discovery."},
        ]},
        "_prior_review_scores": {
            "hook_strength": {"score": 8, "minimum": 7},
            "coherence": {"score": 7, "minimum": 6},
        },
    }
    plan = {
        "format": "sounds_fake", "hook_style": "bold_claim", "voice": "en-US-AndrewNeural",
        "comment_device": "none", "turn_experiment_id": TURN_EXPERIMENT_ID,
        "turn_variant": "consequence_first",
    }
    source = {"title": "Stone spheres of Costa Rica", "text": "The source describes stone spheres. " * 30}

    with pytest.raises(RuntimeError, match="prompt captured"):
        writer.write(topic, plan, source)

    assert "ORDERING REPAIR (consequence_first)" in writer.llm.prompt
    assert "CAUSAL-CHAIN REPAIR" in writer.llm.prompt
    assert "Never invent a causal link" in writer.llm.prompt
    assert "ENTERTAINMENT REPAIR" in writer.llm.prompt
    assert "MIDDLE-BEAT TENSION REPAIR" in writer.llm.prompt
    assert "REVIEWER-FIX REPAIR" in writer.llm.prompt
    assert "WORD-COUNT REPAIR" in writer.llm.prompt
    assert "PREVIOUS INDEPENDENT-REVIEW PASSES (not factual evidence)" in writer.llm.prompt
    assert "hook 8/10 (passed minimum 7)" in writer.llm.prompt
    assert "LATER-REWRITE ESCALATION" in writer.llm.prompt
    assert "PREVIOUS AUTO-TUBE ATTEMPT (our own draft, NOT factual evidence)" in writer.llm.prompt
    assert "The stone spheres hide their makers." in writer.llm.prompt
    assert "do not copy a reviewer example unless SOURCE TEXT supports it" in writer.llm.prompt


def test_unresolved_review_feedback_survives_programmatic_rewrite_failures(monkeypatch):
    cfg = common.load_config()
    writer = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    writer.cfg = cfg
    writer.llm = object()
    writer.lang = "en"
    calls = []

    def fake_write(topic, _plan, _source):
        calls.append(copy.deepcopy(topic))
        return {"title": f"draft {len(calls)}", "segments": [{"text": f"attempt {len(calls)}"}]}

    reviews = [
        (False, {"programmatic_ok": True, "independent": True, "score": 5, "hook_strength": 6,
                 "entertainment": 5, "coherence": 7, "naturalness": 9, "trend_alignment": 9,
                 "issues": ["not entertaining enough (5/10, need 6)",
                            "overall review score too low (5/10, need 7)"]}),
        (False, {"programmatic_ok": False,
                 "issues": ["final payoff is 11 words; it must be 6-9 factual words"]}),
        (False, {"programmatic_ok": False,
                 "issues": ["the hook is 10 words, must be 6-9"]}),
    ]

    def fake_check(_script, _source, _topic):
        return reviews.pop(0)

    writer.write = fake_write
    writer.check = fake_check
    monkeypatch.setattr(scriptwriter, "ground", lambda *_args, **_kwargs: {
        "title": "Stone spheres of Costa Rica", "text": "Stone spheres were carved from rock. " * 20,
    })
    monkeypatch.setattr(scriptwriter, "source_supports_angle", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(scriptwriter, "subject_is_depictable", lambda *_args, **_kwargs: True)

    result = writer.produce({
        "topic": "The Giant Stone Sphere", "script_subject": "Stone Sphere",
        "angle": "the spheres were carved before their discovery", "category": "science",
    }, {}, max_attempts=3)

    assert result is None and len(calls) == 3
    assert calls[1]["_rewrite_attempt"] == 2
    assert calls[1]["_prior_draft"]["title"] == "draft 1"
    assert "not entertaining enough" in calls[1]["_draft_feedback"]
    assert calls[1]["_prior_review_scores"] == {
        "coherence": {"score": 7.0, "minimum": 6.0},
        "naturalness": {"score": 9.0, "minimum": 8.0},
        "trend_alignment": {"score": 9.0, "minimum": 8.0},
    }
    assert calls[2]["_rewrite_attempt"] == 3
    assert calls[2]["_prior_draft"]["title"] == "draft 2"
    assert calls[2]["_prior_review_scores"] == calls[1]["_prior_review_scores"]
    assert "not entertaining enough" in calls[2]["_draft_feedback"]
    assert "final payoff" in calls[2]["_draft_feedback"]


def test_review_retry_preserves_passed_scores_while_targeting_middle_tension(monkeypatch):
    cfg = common.load_config()
    writer = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    writer.cfg = cfg
    writer.llm = object()
    writer.lang = "en"
    calls = []

    def fake_write(topic, _plan, _source):
        calls.append(copy.deepcopy(topic))
        return {"title": f"draft {len(calls)}", "segments": [{"text": f"attempt {len(calls)}"}]}

    reviews = [
        (False, {"programmatic_ok": True, "independent": True, "score": 4, "hook_strength": 6, "entertainment": 4,
                 "coherence": 7, "naturalness": 9, "trend_alignment": 9,
                 "issues": ["hook too weak (6/10, need 7)", "not entertaining enough (4/10, need 6)"]}),
        (False, {"programmatic_ok": True, "independent": True, "score": 8, "hook_strength": 8, "entertainment": 7,
                 "coherence": 7, "naturalness": 9, "trend_alignment": 9,
                 "issues": ["independent reviewer still requests a rewrite: "
                            "Enhance narrative tension in the middle segments to boost entertainment value above 6."]}),
        (True, {"programmatic_ok": True, "independent": True, "score": 8, "hook_strength": 8, "entertainment": 8,
                "coherence": 7, "naturalness": 9, "trend_alignment": 9, "issues": []}),
    ]

    def fake_check(_script, _source, _topic):
        return reviews.pop(0)

    writer.write = fake_write
    writer.check = fake_check
    monkeypatch.setattr(scriptwriter, "ground", lambda *_args, **_kwargs: {
        "title": "Kola Superdeep Borehole", "text": "The borehole reached deep rock. " * 20,
    })
    monkeypatch.setattr(scriptwriter, "source_supports_angle", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(scriptwriter, "subject_is_depictable", lambda *_args, **_kwargs: True)

    result = writer.produce({
        "topic": "They Drilled 12 KM Into Earth And It Got Extremely Hot",
        "script_subject": "Kola Superdeep Borehole", "angle": "temperature rose with depth",
        "category": "science",
    }, {}, max_attempts=3)

    assert result is not None and len(calls) == 3
    assert calls[1]["_prior_review_scores"]["coherence"] == {"score": 7.0, "minimum": 6.0}
    assert "hook_strength" not in calls[1]["_prior_review_scores"]
    assert calls[2]["_prior_review_scores"] == {
        "score": {"score": 8.0, "minimum": 7.0},
        "hook_strength": {"score": 8.0, "minimum": 7.0},
        "entertainment": {"score": 7.0, "minimum": 6.0},
        "coherence": {"score": 7.0, "minimum": 6.0},
        "naturalness": {"score": 9.0, "minimum": 8.0},
        "trend_alignment": {"score": 9.0, "minimum": 8.0},
    }
    assert "narrative tension" in calls[2]["_draft_feedback"]
    assert "not entertaining enough" not in calls[2]["_draft_feedback"]


def test_writer_word_count_errors_are_sent_to_the_model_validation_retry():
    cfg = common.load_config()

    class CaptureThenStop:
        lite = False
        validator = None

        def json(self, _system, _user, validate=None, **_kwargs):
            self.validator = validate
            raise RuntimeError("prompt captured")

    writer = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    writer.cfg = cfg
    writer.llm = CaptureThenStop()
    writer.src_chars = 4000
    topic = {"topic": "Stone Sphere", "script_subject": "Stone Sphere", "category": "science"}
    plan = {"format": "sounds_fake", "hook_style": "bold_claim", "voice": "en-US-AndrewNeural",
            "comment_device": "none"}
    source = {"title": "Stone spheres", "text": "Stone spheres were carved from stone. " * 30}

    with pytest.raises(RuntimeError, match="prompt captured"):
        writer.write(topic, plan, source)

    bad_counts = {
        "title": "Stone spheres hide a strange history",
        "segments": [
            {"text": "Stone spheres hide a strange mystery.", "aside": "", "evidence": "source quote",
             "visual": {"shows": "a stone sphere", "queries": ["stone sphere photo", "stone sphere detail"]}},
            {"text": "Some were carved from hard stone.", "aside": "", "evidence": "source quote",
             "visual": {"shows": "a stone sphere", "queries": ["stone sphere photo", "stone sphere detail"]}},
            {"text": "A stone sphere can weigh several tons and remain mysterious.", "aside": "", "evidence": "source quote",
             "visual": {"shows": "a stone sphere", "queries": ["stone sphere photo", "stone sphere detail"]}},
            {"text": "Their makers remain unknown today.", "aside": "", "evidence": "source quote",
             "visual": {"shows": "a stone sphere", "queries": ["stone sphere photo", "stone sphere detail"]}},
        ],
    }
    with pytest.raises(AssertionError, match="word-count repair required:.*segment 2 turn.*final payoff"):
        writer.llm.validator(bad_counts)


def test_writer_visual_gate_targets_trend_subject_not_research_article():

    cfg = common.load_config()
    script = {
        "title": "iPhone Duo's two cameras map depth",
        "segments": [
            {"text": "The iPhone Duo reads depth with two camera views.", "aside": "",
             "emphasis": [], "reveal": False,
             "visual": {"shows": "close-up of the iPhone Duo camera module",
                        "queries": ["iPhone Duo rear camera", "iPhone Duo camera module"]},
             "evidence": "The iPhone Duo uses two camera views for depth mapping."},
            {"text": "Two camera lenses capture the same scene from different positions.",
             "aside": "", "emphasis": [], "reveal": False,
             "visual": {"shows": "two lenses on the iPhone Duo",
                        "queries": ["iPhone Duo dual cameras", "iPhone Duo camera close-up"]},
             "evidence": "Two camera views capture the same scene from different positions."},
            {"text": "Stereo vision compares those shifted images to estimate distance.",
             "aside": "", "emphasis": [], "reveal": False,
             "visual": {"shows": "a pair of camera lenses aimed at one scene",
                        "queries": ["paired camera lenses", "stereo camera lenses"]},
             "evidence": "Stereo vision compares shifted images to estimate distance."},
            {"text": "A tiny offset helps software place objects nearer or farther.",
             "aside": "", "emphasis": [], "reveal": True,
             "visual": {"shows": "a phone camera focused on a foreground object",
                        "queries": ["phone camera foreground object", "smartphone depth focus"]},
             "evidence": "The shift helps software estimate distance and depth."},
            {"text": "That difference becomes the phone's depth map.", "aside": "",
             "emphasis": [], "reveal": False,
             "visual": {"shows": "the iPhone Duo camera module",
                        "queries": ["iPhone Duo camera module", "iPhone Duo rear lenses"]},
             "evidence": "Two camera views support a depth map."},
        ],
    }
    metadata = {
        "description": "A source-grounded look at how two camera views can help estimate depth.",
        "tags": ["iPhone Duo", "smartphone cameras", "stereo vision", "depth mapping",
                 "computer vision", "optics", "camera sensors", "smartphones"],
        "hashtags": ["#science", "#technology", "#shorts"],
        "thumbnail_text": "Two views",
    }

    class ScriptLLM:
        lite = False
        last_used = "test:model"

        def json(self, _system, user, validate=None, **_kwargs):
            if user.startswith("VIDEO TITLE:"):
                return metadata
            assert "CURRENT TREND SUBJECT (deliver its specific live-trend angle; do not copy the raw headline): iPhone Duo cameras" in user
            assert "RESEARCH SOURCE SUBJECT (facts must come from this article; the trend angle still determines the story): Stereopsis" in user
            assert "Why the iPhone Duo detects depth" not in user, "raw trend wording must not reach the writer"
            if validate:
                validate(script)
            return copy.deepcopy(script)

    writer = scriptwriter.ScriptWriter(cfg, ScriptLLM(), object())
    result = writer.write(
        {"topic": "Why the iPhone Duo detects depth", "script_subject": "iPhone Duo cameras",
         "category": "science", "angle": "two camera views can build stereo depth maps",
         "why_trending": "A qualifying current science trend."},
        {"format": "sounds_fake", "hook_style": "bold_claim", "voice": "en-US-AndrewNeural"},
        {"title": "Stereopsis", "text": "Stereo vision compares two camera views from different positions. " * 40},
    )

    assert result["title"] == script["title"]
    assert "iPhone Duo" in result["segments"][0]["visual"]["shows"]


def test_early_turn_timing_is_recorded_as_runtime_location_not_feed_data():
    result = pipeline._early_turn_diagnostic(
        {"segments": [{"text": "Hook line."}, {"text": "A sourced consequence follows."}]},
        {"segments": [{"start": 0.0}, {"start": 2.4}]},
        20.0,
    )
    assert result == {
        "turn_segment_index": 2,
        "turn_start_seconds": 2.4,
        "turn_start_runtime_pct": 12.0,
        "turn_factual_word_count": 4,
    }


def test_missing_turn_timing_is_optional_and_does_not_raise():
    result = pipeline._early_turn_diagnostic({"segments": [{"text": "Hook."}]}, {}, 0)
    assert result["turn_start_seconds"] is None
    assert result["turn_start_runtime_pct"] is None
    assert result["turn_factual_word_count"] is None
