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
            "not entertaining enough."),
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
    assert "do not copy a reviewer example unless SOURCE TEXT supports it" in writer.llm.prompt


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
             "aside": "Two eyes, pocket-sized.", "emphasis": [], "reveal": False,
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
