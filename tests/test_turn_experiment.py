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

    assert "RANDOMIZED EARLY-TURN TREATMENT (consequence_first)" in writer.llm.prompt
    assert "first clause" in writer.llm.prompt and "source-backed consequence" in writer.llm.prompt
    assert "Keep the script original; do not reuse another upload's topic" in writer.llm.prompt


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
