"""Outlier format mining uses aggregate metadata only; it never supplies creator executions to the writer."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import common, scriptwriter, trends, youtube  # noqa: E402
from autotube.common import load_config, now_utc  # noqa: E402

CFG = load_config()


def test_youtube_duration_parser_handles_iso8601_and_rejects_invalid_values():
    assert trends.youtube_duration_seconds("PT15S") == 15
    assert trends.youtube_duration_seconds("PT1M20S") == 80
    assert trends.youtube_duration_seconds("PT1H2M3.5S") == 3724
    assert trends.youtube_duration_seconds("PT0S") is None
    assert trends.youtube_duration_seconds("P1D") is None
    assert trends.youtube_duration_seconds(True) is None


def test_format_pulse_is_velocity_weighted_by_lane_and_stores_no_examples():
    rows = [
        {"topic": "Why does the sea glow?", "hook": "question", "lane": "history_mystery",
         "vph": 100, "duration_seconds": 18},
        {"topic": "5 reasons a lake turns pink", "hook": "number_first", "lane": "history_mystery",
         "vph": 900, "duration_seconds": 35},
        {"topic": "The quietest animal on Earth", "hook": "bold_claim", "lane": "science_nature",
         "vph": 500, "duration_seconds": 65},
    ]

    pulse = trends.format_pulse(rows)
    history = pulse["by_lane"]["history_mystery"]

    assert pulse["schema_version"] == 2
    assert history["sample"] == 2
    assert history["hook_style"]["number_first"] > history["hook_style"]["question"]
    assert history["duration_sample"] == 2
    assert history["duration_bands"] == {"21_40s": 0.9, "le_20s": 0.1}
    assert pulse["not_measured"] == ["spoken_hook", "beat_count", "caption_density"]
    serialized = json.dumps(pulse)
    for row in rows:
        assert row["topic"] not in serialized, "individual creator titles must not be retained in the aggregate"


def test_youtube_outlier_collection_uses_duration_metadata_not_descriptions(monkeypatch):
    fixed_now = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
    description = "UNIQUE CREATOR DESCRIPTION THAT MUST NOT BE RETAINED OR PASSED ON"
    video_id = "abcdefghijk"
    published_at = (fixed_now - timedelta(hours=10)).isoformat().replace("+00:00", "Z")
    captured = {}

    class Request:
        def __init__(self, data):
            self.data = data

        def execute(self):
            return self.data

    class Search:
        def list(self, **kwargs):
            return Request({"items": [{"id": {"videoId": video_id}}]})

    class Videos:
        def list(self, **kwargs):
            captured["videos_part"] = kwargs["part"]
            return Request({"items": [{
                "id": video_id,
                "snippet": {
                    "title": "Why does the desert bloom at night?", "publishedAt": published_at,
                    "channelId": "channel-1", "channelTitle": "Other Creator", "description": description,
                },
                "statistics": {"viewCount": "50000"},
                "contentDetails": {"duration": "PT15S"},
            }]})

    class Channels:
        def list(self, **kwargs):
            return Request({"items": [{"id": "channel-1", "statistics": {"subscriberCount": "1000"}}]})

    class Service:
        def search(self):
            return Search()

        def videos(self):
            return Videos()

        def channels(self):
            return Channels()

    stored = {}
    monkeypatch.setattr(trends, "_yt_oauth_ready", lambda: True)
    monkeypatch.setattr(trends, "lane_queries", lambda _cfg: [("science_nature", "night desert bloom")])
    monkeypatch.setattr(trends, "now_utc", lambda: fixed_now)
    monkeypatch.setattr(trends, "read_json", lambda *_args, **_kwargs: {"yt_outliers_date": "2026-10-10"})
    monkeypatch.setattr(common, "write_json", lambda name, data: stored.__setitem__(name, data))
    monkeypatch.setattr(youtube, "service", lambda: Service())
    cfg = {
        "channel": {"name": "Archive 13", "region": "US", "language": "en"},
        "trends": {"outlier_min_views": 30000, "outlier_window_hours": 72, "format_pulse": True},
    }

    outliers = trends.youtube_outliers(cfg)

    assert captured["videos_part"] == "snippet,statistics,contentDetails"
    assert len(outliers) == 1
    assert outliers[0]["evidence"]["duration_seconds"] == 15
    assert description not in json.dumps(outliers)
    pulse = stored["format_pulse.json"]
    assert pulse["by_lane"]["science_nature"]["duration_sample"] == 1
    assert description not in json.dumps(stored["trend_cache.json"])


def test_writer_receives_only_fresh_lane_aggregates_not_the_outlier_title(monkeypatch):
    cfg = load_config()
    now = now_utc()
    pulse = {
        "schema_version": 2,
        "updated": now.isoformat(),
        "sample": 30,
        "hook_style": {"question": 0.6, "number_first": 0.4},
        "by_lane": {
            "history_mystery": {
                "sample": 12,
                "hook_style": {"question": 0.7, "number_first": 0.3},
                "duration_sample": 12,
                "duration_bands": {"le_20s": 0.25, "21_40s": 0.75},
            },
        },
        "not_measured": ["spoken_hook", "beat_count", "caption_density"],
    }
    monkeypatch.setattr(scriptwriter, "read_json", lambda *_args, **_kwargs: pulse)
    captured = {}

    class CaptureLLM:
        lite = False
        last_used = "test:model"

        def json(self, _system, user, **_kwargs):
            captured["user"] = user
            raise RuntimeError("captured writer prompt")

    writer = scriptwriter.ScriptWriter(cfg, CaptureLLM(), object())
    creator_title = "Why the hidden lake turns glass at midnight"
    topic = {
        "topic": creator_title,
        "lane": "history_mystery",
        "category": "history",
        "angle": "The lake's night-time color comes from a rare mineral reaction",
        "why_trending": "A recent Short rapidly gained views",
    }
    source = {"title": "Lake chemistry", "text": "Sourced facts only.", "url": "https://example.test/source"}
    plan = {"format": "sounds_fake", "hook_style": "bold_claim", "voice": "voice"}

    try:
        writer.write(topic, plan, source)
    except RuntimeError as exc:
        assert str(exc) == "captured writer prompt"

    prompt = captured["user"]
    assert "OUTLIER-DERIVED FORMAT GUIDANCE" in prompt
    assert "Trend evidence is for choosing a currently interesting subject, not a template" in prompt
    assert "question 70%" in prompt
    assert "Lake chemistry" in prompt
    assert creator_title not in prompt, "competitor title wording must not be sent to the scriptwriter"
    assert "not evidence of causation or viewer retention" in prompt
    assert "caption density" in prompt and "beat count" in prompt
    assert "Keep the assigned format, lane hook" in prompt


def test_writer_ignores_stale_or_underpowered_format_pulse(monkeypatch):
    cfg = load_config()
    now = datetime(2026, 10, 11, 12, tzinfo=timezone.utc)
    stale = {
        "updated": (now - timedelta(days=4)).isoformat(),
        "sample": 100,
        "hook_style": {"question": 1.0},
    }
    monkeypatch.setattr(scriptwriter, "read_json", lambda *_args, **_kwargs: stale)
    monkeypatch.setattr(scriptwriter, "now_utc", lambda: now)
    assert scriptwriter.outlier_structure_guidance({"category": "history"}, cfg) == ""

    fresh_but_small = {"updated": now.isoformat(), "sample": 4, "hook_style": {"question": 1.0}}
    monkeypatch.setattr(scriptwriter, "read_json", lambda *_args, **_kwargs: fresh_but_small)
    assert scriptwriter.outlier_structure_guidance({"category": "history"}, cfg) == ""

    disabled_cfg = {**cfg, "trends": {**cfg["trends"], "format_pulse": False}}
    monkeypatch.setattr(scriptwriter, "read_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("disabled format mining must not read stale state")))
    assert scriptwriter.outlier_structure_guidance({"category": "history"}, disabled_cfg) == ""
