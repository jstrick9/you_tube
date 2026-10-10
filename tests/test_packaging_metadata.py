"""Packaged sidecars keep the committed episode number and trend provenance."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import pipeline, series  # noqa: E402
from autotube.common import load_config  # noqa: E402


def test_sidecar_uses_assigned_episode_and_preserves_trend_evidence(tmp_path, monkeypatch):
    cfg = load_config()
    out = tmp_path / "archive13.mp4"
    out.write_bytes(b"placeholder")
    monkeypatch.setattr(pipeline.render, "thumbnail", lambda *args, **kwargs: tmp_path / "thumb.jpg")
    monkeypatch.setattr(pipeline, "_shots_sheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "read_json", lambda *args, **kwargs: [])
    series.reset()

    topic = {
        "sources": ["youtube_outliers", "google_trends"],
        "source_families": ["youtube", "google_trends"],
        "score": 0.91,
        "viral_score": 8,
        "angle": "the figure doubled after the specific discovery",
        "why_trending": "A related explainer Short is gaining views today.",
        "context": ["views up 4.2x in 12 hours", "Google Trends breakout in US"],
        "signals": [{"source": "youtube_outliers", "source_url": "https://youtube.test/viral",
                     "evidence": {"views": 2000000, "hours": 12}}],
    }
    assets = {
        "script": {"title": "The treaty that ended a war", "description": "", "segments": [],
                   "tags": [], "hashtags": []},
        "source": {"title": "Treaty of Portsmouth", "url": "https://example.test/treaty", "also": []},
        "visuals": {"shots": []},
        "review": {"score": 9, "trend_alignment": 9, "trend_alignment_reason": "Covers the discovery"},
        "plan": {"format": "backstory"},
        "topic": topic,
        "out": out,
        "hook_card": "",
        "title": "The treaty that ended a war",
        "series": "unsolved",
        "episode": 0,
        "tts": {"engine": "edge-tts"},
    }
    render_result = {"first_frame": tmp_path / "first.jpg", "theme": 0, "duration": 18.2,
                     "archetype": "clean", "timeline": []}
    report = {"passed": True, "attempt": 1, "issues": [], "frames": [], "meta": {}}

    result = pipeline.package(assets, render_result, report, cfg)
    sidecar = json.loads(out.with_suffix(".json").read_text())

    assert result["episode"] == sidecar["episode"] == 1
    assert sidecar["series"] == "unsolved"
    assert "CASE #001" in sidecar["meta"]["title"]
    assert sidecar["trend"]["sources"] == topic["sources"]
    assert sidecar["trend"]["source_families"] == topic["source_families"]
    assert sidecar["trend"]["score"] == topic["score"]
    assert sidecar["trend"]["angle"] == topic["angle"]
    assert sidecar["trend"]["signals"] == topic["signals"]
    assert sidecar["review"]["trend_alignment"] == 9
    assert sidecar["trend"]["wikipedia_spike"] is None
    assert sidecar["trend"]["viral_score"] == topic["viral_score"]
    assert sidecar["trend"]["why_trending"] == topic["why_trending"]
    assert sidecar["trend"]["evidence"] == topic["context"]
    history_trend = pipeline._history_trend(topic)
    assert history_trend["trend_sources"] == topic["sources"]
    assert history_trend["trend_source_families"] == topic["source_families"]
    assert history_trend["trend_score"] == topic["score"]
    assert history_trend["trend_angle"] == topic["angle"]
    assert history_trend["trend_signals"] == topic["signals"]
    assert history_trend["wikipedia_spike"] is None
    assert history_trend["viral_score"] == topic["viral_score"]
    assert history_trend["why_trending"] == topic["why_trending"]
    assert history_trend["trend_evidence"] == topic["context"][:2]
