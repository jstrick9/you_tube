"""Packaged sidecars keep the committed episode number and trend provenance."""
import json
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import pipeline, series  # noqa: E402
from autotube.common import load_config  # noqa: E402


def test_package_rejects_truthy_string_qa_pass_report(tmp_path, monkeypatch):
    cfg = load_config()
    out = tmp_path / "unverified.mp4"
    out.write_bytes(b"placeholder")
    monkeypatch.setattr(pipeline.render, "thumbnail", lambda *args, **kwargs: tmp_path / "thumb.jpg")
    monkeypatch.setattr(pipeline, "_shots_sheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "build_description", lambda *args, **kwargs: "desc")
    assets = {
        "script": {"title": "A title", "segments": [{"text": "A sourced fact."}], "tags": []},
        "source": {"title": "Topic", "url": "https://example.test/topic"},
        "visuals": {"shots": []}, "review": {}, "plan": {}, "topic": {},
        "out": out, "hook_card": "", "title": "A title", "tts": {"engine": "test"},
    }

    result = pipeline.package(assets, {"first_frame": "frame.jpg", "theme": 0, "duration": 2, "timeline": []},
                             {"passed": "false", "issues": [], "frames": []}, cfg)

    assert result is None
    rejected = out.with_name("unverified-REJECTED.mp4")
    assert rejected.exists() and not out.exists()
    sidecar = json.loads(rejected.with_suffix(".json").read_text())
    assert sidecar["qa"]["passed"] is False
    assert any("passed verdict was not a boolean" in issue for issue in sidecar["qa"]["issues"])


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
        "script": {"title": "The treaty that ended a war", "description": "",
                   "segments": [{"text": "The treaty ended the war."}], "tags": [], "hashtags": []},
        "source": {"title": "Treaty of Portsmouth", "url": "https://example.test/treaty", "also": []},
        "visuals": {"shots": [{"seg": 0, "path": "approved.jpg", "score": 9, "judge": "gemini",
                                 "want": "the treaty", "shows": "the treaty document",
                                 "credit": {"title": "Treaty document", "author": "Archive staff",
                                            "license": "Public domain", "source": "archive",
                                            "page": "https://example.test/document", "url": "https://example.test/document"}}]},
        "review": {"score": 9, "trend_alignment": 9, "trend_alignment_reason": "Covers the discovery",
                   "turn_strength": 8, "turn_strength_status": "scored",
                   "turn_strength_reason": "The consequence arrives immediately."},
        "plan": {"format": "backstory", "turn_experiment_id": "early_turn_v1",
                 "turn_variant": "consequence_first"},
        "topic": topic,
        "out": out,
        "hook_card": "",
        "title": "The treaty that ended a war",
        "series": "unsolved",
        "episode": 0,
        "tts": {"engine": "edge-tts", "segments": [
            {"start": 0.0}, {"start": 2.4}, {"start": 5.0}, {"start": 8.0},
        ]},
    }
    render_result = {"first_frame": tmp_path / "first.jpg", "theme": 0, "duration": 18.2,
                     "archetype": "clean", "timeline": [{"seg": 0, "start": 0.0, "end": 2.0,
                                                               "path": "approved.jpg"}]}
    report = {"passed": True, "attempt": 1, "issues": [], "failed": [], "repairable": False,
              "frames": [{"timeline_index": 0, "seg": 0, "path": "approved.jpg", "t": 0.5,
                          "shows": "verified subject",
                          "sample": 1, "samples": 1, "match": True, "score": 9,
                          "audit_shows": "verified subject", "audit_match": True, "audit_score": 9,
                          "judge": "gemini", "audit_judge": "gemini"}],
              "meta": {"hook_text_ok": True, "title_ok": True, "visual_variety_ok": True,
                       "notes": [], "visual_variety_notes": ""}}

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
    assert sidecar["plan"]["turn_experiment_id"] == "early_turn_v1"
    assert sidecar["plan"]["turn_variant"] == "consequence_first"
    assert sidecar["early_turn"]["turn_start_runtime_pct"] == 13.2
    assert sidecar["review"]["turn_strength"] == 8
    assert sidecar["review"]["turn_strength_reason"] == "The consequence arrives immediately."
    assert result["early_turn"] == sidecar["early_turn"]
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
