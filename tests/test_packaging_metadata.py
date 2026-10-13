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


def test_package_sets_per_video_synthetic_flag_and_records_runway_provenance(tmp_path, monkeypatch):
    cfg = load_config()
    out = tmp_path / "generated-insert.mp4"
    out.write_bytes(b"placeholder")
    monkeypatch.setattr(pipeline.render, "thumbnail", lambda *args, **kwargs: tmp_path / "thumb.jpg")
    monkeypatch.setattr(pipeline, "_shots_sheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "read_json", lambda *args, **kwargs: [])

    credit = {
        "title": "AI-generated illustration (veo3.1_fast)", "author": "Runway Dev generated media",
        "license": "AI-generated; account terms confirmed", "source": "Runway Dev API (veo3.1_fast)",
        "url": "runway://task/example", "page": "https://docs.dev.runwayml.com/api/",
        "synthetic": True, "provider": "Runway Dev API", "model": "veo3.1_fast",
        "api_version": "2024-11-06", "task_id": "example-task",
        "prompt": "A visibly stylized fossil illustration.", "generated_at": "2026-10-13T00:00:00+00:00",
        "ratio": "1080:1920", "audio": False, "requested_duration_seconds": 8,
        "actual_duration_seconds": 8.0, "output_width": 1080, "output_height": 1920,
        "credits_reserved": 80, "estimated_cost_credits": 80, "actual_cost_credits": 80,
        "estimated_cost_usd": 0.8, "file_sha256": "b" * 64,
        "rights_confirmation": {"runway_account_terms_confirmed": True,
                                "hosted_veo_terms_confirmed": True,
                                "terms_url": "https://runway.com/terms-of-use"},
    }
    clip = tmp_path / "generated-clip.mp4"
    assets = {
        "script": {"title": "The fossil in stone", "description": "A sourced fossil fact.",
                   "segments": [{"text": "A fossil is trapped in stone."}], "tags": [], "hashtags": ["#science"]},
        "source": {"title": "Fossil formation", "url": "https://example.test/fossils", "also": []},
        "visuals": {"shots": [{"seg": 0, "path": clip, "kind": "video", "synthetic": True,
                                 "score": 9, "judge": "gemini", "shows": "a fossil illustration",
                                 "want": "a stylized fossil", "credit": credit}]},
        "review": {"score": 9}, "plan": {}, "topic": {"topic": "Fossil formation"},
        "out": out, "hook_card": "", "title": "A fossil in stone", "tts": {"engine": "test"},
        "series": None, "episode": 0,
    }
    render_result = {"first_frame": tmp_path / "frame.jpg", "theme": 0, "duration": 3.0,
                     "timeline": [{"seg": 0, "start": 0.0, "end": 2.0, "path": str(clip)}]}
    frames = [{"timeline_index": 0, "seg": 0, "path": str(clip), "t": t,
               "sample": i, "samples": 3, "match": True, "score": 9,
               "shows": "a clearly stylized fossil", "audit_shows": "a clearly stylized fossil",
               "audit_match": True, "audit_score": 9, "judge": "gemini", "audit_judge": "gemini"}
              for i, t in enumerate((0.25, 1.0, 1.75), start=1)]
    report = {"passed": True, "attempt": 1, "issues": [], "failed": [], "repairable": False,
              "frames": frames, "meta": {"hook_text_ok": True, "title_ok": True,
                                           "visual_variety_ok": True, "notes": [],
                                           "visual_variety_notes": ""}}

    result = pipeline.package(assets, render_result, report, cfg)
    sidecar = json.loads(out.with_suffix(".json").read_text())

    assert result["meta"]["contains_synthetic_media"] is True
    assert "not archival footage" in result["meta"]["description"]
    assert sidecar["contains_synthetic_media"] is True
    generation = sidecar["shots"][0]["generation"]
    assert generation["model"] == "veo3.1_fast"
    assert generation["task_id"] == "example-task"
    assert generation["prompt"] == credit["prompt"]
    assert generation["file_sha256"] == "b" * 64
    assert generation["rights_confirmation"]["hosted_veo_terms_confirmed"] is True
