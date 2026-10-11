"""Regression guards for AUDIT section 5 (architecture & code quality)."""
import time
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def test_final_qa_pass_requires_complete_primary_and_adversarial_verdicts():
    import copy
    import yaml

    from autotube.pipeline import _normalize_qa_report

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    complete = {"passed": True, "issues": [], "failed": [], "repairable": False,
                "frames": [{"timeline_index": 0, "seg": 0, "path": "shot.jpg", "t": 1.0,
                            "sample": 1, "samples": 1,
                            "shows": "the actual subject", "match": True, "score": 9,
                            "judge": "gemini", "audit_shows": "the actual subject",
                            "audit_match": True, "audit_score": 9, "audit_judge": "gemini"}],
                "meta": {"hook_text_ok": True, "title_ok": True, "visual_variety_ok": True,
                         "notes": [], "visual_variety_notes": ""}}
    timeline = [{"seg": 0, "start": 0.0, "end": 2.0, "path": "shot.jpg"}]
    shots = [{"seg": 0, "path": "shot.jpg", "score": 9, "judge": "gemini"}]
    assert _normalize_qa_report(complete, cfg, timeline, shots, 1)["passed"] is True
    for field in ("timeline_index", "judge", "audit_judge", "audit_shows", "audit_match",
                  "audit_score", "sample", "samples"):
        malformed = copy.deepcopy(complete)
        malformed["frames"][0].pop(field)
        assert _normalize_qa_report(malformed, cfg, timeline, shots, 1)["passed"] is False, field
    assert _normalize_qa_report(complete, cfg, timeline, [], 1)["passed"] is False
    unlinked = [{"seg": 0, "start": 0.0, "end": 2.0, "path": "different.jpg"}]
    assert _normalize_qa_report(complete, cfg, unlinked, shots, 1)["passed"] is False

    duplicate_intervals = [
        {"seg": 0, "start": 0.0, "end": 1.0, "path": "shot.jpg"},
        {"seg": 0, "start": 1.0, "end": 2.0, "path": "shot.jpg"},
    ]
    assert _normalize_qa_report(complete, cfg, duplicate_intervals, shots, 1)["passed"] is False, (
        "one QA frame must not stand in for two separately rendered intervals of the same approved shot")

    moving_timeline = [{"seg": 0, "start": 0.0, "end": 3.0, "path": "clip.mp4"}]
    moving_shots = [{"seg": 0, "path": "clip.mp4", "score": 9, "judge": "gemini", "kind": "video"}]
    moving_report = copy.deepcopy(complete)
    moving_report["frames"] = [
        {**complete["frames"][0], "timeline_index": 0, "path": "clip.mp4", "t": t,
         "sample": i, "samples": 3}
        for i, t in enumerate((0.6, 1.5, 2.4), start=1)
    ]
    assert _normalize_qa_report(moving_report, cfg, moving_timeline, moving_shots, 1)["passed"] is True
    moving_report["frames"].pop()
    assert _normalize_qa_report(moving_report, cfg, moving_timeline, moving_shots, 1)["passed"] is False, (
        "moving footage must have QA for its start, middle and end")

    weak_thresholds = copy.deepcopy(cfg)
    weak_thresholds["media"]["min_match_score"] = 0
    weak_thresholds["qa"]["min_frame_score"] = 0
    assert _normalize_qa_report(complete, weak_thresholds, timeline, shots, 1)["passed"] is False


def test_upload_guard_requires_an_explicit_boolean_qa_pass():
    from autotube.pipeline import _qa_pass_is_explicit

    assert _qa_pass_is_explicit({"qa": {"passed": True}})
    for verdict in ("true", "false", 1, 0, [], {}, None):
        assert not _qa_pass_is_explicit({"qa": {"passed": verdict}}), repr(verdict)
    assert not _qa_pass_is_explicit({"qa": "passed"})
    assert not _qa_pass_is_explicit({})


# ── the render/QA repair loop ────────────────────────────────────────────────
def test_render_with_qa_always_renders_at_least_once(monkeypatch):
    """`r` is bound inside the repair loop and used after it, so a loop that never runs is a NameError."""
    import autotube.pipeline as pl

    calls = []
    fake_r = {"timeline": [{"seg": 0, "start": 0.0, "end": 2.0, "path": "a"}],
              "duration": 25.0, "first_frame": "f.png", "theme": {}, "archetype": "bold"}
    monkeypatch.setattr(pl.render, "render", lambda *a, **k: (calls.append(1), fake_r)[1])
    good_report = {"passed": True, "frames": [{"timeline_index": 0, "seg": 0, "path": "a", "t": 0.5,
                                             "shows": "subject",
                                             "sample": 1, "samples": 1, "match": True, "score": 9,
                                             "audit_shows": "subject", "audit_match": True, "audit_score": 9,
                                             "judge": "gemini", "audit_judge": "gemini"}],
                   "issues": [], "failed": [], "repairable": False,
                   "meta": {"hook_text_ok": True, "title_ok": True, "visual_variety_ok": True,
                            "notes": [], "visual_variety_notes": ""}}
    monkeypatch.setattr(pl.qa, "verify", lambda *a, **k: good_report)

    assets = {k: None for k in ("tts", "visuals", "music", "hook_card", "work", "out", "seed", "fx",
                                "script", "title", "subject")}
    assets["visuals"] = {"shots": [{"seg": 0, "path": "a", "score": 9, "judge": "gemini"}]}
    assets["script"] = {"segments": [{"text": "The subject is visible."}]}
    writer = type("W", (), {"llm": object()})()
    for bad in (-5, -1, 0):
        calls.clear()
        r, report = pl.render_with_qa(assets, {"qa": {"max_repairs": bad},
                                                "media": {"min_match_score": 7}, "channel": {"name": "x"}}, writer)
        assert len(calls) == 1, f"max_repairs={bad} rendered {len(calls)} times"
        assert r is fake_r and report["passed"]


def test_render_with_qa_rejects_truthy_string_pass_verdict(monkeypatch, tmp_path):
    import autotube.pipeline as pl

    monkeypatch.setattr(pl.render, "render", lambda *a, **k: {"timeline": []})
    monkeypatch.setattr(pl.qa, "verify", lambda *a, **k: {"passed": "false", "issues": [], "frames": []})
    assets = {"tts": {}, "visuals": {"shots": []}, "music": None, "hook_card": "hook",
              "work": tmp_path, "out": tmp_path / "out.mp4", "seed": 1, "fx": [],
              "script": {"segments": []}, "title": "Title", "subject": "Subject"}
    writer = type("W", (), {"llm": object()})()

    _, report = pl.render_with_qa(
        assets, {"qa": {"max_repairs": 0}, "channel": {"name": "Archive 13"}}, writer)

    assert report["passed"] is False
    assert any("passed verdict was not a boolean" in issue for issue in report["issues"])


# ── LLM failover ─────────────────────────────────────────────────────────────
def test_retry_after_is_honoured_and_bounded():
    from autotube.llm import _retry_after
    assert _retry_after('HTTP 429 {"retry_after": 3}', 6) == 3.0      # provider told us
    assert _retry_after("HTTP 429 Retry-After: 900", 6) == 45.0        # hostile value capped
    assert _retry_after("HTTP 429 retry-after: 0", 6) == 1.0           # never a busy-loop
    assert _retry_after("HTTP 429 slow down", 6) == 6                  # nothing to parse
    assert _retry_after("HTTP 429 retry-after: soon", 6) == 6          # unparseable


def test_no_backoff_sleep_after_the_final_attempt(monkeypatch):
    """Sleeping then giving up only delays failover; it never buys another try."""
    from autotube import llm

    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    def always_429(model, system, user, temp, images=None):
        raise llm.LLMError("gemini m HTTP 429: rate limited")

    monkeypatch.setitem(llm.PROVIDERS, "fake", (always_429, "fake_models"))
    client = llm.LLM.__new__(llm.LLM)
    client.cfg = {"fake_models": ["m1", "m2"]}
    client.order = ["fake"]
    client.temperature = 0.7
    client._dead = set()
    client.last_used = None

    with pytest.raises(llm.LLMError):
        client.json("sys", "user", attempts_per_model=2)

    # 2 models x 2 attempts = 4 failures, but only the non-final attempt of each model may sleep
    assert len(slept) == 2, f"slept {len(slept)} times, expected 2 (one per model): {slept}"


# ── history / analytics windows ──────────────────────────────────────────────
def test_history_is_kept_at_least_as_long_as_the_refresh_window():
    """Trimming history below the refresh window would discard videos before they were ever measured."""
    a = _cfg()["analytics"]
    assert a["history_keep"] >= a["refresh_videos"]


def test_both_writers_trim_history_with_the_same_configured_cap():
    src = (ROOT / "autotube" / "pipeline.py").read_text() + (ROOT / "autotube" / "analytics.py").read_text()
    assert "[-2000:]" not in src, "hardcoded history tail is back; it must come from analytics.history_keep"
    assert src.count('history_keep", 2000)') == 2


# ── make_one decomposition ───────────────────────────────────────────────────
def test_pipeline_stages_stay_separately_callable_and_small():
    import autotube.pipeline as pl
    import inspect

    for name in ("produce_assets", "render_with_qa", "package", "make_one"):
        fn = getattr(pl, name)
        n = len(inspect.getsource(fn).splitlines())
        assert n <= 60, f"{name} has grown back to {n} lines"
