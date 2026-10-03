"""Regression guards for AUDIT section 5 (architecture & code quality)."""
import time
import pytest
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


# ── the render/QA repair loop ────────────────────────────────────────────────
def test_render_with_qa_always_renders_at_least_once(monkeypatch):
    """`r` is bound inside the repair loop and used after it, so a loop that never runs is a NameError."""
    import autotube.pipeline as pl

    calls = []
    fake_r = {"timeline": [], "duration": 25.0, "first_frame": "f.png", "theme": {}, "archetype": "bold"}
    monkeypatch.setattr(pl.render, "render", lambda *a, **k: (calls.append(1), fake_r)[1])
    monkeypatch.setattr(pl.qa, "verify", lambda *a, **k: {"passed": True, "frames": [], "issues": [], "failed": []})

    assets = {k: None for k in ("tts", "visuals", "music", "hook_card", "work", "out", "seed", "fx",
                                "script", "title", "subject")}
    assets["visuals"] = {"shots": []}
    writer = type("W", (), {"llm": object()})()
    for bad in (-5, -1, 0):
        calls.clear()
        r, report = pl.render_with_qa(assets, {"qa": {"max_repairs": bad}, "channel": {"name": "x"}}, writer)
        assert len(calls) == 1, f"max_repairs={bad} rendered {len(calls)} times"
        assert r is fake_r and report["passed"]


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
