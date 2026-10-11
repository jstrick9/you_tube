"""Rendered-frame repair regressions: better sourcing is allowed, a weaker QA gate is not."""
from pathlib import Path


def test_final_qa_feedback_drives_a_fresh_exact_asset_search(monkeypatch, tmp_path):
    from autotube import media, vision

    class RequeryLLM:
        user = ""

        def json(self, system, user, temperature=None, validate=None):
            self.user = user
            result = {
                "shows": "clear close-up of a pre-1992 British two pence coin, no hand required",
                "queries": ["pre-1992 British two pence", "1979 British 2p coin"],
            }
            validate and validate(result)
            return result

    class FakeJudge:
        def __init__(self, cfg, llm):
            pass

        def available(self):
            return True

        def score(self, candidates, want, narration, subject, context, keep=None, page=0):
            for c in candidates:
                c["vscore"] = 7.7 if c["url"].endswith("weak") else 8.4
                c["vshows"] = "correct 1979 two-pence coin" if c["url"].endswith("strong") else "similar coin"
                c["vjudge"] = "fake:vision"
            return list(candidates)

    llm = RequeryLLM()
    monkeypatch.setattr(vision, "Judge", FakeJudge)
    captured = {}

    def candidates_for(queries, sources, allowed, min_width):
        captured.update(queries=queries, sources=sources)
        return [
            {"url": "https://assets.test/weak", "title": "wrong coin", "license": "CC BY",
             "author": "a", "source": "Wikimedia Commons", "page": "weak"},
            {"url": "https://assets.test/strong", "title": "1979 two pence", "license": "CC BY",
             "author": "b", "source": "Wikimedia Commons", "page": "strong"},
        ]

    monkeypatch.setattr(media, "_candidates_for", candidates_for)
    monkeypatch.setattr(
        media, "_usable",
        lambda candidate, work, *args: tmp_path / ("new-strong.jpg" if candidate["url"].endswith("strong")
                                                   else "new-weak.jpg"),
    )

    old_path = tmp_path / "wrong-one-penny.jpg"
    visuals = {
        "shots": [{"seg": 1, "path": old_path, "want": "a hand holding a two pence coin",
                   "shows": "one-penny coin", "score": 9,
                   "credit": {"url": "https://assets.test/old", "title": "one penny"}}],
        "alts": {1: []}, "need": {1: 2.5}, "max_mb": 80,
    }
    segment = {"text": "Because these pre-1992 coins contain 97 percent copper.",
               "visual": {"shows": "a hand holding a two pence coin", "queries": ["old British coin"]}}
    cfg = {"media": {"requery": True, "sources": ["wikimedia", "pixazo"], "min_image_width": 480,
                     "prefer_video": False, "shape_bonus": 1},
           "compliance": {"allowed_licenses": ["CC BY"]},
           "qa": {"repair_min_match_score": 8}}

    repaired = media.use_alternative(
        visuals, 1, str(old_path), tmp_path, source={"title": "Two pence", "description": "British decimal coin"},
        segment=segment, cfg=cfg, llm=llm,
        feedback="rendered pixels show a ONE PENNY coin; the two-pence design is absent",
    )

    assert repaired
    assert "ONE PENNY" in llm.user and "EXACT NARRATION" in llm.user
    assert captured["queries"] == ["pre-1992 British two pence", "1979 British 2p coin"]
    assert "pixazo" not in captured["sources"]  # do not replace factual evidence with fabricated scenes
    assert visuals["shots"][0]["path"] == tmp_path / "new-strong.jpg"
    assert visuals["shots"][0]["want"] == "clear close-up of a pre-1992 British two pence coin, no hand required"
    assert segment["visual"]["shows"] == visuals["shots"][0]["want"]


def test_visual_requery_rejects_malformed_provider_types_even_if_callback_is_ignored():
    import pytest
    from autotube import media

    class Malformed:
        def json(self, *args, **kwargs):
            return {"shows": "a real close-up", "queries": "not-an-array"}

    with pytest.raises(ValueError):
        media._requery(Malformed(), "subject", "exact narration", "old shot", [])
    with pytest.raises(ValueError):
        media._requery_after_final_qa(Malformed(), "subject", "exact narration", "old shot", "bad asset", "mismatch")


def test_final_qa_requery_fails_closed_below_strong_match_threshold(monkeypatch, tmp_path):
    from autotube import media, vision

    class FakeLLM:
        def json(self, system, user, temperature=None, validate=None):
            result = {"shows": "a real image of the exact object", "queries": ["exact object photo"]}
            validate and validate(result)
            return result

    class WeakJudge:
        def __init__(self, cfg, llm):
            pass

        def available(self):
            return True

        def score(self, candidates, *args, **kwargs):
            for c in candidates:
                c.update(vscore=7.9, vshows="loosely related object", vjudge="fake:vision")
            return list(candidates)

    monkeypatch.setattr(vision, "Judge", WeakJudge)
    monkeypatch.setattr(media, "_candidates_for", lambda *a, **k: [
        {"url": "https://assets.test/weak", "title": "near match", "license": "CC BY",
         "author": "a", "source": "Wikimedia Commons", "page": "weak"}])
    monkeypatch.setattr(media, "_usable", lambda *a, **k: tmp_path / "weak.jpg")
    old = tmp_path / "bad.jpg"
    visuals = {"shots": [{"seg": 0, "path": old, "want": "requested object",
                           "credit": {"url": "https://assets.test/bad", "title": "bad"}}],
               "alts": {0: []}, "need": {0: 2}, "max_mb": 80}
    cfg = {"media": {"requery": True, "sources": ["wikimedia"], "min_image_width": 480,
                     "prefer_video": False},
           "compliance": {"allowed_licenses": ["CC BY"]}, "qa": {"repair_min_match_score": 8}}

    assert not media.use_alternative(
        visuals, 0, str(old), tmp_path, source={"title": "Subject"},
        segment={"text": "Exact narration", "visual": {}}, cfg=cfg, llm=FakeLLM(), feedback="wrong pixels",
    )
    assert visuals["shots"][0]["path"] == old


def test_render_qa_attempts_every_failed_shot_without_short_circuit(monkeypatch, tmp_path):
    from autotube import pipeline

    renders, repairs, verify_calls = [], [], []
    render_result = {"timeline": [{"seg": i, "start": float(i), "end": float(i + 1), "path": f"good-{i}.jpg"}
                                 for i in range(3)]}
    monkeypatch.setattr(pipeline.render, "render", lambda *a, **k: (renders.append(1), render_result)[1])

    failed = {
        "passed": False, "repairable": True, "issues": ["mismatched frames"],
        "failed": [(0, "bad-0"), (1, "bad-1"), (2, "bad-2")],
        "frames": [
            {"seg": i, "path": f"bad-{i}", "shows": f"wrong image {i}",
             "intended_shot": f"intended {i}", "issue": f"mismatch {i}"}
            for i in range(3)
        ],
    }

    success = {"passed": True, "frames": [{"timeline_index": i, "seg": i, "path": f"good-{i}.jpg", "t": float(i) + 0.5,
                                              "shows": "verified subject", "sample": 1, "samples": 1,
                                              "match": True, "score": 9, "audit_shows": "verified subject",
                                              "audit_match": True, "audit_score": 9, "judge": "gemini",
                                              "audit_judge": "gemini"} for i in range(3)],
               "issues": [], "failed": [], "repairable": False,
               "meta": {"hook_text_ok": True, "title_ok": True, "visual_variety_ok": True,
                        "notes": [], "visual_variety_notes": ""}}

    def verify(*args, **kwargs):
        verify_calls.append(1)
        return failed if len(verify_calls) == 1 else success

    monkeypatch.setattr(pipeline.qa, "verify", verify)

    def repair(visuals, seg, path, work, **kwargs):
        repairs.append((seg, kwargs.get("feedback", "")))
        return seg == 1  # first and last fail; middle succeeds

    monkeypatch.setattr(pipeline.media, "use_alternative", repair)
    assets = {"tts": {}, "visuals": {"shots": [{"seg": i, "path": f"good-{i}.jpg", "score": 9,
                                                   "judge": "gemini"} for i in range(3)]},
              "music": None, "hook_card": "hook",
              "work": tmp_path, "out": tmp_path / "out.mp4", "seed": 1, "fx": [],
              "script": {"segments": [{"text": f"line {i}"} for i in range(3)]},
              "source": {"title": "Topic"}, "title": "Title", "subject": "Topic"}
    writer = type("Writer", (), {"llm": object()})()

    _, report = pipeline.render_with_qa(
        assets, {"qa": {"max_repairs": 1}, "media": {"min_match_score": 7},
                 "channel": {"name": "Archive 13"}}, writer)

    assert [seg for seg, _ in repairs] == [0, 1, 2]
    assert all(feedback for _, feedback in repairs)
    assert len(renders) == 2 and report["passed"]


def test_render_qa_does_not_repair_after_final_failed_attempt(monkeypatch, tmp_path):
    from autotube import pipeline

    renders, repairs, verify_calls = [], [], []
    monkeypatch.setattr(
        pipeline.render, "render",
        lambda *a, **k: (renders.append(1), {"timeline": []})[1],
    )
    failed = {
        "passed": False, "repairable": True, "issues": ["mismatched frames"],
        "failed": [(0, "bad-0")],
        "frames": [{"seg": 0, "path": "bad-0", "shows": "wrong image",
                    "intended_shot": "intended image", "issue": "mismatch"}],
    }
    monkeypatch.setattr(pipeline.qa, "verify", lambda *a, **k: (verify_calls.append(1), failed)[1])
    monkeypatch.setattr(
        pipeline.media, "use_alternative",
        lambda *a, **k: (repairs.append(1), True)[1],
    )
    assets = {
        "tts": {}, "visuals": {"shots": []}, "music": None, "hook_card": "hook",
        "work": tmp_path, "out": tmp_path / "out.mp4", "seed": 1, "fx": [],
        "script": {"segments": [{"text": "line"}]}, "source": {"title": "Topic"},
        "title": "Title", "subject": "Topic",
    }
    writer = type("Writer", (), {"llm": object()})()

    _, report = pipeline.render_with_qa(
        assets, {"qa": {"max_repairs": 1}, "media": {"min_match_score": 7},
                 "channel": {"name": "Archive 13"}}, writer)

    assert len(renders) == 2
    assert len(verify_calls) == 2
    assert len(repairs) == 1  # repair after attempt 1 was re-rendered and checked on attempt 2
    assert report["attempt"] == 2
    assert not report["passed"]
