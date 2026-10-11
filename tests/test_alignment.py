"""Guarantees of the strict alignment pipeline (no network, no API keys, no CLIP).

A fake vision model approves an image for a line only when the image's hidden "truth" word appears in that line —
so these tests prove the plumbing never lets an unapproved / mismatched image through.
"""
import copy
import re
import subprocess
import zlib
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, media, qa, vision  # noqa: E402

CFG = common.load_config()
CFG["media"]["clip"] = False
CFG["media"]["sources"] = ["wikimedia"]


def cand(truth, i=0):
    import random
    rnd = random.Random(zlib.crc32(f"{truth}{i}".encode()))
    im = Image.new("RGB", (320, 240), (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    for x in range(0, 320, 16):                     # random block pattern → visually distinct images
        for y in range(0, 240, 16):
            if rnd.random() < 0.5:
                im.paste((rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)), (x, y, x + 16, y + 16))
    return {"url": f"https://img/{truth}/{i}.jpg", "title": f"{truth} photo {i}", "_img": im,
            "license": "CC BY", "author": "a", "source": "Wikimedia Commons", "page": "p"}


class FakeLLM:
    last_used = "fake:vision"

    def __init__(self, down=False, requery=None):
        self.down, self.requery, self.vision_calls = down, requery, 0

    def vision_available(self):
        return True

    def vision_json(self, system, user, images, validate=None):
        self.vision_calls += 1
        if self.down:
            raise RuntimeError("gemini HTTP 503")
        line = re.search(r'NARRATION LINE: "(.*?)"', user).group(1).lower()
        titles = re.findall(r"^\s+(\d+)\. (\S+) photo", user, re.M)
        out = {"images": [{"n": int(n), "shows": t, "score": 9 if t in line else 2,
                           "legible": True} for n, t in titles]}
        validate and validate(out)
        return out

    def json(self, system, user, temperature=None, validate=None, **kw):
        o = self.requery or {"shows": "x", "queries": ["x"]}
        validate and validate(o)
        return o


@pytest.fixture
def patched(monkeypatch, tmp_path):
    bank = {}

    def search(queries, *a, **k):
        out = []
        for q in queries:
            out += bank.get(q, [])
        return out

    monkeypatch.setattr(media, "_candidates_for", search)
    monkeypatch.setattr(media, "_commons_info", lambda *a, **k: bank.get("_article", []))
    monkeypatch.setattr(media, "commons_search", lambda *a, **k: [])
    monkeypatch.setattr(media, "openverse_search", lambda *a, **k: [])

    def usable(c, work, *a):
        p = tmp_path / (re.sub(r"\W", "_", c["url"]) + ".jpg")
        c["_img"].save(p)
        return p

    monkeypatch.setattr(media, "_usable", usable)
    monkeypatch.setattr(vision.time, "sleep", lambda s: None)
    return bank


def segs(*pairs):
    return [{"text": t, "visual": {"shows": t, "queries": [q]}} for t, q in pairs]


SRC = {"title": "Whiskers", "description": "hair", "images": []}


def test_every_line_gets_an_image_approved_for_that_line(patched, tmp_path):
    patched["q-cat"] = [cand("building"), cand("cat", 1), cand("cat", 2)]
    patched["q-fossil"] = [cand("building", 3), cand("fossil", 4)]
    patched["q-paw"] = [cand("street"), cand("paw", 5)]
    s = segs(("a cat with whiskers", "q-cat"), ("an ancient fossil skull", "q-fossil"), ("a cat paw up close", "q-paw"))
    v = media.gather(SRC, s, CFG, tmp_path, llm=FakeLLM())
    for shot in v["shots"]:
        assert shot["credit"]["title"].split()[0] in s[shot["seg"]]["text"]    # the image matches ITS line
        assert shot["score"] >= 7 and shot["judge"] == "fake:vision"
    assert {sh["seg"] for sh in v["shots"]} == {0, 1, 2}
    assert not any("building" in sh["credit"]["title"] or "street" in sh["credit"]["title"] for sh in v["shots"])


def test_pixazo_is_suppressed_without_synthetic_disclosure(patched, tmp_path, monkeypatch):
    cfg = copy.deepcopy(CFG)
    cfg["media"]["sources"] = ["pixazo"]  # even an accidental config entry must not leak through
    cfg["compliance"]["contains_synthetic_media"] = False
    searched_sources = []
    search = media._candidates_for

    def capture(queries, sources, allowed, min_w):
        searched_sources.append(list(sources))
        return search(queries, sources, allowed, min_w)

    monkeypatch.setattr(media, "_candidates_for", capture)
    patched["q-cat"] = [cand("cat", 11)]
    patched["q-fossil"] = [cand("fossil", 12)]
    patched["q-paw"] = [cand("paw", 13)]
    media.gather(SRC, segs(("a cat", "q-cat"), ("a fossil", "q-fossil"), ("a paw", "q-paw")),
                 cfg, tmp_path, llm=FakeLLM())
    assert searched_sources and all("pixazo" not in sources for sources in searched_sources)


def test_line_without_match_requeries_then_skips_topic(patched, tmp_path):
    patched["q-cat"] = [cand("cat", 1), cand("cat", 2), cand("cat", 3)]
    patched["q-nerve"] = [cand("building", 9)]
    patched["q-retry"] = [cand("street", 8)]
    s = segs(("a cat face", "q-cat"), ("nerve endings fire", "q-nerve"), ("a cat again", "q-cat"))
    llm = FakeLLM(requery={"shows": "nerve", "queries": ["q-retry"]})
    with pytest.raises(media.NoVisualMatch):
        media.gather(SRC, s, CFG, tmp_path, llm=llm)


def test_requery_can_rescue_a_line(patched, tmp_path):
    patched["q-cat"] = [cand("cat", 1), cand("cat", 2)]
    patched["q-nerve"] = [cand("building", 9)]
    patched["q-retry"] = [cand("nerve", 7)]
    patched["q-paw"] = [cand("paw", 3)]
    s = segs(("a cat face", "q-cat"), ("nerve endings fire", "q-nerve"), ("a paw", "q-paw"))
    v = media.gather(SRC, s, CFG, tmp_path, llm=FakeLLM(requery={"shows": "nerve", "queries": ["q-retry"]}))
    assert [sh["credit"]["title"].split()[0] for sh in v["shots"] if sh["seg"] == 1] == ["nerve"]


def test_vision_outage_never_falls_back_to_unverified(patched, tmp_path):
    patched["q-cat"] = [cand("cat", 1)]
    with pytest.raises(vision.VisionUnavailable):
        media.gather(SRC, segs(("a cat", "q-cat")), CFG, tmp_path, llm=FakeLLM(down=True))


def test_no_vision_model_configured_refuses(tmp_path):
    with pytest.raises(vision.VisionUnavailable):
        media.gather(SRC, segs(("a cat", "q")), CFG, tmp_path, llm=None)


def test_repeat_only_if_approved_for_that_line_and_not_adjacent(patched, tmp_path):
    patched["q-cat"] = [cand("cat", 1)]
    patched["q-fossil"] = [cand("fossil", 2)]
    patched["q-paw"] = [cand("paw", 3)]
    s = segs(("a cat", "q-cat"), ("a fossil", "q-fossil"), ("a paw", "q-paw"), ("the cat again", "q-cat"))
    v = media.gather(SRC, s, CFG, tmp_path, llm=FakeLLM())
    last = [sh for sh in v["shots"] if sh["seg"] == 3][0]
    assert last["reused"] and last["credit"]["title"].startswith("cat")


# ── final QA ─────────────────────────────────────────────────────────────────
def _tts(lines):
    t, segs_ = 0.0, []
    for ln in lines:
        words = []
        for w in ln.split():
            words.append({"word": w, "start": t, "end": t + 0.3})
            t += 0.35
        segs_.append({"text": ln, "start": words[0]["start"], "end": t, "words": words})
        t += 0.2
    return {"segments": segs_, "duration": t}


def test_narration_check():
    script = {"segments": [{"text": "Cats have whiskers."}, {"text": "They sense air."}]}
    assert qa.narration_check(_tts(["Cats have whiskers.", "They sense air."]), script) == []
    bad = _tts(["Cats have whiskers.", "Dogs bark loudly at night."])
    assert qa.narration_check(bad, script)
    misplaced = _tts(["Cats have whiskers.", "They sense air."])
    misplaced["segments"][0]["words"][-1]["end"] = 999
    assert any("word timings are malformed" in issue for issue in qa.narration_check(misplaced, script))

    overlapping = _tts(["Cats have whiskers.", "They sense air."])
    overlapping["segments"][0]["words"][1]["start"] = 0.1
    assert any("word timings are malformed" in issue for issue in qa.narration_check(overlapping, script))

    missing_duration = _tts(["Cats have whiskers.", "They sense air."])
    missing_duration.pop("duration")
    assert any("audio duration" in issue for issue in qa.narration_check(missing_duration, script))


def test_coverage_check_rejects_unapproved_or_missing():
    tl = [{"seg": 0, "start": 0, "end": 1, "path": "a"}]
    shots = [{"seg": 0, "path": "a", "score": 9, "judge": "gemini"},
             {"seg": 1, "path": "b", "score": 9, "judge": "clip"}]
    issues = qa.coverage_check(tl, shots, 2, 7)
    assert any("line 2 has no shot" in i for i in issues)
    assert any("wasn't approved" in i for i in issues)


class QALLM:
    last_used = "fake:qa"

    def __init__(self, bad_frames=(), down=False):
        self.bad, self.down = set(bad_frames), down
        self.prompts = []

    def vision_json(self, system, user, images, validate=None):
        if self.down:
            raise RuntimeError("503")
        self.prompts.append((system, user))
        if "SEQUENCE-LEVEL VISUAL VARIETY REVIEW" in user:
            out = {"visual_variety_ok": True, "repetitive_frames": [], "notes": ""}
            validate and validate(out)
            return out
        n = len(re.findall(r"^\s+\d+ \| line", user, re.M))
        out = {"frames": [{"n": i + 1, "shows": "x", "match": (i + 1) not in self.bad,
                           "score": 2 if (i + 1) in self.bad else 9, "issue": ""} for i in range(n)],
               "hook_text_ok": True, "title_ok": True, "notes": ""}
        validate and validate(out)
        return out


def _passing_qa_report(path="a"):
    return {"passed": True, "issues": [], "failed": [], "repairable": False,
            "frames": [{"timeline_index": 0, "seg": 0, "path": path, "t": 0.5,
                        "shows": "verified subject",
                        "sample": 1, "samples": 1, "match": True, "score": 9,
                        "audit_shows": "verified subject", "audit_match": True, "audit_score": 9,
                        "judge": "gemini", "audit_judge": "gemini"}],
            "meta": {"hook_text_ok": True, "title_ok": True, "visual_variety_ok": True,
                     "notes": [], "visual_variety_notes": ""}}


@pytest.fixture
def tiny_video(tmp_path):
    mp4 = tmp_path / "v.mp4"
    subprocess.run([common.ffmpeg_bin(), "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=180x320:rate=10",
                    "-t", "4", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    return mp4


def test_final_qa_passes_and_fails_per_frame(tiny_video):
    lines = ["Cats have whiskers.", "They sense air."]
    tts, script = _tts(lines), {"segments": [{"text": x} for x in lines]}
    tl = [{"seg": 0, "start": 0.0, "end": 1.5, "path": "a"}, {"seg": 1, "start": 1.5, "end": 3.0, "path": "b"}]
    shots = [{"seg": 0, "path": "a", "score": 9, "judge": "g"},
             {"seg": 1, "path": "b", "score": 9, "judge": "g"}]
    ok = qa.verify(tiny_video, tl, shots, tts, script, "T", "HOOK", "Whiskers", QALLM(), CFG)
    assert ok["passed"] and len(ok["frames"]) == 2
    bad = qa.verify(tiny_video, tl, shots, tts, script, "T", "HOOK", "Whiskers", QALLM(bad_frames=[2]), CFG)
    assert not bad["passed"] and bad["failed"] == [(1, "b")]


@pytest.mark.parametrize("defect", ["truthy_match", "nan_score", "missing_hook_verdict", "missing_title_verdict", "wrong_frame_number"])
def test_final_qa_fails_closed_on_malformed_primary_provider_output(tiny_video, monkeypatch, defect):
    monkeypatch.setattr(qa.time, "sleep", lambda _seconds: None)

    class MalformedPrimary(QALLM):
        def vision_json(self, system, user, images, validate=None):
            if "SEQUENCE-LEVEL VISUAL VARIETY REVIEW" in user:
                return {"visual_variety_ok": True, "repetitive_frames": [], "notes": ""}
            n = len(re.findall(r"^\s+\d+ \| line", user, re.M))
            out = {"frames": [{"n": i + 1, "shows": "cat", "match": True, "score": 9, "issue": ""}
                              for i in range(n)],
                   "hook_text_ok": True, "title_ok": True, "notes": ""}
            if defect == "truthy_match":
                out["frames"][0]["match"] = "false"
            elif defect == "nan_score":
                out["frames"][0]["score"] = float("nan")
            elif defect == "missing_hook_verdict":
                out.pop("hook_text_ok")
            elif defect == "missing_title_verdict":
                out.pop("title_ok")
            elif defect == "wrong_frame_number":
                out["frames"][0]["n"] = 0
            return out  # deliberately ignores the validation callback

    lines = ["Cats have whiskers."]
    timeline = [{"seg": 0, "start": 0.0, "end": 2.0, "path": "cat.jpg"}]
    shots = [{"seg": 0, "path": "cat.jpg", "score": 9, "judge": "gemini"}]
    with pytest.raises(vision.VisionUnavailable, match="no valid vision verdict"):
        qa.verify(tiny_video, timeline, shots, _tts(lines), {"segments": [{"text": lines[0]}]},
                  "Cats", "Cats", "Cats", MalformedPrimary(), CFG)


def test_sequence_qa_fails_closed_on_truthy_non_boolean_provider_verdict(tiny_video, monkeypatch):
    monkeypatch.setattr(qa.time, "sleep", lambda _seconds: None)

    class MalformedSequence(QALLM):
        def vision_json(self, system, user, images, validate=None):
            if "SEQUENCE-LEVEL VISUAL VARIETY REVIEW" in user:
                return {"visual_variety_ok": "false", "repetitive_frames": [], "notes": ""}
            return super().vision_json(system, user, images, validate)

    lines = ["Cats have whiskers."]
    timeline = [{"seg": 0, "start": 0.0, "end": 2.0, "path": "cat.jpg"}]
    shots = [{"seg": 0, "path": "cat.jpg", "score": 9, "judge": "gemini"}]
    with pytest.raises(vision.VisionUnavailable, match="sequence-level visual QA unavailable"):
        qa.verify(tiny_video, timeline, shots, _tts(lines), {"segments": [{"text": lines[0]}]},
                  "Cats", "Cats", "Cats", MalformedSequence(), CFG)


def test_coverage_requires_the_same_segment_and_media_path_to_be_approved():
    timeline = [{"seg": 0, "start": 0.0, "end": 2.0, "path": "rendered-final.jpg"}]
    approved = {"seg": 0, "path": "approved-original.jpg", "score": 9, "judge": "gemini"}

    issues = qa.coverage_check(timeline, [approved], 1, 7)

    assert any("rendered-final.jpg" in issue and "without a selected-shot approval" in issue
               for issue in issues)
    approved["path"] = "rendered-final.jpg"
    assert qa.coverage_check(timeline, [approved], 1, 7) == []


def test_coverage_gate_rejects_nan_scores_and_malformed_times():
    timeline = [{"seg": 0, "start": 0.0, "end": 1.0, "path": "cat.jpg"}]
    shot = {"seg": 0, "path": "cat.jpg", "score": float("nan"), "judge": "gemini"}
    issues = qa.coverage_check(timeline, [shot], 1, 7)
    assert any("wasn't approved" in issue for issue in issues)
    timeline[0]["end"] = float("nan")
    assert any("invalid time bounds" in issue for issue in qa.coverage_check(
        timeline, [{"seg": 0, "path": "cat.jpg", "score": 9, "judge": "g"}], 1, 7))


def test_sequence_level_variety_gate_marks_redundant_shots_for_repair(tiny_video):
    class RepetitiveSequence(QALLM):
        def vision_json(self, system, user, images, validate=None):
            if "SEQUENCE-LEVEL VISUAL VARIETY REVIEW" in user:
                self.prompts.append((system, user))
                out = {"visual_variety_ok": False, "repetitive_frames": [2],
                       "notes": "the second frame repeats the same static aerial composition"}
                validate and validate(out)
                return out
            return super().vision_json(system, user, images, validate)

    lines = ["The lake fills an ancient crater.", "A second view shows the crater rim."]
    tts, script = _tts(lines), {"segments": [{"text": x} for x in lines]}
    timeline = [{"seg": 0, "start": 0.0, "end": 1.5, "path": "first-aerial.jpg"},
                {"seg": 1, "start": 1.5, "end": 3.0, "path": "second-aerial.jpg"}]
    shots = [{"seg": 0, "path": "first-aerial.jpg", "score": 9, "judge": "g", "kind": "image"},
             {"seg": 1, "path": "second-aerial.jpg", "score": 9, "judge": "g", "kind": "image"}]

    result = qa.verify(tiny_video, timeline, shots, tts, script, "The crater lake", "A lake in a crater",
                       "crater lake", RepetitiveSequence(), CFG)

    assert not result["passed"] and result["repairable"]
    assert result["failed"] == [(1, "second-aerial.jpg")]
    assert "sequence-level visual review" in result["issues"][0]
    assert result["meta"]["visual_variety_ok"] is False


def test_render_pipeline_passes_trend_angle_and_linked_evidence_to_final_qa(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from autotube import pipeline

    captured = {}
    monkeypatch.setattr(pipeline.render, "render", lambda *args, **kwargs: {
        "timeline": [{"seg": 0, "start": 0.0, "end": 2.0, "path": "a"}]})

    def fake_verify(*args, **kwargs):
        captured.update(kwargs)
        return _passing_qa_report()

    monkeypatch.setattr(pipeline.qa, "verify", fake_verify)
    angle = "the animal uses the reef to ambush prey"
    url = "https://youtube.test/viral-short"
    topic = {
        "angle": angle,
        "context": ["viral Short now: 2,000,000 views in 20h"],
        "signals": [{"source": "youtube_outliers", "source_url": url, "captured_at": "2026-10-10T12:00:00Z",
                     "evidence": {"views": 2_000_000, "hours": 20, "vph": 100_000, "breakout": 4.0}}],
    }
    assets = {"tts": {}, "visuals": {"shots": [{"seg": 0, "path": "a", "score": 9, "judge": "gemini"}]},
              "music": {}, "hook_card": "Seal ambushes its prey",
              "script": {"segments": [{"text": "A seal ambushes prey."}]},
              "title": "A seal's hunting trick", "subject": "seal",
              "work": tmp_path, "out": tmp_path / "render.mp4", "seed": 1, "fx": {}, "topic": topic}
    cfg = {"qa": {"max_repairs": 0}, "channel": {"name": "Archive 13"}}

    pipeline.render_with_qa(assets, cfg, SimpleNamespace(llm=object()))

    assert captured["trend_angle"] == angle
    assert "2,000,000 views in 20h" in captured["trend_evidence"][0]
    assert any(url in line for line in captured["trend_evidence"])


def test_moving_footage_is_sampled_near_start_middle_and_end(monkeypatch):
    lines = ["A seal swims past the reef while divers watch."]
    tts, script = _tts(lines), {"segments": [{"text": lines[0]}]}
    timeline = [{"seg": 0, "start": 0.0, "end": 3.5, "path": "clip.mp4"}]
    shots = [{"seg": 0, "path": "clip.mp4", "score": 9, "judge": "gemini", "kind": "video",
              "want": "a seal swimming over a reef", "shows": "seal underwater",
              "credit": {"title": "Seal in reef footage", "source": "archive", "page": "https://example.test/clip"}}]
    samples = []
    monkeypatch.setattr(qa, "_frame_at", lambda mp4, t: (samples.append(t), Image.new("RGB", (10, 10)))[1])

    llm = QALLM(bad_frames=[2])
    result = qa.verify(Path("unused.mp4"), timeline, shots, tts, script, "Seal", "Seal reef",
                       "Seal", llm, CFG, trend_angle="the animal uses the reef to ambush prey")

    assert samples == pytest.approx([0.7, 1.75, 2.8])
    assert [f["sample"] for f in result["frames"]] == [1, 2, 3]
    assert not result["passed"] and result["repairable"]
    assert result["failed"] == [(0, "clip.mp4")], "three failed samples from one clip create one repair request"
    assert "CURRENT TREND ANGLE" in llm.prompts[0][1]


def test_final_qa_rejects_false_positive_and_supplies_shot_context(tiny_video):
    class FirstPassFalsePositive(QALLM):
        def vision_json(self, system, user, images, validate=None):
            self.prompts.append((system, user))
            if "SEQUENCE-LEVEL VISUAL VARIETY REVIEW" in user:
                out = {"visual_variety_ok": True, "repetitive_frames": [], "notes": ""}
                validate and validate(out)
                return out
            n = len(re.findall(r"^\s+\d+ \| line", user, re.M))
            adversarial = "ADVERSARIAL SECOND LOOK" in user
            out = {"frames": [{"n": i + 1, "shows": "modern climate summit", "match": not adversarial,
                               "score": 10 if not adversarial else 2,
                               "issue": "modern meeting, not the historical treaty negotiation" if adversarial else ""}
                              for i in range(n)],
                   "hook_text_ok": True, "title_ok": True, "notes": ""}
            validate and validate(out)
            return out

    line = "The 1905 Treaty of Portsmouth ended the Russo-Japanese War."
    tts = _tts([line])
    script = {"segments": [{"text": line}]}
    timeline = [{"seg": 0, "start": 0.0, "end": 2.0, "path": "modern-meeting.jpg"}]
    shots = [{"seg": 0, "path": "modern-meeting.jpg", "score": 9, "judge": "gemini",
              "want": "1905 historic treaty negotiation room",
              "shows": "modern leaders at a climate summit",
              "credit": {"title": "Oxford climate change meeting 2024",
                         "source": "news archive", "page": "https://example.test/2024-meeting"}}]
    llm = FirstPassFalsePositive()

    result = qa.verify(tiny_video, timeline, shots, tts, script, "The 1905 Treaty", "Treaty of Portsmouth",
                       "Treaty of Portsmouth", llm, CFG,
                       trend_angle="the actual 1905 treaty negotiations, not a modern summit",
                       trend_evidence=["recent history spike; verified trend record"])

    assert len(llm.prompts) == 3, "a positive frame gets an adversarial look, then the full sequence is checked for variety"
    assert "CURRENT TREND ANGLE" in llm.prompts[0][1]
    assert 'intended shot: "1905 historic treaty negotiation room"' in llm.prompts[0][1]
    assert 'asset title: "Oxford climate change meeting 2024"' in llm.prompts[0][1]
    assert "ADVERSARIAL SECOND LOOK" in llm.prompts[1][1]
    assert not result["passed"] and result["failed"] == [(0, "modern-meeting.jpg")]
    assert "adversarial audit" in result["issues"][0]


def test_final_qa_outage_raises(tiny_video, monkeypatch):
    monkeypatch.setattr(qa.time, "sleep", lambda s: None)
    lines = ["Cats have whiskers."]
    with pytest.raises(vision.VisionUnavailable):
        qa.verify(tiny_video, [{"seg": 0, "start": 0, "end": 2, "path": "a"}], [{"seg": 0, "path": "a", "score": 9, "judge": "g"}],
                  _tts(lines), {"segments": [{"text": lines[0]}]}, "T", "H", "S", QALLM(down=True), CFG)


def test_adversarial_audit_outage_fails_closed(tiny_video, monkeypatch):
    monkeypatch.setattr(qa.time, "sleep", lambda s: None)

    class AuditDown(QALLM):
        def vision_json(self, system, user, images, validate=None):
            if "ADVERSARIAL SECOND LOOK" in user:
                raise RuntimeError("audit provider unavailable")
            return super().vision_json(system, user, images, validate)

    lines = ["Cats have whiskers."]
    timeline = [{"seg": 0, "start": 0, "end": 2, "path": "cat.jpg"}]
    shots = [{"seg": 0, "path": "cat.jpg", "score": 9, "judge": "g", "want": "a cat",
              "shows": "a cat", "credit": {"title": "cat", "source": "archive"}}]
    with pytest.raises(vision.VisionUnavailable, match="adversarial final-frame audit unavailable"):
        qa.verify(tiny_video, timeline, shots, _tts(lines), {"segments": [{"text": lines[0]}]},
                  "Cats", "Cats", "Cats", AuditDown(), CFG)


def test_script_fact_check_outage_fails_closed(monkeypatch):
    """If the fact-checking LLM is unavailable the script is NOT approved (and not rewritten)."""
    from autotube import scriptwriter
    from autotube.llm import LLMError
    cfg = common.load_config()

    class DownLLM:
        def json(self, *a, **k):
            raise LLMError("all providers down")

    sw = scriptwriter.ScriptWriter.__new__(scriptwriter.ScriptWriter)
    sw.cfg, sw.llm, sw.src_chars = cfg, DownLLM(), 4000
    src = {"title": "Honey", "text": "Honey is a sweet substance made by bees. " * 20}
    ev = "Honey is a sweet substance made by bees."
    # gate-clean script: the only reason it must fail is that the fact-checker is unreachable
    script = {"title": "Honey facts", "segments": [
        {"text": "Bees visit two million flowers to fill a single jar", "evidence": ev},
        {"text": "A worker bee makes one twelfth of a teaspoon in life", "evidence": ev},
        # aside on a BODY line, never the hook: asides count toward hook length, and
        # content.min_asides requires the narration to have a voice of its own
        {"text": "The hive fans its wings to dry the nectar down", "evidence": ev,
         "aside": "Busy little things."},
        {"text": "so the jar in your cupboard will never spoil", "evidence": ev}]}
    monkeypatch.setitem(cfg["content"], "require_grounding", False)
    ok, review = sw.check(script, src)
    assert ok is False and review.get("unavailable")


def test_only_scheduled_videos_hold_publish_slots():
    from datetime import timedelta
    from autotube import pipeline
    fut = (common.now_utc() + timedelta(days=1)).replace(second=0, microsecond=0)
    hist = [{"video_id": v, "publish_at": (fut + timedelta(hours=i)).isoformat(), "status": st}
            for i, (v, st) in enumerate([("a", "scheduled"), ("b", "withdrawn"), ("c", "missing"),
                                          ("d", "rejected"), ("e", "upload_failed: x")])]
    assert pipeline.taken_slots(hist) == {pipeline._slot_key(fut)}


def test_approved_but_unusable_image_triggers_requery(patched, tmp_path, monkeypatch):
    """Regression (run 36506122073): a line whose approved image couldn't be used was dropped without a re-search."""
    patched["q-cat"] = [cand("cat", 1), cand("cat", 2)]
    patched["q-nerve"] = [cand("nerve", 9)]            # approved 9/10 … but its download will fail
    patched["q-retry"] = [cand("nerve", 7)]
    patched["q-paw"] = [cand("paw", 3)]
    ok_usable = media._usable
    monkeypatch.setattr(media, "_usable", lambda c, w, *a: None if c["url"].endswith("nerve/9.jpg") else ok_usable(c, w, *a))
    s = segs(("a cat face", "q-cat"), ("nerve endings fire", "q-nerve"), ("a paw", "q-paw"))
    v = media.gather(SRC, s, CFG, tmp_path, llm=FakeLLM(requery={"shows": "nerve", "queries": ["q-retry"]}))
    assert [sh["credit"]["url"] for sh in v["shots"] if sh["seg"] == 1] == ["https://img/nerve/7.jpg"]


def test_black_and_white_photo_is_not_a_document():
    """Regression: the colour-count rule threw away every B&W photo/engraving after the judge approved it."""
    import random
    rnd = random.Random(3)
    im = Image.new("L", (400, 300))
    im.putdata([min(255, max(0, int(40 + 170 * (x / 400) + rnd.gauss(0, 25)))) for y in range(300) for x in range(400)])
    assert not media._looks_like_document(im.convert("RGB"))


def test_vision_approved_image_skips_document_heuristic(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(media, "_download", lambda a, d, doc_filter=True: seen.setdefault("f", doc_filter) and None)
    media._usable({"url": "u", "vscore": 9}, tmp_path)
    assert seen["f"] is False


def test_sign_off_line_detection():
    for t in ["Follow for more science facts", "Subscribe for more!", "Like and follow for part two"]:
        assert media.CTA_RE.search(t)
    for t in ["Follow the money trail to 1920", "The ship sank in 1912", "Bees like flowers"]:
        assert not media.CTA_RE.search(t)
