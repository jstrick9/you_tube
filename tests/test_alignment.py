"""Guarantees of the strict alignment pipeline (no network, no API keys, no CLIP).

A fake vision model approves an image for a line only when the image's hidden "truth" word appears in that line —
so these tests prove the plumbing never lets an unapproved / mismatched image through.
"""
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
        out = {"images": [{"n": int(n), "shows": t, "score": 9 if t in line else 2} for n, t in titles]}
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

    def usable(c, work):
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


def test_coverage_check_rejects_unapproved_or_missing():
    tl = [{"seg": 0, "start": 0, "end": 1, "path": "a"}]
    shots = [{"seg": 0, "score": 9, "judge": "gemini"}, {"seg": 1, "score": 9, "judge": "clip"}]
    issues = qa.coverage_check(tl, shots, 2, 7)
    assert any("line 2 has no shot" in i for i in issues)
    assert any("wasn't approved" in i for i in issues)


class QALLM:
    last_used = "fake:qa"

    def __init__(self, bad_frames=(), down=False):
        self.bad, self.down = set(bad_frames), down

    def vision_json(self, system, user, images, validate=None):
        if self.down:
            raise RuntimeError("503")
        n = len(re.findall(r"^\s+\d+ \| line", user, re.M))
        out = {"frames": [{"n": i + 1, "shows": "x", "match": (i + 1) not in self.bad,
                           "score": 2 if (i + 1) in self.bad else 9, "issue": ""} for i in range(n)],
               "hook_text_ok": True, "title_ok": True}
        validate and validate(out)
        return out


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
    shots = [{"seg": 0, "score": 9, "judge": "g"}, {"seg": 1, "score": 9, "judge": "g"}]
    ok = qa.verify(tiny_video, tl, shots, tts, script, "T", "HOOK", "Whiskers", QALLM(), CFG)
    assert ok["passed"] and len(ok["frames"]) == 2
    bad = qa.verify(tiny_video, tl, shots, tts, script, "T", "HOOK", "Whiskers", QALLM(bad_frames=[2]), CFG)
    assert not bad["passed"] and bad["failed"] == [(1, "b")]


def test_final_qa_outage_raises(tiny_video, monkeypatch):
    monkeypatch.setattr(qa.time, "sleep", lambda s: None)
    lines = ["Cats have whiskers."]
    with pytest.raises(vision.VisionUnavailable):
        qa.verify(tiny_video, [{"seg": 0, "start": 0, "end": 2, "path": "a"}], [{"seg": 0, "score": 9, "judge": "g"}],
                  _tts(lines), {"segments": [{"text": lines[0]}]}, "T", "H", "S", QALLM(down=True), CFG)


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
    script = {"title": "Honey facts", "segments": [{"text": "Bees make honey.", "evidence": ev}] * 4}
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
    monkeypatch.setattr(media, "_usable", lambda c, w: None if c["url"].endswith("nerve/9.jpg") else ok_usable(c, w))
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
