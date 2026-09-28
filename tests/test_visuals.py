"""Tests for the visual relevance pipeline (no network, no CLIP, no API keys needed)."""
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import render, vision  # noqa: E402


class FakeLLM:
    """Pretends to be a vision model: scores tile n by a fixed table."""
    last_used = "fake:vision"

    def __init__(self, scores, fail=False):
        self.scores, self.fail, self.calls = scores, fail, []

    def vision_available(self):
        return True

    def vision_json(self, system, user, images, validate=None):
        self.calls.append(user)
        if self.fail:
            raise RuntimeError("quota")
        out = {"images": [{"n": i + 1, "shows": f"thing {i + 1}", "score": s} for i, s in enumerate(self.scores)]}
        if validate:
            validate(out)
        return out


def _cands(n):
    cs = []
    for i in range(n):
        im = Image.new("RGB", (300, 200), (i * 40 % 255, 100, 200 - i * 20))
        cs.append({"url": f"https://x/{i}.jpg", "title": f"img {i}", "_img": im})
    return cs


def test_llm_scores_are_applied_and_sorted():
    j = vision.Judge({"media": {"clip": False}}, FakeLLM([2, 9, 7.5]))
    out = j.score(_cands(3), "a cat's whiskers", "Cats have whiskers.", "Whiskers")
    assert [c["title"] for c in out] == ["img 1", "img 2", "img 0"]
    assert out[0]["vscore"] == 9 and out[0]["vjudge"] == "fake:vision"
    assert "Cats have whiskers." in j.llm.calls[0]


def test_no_judge_means_no_verdict():
    import pytest
    j = vision.Judge({"media": {"clip": False}}, FakeLLM([9, 9], fail=True))       # strict (default)
    j.retry_pauses = []
    with pytest.raises(vision.VisionUnavailable):
        j.score(_cands(2), "x", "y", "z")
    j = vision.Judge({"media": {"clip": False, "require_llm_verdict": False}}, FakeLLM([9, 9], fail=True))
    j.retry_pauses = []
    assert j.score(_cands(2), "x", "y", "z") == []      # non-strict, no CLIP → still nothing "verified"


def test_contact_sheet_is_jpeg():
    b = vision.contact_sheet([c["_img"] for c in _cands(5)])
    assert b[:2] == b"\xff\xd8"


def test_thumb_url_for_commons_original():
    u = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Cat_face.jpg"
    assert vision.thumb_url({"url": u}) == \
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/Cat_face.jpg/500px-Cat_face.jpg"
    assert vision.thumb_url({"url": u, "thumb": "T"}) == "T"


def test_same_image_hash():
    a = _cands(1)[0]["_img"]
    assert vision.same_image(vision.ahash(a), vision.ahash(a.resize((150, 100))))


def test_shot_plan_multi_and_legacy():
    shots = [{"path": "a", "seg": 0}, {"path": "b", "seg": 1}, {"path": "c", "seg": 1}]
    plan = render.shot_plan(shots, 3)
    assert [s["seg"] for s in plan] == [0, 1, 1, 2]           # seg 2 filled by reuse
    legacy = render.shot_plan([{"path": "a"}, {"path": "b"}], 2)
    assert [s["seg"] for s in legacy] == [0, 1]


def test_writer_validation_requires_visuals():
    from autotube.scriptwriter import ScriptWriter
    import inspect
    src = inspect.getsource(ScriptWriter.write)
    assert '"visual"' in src and "queries" in src


def test_near_duplicate():
    assert vision.near_duplicate([1.0, 0.0], [0.99, 0.141])
    assert not vision.near_duplicate([1.0, 0.0], [0.0, 1.0])
    assert not vision.near_duplicate(None, [1.0])


def test_near_duplicate_same_title():
    a, b = [1.0, 0.0], [0.87, 0.493]            # cos 0.87: similar but not identical pixels
    assert vision.near_duplicate(a, b, t1="Tardigrade", t2="A Tardigrade")
    assert not vision.near_duplicate(a, b, t1="Cat face", t2="Tabby kitten")
