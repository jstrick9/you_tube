"""9:16 fitness: what survives the full-bleed cover-crop (AUDIT 3, render follow-up)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import media  # noqa: E402


# ── the measurement ───────────────────────────────────────────────────────────
def test_crop_px_counts_pixels_that_survive_the_crop_not_raw_width():
    """A 1920-wide clip sounds like plenty and is not: cover-cropping to 9:16 keeps 608 of it."""
    assert media.crop_px(1920, 1080) == 607
    assert media.crop_px(1280, 720) == 405
    assert media.crop_px(1080, 1920) == 1080        # portrait: nothing is thrown away
    assert media.crop_px(3840, 2160) == 1215        # 4K landscape still clears 1080


def test_crop_px_is_safe_on_missing_or_junk_dimensions():
    for w, h in [(None, None), (0, 0), (-5, 10), ("x", "y"), (100, None)]:
        assert media.crop_px(w, h) == 0


def test_crop_quality_is_the_fraction_of_the_frame_width_available():
    assert media.crop_quality(1080, 1920) == 1.0
    assert 0.55 < media.crop_quality(1920, 1080) < 0.57
    assert media.crop_quality(0, 0) == 0


# ── the ranking nudge ─────────────────────────────────────────────────────────
def test_assets_that_fill_the_frame_are_rewarded_and_soft_ones_penalised():
    assert media.shape_bonus({"w": 1080, "h": 1920}) == 1.0
    assert media.shape_bonus({"w": 3840, "h": 2160}) == 1.0
    assert media.shape_bonus({"w": 1920, "h": 1080}) < 0
    assert media.shape_bonus({"w": 1280, "h": 720}) < media.shape_bonus({"w": 1920, "h": 1080})


def test_unknown_dimensions_get_no_opinion_either_way():
    """Commons and NASA often omit sizes; they must not be silently penalised into last place."""
    assert media.shape_bonus({}) == 0.0
    assert media.shape_bonus({"w": None, "h": None}) == 0.0


def test_the_bonus_cannot_outweigh_relevance():
    """A sharp picture of the wrong thing is worse than a soft picture of the right one."""
    weight = 1.0
    best_shape = max(media.shape_bonus({"w": 1080, "h": 1920}, weight),
                     media.shape_bonus({"w": 3840, "h": 2160}, weight))
    worst_shape = media.shape_bonus({"w": 1280, "h": 720}, weight)
    assert (best_shape - worst_shape) < 2.0, "shape must not span more than the judge's own scale"
    # a candidate two points better on relevance still wins despite the worst possible shape
    assert 9.0 + worst_shape > 7.0 + best_shape


def test_bonus_can_be_disabled():
    assert media.shape_bonus({"w": 1280, "h": 720}, 0.0) == 0.0


# ── video rendition choice ────────────────────────────────────────────────────
def test_picks_the_cheapest_rendition_that_still_fills_the_frame():
    files = [{"link": "sd", "width": 1280, "height": 720},
             {"link": "hd", "width": 1920, "height": 1080},
             {"link": "4k", "width": 3840, "height": 2160},
             {"link": "vert", "width": 1080, "height": 1920}]
    assert media._best_video_file(files)["link"] == "vert"


def test_prefers_4k_over_a_soft_landscape_when_no_portrait_exists():
    files = [{"link": "sd", "width": 1280, "height": 720},
             {"link": "4k", "width": 3840, "height": 2160}]
    assert media._best_video_file(files)["link"] == "4k"


def test_falls_back_to_the_sharpest_when_nothing_clears_the_bar():
    """The old rule took the SMALLEST file, which is the worst choice once shots are cover-cropped."""
    files = [{"link": "tiny", "width": 1280, "height": 720},
             {"link": "better", "width": 1920, "height": 1080}]
    assert media._best_video_file(files)["link"] == "better"


# ── pool widening ─────────────────────────────────────────────────────────────
def _fake_photos(n, start=0):
    return {"photos": [{"src": {"large2x": f"u{i}", "medium": f"m{i}"}, "url": f"p{i}", "alt": "a",
                        "photographer": "x", "width": 1080, "height": 1920}
                       for i in range(start, start + n)]}


def test_photo_search_queries_both_orientations_and_dedupes(monkeypatch):
    """Hard-filtering to portrait starved the pool; both orientations are fetched now."""
    calls = []

    def fake(url, params=None, headers=None, **k):
        calls.append(params.get("orientation"))
        return _fake_photos(3, start=0 if params.get("orientation") else 2)

    monkeypatch.setenv("PEXELS_API_KEY", "k")
    monkeypatch.setattr(media, "get_json", fake)
    out = media.pexels_search("otter", 3)
    assert calls == ["portrait", None], "must ask for portrait AND unrestricted"
    assert len(out) == 5, "overlapping results must be deduped by url"
    assert len({a["url"] for a in out}) == 5


def test_photo_search_survives_one_orientation_failing(monkeypatch):
    def fake(url, params=None, headers=None, **k):
        if params.get("orientation"):
            raise RuntimeError("pexels 500")
        return _fake_photos(2)

    monkeypatch.setenv("PEXELS_API_KEY", "k")
    monkeypatch.setattr(media, "get_json", fake)
    assert len(media.pexels_search("otter")) == 2


def test_no_api_key_means_no_calls(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    assert media.pexels_search("otter") == []
    assert media.pexels_video_search("otter") == []
