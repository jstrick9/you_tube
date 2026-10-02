"""Moving footage: real video clips (Commons / NASA / Pexels) cut around the frame the vision judge approved,
2.5D 'living photo' motion for stills, and the memory-flat piecewise video track."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import media, render  # noqa: E402
from autotube.common import ffmpeg_bin, load_config  # noqa: E402

CFG = load_config()


def test_seek_thumb_rewrites_commons_video_thumbnail():
    u = "https://thumb.wikimedia.org/wikipedia/commons/thumb/9/9b/Hairy_octopus.webm/500px--Hairy_octopus.webm.jpg?x=1"
    assert media._seek_thumb(u, 7.5) == ("https://thumb.wikimedia.org/wikipedia/commons/thumb/9/9b/Hairy_octopus.webm/"
                                         "500px-seek=7.5-Hairy_octopus.webm.jpg?x=1")


def test_pick_derivative_prefers_720p_transcode():
    vi = {"url": "orig.webm", "derivatives": [
        {"src": "a.240p.webm", "height": 240, "type": "video/webm"},
        {"src": "a.480p.webm", "height": 480, "type": "video/webm"},
        {"src": "a.720p.webm", "height": 720, "type": "video/webm"},
        {"src": "a.1080p.webm", "height": 1080, "type": "video/webm"}]}
    assert media._pick_derivative(vi) == "a.720p.webm"
    assert media._pick_derivative({"url": "orig.webm", "derivatives": []}) == "orig.webm"


def test_commons_video_search_filters_license_and_length(monkeypatch):
    def fake(url, params=None, **k):
        assert params["gsrsearch"].startswith("filetype:video")
        mk = lambda t, lic, dur, w=1280: {"title": f"File:{t}", "videoinfo": [{  # noqa: E731
            "mime": "video/webm", "duration": dur, "width": w, "height": 720, "url": f"https://u/{t}",
            "thumburl": f"https://thumb/x/500px--{t}.jpg", "descriptionurl": f"https://c/{t}",
            "extmetadata": {"LicenseShortName": {"value": lic}, "Artist": {"value": "<b>Ann</b>"}},
            "derivatives": []}]}
        return {"query": {"pages": {"1": mk("good.webm", "CC BY 4.0", 20), "2": mk("nc.webm", "CC BY-NC 4.0", 20),
                                    "3": mk("short.webm", "CC0", 1.5), "4": mk("tiny.webm", "CC0", 20, w=320)}}}
    monkeypatch.setattr(media, "get_json", fake)
    out = media.commons_video_search("octopus", CFG["compliance"]["allowed_licenses"])
    assert [o["title"] for o in out] == ["good.webm (video)"]
    o = out[0]
    assert o["kind"] == "video" and o["author"] == "Ann" and "seek=8.0" in o["thumb"] and o["t_ref"] == 8.0


def test_nasa_video_search_parses_items(monkeypatch):
    monkeypatch.setattr(media, "get_json", lambda url, params=None, **k: {"collection": {"items": [
        {"data": [{"nasa_id": "GSFC Rings", "title": "Saturn's Rings", "center": "GSFC"}],
         "links": [{"href": "https://images-assets.nasa.gov/video/GSFC Rings/GSFC Rings~thumb.jpg", "render": "image"}]}]}})
    o = media.nasa_video_search("saturn")[0]
    assert o["kind"] == "video" and o["license"].startswith("Public domain") and o["author"] == "NASA/GSFC"
    assert o["variants"][0].endswith("GSFC%20Rings~medium.mp4") and " " not in o["thumb"]


def _samples(scenes):
    """Fake 4-fps grayscale samples: list of (n_samples, base_level, pattern_seed)."""
    rows = []
    for n, level, seed in scenes:
        rng = np.random.default_rng(seed)
        pat = rng.integers(0, 120, 576)
        drift = rng.random(576) * 2.0          # the picture slowly changes inside a scene
        for k in range(n):
            rows.append(np.clip(pat + level + k * drift, 0, 255))
    return np.array(rows, dtype=float)


def test_choose_window_stays_inside_one_scene_around_the_approved_frame():
    s = _samples([(20, 40, 1), (40, 90, 2), (20, 10, 3)])       # cuts at 5 s and 15 s
    ref = s[35]                                                   # judge approved a frame from scene 2
    start, end, speed = media.choose_window(s, 4.0, 4.0, ref=ref)
    assert 5.0 <= start and end <= 15.0 and speed == 1.0 and abs((start + end) / 2 - 35 / 4) < 0.6


def test_choose_window_rejects_clips_without_the_approved_frame():
    s = _samples([(40, 40, 1)])
    other = np.random.default_rng(99).integers(0, 255, 576).astype(float)
    assert media.choose_window(s, 4.0, 4.0, ref=other) is None


def test_choose_window_short_scene_uses_gentle_slow_motion_or_rejects():
    s = _samples([(20, 40, 1), (14, 90, 2), (20, 10, 3)])       # middle scene ~3.5 s
    w = media.choose_window(s, 4.0, 4.0, ref=s[27])
    assert w and w[2] < 1.0 and w[1] - w[0] >= 4.0 / 1.6
    s2 = _samples([(20, 40, 1), (6, 90, 2), (20, 10, 3)])        # 1.5 s scene: too short
    assert media.choose_window(s2, 4.0, 4.0, ref=s2[23]) is None


def test_config_enables_moving_footage():
    m = CFG["media"]
    assert m["sources"][:2] == ["commons_video", "nasa_video"] and m["video_bonus"] > 0 and m["max_video_mb"] <= 100
    assert CFG["video"]["animate_stills"] is True


def _clip(path: Path, seconds: float = 3.0, size=(320, 180)):
    subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc=size={size[0]}x{size[1]}:rate=30:duration={seconds}", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


def test_video_track_is_frame_exact_with_clips_and_stills(tmp_path):
    W, H, fps = 108, 192, 30
    frames, clip_aspect, clip_src = [], {}, {}
    for i in range(3):
        p = tmp_path / f"f{i}.jpg"
        im = Image.new("RGB", (W * 2, H * 2), (60 * i, 90, 150))
        ImageDraw.Draw(im).ellipse((40, 40, 160, 160), fill=(250, 250, 250))
        im.save(p)
        frames.append(p)
    clip_src[1] = _clip(tmp_path / "c.mp4", 1.0)                 # shorter than its shot → looped
    clip_aspect[1] = 16 / 9
    durs = [1.2, 1.5, 1.1]
    total = sum(durs)
    track = render.build_video_track(frames, clip_aspect, clip_src, durs, ["in", "left", "out"],
                                     ["fade", "fade", "fade"], total, W, H, fps, tmp_path)
    out = subprocess.run([ffmpeg_bin(), "-i", str(track), "-map", "0:v", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    import re
    n = int(re.findall(r"frame=\s*(\d+)", out)[-1])
    assert n == round(total * fps)


def test_video_chain_layouts():
    # full-bleed: ordinary landscape AND portrait clips both cover the whole 9:16 frame — no letterbox
    for aspect in (16 / 9, 4 / 3, 1.0, 9 / 16):
        ch = render.video_chain(0, aspect, 1080, 1920, 30)
        assert "overlay" not in ch and "boxblur" not in ch, aspect
        assert "crop=1080:1920" in ch, aspect
    # only an extreme panorama keeps the inset-over-blurred-bed layout
    pano = render.video_chain(0, 4.0, 1080, 1920, 30)
    assert "overlay" in pano and "boxblur" in pano


def test_compose_frame_is_full_bleed(tmp_path):
    from PIL import Image
    src = tmp_path / "wide.jpg"
    Image.new("RGB", (1600, 900), (20, 120, 200)).save(src)
    out = render.compose_frame(src, tmp_path / "out.jpg", 1080, 1920)
    with Image.open(out) as im:
        assert im.size == (int(1080 * render.SS), int(1920 * render.SS))
        # every pixel comes from the photo: no blurred/darkened bars top or bottom
        top = im.crop((0, 0, im.width, 40)).convert("L").resize((1, 1)).getpixel((0, 0))
        mid = im.crop((0, im.height // 2 - 20, im.width, im.height // 2 + 20)).convert("L").resize((1, 1)).getpixel((0, 0))
        assert abs(top - mid) < 12


def test_saliency_offset_finds_the_subject(tmp_path):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1200, 600), (10, 10, 10))
    d = ImageDraw.Draw(im)
    for x in range(60, 300, 12):                      # dense detail on the LEFT
        d.line((x, 80, x, 520), fill=(255, 255, 255), width=4)
    x, _ = render.saliency_offset(im, 600, 600)
    assert x < 300, f"crop window should follow the detail, got x={x}"


def test_is_video():
    assert render.is_video({"path": "x/clip_1.mp4"}) and render.is_video({"path": "a.jpg", "kind": "video"})
    assert not render.is_video({"path": "a.jpg"})


def test_living_photo_motion(tmp_path):
    from autotube import motion
    if not motion.available():
        pytest.skip("depth model / onnxruntime not available")
    p = tmp_path / "p.jpg"
    im = Image.new("RGB", (320, 200), (90, 140, 200))
    ImageDraw.Draw(im).rectangle((120, 60, 200, 200), fill=(200, 80, 40))
    im.save(p)
    out = motion.animate_still(p, tmp_path / "a.mp4", 1.0, fps=30, move="drift_left")
    assert media._probe_duration(out) >= 0.95
