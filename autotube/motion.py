"""Turn a verified still photo into moving footage: a 2.5D "living photo" camera move.

A small monocular depth model (Depth Anything V2 Small, Apache-2.0, quantized ONNX, ~27 MB, CPU) estimates how
near each pixel is; each frame is then re-sampled so near things move more than far things while the camera
slowly dollies or drifts. The result looks like real camera movement instead of a flat Ken Burns zoom, and
nothing is invented: every pixel still comes from the same approved photo (so the vision check on that photo, and
the final QA on the rendered frames, still apply unchanged). No AI-generated content → no synthetic-media issue.

If the model or onnxruntime is unavailable, `available()` is False and the renderer keeps its classic Ken Burns
motion — the pipeline never fails because of this module.
"""
from __future__ import annotations

import logging
import os
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .common import ffmpeg_bin, http

log = logging.getLogger("autotube.motion")

MODEL_URL = ("https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/"
             "model_quantized.onnx")
MODEL_PATH = Path(os.environ.get("AUTOTUBE_DEPTH_MODEL",
                                 Path.home() / ".cache" / "autotube" / "depth-anything-v2-small-q.onnx"))
MOVES = ("dolly_in", "dolly_out", "drift_left", "drift_right", "rise")

_session = None
_failed = False


def _get_session():
    global _session, _failed
    if _session is not None or _failed:
        return _session
    try:
        import onnxruntime as ort
        if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading depth model (one-time, ~27 MB)…")
            r = http().get(MODEL_URL, timeout=300)
            r.raise_for_status()
            tmp = MODEL_PATH.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(MODEL_PATH)
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2))
        _session = ort.InferenceSession(str(MODEL_PATH), so, providers=["CPUExecutionProvider"])
    except Exception as e:  # noqa: BLE001
        log.warning("2.5D motion unavailable (%s) — using Ken Burns motion", str(e)[:200])
        _failed = True
        _session = None
    return _session


def available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return _get_session() is not None


def depth_map(rgb: np.ndarray) -> np.ndarray:
    """Relative nearness in [0, 1] (1 = closest), same size as the image, smoothed to avoid tearing."""
    import cv2
    sess = _get_session()
    h, w = rgb.shape[:2]
    side = 518                                     # multiple of 14 (ViT patch size)
    x = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)[None]
    name = sess.get_inputs()[0].name
    d = np.squeeze(sess.run(None, {name: x})[0]).astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    k = max(3, int(min(w, h) * 0.03) | 1)
    return cv2.GaussianBlur(d, (k, k), 0)


def _ease(t: np.ndarray | float):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def animate_still(img_path: Path, out: Path, dur: float, fps: int = 30, seed: int = 0, max_w: int = 1080,
                  strength: float = 1.0, move: str | None = None) -> Path:
    """Render `dur` seconds of 2.5D camera motion from one photo into an H.264 clip (no audio)."""
    import cv2
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, max(2, int(im.height * max_w / im.width))), Image.LANCZOS)
        rgb = np.asarray(im)
    h, w = rgb.shape[:2]
    h, w = h // 2 * 2, w // 2 * 2
    rgb = np.ascontiguousarray(rgb[:h, :w])
    depth = depth_map(rgb)
    rng = random.Random(seed)
    move = move or rng.choice(MOVES)
    n = max(2, int(round(dur * fps)))
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    base_zoom = 1.07                                   # hides the borders that parallax would reveal
    par = 0.022 * strength * w                         # max near-vs-far horizontal offset in px
    dz = 0.075 * strength
    near = depth - 0.35                                # far background moves a little the other way

    cmd = [ffmpeg_bin(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in range(n):
            t = float(_ease(f / (n - 1)))
            zoom = base_zoom
            sx = sy = 0.0
            if move == "dolly_in":
                zoom = base_zoom * (1 + dz * t * (0.6 + 0.8 * depth))
            elif move == "dolly_out":
                zoom = base_zoom * (1 + dz * (1 - t) * (0.6 + 0.8 * depth))
            elif move == "drift_left":
                sx = par * (t - 0.5) * 2 * near
            elif move == "drift_right":
                sx = -par * (t - 0.5) * 2 * near
            else:                                      # rise: camera moves up, near things drop faster
                sy = -par * 0.8 * (t - 0.5) * 2 * near
                zoom = base_zoom * (1 + dz * 0.4 * t * depth)
            map_x = (cx + (xs - cx) / zoom + sx).astype(np.float32)
            map_y = (cy + (ys - cy) / zoom + sy).astype(np.float32)
            frame = cv2.remap(rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg (motion) failed: {err[-300:]}")
    except BaseException:
        proc.kill()
        out.unlink(missing_ok=True)
        raise
    return out
