"""Final pre-upload quality gate: does the FINISHED video say and show the same thing?

Runs on the rendered MP4 itself (not on the plan), so it also catches render/timing mistakes.

  1. Narration  – the synthesized speech's word timings must match the script text (captions come from them).
  2. Coverage   – every narration line has at least one shot on screen, each verified ≥ threshold by a vision
                  model for that line (defensive re-check of the selection step).
  3. Frames     – one frame from the middle of every shot is pulled from the MP4 and shown to a vision model
                  together with the exact words being spoken at that moment, the title and the hook text.
                  Every frame must clearly show what is being said. When unsure → fail.

`verify()` returns a report; the pipeline only uploads when report["passed"] is True. If no vision model can be
reached the gate raises vision.VisionUnavailable — an unverified video is never published.
"""
from __future__ import annotations

import difflib
import io
import logging
import math
import re
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .common import ffmpeg_bin, font_path
from .vision import VisionUnavailable

log = logging.getLogger("autotube.qa")

QA_SYSTEM = ("You are the final quality inspector for an educational short-video channel. A video may only be "
             "published if EVERY frame's main photo clearly shows what the narrator is saying at that moment. "
             "Be strict: when unsure, fail the frame. Return strict JSON only.")


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower().replace("'", ""))


def narration_check(tts: dict, script: dict, min_ratio: float = 0.9) -> list[str]:
    """Speech (word timings) vs the text that was sent to TTS vs the script. Returns a list of problems."""
    issues = []
    segs = script["segments"]
    if len(tts["segments"]) != len(segs):
        return [f"narration has {len(tts['segments'])} parts but the script has {len(segs)} lines"]
    for i, (ts, ss) in enumerate(zip(tts["segments"], segs)):
        spoken = _norm_tokens(" ".join(w["word"] for w in ts["words"]))
        sent = _norm_tokens(ts["text"])
        written = _norm_tokens(ss["text"])
        r1 = difflib.SequenceMatcher(None, spoken, sent).ratio() if sent else 0
        r2 = difflib.SequenceMatcher(None, sent, written).ratio() if written else 0
        if r1 < min_ratio:
            issues.append(f"line {i + 1}: spoken words differ from the script ({r1:.0%} match)")
        if r2 < 0.8:
            issues.append(f"line {i + 1}: text sent to speech differs from the script ({r2:.0%} match)")
        if ts["end"] <= ts["start"]:
            issues.append(f"line {i + 1}: empty audio")
    return issues


def coverage_check(timeline: list[dict], shots: list[dict], n_segs: int, min_score: float) -> list[str]:
    issues = []
    on_screen = {t["seg"] for t in timeline}
    for i in range(n_segs):
        if i not in on_screen:
            issues.append(f"line {i + 1} has no shot on screen")
    for s in shots:
        if s.get("judge", "") in ("", "clip") or float(s.get("score") or 0) < min_score:
            issues.append(f"line {s['seg'] + 1} uses an image that wasn't approved by the vision model "
                          f"(score {s.get('score')}, judge {s.get('judge') or 'none'})")
    return issues


def _frame_at(mp4: Path, t: float, w: int = 360, h: int = 640) -> Image.Image:
    out = subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(mp4),
                          "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(out)).convert("RGB")


def _sheet(frames: list[Image.Image], cols: int = 4) -> bytes:
    tw, th, gap = 300, 533, 8
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(font_path(), 44)
    for i, f in enumerate(frames):
        x = gap + (i % cols) * (tw + gap)
        y = gap + (i // cols) * (th + gap)
        sheet.paste(f.resize((tw, th)), (x, y))
        d.rectangle((x, y + th - 62, x + 62, y + th), fill=(255, 215, 0))
        d.text((x + 31, y + th - 31), str(i + 1), font=font, fill=(0, 0, 0), anchor="mm")
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _spoken_during(seg: dict, start: float, end: float) -> str:
    words = [w["word"] for w in seg["words"] if start - 0.05 <= w["start"] < end]
    return " ".join(words).strip()


def frame_check(mp4: Path, timeline: list[dict], tts: dict, script: dict, title: str, hook: str, subject: str,
                llm, cfg: dict, retry_pauses=(20, 60)) -> tuple[list[dict], dict]:
    segs = tts["segments"]
    n_lines = len(segs)
    items = []
    for k, t in enumerate(timeline):
        mid = t["start"] + 0.55 * (t["end"] - t["start"])
        seg = segs[t["seg"]]
        spoken = _spoken_during(seg, t["start"], t["end"]) or seg["text"]
        role = "hook" if t["seg"] == 0 else ("call to action" if t["seg"] == n_lines - 1 else "")
        items.append({"k": k, "seg": t["seg"], "path": t["path"], "t": round(mid, 2), "line": seg["text"],
                      "spoken": spoken, "role": role, "img": _frame_at(mp4, mid)})

    results, meta = [], {"hook_text_ok": True, "title_ok": True, "notes": []}
    per_sheet = int(cfg.get("qa", {}).get("frames_per_sheet", 8))
    for b in range(0, len(items), per_sheet):
        batch = items[b:b + per_sheet]
        first = b == 0
        rows = "\n".join(
            f'  {j + 1} | line {it["seg"] + 1}/{n_lines}{" (" + it["role"] + ")" if it["role"] else ""} | '
            f'while on screen the narrator says: "{it["spoken"]}"  (full line: "{it["line"]}")'
            for j, it in enumerate(batch))
        extra = (f'\nFrame 1 also shows the on-screen hook text "{hook}". Is that text accurate for this video, and '
                 f'does the title "{title}" describe what this video actually shows and says?') if first else ""
        user = f"""VIDEO SUBJECT: {subject}
VIDEO TITLE: {title}

The attached sheet shows {len(batch)} frames from the finished vertical video, numbered 1-{len(batch)} (number in the
yellow box at each frame's bottom-left). Ignore the burned-in captions, the small channel name at the top, the
progress bar, the hook title box and the blurred background fill — judge the MAIN PHOTO.

{rows}

For each frame: does the main photo clearly show what that narration is about — the specific subject, thing, place,
species, person, object or event being described? For a hook or call-to-action line, a clear photo of the video's
subject counts as a match.
FAIL a frame (match=false) if it shows: a different thing that merely shares a name (building, bar, street, sign,
logo, product, film), a generic stand-in for a named subject, unrelated identifiable people, mostly text/map/diagram,
something that contradicts or could mislead about what is being said, or if you are unsure.{extra}
Return JSON: {{"frames": [{{"n": 1, "shows": "<= 12 words", "match": true, "score": 0-10, "issue": ""}}],
 "hook_text_ok": true, "title_ok": true, "notes": ""}}"""

        def validate(o):
            fr = o.get("frames")
            assert isinstance(fr, list) and len(fr) == len(batch), f"need exactly {len(batch)} entries in 'frames'"
            for f in fr:
                int(f["n"]), float(f["score"]), bool(f["match"])

        out, err = None, None
        for attempt in range(len(retry_pauses) + 1):
            try:
                out = llm.vision_json(QA_SYSTEM, user, [_sheet([it["img"] for it in batch])], validate)
                break
            except Exception as e:  # noqa: BLE001
                err = e
                if attempt < len(retry_pauses):
                    time.sleep(retry_pauses[attempt])
        if out is None:
            raise VisionUnavailable(f"final QA could not reach a vision model: {str(err)[:300]}")
        by_n = {int(f["n"]): f for f in out["frames"]}
        for j, it in enumerate(batch):
            f = by_n.get(j + 1, {"match": False, "score": 0, "shows": "", "issue": "no verdict"})
            results.append({"seg": it["seg"], "path": it["path"], "t": it["t"], "spoken": it["spoken"][:140],
                            "shows": str(f.get("shows", ""))[:120], "match": bool(f.get("match")),
                            "score": float(f.get("score") or 0), "issue": str(f.get("issue", ""))[:160],
                            "judge": llm.last_used})
        if first:
            meta["hook_text_ok"] = bool(out.get("hook_text_ok", True))
            meta["title_ok"] = bool(out.get("title_ok", True))
        if out.get("notes"):
            meta["notes"].append(str(out["notes"])[:200])
    return results, meta


def verify(mp4: Path, timeline: list[dict], shots: list[dict], tts: dict, script: dict, title: str, hook: str,
           subject: str, llm, cfg: dict) -> dict:
    min_score = float(cfg["media"].get("min_match_score", 7))
    qa_min = float(cfg.get("qa", {}).get("min_frame_score", 7))
    report = {"passed": False, "issues": [], "frames": [], "failed": []}
    report["issues"] += narration_check(tts, script)
    report["issues"] += coverage_check(timeline, shots, len(script["segments"]), min_score)
    if report["issues"]:
        return report                                  # structural problems: don't even spend a vision call
    frames, meta = frame_check(mp4, timeline, tts, script, title, hook, subject, llm, cfg)
    report["frames"] = frames
    for f in frames:
        if not f["match"] or f["score"] < qa_min:
            report["failed"].append((f["seg"], f["path"]))
            report["issues"].append(f"line {f['seg'] + 1} @ {f['t']}s shows '{f['shows']}' while saying "
                                    f"'{f['spoken'][:70]}' ({f['score']:.0f}/10: {f['issue']})")
    if not meta["hook_text_ok"]:
        report["issues"].append(f"hook text '{hook}' judged inaccurate")
    if not meta["title_ok"]:
        report["issues"].append(f"title '{title}' judged not to match the video")
    report["meta"] = meta
    report["passed"] = not report["issues"]
    return report
