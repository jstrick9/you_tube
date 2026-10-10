"""Final pre-upload quality gate: does the FINISHED video say and show the same thing?

Runs on the rendered MP4 itself (not on the plan), so it also catches render/timing mistakes.

  1. Narration  – the synthesized speech's word timings must match the script text (captions come from them).
  2. Coverage   – every narration line has at least one shot on screen, each verified ≥ threshold by a vision
                  model for that line (defensive re-check of the selection step).
  3. Frames     – still shots are sampled once; moving clips are sampled near the start, middle and end. Each
                  rendered frame is shown to a vision model with the exact words, trend angle, intended-shot
                  request and selected-asset provenance. A blind adversarial second look challenges positive
                  matches. Every sample must clearly fit the words and intended scene.

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
from .scriptwriter import spoken_text
from .vision import VisionUnavailable

log = logging.getLogger("autotube.qa")

QA_SYSTEM = ("You are the final quality inspector for an educational short-video channel. A video may only be "
             "published if EVERY frame's main photo clearly matches both the narration and the intended shot. "
             "Check identifiable people, event, place, era and subject carefully; asset titles are context, not proof. "
             "Be strict: when unsure, fail the frame. Return strict JSON only.")
QA_AUDIT_SYSTEM = ("You are an adversarial second-look inspector for an educational short-video channel. Your job is to find "
                   "false-positive image matches. Independently inspect the pixels; do not trust file titles, captions, "
                   "or a prior positive verdict. Check the exact person, event, era, place and setting against the "
                   "narration and intended shot. If relevance is not visually defensible, return match=false. "
                   "Return strict JSON only.")


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
        written = _norm_tokens(spoken_text(ss))
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


def _shot_context(t: dict, shots: list[dict]) -> dict:
    """Find the approved shot metadata used for this rendered frame."""
    path = str(t.get("path", ""))
    candidates = [s for s in shots if s.get("seg") == t.get("seg")]
    shot = next((s for s in candidates if str(s.get("path", "")) == path), None)
    if shot is None and candidates:
        shot = candidates[0]
    shot = shot or {}
    credit = shot.get("credit") if isinstance(shot.get("credit"), dict) else {}
    return {
        "intended": str(shot.get("want") or ""),
        "selected_description": str(shot.get("shows") or ""),
        "asset_title": str(credit.get("title") or ""),
        "asset_source": str(credit.get("source") or ""),
        "asset_page": str(credit.get("page") or credit.get("url") or ""),
        "kind": str(shot.get("kind") or credit.get("kind") or ""),
    }


def frame_check(mp4: Path, timeline: list[dict], shots: list[dict], tts: dict, script: dict,
                title: str, hook: str, subject: str, llm, cfg: dict,
                retry_pauses=(20, 60), trend_angle: str = "",
                trend_evidence: list[str] | None = None) -> tuple[list[dict], dict]:
    segs = tts["segments"]
    n_lines = len(segs)
    items = []
    for k, t in enumerate(timeline):
        context = _shot_context(t, shots)
        path = str(t.get("path", ""))
        moving = context["kind"].lower() == "video" or path.lower().endswith((".mp4", ".webm", ".mov", ".mkv"))
        # A single midpoint can miss a scene change or a misleading insert in moving footage.
        # Stills get one representative frame; each finished clip is checked near its start,
        # middle and end (away from the crossfade edges).
        fractions = (0.2, 0.5, 0.8) if moving and t["end"] - t["start"] >= 1.0 else (0.55,)
        seg = segs[t["seg"]]
        spoken = _spoken_during(seg, t["start"], t["end"]) or seg["text"]
        role = "hook" if t["seg"] == 0 else ("call to action" if t["seg"] == n_lines - 1 else "")
        fact = script["segments"][t["seg"]]["text"] if t["seg"] < len(script["segments"]) else seg["text"]
        for sample_idx, fraction in enumerate(fractions, start=1):
            stamp = t["start"] + fraction * (t["end"] - t["start"])
            items.append({"k": k, "seg": t["seg"], "path": path, "t": round(stamp, 2), "line": fact,
                          "spoken": spoken, "role": role, "context": context,
                          "sample": sample_idx, "samples": len(fractions), "fraction": fraction,
                          "img": _frame_at(mp4, stamp)})

    trend_evidence = [str(x)[:220] for x in (trend_evidence or []) if x][:4]
    results, meta = [], {"hook_text_ok": True, "title_ok": True, "notes": [],
                         "trend_angle": str(trend_angle or "")[:400], "trend_evidence": trend_evidence}
    per_sheet = int(cfg.get("qa", {}).get("frames_per_sheet", 8))
    for b in range(0, len(items), per_sheet):
        batch = items[b:b + per_sheet]
        first = b == 0
        rows = "\n".join(
            f'  {j + 1} | line {it["seg"] + 1}/{n_lines}{" (" + it["role"] + ")" if it["role"] else ""} | '
            f'sample {it["sample"]}/{it["samples"]} at {it["t"]:.2f}s | '
            f'intended shot: "{it["context"]["intended"] or "not recorded"}" | '
            f'selected image description: "{it["context"]["selected_description"] or "not recorded"}" | '
            f'asset title: "{it["context"]["asset_title"] or "not recorded"}" | '
            f'asset source/page: "{(it["context"]["asset_source"] + " " + it["context"]["asset_page"]).strip() or "not recorded"}" | '
            f'while on screen the narrator says: "{it["spoken"]}" (full factual line: "{it["line"]}")'
            for j, it in enumerate(batch))
        extra = (f'\nFrame 1 also shows the on-screen hook text "{hook}". Is that text accurate for this video, and '
                 f'does the title "{title}" describe what this video actually shows and says?') if first else ""
        evidence_block = "\n".join(f"- {x}" for x in trend_evidence) or "- see the cited trend record"
        user = f"""VIDEO SUBJECT: {subject}
VIDEO TITLE: {title}
CURRENT TREND ANGLE: {trend_angle or "not recorded"}
TREND SIGNAL CONTEXT (discovery evidence, not factual source text):
{evidence_block}

The attached sheet shows {len(batch)} frames from the finished vertical video, numbered 1-{len(batch)} (number in the
yellow box at each frame's bottom-left). For video assets, several samples from the same clip are listed separately.
Ignore the burned-in captions, the small channel name at the top, the progress bar, the hook title box, big animated
numbers, a "PROVEN" stamp / source card, countdown digits, a white flash and the blurred background fill — judge the
MAIN PHOTO. The narrator sometimes adds a short joke after a line; judge the photo against the factual "full line".
The story sequence must preserve the specific trend angle, not merely share the broad subject; each frame must still
match the exact narration and intended shot, and trend evidence itself never proves a factual claim.

For EACH frame, compare the visible main photo against all of: (1) the factual narration, (2) the intended shot request,
and (3) the selected asset description and provenance shown below. The descriptions and asset metadata are clues, not
proof: the pixels must actually fit. A historically named asset can still be the wrong battle, person, meeting, era,
or setting. If the image only shares a broad topic keyword but depicts an irrelevant event or modern substitute for
a named historical scene, mark it match=false.

{rows}

For each frame: does the main photo clearly show what that narration and intended shot are about — the specific subject,
thing, place, species, person, object or event being described? For a hook or call-to-action line, a clear photo of the
video's subject counts as a match only if the intended shot is also subject-level and not a conflicting scene.
FAIL a frame (match=false) if it shows: a different thing that merely shares a name (building, bar, street, sign,
logo, product, film), a generic stand-in for a named subject, symbolic/mood imagery (an eye, a coffee cup, a crowd,
a sunset, books) where an ordinary viewer would not immediately see why the picture goes with the words, unrelated
identifiable people, a conflicting era/event/location, mostly text/map/diagram/printed pages, something that contradicts
or could mislead about what is being said, or if you are unsure.{extra}
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
        primary = {int(f["n"]): f for f in out["frames"]}
        audit = {}
        # Challenge every positive frame individually. A contact sheet can hide the exact person,
        # event or era mismatch that a batch judge overlooks; the blind second look sees one frame only.
        for j, it in enumerate(batch):
            if not bool(primary.get(j + 1, {}).get("match")):
                continue
            ctx = it["context"]
            audit_user = f"""ADVERSARIAL SECOND LOOK — INSPECT THIS SINGLE FRAME, NOT A CONTACT SHEET.
VIDEO SUBJECT: {subject}
CURRENT TREND ANGLE: {trend_angle or "not recorded"}
  1 | line {it['seg'] + 1}/{n_lines}, sample {it['sample']}/{it['samples']} at {it['t']:.2f}s | intended shot: "{ctx['intended'] or 'not recorded'}"
Selected image description: "{ctx['selected_description'] or 'not recorded'}"
Asset title/source: "{ctx['asset_title'] or 'not recorded'}" / "{ctx['asset_source'] or 'not recorded'} {ctx['asset_page']}"
Exact narration while this frame is visible: "{it['spoken']}"
Full factual line: "{it['line']}"

The attached image is the actual rendered frame. Independently try to DISPROVE the match: check whether the pixels
show the correct named subject, person, event, place and historical era. Do not trust asset titles, the selected-image
description, or a prior positive decision. A generic meeting is not proof of a specific treaty negotiation; a modern
photo of unrelated leaders is not a historical scene. Mark match=false if the scene is irrelevant, misleading, or
not visually defensible. If uncertain, fail it.
Return JSON: {{"frames": [{{"n": 1, "shows": "<= 12 words", "match": true, "score": 0-10, "issue": ""}}]}}"""

            def validate_single(o):
                frames = o.get("frames")
                assert isinstance(frames, list) and len(frames) == 1, "audit must return exactly one frame verdict"
                assert int(frames[0]["n"]) == 1, "single-frame audit must be numbered 1"
                float(frames[0]["score"]), bool(frames[0]["match"])

            audit_out, audit_err = None, None
            for attempt in range(len(retry_pauses) + 1):
                try:
                    audit_out = llm.vision_json(QA_AUDIT_SYSTEM, audit_user,
                                                [_sheet([it["img"]])], validate_single)
                    break
                except Exception as e:  # noqa: BLE001
                    audit_err = e
                    if attempt < len(retry_pauses):
                        time.sleep(retry_pauses[attempt])
            if audit_out is None:
                raise VisionUnavailable(
                    f"adversarial final-frame audit unavailable: {str(audit_err)[:250]}") from audit_err
            audit[j + 1] = audit_out["frames"][0]
        for j, it in enumerate(batch):
            f = primary.get(j + 1, {"match": False, "score": 0, "shows": "", "issue": "no verdict"})
            a = audit.get(j + 1)
            match = bool(f.get("match")) and (bool(a.get("match")) if a is not None else False)
            score = min(float(f.get("score") or 0), float(a.get("score") or 0)) if a is not None else float(f.get("score") or 0)
            issue = str(f.get("issue", ""))
            if a is not None and (not a.get("match") or float(a.get("score") or 0) < float(cfg.get("qa", {}).get("min_frame_score", 7))):
                audit_issue = str(a.get("issue") or "adversarial audit could not confirm this image")
                issue = (issue + "; adversarial audit: " + audit_issue).strip("; ")
            results.append({"seg": it["seg"], "path": it["path"], "t": it["t"],
                            "sample": it["sample"], "samples": it["samples"], "spoken": it["spoken"][:140],
                            "intended_shot": it["context"]["intended"][:160],
                            "selected_image": it["context"]["selected_description"][:120],
                            "asset_title": it["context"]["asset_title"][:120],
                            "shows": str(f.get("shows", ""))[:120],
                            "audit_shows": str(a.get("shows", ""))[:120] if a else "",
                            "match": match, "score": score, "issue": issue[:200],
                            "judge": llm.last_used})
        if first:
            meta["hook_text_ok"] = bool(out.get("hook_text_ok", True))
            meta["title_ok"] = bool(out.get("title_ok", True))
        if out.get("notes"):
            meta["notes"].append(str(out["notes"])[:200])
    return results, meta

def verify(mp4: Path, timeline: list[dict], shots: list[dict], tts: dict, script: dict, title: str, hook: str,
           subject: str, llm, cfg: dict, trend_angle: str = "",
           trend_evidence: list[str] | None = None) -> dict:
    min_score = float(cfg["media"].get("min_match_score", 7))
    qa_min = float(cfg.get("qa", {}).get("min_frame_score", 7))
    report = {"passed": False, "issues": [], "frames": [], "failed": [], "repairable": False}
    report["issues"] += narration_check(tts, script)
    report["issues"] += coverage_check(timeline, shots, len(script["segments"]), min_score)
    if report["issues"]:
        return report                                  # structural problems: don't even spend a vision call
    frames, meta = frame_check(mp4, timeline, shots, tts, script, title, hook, subject, llm, cfg,
                               trend_angle=trend_angle, trend_evidence=trend_evidence)
    report["frames"] = frames
    failed_shots = set()
    for f in frames:
        if not f["match"] or f["score"] < qa_min:
            failed_shots.add((f["seg"], f["path"]))
            report["issues"].append(f"line {f['seg'] + 1} sample {f.get('sample', 1)}/{f.get('samples', 1)} "
                                    f"@ {f['t']}s shows '{f['shows']}' while saying "
                                    f"'{f['spoken'][:70]}' ({f['score']:.0f}/10: {f['issue']})")
    report["failed"] = sorted(failed_shots)
    if not meta["hook_text_ok"]:
        report["issues"].append(f"hook text '{hook}' judged inaccurate")
    if not meta["title_ok"]:
        report["issues"].append(f"title '{title}' judged not to match the video")
    report["repairable"] = bool(report["failed"]) and meta["hook_text_ok"] and meta["title_ok"]
    report["meta"] = meta
    report["passed"] = not report["issues"]
    return report
