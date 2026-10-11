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
QA_SEQUENCE_SYSTEM = ("You are the sequence-level visual pacing inspector for an educational Short. Inspect every tile in "
                      "chronological order. Do not confuse subject consistency with visual variety: the core subject may "
                      "recur, but near-identical views/compositions should not dominate the video. Identify redundant "
                      "shots that should be replaced with a distinct, truthful view or visual type. Return strict JSON only.")


def _norm_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower().replace("'", ""))


def _finite_number(value, low: float, high: float) -> bool:
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and low <= value <= high)
    except (OverflowError, TypeError, ValueError):
        return False


def _validate_frame_output(value, expected: int, require_video_verdicts: bool = True) -> None:
    """Reject ambiguous/partial provider verdicts before a rendered frame can be approved."""
    assert isinstance(value, dict), "vision response must be a JSON object"
    frames = value.get("frames")
    assert isinstance(frames, list) and len(frames) == expected, f"need exactly {expected} frame verdicts"
    seen = set()
    for frame in frames:
        assert isinstance(frame, dict), "each frame verdict must be a JSON object"
        number = frame.get("n")
        assert type(number) is int and 1 <= number <= expected, "frame number must be an in-range integer"
        assert number not in seen, "frame numbers must be unique"
        seen.add(number)
        assert type(frame.get("match")) is bool, "frame match must be a boolean"
        assert _finite_number(frame.get("score"), 0, 10), "frame score must be a finite number from 0 to 10"
        assert isinstance(frame.get("shows"), str) and frame["shows"].strip(), "frame shows must be non-empty text"
        assert len(frame["shows"].split()) <= 12, "frame shows must be at most 12 words"
        assert isinstance(frame.get("issue"), str), "frame issue must be text"
    assert seen == set(range(1, expected + 1)), "frame verdicts must cover every numbered frame"
    if require_video_verdicts:
        assert type(value.get("hook_text_ok")) is bool, "hook_text_ok must be a boolean"
        assert type(value.get("title_ok")) is bool, "title_ok must be a boolean"
        assert isinstance(value.get("notes"), str), "notes must be text"


def _validate_sequence_output(value, expected: int) -> None:
    assert isinstance(value, dict), "sequence response must be a JSON object"
    assert type(value.get("visual_variety_ok")) is bool, "need boolean visual_variety_ok"
    repetitive = value.get("repetitive_frames")
    assert isinstance(repetitive, list), "need repetitive_frames list"
    assert all(type(number) is int for number in repetitive), "repetitive frame numbers must be integers"
    assert len(repetitive) == len(set(repetitive)), "repetitive frame numbers must be unique"
    assert all(1 <= number <= expected for number in repetitive), "repetitive frame number is out of range"
    assert isinstance(value.get("notes"), str), "sequence notes must be text"
    assert not repetitive if value["visual_variety_ok"] else bool(repetitive), (
        "a failed sequence must identify redundant frames; a passing sequence must have none")


def narration_check(tts: dict, script: dict, min_ratio: float = 0.9) -> list[str]:
    """Speech (word timings) vs the text sent to TTS vs the script; malformed timing data fails closed."""
    if not _finite_number(min_ratio, 0, 1):
        return ["narration match threshold is invalid"]
    if not isinstance(script, dict):
        return ["script must be an object for final narration QA"]
    segs = script.get("segments")
    if not isinstance(segs, list) or not segs:
        return ["script must contain a non-empty segment list for final narration QA"]
    if any(not isinstance(seg, dict) or not isinstance(seg.get("text"), str) or not seg["text"].strip()
           or ("aside" in seg and not isinstance(seg["aside"], str)) for seg in segs):
        return ["script contains a malformed narration segment"]
    if (not isinstance(tts, dict) or not isinstance(tts.get("segments"), list)
            or not _finite_number(tts.get("duration"), 0.1, float("inf"))):
        return ["narration timing data or audio duration is missing or malformed"]
    duration = tts["duration"]
    tts_segs = tts["segments"]
    if len(tts_segs) != len(segs):
        return [f"narration has {len(tts_segs)} parts but the script has {len(segs)} lines"]

    issues = []
    previous_segment_end = 0.0
    for i, (ts, ss) in enumerate(zip(tts_segs, segs)):
        if not isinstance(ts, dict):
            issues.append(f"line {i + 1}: narration segment is not an object")
            continue
        start, end = ts.get("start"), ts.get("end")
        if (not _finite_number(start, 0, duration) or not _finite_number(end, 0, duration + 0.1)
                or end <= start or start < previous_segment_end - 0.1):
            issues.append(f"line {i + 1}: invalid, empty, or overlapping audio timing")
            continue
        previous_segment_end = end
        sent_text = ts.get("text")
        words = ts.get("words")
        if not isinstance(sent_text, str) or not isinstance(words, list) or not words:
            issues.append(f"line {i + 1}: narration text or word timings are missing")
            continue
        word_text = []
        malformed_words = False
        previous_start = -1.0
        previous_word_end = start
        for word in words:
            if not isinstance(word, dict) or not isinstance(word.get("word"), str) or not word["word"].strip() \
                    or not _finite_number(word.get("start"), 0, float("inf")) \
                    or not _finite_number(word.get("end"), 0, float("inf")) \
                    or word["end"] <= word["start"] or word["start"] < previous_start \
                    or word["start"] < previous_word_end - 0.1 \
                    or word["start"] < start - 0.1 or word["start"] >= end + 0.1 \
                    or word["end"] > end + 0.1:
                malformed_words = True
                break
            previous_start = word["start"]
            previous_word_end = max(previous_word_end, word["end"])
            word_text.append(word["word"])
        if malformed_words:
            issues.append(f"line {i + 1}: word timings are malformed or out of order")
            continue
        spoken = _norm_tokens(" ".join(word_text))
        sent = _norm_tokens(sent_text)
        written = _norm_tokens(spoken_text(ss))
        r1 = difflib.SequenceMatcher(None, spoken, sent).ratio() if sent else 0
        r2 = difflib.SequenceMatcher(None, sent, written).ratio() if written else 0
        if r1 < min_ratio:
            issues.append(f"line {i + 1}: spoken words differ from the script ({r1:.0%} match)")
        if r2 < 0.8:
            issues.append(f"line {i + 1}: text sent to speech differs from the script ({r2:.0%} match)")
    return issues


def coverage_check(timeline: list[dict], shots: list[dict], n_segs: int, min_score: float) -> list[str]:
    """Validate rendered coverage and selected-image verdicts; bad types/numbers never pass as truthy."""
    if type(n_segs) is not int or n_segs < 1:
        return ["script has no valid narration segments for visual coverage"]
    if not _finite_number(min_score, 7, 10):
        return ["minimum visual-match score must be a finite number from 7 to 10"]
    if not isinstance(timeline, list) or not isinstance(shots, list):
        return ["rendered timeline or selected-shot list is malformed"]

    issues = []
    on_screen = set()
    timeline_keys = set()
    selected_keys = set()
    for i, entry in enumerate(timeline):
        if not isinstance(entry, dict):
            issues.append(f"timeline entry {i + 1} is not an object")
            continue
        seg = entry.get("seg")
        if type(seg) is not int or not 0 <= seg < n_segs:
            issues.append(f"timeline entry {i + 1} has an invalid segment number")
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path.strip():
            issues.append(f"timeline entry {i + 1} has no media path")
        else:
            timeline_keys.add((seg, path))
        start, end = entry.get("start"), entry.get("end")
        if (not _finite_number(start, 0, float("inf"))
                or not _finite_number(end, 0, float("inf")) or end <= start):
            issues.append(f"timeline entry {i + 1} has invalid time bounds")
        on_screen.add(seg)
    for i in range(n_segs):
        if i not in on_screen:
            issues.append(f"line {i + 1} has no shot on screen")

    for i, shot in enumerate(shots):
        if not isinstance(shot, dict):
            issues.append(f"selected shot {i + 1} is not an object and wasn't approved by the vision model")
            continue
        seg = shot.get("seg")
        if type(seg) is not int or not 0 <= seg < n_segs:
            issues.append(f"selected shot {i + 1} has an invalid segment number and wasn't approved by the vision model")
            continue
        raw_path = shot.get("path")
        path = str(raw_path) if isinstance(raw_path, (str, Path)) else ""
        if not path.strip():
            issues.append(f"line {seg + 1} has no selected media path")
        else:
            selected_keys.add((seg, path))
        score, judge = shot.get("score"), shot.get("judge")
        if (not _finite_number(score, 0, 10) or not isinstance(judge, str) or not judge.strip()
                or judge.strip().lower() == "clip" or score < min_score):
            issues.append(f"line {seg + 1} uses an image that wasn't approved by the vision model "
                          f"(score {score!r}, judge {judge or 'none'})")
    for seg, path in sorted(timeline_keys - selected_keys):
        issues.append(f"line {seg + 1} renders {path!r} without a selected-shot approval record")
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


def _sequence_representatives(items: list[dict]) -> list[dict]:
    """Choose the midpoint of each rendered shot, preserving timeline order for pacing QA."""
    chosen, seen = [], set()
    for it in items:
        if int(it["sample"]) != (int(it["samples"]) + 1) // 2:
            continue
        key = (it["seg"], it["path"])
        if key in seen:
            continue
        seen.add(key)
        chosen.append(it)
    return chosen


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
    qa_cfg = cfg.get("qa", {}) if isinstance(cfg, dict) else {}
    if not isinstance(qa_cfg, dict):
        raise VisionUnavailable("final QA thresholds are malformed")
    per_sheet = qa_cfg.get("frames_per_sheet", 8)
    qa_min = qa_cfg.get("min_frame_score", 7)
    if type(per_sheet) is not int or not 1 <= per_sheet <= 32 or not _finite_number(qa_min, 7, 10):
        raise VisionUnavailable("final QA batch size or frame-score threshold is invalid")
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
            _validate_frame_output(o, len(batch), require_video_verdicts=True)

        out, err = None, None
        for attempt in range(len(retry_pauses) + 1):
            try:
                out = llm.vision_json(QA_SYSTEM, user, [_sheet([it["img"] for it in batch])], validate)
                validate(out)  # also defend against adapters that ignore the validation callback
                break
            except Exception as e:  # noqa: BLE001
                out = None
                err = e
                if attempt < len(retry_pauses):
                    time.sleep(retry_pauses[attempt])
        if out is None:
            raise VisionUnavailable(f"final QA received no valid vision verdict: {str(err)[:300]}")
        primary_judge = str(getattr(llm, "last_used", "") or "")
        primary = {f["n"]: f for f in out["frames"]}
        audit = {}
        audit_judges = {}
        # Challenge every positive frame individually. A contact sheet can hide the exact person,
        # event or era mismatch that a batch judge overlooks; the blind second look sees one frame only.
        for j, it in enumerate(batch):
            if primary[j + 1]["match"] is not True:
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
                _validate_frame_output(o, 1, require_video_verdicts=False)
                assert o["frames"][0]["n"] == 1, "single-frame audit must be numbered 1"

            audit_out, audit_err = None, None
            for attempt in range(len(retry_pauses) + 1):
                try:
                    audit_out = llm.vision_json(QA_AUDIT_SYSTEM, audit_user,
                                                [_sheet([it["img"]])], validate_single)
                    validate_single(audit_out)  # revalidate at the trust boundary
                    break
                except Exception as e:  # noqa: BLE001
                    audit_out = None
                    audit_err = e
                    if attempt < len(retry_pauses):
                        time.sleep(retry_pauses[attempt])
            if audit_out is None:
                raise VisionUnavailable(
                    f"adversarial final-frame audit unavailable: {str(audit_err)[:250]}") from audit_err
            audit[j + 1] = audit_out["frames"][0]
            audit_judges[j + 1] = str(getattr(llm, "last_used", "") or "")
        for j, it in enumerate(batch):
            f = primary[j + 1]
            a = audit.get(j + 1)
            match = f["match"] is True and (a["match"] is True if a is not None else False)
            score = min(f["score"], a["score"]) if a is not None else f["score"]
            issue = f["issue"]
            if a is not None and (a["match"] is not True or a["score"] < qa_min):
                audit_issue = a["issue"] or "adversarial audit could not confirm this image"
                issue = (issue + "; adversarial audit: " + audit_issue).strip("; ")
            results.append({"timeline_index": it["k"], "seg": it["seg"], "path": it["path"], "t": it["t"],
                            "sample": it["sample"], "samples": it["samples"], "spoken": it["spoken"][:140],
                            "intended_shot": it["context"]["intended"][:160],
                            "selected_image": it["context"]["selected_description"][:120],
                            "asset_title": it["context"]["asset_title"][:120],
                            "shows": f["shows"][:120],
                            "audit_shows": a["shows"][:120] if a else "",
                            "audit_match": a["match"] if a else False,
                            "audit_score": a["score"] if a else None,
                            "match": match, "score": score, "issue": issue[:200],
                            "judge": primary_judge, "audit_judge": audit_judges.get(j + 1, "")})
        if first:
            meta["hook_text_ok"] = out["hook_text_ok"]
            meta["title_ok"] = out["title_ok"]
        if out["notes"]:
            meta["notes"].append(out["notes"][:200])

    # Per-frame relevance does not catch a factually correct but monotonous slideshow. Review one
    # representative midpoint from every shot together, in timeline order, and fail the shots the
    # sequence judge identifies as redundant so the existing visual-repair path can replace them.
    sequence_items = _sequence_representatives(items)
    meta["visual_variety_ok"] = True
    meta["visual_variety_notes"] = ""
    if sequence_items:
        sequence_rows = "\n".join(
            f'  {j + 1} | line {it["seg"] + 1}/{n_lines} | '
            f'intended shot: "{it["context"]["intended"] or "not recorded"}" | '
            f'asset description: "{it["context"]["selected_description"] or "not recorded"}" | '
            f'narration: "{it["line"]}"'
            for j, it in enumerate(sequence_items))
        sequence_user = f"""SEQUENCE-LEVEL VISUAL VARIETY REVIEW
VIDEO SUBJECT: {subject}
VIDEO TITLE: {title}
CURRENT TREND ANGLE: {trend_angle or "not recorded"}

The attached sheet shows one midpoint frame from EACH shot in the finished video, in chronological order, numbered
1-{len(sequence_items)} in yellow boxes. Assess the whole sequence, not just whether each image is individually relevant.
The same reservoir, animal, artifact or historical subject may recur; do not demand unrelated imagery. But multiple
near-identical reservoir aerials, repeated static views, or several frames with the same composition and no new visual
information make a Short feel like a slideshow, even when each picture is technically relevant. A repeated visual motif
is fine once or twice when it serves the story; it should not dominate. Prefer truthful variety in scale, angle, action,
map/diagram, close detail, process or documented context—never a generic or misleading substitute.

SHOT ORDER AND CONTEXT:
{sequence_rows}

Is the sequence visually varied enough to hold attention while remaining faithful to the subject? If it is, set
visual_variety_ok=true and repetitive_frames=[]. If it is not, set visual_variety_ok=false and list every numbered frame
that is visually redundant and should be replaced. Do not fail a shot solely because it shares the video's subject.
Return strict JSON: {{"visual_variety_ok": true, "repetitive_frames": [], "notes": "short reason"}}"""

        def validate_sequence(o):
            _validate_sequence_output(o, len(sequence_items))

        sequence_out, sequence_err = None, None
        for attempt in range(len(retry_pauses) + 1):
            try:
                sequence_out = llm.vision_json(QA_SEQUENCE_SYSTEM, sequence_user,
                                               [_sheet([it["img"] for it in sequence_items])], validate_sequence)
                validate_sequence(sequence_out)  # revalidate adapters that omit callback enforcement
                break
            except Exception as e:  # noqa: BLE001
                sequence_out = None
                sequence_err = e
                if attempt < len(retry_pauses):
                    time.sleep(retry_pauses[attempt])
        if sequence_out is None:
            raise VisionUnavailable(f"sequence-level visual QA unavailable: {str(sequence_err)[:250]}") from sequence_err
        repetitive = sequence_out["repetitive_frames"]
        variety_notes = sequence_out["notes"][:250]
        meta["visual_variety_ok"] = sequence_out["visual_variety_ok"]
        meta["visual_variety_notes"] = variety_notes
        if variety_notes:
            meta["notes"].append("visual sequence: " + variety_notes)
        if not meta["visual_variety_ok"]:
            for number in repetitive:
                redundant = sequence_items[number - 1]
                reason = "sequence-level visual review found a repetitive shot"
                if variety_notes:
                    reason += ": " + variety_notes
                for frame in results:
                    if frame["seg"] == redundant["seg"] and frame["path"] == redundant["path"]:
                        frame["match"] = False
                        frame["score"] = 0.0
                        frame["issue"] = (frame["issue"] + "; " + reason).strip("; ")[:200]
    return results, meta

def verify(mp4: Path, timeline: list[dict], shots: list[dict], tts: dict, script: dict, title: str, hook: str,
           subject: str, llm, cfg: dict, trend_angle: str = "",
           trend_evidence: list[str] | None = None) -> dict:
    report = {"passed": False, "issues": [], "frames": [], "failed": [], "repairable": False}
    if not isinstance(cfg, dict):
        report["issues"].append("final QA configuration is malformed")
        return report
    media_cfg = cfg.get("media", {})
    qa_cfg = cfg.get("qa", {})
    if not isinstance(media_cfg, dict) or not isinstance(qa_cfg, dict):
        report["issues"].append("final QA thresholds are malformed")
        return report
    min_score = media_cfg.get("min_match_score", 7)
    qa_min = qa_cfg.get("min_frame_score", 7)
    if not _finite_number(min_score, 7, 10) or not _finite_number(qa_min, 7, 10):
        report["issues"].append("final QA score thresholds must be finite numbers from 7 to 10")
        return report
    report["issues"] += narration_check(tts, script)
    segs = script.get("segments") if isinstance(script, dict) else []
    report["issues"] += coverage_check(timeline, shots, len(segs) if isinstance(segs, list) else 0, min_score)
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
