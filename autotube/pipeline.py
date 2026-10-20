"""Orchestrator: trends → plan → ground → script → review → TTS → visuals → render → upload.

One run produces `schedule.videos_per_day` videos and schedules each to go public
at the configured local time slots (YouTube handles publishing via publishAt), so
a single daily job is enough — no always-on server needed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import media, music, qa, render, trends, run_report
from . import series as series_mod
from . import originality, provenance
from .vision import VisionUnavailable
from .target import daily_target
from .common import OUTPUT_DIR, WORK_DIR, now_utc, read_json, slugify, write_json
from .llm import LLM
from .recovery import RecoveryBudgets
from .scriptwriter import ScriptWriter, spoken_text, trend_evidence_summary, visual_subject_for_topic
from .strategy import Strategy
from .tts import synthesize

log = logging.getLogger("autotube.pipeline")


def _report_stage(stage: str) -> None:
    receipt = run_report.current_receipt()
    if receipt:
        receipt.set_stage(stage)


def _report_failure(stage: str, category: str, exc: Exception | None = None, attempt: int | None = None) -> None:
    receipt = run_report.current_receipt()
    if receipt:
        receipt.note_failure(stage, category, exception_type=type(exc).__name__ if exc else None,
                             attempt=attempt)


def filter_topic_candidates_by_hint(candidates: list[dict], hint: str | None) -> list[dict]:
    """Optionally focus selection on a live candidate, without creating or approving one.

    The hint may be a topic-title phrase or an exact source identifier such as a YouTube video ID.
    Matching is against trend metadata only; candidates still pass the normal live-evidence, approved-lane,
    virality, safety, research, originality, and final-render gates after this discovery-time filter.
    """
    if not isinstance(candidates, list):
        return []
    hint_terms = set(originality.tokens(hint or ""))
    if not hint_terms:
        return candidates
    matched = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        parts = [str(candidate.get("topic") or ""), str(candidate.get("topic_key") or "")]
        for context in candidate.get("context") or []:
            if isinstance(context, str):
                parts.append(context)
        for signal in candidate.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            parts.extend(str(signal.get(key) or "") for key in ("topic", "source_url"))
            evidence = signal.get("evidence")
            if isinstance(evidence, dict):
                parts.extend(str(evidence.get(key) or "") for key in ("video_id", "title", "url"))
        if hint_terms.issubset(set(originality.tokens(" ".join(parts)))):
            matched.append(candidate)
    return matched


def _slot_key(ts) -> str:
    d = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
    return d.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M")


def taken_slots(hist: list[dict]) -> set[str]:
    """Publish slots booked by live uploads that haven't gone public yet.
    Only status 'scheduled' holds a slot — withdrawn (set Private), missing (deleted) or rejected
    videos free their slot so it can be refilled."""
    now = now_utc()
    out = set()
    for h in hist:
        if h.get("video_id") and h.get("publish_at") and h.get("status") == "scheduled":
            try:
                if datetime.fromisoformat(h["publish_at"]) > now:
                    out.add(_slot_key(h["publish_at"]))
            except ValueError:
                pass
    return out


def slot_jitter(day, idx: int, spread: int) -> int:
    """Deterministic ±spread minute offset for one slot on one day.

    Publishing at 12:00:00, 17:30:00 and 20:30:00 to the second, every single day, is a machine
    fingerprint and one of the cadence signals a channel review looks at. The offset has to be
    deterministic rather than random, because top-up runs recompute slots several times a day and
    must agree with the booking the earlier run already made.
    """
    if spread <= 0:
        return 0
    h = hashlib.md5(f"{day.isoformat()}:{idx}".encode()).hexdigest()
    return int(h[:8], 16) % (2 * spread + 1) - spread


def publish_slots(cfg: dict, n: int, taken: set[str] | None = None) -> list[datetime | None]:
    """Next n free publish slots (local schedule times, jittered), skipping past and booked ones."""
    tz = ZoneInfo(cfg["channel"].get("timezone", "UTC"))
    now_local = now_utc().astimezone(tz)
    taken = taken or set()
    slots = []
    times = cfg["schedule"]["publish_times"]
    spread = int(cfg["schedule"].get("jitter_minutes", 0))
    day = now_local.date()
    for _ in range(60):                                  # hard stop: never loop forever
        for idx, t in enumerate(times):
            hh, mm = map(int, t.split(":"))
            dt = (datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)
                  + timedelta(minutes=slot_jitter(day, idx, spread)))
            if dt > now_local + timedelta(minutes=20) and _slot_key(dt) not in taken:   # publishAt must be future
                slots.append(dt.astimezone(ZoneInfo("UTC")))
            if len(slots) >= n:
                return slots
        day += timedelta(days=1)
    return slots


def remaining_today(cfg: dict, upload: bool = True) -> int:
    """How many of today's videos (channel timezone) still need to be made. Used by scheduled top-up runs."""
    tz = ZoneInfo(cfg["channel"].get("timezone", "UTC"))
    today = now_utc().astimezone(tz).date()
    ok = {"scheduled"} if upload else {"scheduled", "rendered"}
    done = sum(1 for h in read_json("history.json", [])
               if h.get("status") in ok and h.get("created_at")
               and datetime.fromisoformat(h["created_at"]).astimezone(tz).date() == today)
    return max(0, daily_target(cfg) - done)


def _max_similarity(script: dict, history: list[dict]) -> float:
    """How close this script came to the nearest thing already published."""
    mine = originality.sketch(originality.narration_of(script))
    sims = [originality.similarity(mine, h.get("sketch") or []) for h in history[-50:]]
    return round(max(sims), 3) if sims else 0.0


def weakest_segment(segments: list[dict]) -> int | None:
    """Index of the body line we can lose with the least damage, or None if there's nothing safe to drop.

    Protected: the hook (0), the payoff/loop (last), and the reveal. Among the rest we drop the line that
    carries the least new information — no number, no aside, shortest evidence quote, fewest words.
    """
    n = len(segments)
    if n <= 4:
        return None
    best, best_key = None, None
    for i in range(1, n - 1):
        s = segments[i]
        if s.get("reveal"):
            continue
        key = (
            bool(re.search(r"\d", s.get("text", ""))),        # keep lines with numbers
            bool((s.get("aside") or "").strip()),             # keep lines carrying a joke
            len((s.get("evidence") or "").split()),           # keep the best-sourced lines
            len(s.get("text", "").split()),                   # keep the densest lines
        )
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


_DESC_BAD = re.compile(r"[<>\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_description(text: str, limit: int = 4900) -> str:
    """YouTube rejects descriptions containing angle brackets or control characters with a 400.

    LLM-written descriptions and Commons credit strings both leak them, which cost us an upload on
    2026-09-30 ("The request metadata specifies an invalid video description").
    """
    text = _DESC_BAD.sub(" ", text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > limit:                       # cut on a line boundary so credits never end mid-URL
        text = text[:limit].rsplit("\n", 1)[0].rstrip()
    return text


def build_description(script: dict, source: dict, visuals: list[dict], cfg: dict, tts_engine: str) -> str:
    tags = " ".join(h if h.startswith("#") else f"#{h}" for h in script.get("hashtags", [])[:3])
    synthetic_visuals = any(
        isinstance(shot, dict) and (shot.get("synthetic") is True
                                    or (isinstance(shot.get("credit"), dict)
                                        and shot["credit"].get("synthetic") is True))
        for shot in visuals
    )
    catch = (cfg.get("persona") or {}).get("catchphrase")
    parts = [
        script.get("description", "").strip(),
        "",
        *([f"✅ {catch} Every claim in this video is checked against the source below.", ""] if catch else []),
        f"📚 Source: {source['title']} — {source['url']}",
        *[f"📚 Also: {a['title']} — {a['url']}" for a in source.get("also", [])[:4]],
    ]
    if cfg["compliance"].get("ai_disclosure_in_description", True) or synthetic_visuals:
        parts += ["", "ℹ️ This video was researched from the cited source and produced with AI-assisted "
                      "scripting and a synthetic narrator voice. Facts are checked against the source before publishing."]
    if synthetic_visuals:
        parts += ["AI-generated visual illustration(s) are clearly labelled on screen. They are illustrative only, "
                  "not archival footage, eyewitness material, documentary evidence, or a source for factual claims."]
    parts += ["", "🖼️ Image & video credits:", media.credits_text(visuals),
              "🎵 Music: original, procedurally generated for this video.", "",
              f"Follow {cfg['channel'].get('handle') or cfg['channel']['name']} for a surprising fact, with sources, every day!", "", f"{tags} #shorts"]
    return clean_description("\n".join(parts))


MAX_SPEEDUP_PCT = 10        # beyond roughly this the voice starts to sound robotic, which costs
                            # more retention than the extra seconds do


def pause_plan(script: dict, plan: dict, cfg: dict) -> list[float]:
    """Silence before each line: normal, except a beat before the reveal (longer for the 3-2-1 countdown)."""
    gaps = [0.0] * len(script["segments"])
    for i, s in enumerate(script["segments"]):
        if s.get("reveal") and i > 0:
            key = "countdown_pause" if plan.get("format") == "guess_reveal" else "reveal_pause"
            gaps[i] = float(cfg["content"].get(key, 0.55))
    return gaps



def shown_duration(tts: dict) -> float:
    """What the viewer actually sits through: narration plus the silence the renderer appends.

    Checking tts["duration"] alone meant every video ran TAIL seconds longer than the target it
    had just been measured against.
    """
    return tts["duration"] + render.TAIL


def fit_length(script: dict, plan: dict, cfg: dict, work, synth) -> tuple[dict | None, list[float]]:
    """Synthesize the narration and bring it inside the configured band.

    The configured duration is an internal creative experiment, not an asserted YouTube ranking rule.
    This function holds it as a production target with three escalating steps:

      1. Drop the least load-bearing body line and re-synthesize (never the hook, reveal or payoff).
      2. If nothing is safe to drop, add a bounded amount of pace while preserving natural narration.
      3. Skip only if it exceeds the internal 179-second Shorts ceiling; the configured target is much shorter.

    The speed-up is capped because past roughly ten percent the voice sounds robotic, and that
    costs more retention than the seconds it saves.
    """
    lo, hi = cfg["video"]["target_seconds"]
    base_rate = cfg["video"]["speech_rate"]
    gaps = pause_plan(script, plan, cfg)
    tts = synth([spoken_text(s) for s in script["segments"]], plan["voice"], base_rate, work, gaps=gaps)

    for _ in range(3):
        if shown_duration(tts) <= hi:
            break
        i = weakest_segment(script["segments"])
        if i is None:
            break
        log.info("  narration %.1fs > %.0fs target → dropping body line %d: %r",
                 shown_duration(tts), float(hi), i + 1, script["segments"][i]["text"][:60])
        script["segments"].pop(i)
        gaps = pause_plan(script, plan, cfg)
        tts = synth([spoken_text(s) for s in script["segments"]], plan["voice"], base_rate, work, gaps=gaps)

    if shown_duration(tts) > hi:
        need = shown_duration(tts) / max(float(hi) - 0.1, 1.0)
        bump = min(MAX_SPEEDUP_PCT, int((need - 1) * 100) + 2)
        if bump > 0:
            faster = f"+{int(str(base_rate).strip('+%')) + bump}%"
            log.info("  narration %.1fs > %.0fs and nothing safe to drop → pacing %s",
                     shown_duration(tts), float(hi), faster)
            tts = synth([spoken_text(s) for s in script["segments"]], plan["voice"], faster, work, gaps=gaps)

    if shown_duration(tts) > 179.0:         # upload ceiling leaves margin below YouTube's 3-minute Shorts maximum
        log.info("  ✗ exceeds the internal Shorts ceiling (%.1fs), skipping", shown_duration(tts))
        return None, gaps
    if shown_duration(tts) > hi:
        log.info("  note: %.1fs is over the %.0fs target but within the Shorts limit — keeping",
                 shown_duration(tts), float(hi))
    return tts, gaps


def effects_plan(script: dict, plan: dict, source: dict, cfg: dict) -> dict:
    """Everything the renderer needs for kinetic captions, SFX and the proof stamp."""
    persona = cfg.get("persona") or {}
    segs = script["segments"]
    reveal = next((i for i, s in enumerate(segs) if s.get("reveal")), None)
    return {
        "emphasis": [s.get("emphasis") or [] for s in segs],
        "asides": [s.get("aside") or "" for s in segs],
        "reveal": reveal,
        "countdown": plan.get("format") == "guess_reveal" and reveal is not None,
        "stamp": ({"source": f"Source: Wikipedia · {source['title']}", "seconds": float(persona.get("stamp_seconds", 2.0))}
                  if persona.get("proof_stamp", True) else None),
        "kinetic": bool(cfg["video"].get("kinetic_captions", True)),
        "sfx": bool(cfg["video"].get("sound_effects", True)),
    }


def produce_assets(cfg: dict, writer: ScriptWriter, topic: dict, plan: dict,
                   idx: int, run_id: str) -> dict | None:
    """Script → narration → visuals → music. Everything needed to render, or None to skip the topic."""
    _report_stage("script_research")
    produced = writer.produce(topic, plan)
    if not produced:
        _report_failure("script_research", "script_or_grounding_rejected")
        log.info("  ✗ could not produce a verified script for %r", topic["topic"])
        return None
    script, source, review = produced
    slug = slugify(source["title"])
    work = WORK_DIR / run_id / f"{idx:02d}-{slug}"
    work.mkdir(parents=True, exist_ok=True)
    seed = int(hashlib.md5(f"{run_id}{slug}".encode()).hexdigest()[:8], 16)

    _report_stage("tts")
    tts, _gaps = fit_length(script, plan, cfg, work, synthesize)
    if tts is None:
        _report_failure("tts", "shorts_duration_gate")
        return None

    _report_stage("visuals")
    tsegs = tts["segments"]
    starts = [x["start"] for x in tsegs] + [tts["duration"]]
    seg_durs = [starts[i + 1] - starts[i] for i in range(len(tsegs))]
    try:
        visuals = media.gather(source, script["segments"], cfg, work / "img", llm=writer.llm, seg_durs=seg_durs)
    except media.NoVisualMatch as e:
        _report_failure("visuals", "no_verified_visual_match", e)
        log.info("  ✗ visuals: %s — skipping topic", e)
        return None

    _report_stage("music")
    mus = music.generate(tts["duration"] + 1, work / "music.wav", seed) if cfg["video"]["background_music"] else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    title = script["title"].strip()
    visual_subject = visual_subject_for_topic(topic, source)
    # Which series this belongs to is known now; which episode number it gets is not.
    # Numbering is deferred to package(), after final QA, because a number reserved
    # here is spent even when the video is thrown away minutes later. That is how the
    # channel published FIELD #001, #002, #004, #005, #006, #008, #011 - four missing
    # numbers, each burned by a video that never existed. An archive that skips
    # episodes tells a viewer the archive is a prop, which is the opposite of the
    # point of numbering it.
    s_key = series_mod.assign(topic.get("category"), cfg)
    return {"script": script, "source": source, "review": review, "plan": plan, "topic": topic,
            "work": work, "seed": seed, "tts": tts, "visuals": visuals, "music": mus,
            "out": OUTPUT_DIR / f"{run_id}-{idx:02d}-{slug}.mp4",
            "hook_card": script.get("thumbnail_text") or "", "title": title,
            "series": s_key, "episode": 0,
            "subject": source["title"], "visual_subject": visual_subject,
            "fx": effects_plan(script, plan, source, cfg)}


def _trend_qa_evidence(topic: dict) -> list[str]:
    """Pass measured trend rationale and auditable source links to final rendered-frame QA."""
    rows = [trend_evidence_summary(topic)] if topic else []
    for signal in (topic.get("signals") or []) if topic else []:
        if not isinstance(signal, dict):
            continue
        source = str(signal.get("source") or "trend source")
        url = str(signal.get("source_url") or "")
        captured = str(signal.get("captured_at") or "")
        if url:
            rows.append(f"{source} source link: {url}" + (f" (captured {captured})" if captured else ""))
        if len(rows) >= 4:
            break
    return rows[:4]


def _finite_range(value, low: float, high: float) -> bool:
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and low <= value <= high)
    except (OverflowError, TypeError, ValueError):
        return False


def _qa_pass_is_explicit(result) -> bool:
    """An upload result must carry an actual Boolean True, never a truthy string or number."""
    return (isinstance(result, dict) and isinstance(result.get("qa"), dict)
            and result["qa"].get("passed") is True)


def _qa_report_publishable(report: dict, cfg: dict, timeline, shots, n_segs: int) -> bool:
    """Trust-boundary check: QA must cover each rendered interval and its approved shot."""
    if (not isinstance(report, dict) or report.get("passed") is not True or not isinstance(cfg, dict)
            or type(n_segs) is not int or n_segs < 1 or not isinstance(timeline, list) or not isinstance(shots, list)):
        return False
    media_cfg, qa_cfg = cfg.get("media"), cfg.get("qa")
    if not isinstance(media_cfg, dict) or not isinstance(qa_cfg, dict):
        return False
    min_match = media_cfg.get("min_match_score", 7)
    minimum = qa_cfg.get("min_frame_score", 7)
    if not _finite_range(min_match, 7, 10) or not _finite_range(minimum, 7, 10):
        return False
    if qa.coverage_check(timeline, shots, n_segs, min_match):
        return False
    issues = report.get("issues")
    if not isinstance(issues, list) or issues or report.get("failed") != [] or report.get("repairable") is not False:
        return False
    frames = report.get("frames")
    meta = report.get("meta")
    if not isinstance(frames, list) or not frames or not isinstance(meta, dict):
        return False
    if any(meta.get(key) is not True for key in ("hook_text_ok", "title_ok", "visual_variety_ok")):
        return False
    if (not isinstance(meta.get("notes"), list)
            or any(not isinstance(note, str) for note in meta["notes"])
            or not isinstance(meta.get("visual_variety_notes"), str)):
        return False

    frames_by_timeline: dict[int, list[dict]] = {}
    for frame in frames:
        if not isinstance(frame, dict) or frame.get("match") is not True:
            return False
        score = frame.get("score")
        if not _finite_range(score, minimum, 10):
            return False
        timeline_index = frame.get("timeline_index")
        if type(timeline_index) is not int or not 0 <= timeline_index < len(timeline):
            return False
        entry = timeline[timeline_index]
        if (not isinstance(entry, dict) or type(frame.get("seg")) is not int
                or frame["seg"] != entry.get("seg") or frame.get("path") != entry.get("path")):
            return False
        if not _finite_range(frame.get("t"), entry["start"], entry["end"]):
            return False
        sample, samples = frame.get("sample"), frame.get("samples")
        if type(sample) is not int or type(samples) is not int or samples < 1 or not 1 <= sample <= samples:
            return False
        if not isinstance(frame.get("shows"), str) or not frame["shows"].strip():
            return False
        if (not isinstance(frame.get("audit_shows"), str) or not frame["audit_shows"].strip()
                or frame.get("audit_match") is not True
                or not _finite_range(frame.get("audit_score"), minimum, 10)):
            return False
        for key in ("judge", "audit_judge"):
            judge = frame.get(key)
            if not isinstance(judge, str) or not judge.strip() or judge.strip().lower() == "clip":
                return False
        frames_by_timeline.setdefault(timeline_index, []).append(frame)

    # Every timeline entry is inspected; a repeated path on the same line is still a separate
    # rendered interval. Moving footage must receive start/middle/end review, not one arbitrary tile.
    for index, entry in enumerate(timeline):
        seg, path = entry["seg"], entry["path"]
        shot = next((s for s in shots if s["seg"] == seg and str(s["path"]) == path), None)
        if shot is None:
            return False  # coverage_check should already catch this; keep the report boundary explicit
        credit = shot.get("credit") if isinstance(shot.get("credit"), dict) else {}
        kind = str(shot.get("kind") or credit.get("kind") or "").lower()
        moving = kind == "video" or path.lower().endswith((".mp4", ".webm", ".mov", ".mkv"))
        expected_samples = 3 if moving and entry["end"] - entry["start"] >= 1.0 else 1
        observed = frames_by_timeline.get(index, [])
        if (len(observed) != expected_samples
                or {frame["sample"] for frame in observed} != set(range(1, expected_samples + 1))
                or any(frame["samples"] != expected_samples for frame in observed)):
            return False
    if set(frames_by_timeline) != set(range(len(timeline))):
        return False
    return True


def _normalize_qa_report(report, cfg: dict, timeline=None, shots=None, n_segs: int | None = None) -> dict:
    """Convert malformed QA result shapes to an explicit, non-repairable rejection."""
    if not isinstance(report, dict):
        return {"passed": False, "issues": ["final QA returned a malformed report; rejecting"],
                "frames": [], "failed": [], "repairable": False}
    result = dict(report)
    malformed = []
    if type(result.get("passed")) is not bool:
        result["passed"] = False
        malformed.append("final QA passed verdict was not a boolean")
    if not isinstance(result.get("issues"), list) or any(not isinstance(item, str) for item in result["issues"]):
        result["issues"] = []
        malformed.append("final QA issues were missing or malformed")
    if type(result.get("repairable")) is not bool:
        result["repairable"] = False
        malformed.append("final QA repairable verdict was not a boolean")
    if not isinstance(result.get("failed"), list):
        result["failed"] = []
        result["repairable"] = False
        malformed.append("final QA failed-shot list was malformed")
    if result["passed"] and not _qa_report_publishable(result, cfg, timeline, shots, n_segs):
        result["passed"] = False
        result["repairable"] = False
        malformed.append("final QA pass report was incomplete or inconsistent")
    if malformed:
        result["passed"] = False
        result["repairable"] = False
        result["issues"] = [*result["issues"], *malformed]
    return result


def _qa_repairable(report: dict) -> bool:
    failed = report.get("failed") if isinstance(report, dict) else None
    return (isinstance(report, dict) and report.get("passed") is False
            and report.get("repairable") is True and isinstance(failed, list) and bool(failed)
            and all(isinstance(item, (list, tuple)) and len(item) == 2
                    and type(item[0]) is int and item[0] >= 0
                    and isinstance(item[1], str) and item[1].strip() for item in failed))


def _qa_repair_feedback(report: dict, seg: int, path: str) -> str:
    """Turn final-frame findings into concrete search guidance, not a generic retry."""
    frames = [f for f in report.get("frames", [])
              if f.get("seg") == seg and str(f.get("path", "")) == str(path)]
    details = [
        f"pixels show {f.get('shows', 'unknown')!r}; intended shot was {f.get('intended_shot', 'unspecified')!r}; "
        f"QA issue: {f.get('issue', 'mismatch')}"
        for f in frames[:3]
    ]
    return "; ".join(details)[:900]


def _repair_failed_shots(a: dict, cfg: dict, writer: ScriptWriter, report: dict) -> bool:
    """Attempt every failed shot; a missing spare for one segment must not short-circuit the rest."""
    repaired = False
    segments = a["script"].get("segments", [])
    for seg, path in report.get("failed", []):
        segment = segments[seg] if 0 <= seg < len(segments) else None
        changed = media.use_alternative(
            a["visuals"], seg, path, a["work"] / "img", source=a.get("source"), segment=segment,
            cfg=cfg, llm=writer.llm, feedback=_qa_repair_feedback(report, seg, path),
            visual_subject=a.get("visual_subject", ""))
        repaired = bool(changed) or repaired
    return repaired


def render_with_qa(a: dict, cfg: dict, writer: ScriptWriter) -> tuple[dict, dict]:
    """Render and verify the finished MP4; repair only when another QA pass remains."""
    # Enforce the stage-specific hard ceiling even if config is accidentally raised.
    repairs = RecoveryBudgets.from_config(cfg).visual_repairs
    report: dict = {"passed": False}
    r: dict = {}
    for attempt in range(repairs + 1):
        _report_stage("render_qa")
        r = render.render(a["tts"], a["visuals"]["shots"], a["music"], a["hook_card"],
                          cfg["channel"]["name"], cfg, a["work"], a["out"], a["seed"], fx=a["fx"])
        report = _normalize_qa_report(
            qa.verify(a["out"], r["timeline"], a["visuals"]["shots"], a["tts"], a["script"],
                      a["title"], a["hook_card"], a["subject"], writer.llm, cfg,
                      trend_angle=(a.get("topic") or {}).get("angle") or "",
                      trend_evidence=_trend_qa_evidence(a.get("topic") or {})), cfg,
            timeline=r.get("timeline") if isinstance(r, dict) else None,
            shots=(a.get("visuals") or {}).get("shots") if isinstance(a.get("visuals"), dict) else None,
            n_segs=len(a["script"].get("segments", [])) if isinstance(a.get("script"), dict)
            and isinstance(a["script"].get("segments"), list) else None)
        report["attempt"] = attempt + 1
        if report["passed"] is True:
            log.info("  ✓ final QA passed: %d frames checked against the narration", len(report["frames"]))
            break
        log.info("  ✗ final QA attempt %d: %s", attempt + 1, "; ".join(report["issues"])[:500])
        if not _qa_repairable(report):
            _report_failure("render_qa", "final_qa_nonrepairable", attempt=attempt + 1)
            break                    # structural or malformed reports cannot be repaired by guessing at a shot
        if attempt >= repairs:
            _report_failure("render_qa", "visual_repair_budget_exhausted", attempt=attempt + 1)
            log.warning("  final QA attempt failed; leaving the rendered frames unchanged and unverified")
            break                    # every replacement must get a fresh render and QA pass
        if not _repair_failed_shots(a, cfg, writer, report):
            _report_failure("render_qa", "no_verified_shot_replacement", attempt=attempt + 1)
            break
        receipt = run_report.current_receipt()
        if receipt:
            receipt.note_retry("visual_repair", attempt + 1, repairs + 1, "replace_rejected_shots_and_rerender")
    return r, report


def _shots_sheet(visuals: dict, out) -> None:
    try:   # contact sheet of the shots actually used (kept with the run artifact)
        from PIL import Image
        from .vision import contact_sheet
        ims = []
        for v in visuals["shots"]:
            with Image.open(v.get("poster") or v["path"]) as im:
                im = im.convert("RGB")
                im.thumbnail((400, 400))
                ims.append(im)
        out.with_suffix(".shots.jpg").write_bytes(contact_sheet(ims, tile=300, cols=4))
    except Exception as e:  # noqa: BLE001
        log.debug("shots sheet failed: %s", e)


def _trend_provenance(topic: dict) -> dict:
    return {"sources": list(topic.get("sources") or []),
            "source_families": list(topic.get("source_families") or []),
            "score": topic.get("score"), "viral_score": topic.get("viral_score"),
            "angle": topic.get("angle"), "wikipedia_spike": topic.get("spike"),
            "why_trending": topic.get("why_trending"),
            "evidence": (topic.get("context") or [])[:4],
            "signals": (topic.get("signals") or [])[:8]}


def _history_trend(topic: dict) -> dict:
    return {"trend_sources": list(topic.get("sources") or []),
            "trend_source_families": list(topic.get("source_families") or []),
            "trend_score": topic.get("score"), "trend_angle": topic.get("angle"), "wikipedia_spike": topic.get("spike"),
            "why_trending": topic.get("why_trending"), "viral_score": topic.get("viral_score"),
            "trend_evidence": (topic.get("context") or [])[:2],
            "trend_signals": (topic.get("signals") or [])[:6]}


def _shot_provenance(shots: list[dict]) -> list[dict]:
    rows = []
    generated_fields = (
        "provider", "model", "api_version", "task_id", "prompt", "generated_at", "ratio", "audio",
        "requested_duration_seconds", "actual_duration_seconds", "output_width", "output_height",
        "credits_reserved", "estimated_cost_credits", "actual_cost_credits", "estimated_cost_usd",
        "file_sha256", "rights_confirmation",
    )
    for shot in shots:
        credit = shot.get("credit") if isinstance(shot.get("credit"), dict) else {}
        synthetic = shot.get("synthetic") is True or credit.get("synthetic") is True
        row = {"seg": shot["seg"], "kind": shot.get("kind", "image"), "window": shot.get("window"),
               "score": shot.get("score"), "judge": shot.get("judge"), "shows": shot.get("shows"),
               "want": shot.get("want"), "reused": shot.get("reused", False),
               "image": credit.get("title"), "page": credit.get("page"), "synthetic": synthetic}
        if synthetic:
            row["generation"] = {key: credit.get(key) for key in generated_fields if key in credit}
        rows.append(row)
    return rows


def _early_turn_diagnostic(script: dict, tts: dict, duration) -> dict:
    """Record where beat two starts in the finished Short; missing timing is non-blocking."""
    script_segments = script.get("segments") if isinstance(script, dict) else None
    tts_segments = tts.get("segments") if isinstance(tts, dict) else None
    start = None
    if isinstance(tts_segments, list) and len(tts_segments) > 1 and isinstance(tts_segments[1], dict):
        candidate = tts_segments[1].get("start")
        if _finite_range(candidate, 0, float("inf")):
            start = float(candidate)
    runtime_pct = None
    if start is not None and _finite_range(duration, 0.001, float("inf")):
        runtime_pct = round(start / float(duration) * 100, 1)
    turn_words = None
    if isinstance(script_segments, list) and len(script_segments) > 1 and isinstance(script_segments[1], dict):
        text = script_segments[1].get("text")
        if isinstance(text, str):
            turn_words = len(text.split())
    return {
        "turn_segment_index": 2,
        "turn_start_seconds": round(start, 3) if start is not None else None,
        "turn_start_runtime_pct": runtime_pct,
        "turn_factual_word_count": turn_words,
    }


def package(a: dict, r: dict, report: dict, cfg: dict) -> dict | None:
    """Thumbnail, metadata and the run artifact. Only a complete, explicit QA pass can be uploaded."""
    script, source, visuals, out = a["script"], a["source"], a["visuals"], a["out"]
    report = _normalize_qa_report(
        report, cfg, timeline=r.get("timeline") if isinstance(r, dict) else None,
        shots=visuals.get("shots") if isinstance(visuals, dict) else None,
        n_segs=len(script.get("segments", [])) if isinstance(script, dict)
        and isinstance(script.get("segments"), list) else None)
    thumb = render.thumbnail(r["first_frame"], a["hook_card"] or source["title"],
                             out.with_suffix(".jpg"), r["theme"])
    _shots_sheet(visuals, out)
    words = sum(len(spoken_text(x).split()) for x in script["segments"])
    early_turn = _early_turn_diagnostic(script, a.get("tts") or {}, r.get("duration"))
    contains_synthetic = bool(
        (cfg.get("compliance") or {}).get("contains_synthetic_media", False)
        or any(shot.get("synthetic") is True
               or (isinstance(shot.get("credit"), dict)
                   and shot["credit"].get("synthetic") is True)
               for shot in visuals.get("shots", []) if isinstance(shot, dict))
    )
    meta = {
        "title": a["title"],
        "description": build_description(script, source, visuals["shots"], cfg, a["tts"]["engine"]),
        "tags": list(dict.fromkeys([t.strip("#") for t in script.get("tags", [])] + ["shorts", "facts"]))[:25],
        "contains_synthetic_media": contains_synthetic,
    }
    record = {"meta": meta, "script": script, "review": a["review"],
              "series": a.get("series"), "episode": a.get("episode"),
              "source": {k: source[k] for k in ("title", "url")},
              "trend": _trend_provenance(a["topic"]),
              "plan": a["plan"], "tts_engine": a["tts"]["engine"], "duration": r["duration"],
              "archetype": r.get("archetype"), "contains_synthetic_media": contains_synthetic,
              "words": words, "early_turn": early_turn,
              "qa": {k: report.get(k) for k in ("passed", "attempt", "issues", "frames", "meta")},
              "shots": _shot_provenance(visuals["shots"]), "timeline": r["timeline"]}
    if report.get("passed") is not True:
        # keep the evidence in the run artifact, clearly marked, and never hand it to the uploader
        rejected = out.with_name(out.stem + "-REJECTED.mp4")
        out.replace(rejected)
        rejected.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
        log.info("  ✗ NOT publishing %r — final QA failed", source["title"])
        return None
    # Survived final QA, so it is a real episode and may take the next number.
    s_key = a.get("series")
    s_num = 0
    if s_key:
        s_num = series_mod.episode_number(s_key, read_json("history.json", []))
        meta["title"] = series_mod.decorate_title(meta["title"], s_key, s_num, cfg, limit=90)
    if "#shorts" not in meta["title"].lower() and len(meta["title"]) <= 90:
        meta["title"] = f"{meta['title']} #shorts"
    record["series"] = s_key
    record["episode"] = s_num
    record["meta"] = meta
    out.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
    log.info("  ✓ rendered %s (%.1fs, review=%s)", out.name, r["duration"], a["review"].get("score"))
    return {"file": out, "thumb": thumb, "meta": meta, "script": script, "source": source,
            "review": a["review"], "topic": a["topic"], "plan": a["plan"], "duration": r["duration"],
            "archetype": r.get("archetype"), "tts_engine": a["tts"]["engine"], "words": words,
            "early_turn": early_turn, "series": s_key, "episode": s_num,
            "qa": report}


def make_one(cfg: dict, writer: ScriptWriter, topic: dict, plan: dict, idx: int, run_id: str) -> dict | None:
    _report_stage("topic_validation")
    if not isinstance(cfg, dict) or not isinstance(topic, dict) or not isinstance(plan, dict):
        _report_failure("topic_validation", "malformed_topic_plan_or_config")
        log.warning("refusing malformed topic, plan, or configuration")
        return None
    trend_cfg = cfg.get("trends") or {}
    content_cfg = cfg.get("content") or {}
    if not isinstance(trend_cfg, dict) or not isinstance(content_cfg, dict):
        _report_failure("topic_validation", "malformed_trend_or_content_config")
        log.warning("refusing malformed trend or content configuration")
        return None
    min_wiki_raw = trend_cfg.get("min_wikipedia_spike", 3.0)
    min_viral_raw = content_cfg.get("min_viral_score", 7)
    if (not _finite_range(min_viral_raw, 7, 10) or not _finite_range(min_wiki_raw, 0, float("inf"))
            or min_wiki_raw <= 0):
        _report_failure("topic_validation", "invalid_trend_or_viral_threshold")
        log.warning("refusing invalid trend or viral-potential threshold configuration")
        return None
    min_wiki_spike = float(min_wiki_raw)
    min_viral = float(min_viral_raw)
    viral_raw = topic.get("viral_score")
    if not _finite_range(viral_raw, 0, 10):
        _report_failure("topic_validation", "invalid_viral_score")
        log.info("  ✗ refusing topic with invalid viral score")
        return None
    viral_score = float(viral_raw)
    if not isinstance(topic.get("topic"), str) or not topic["topic"].strip() \
            or not isinstance(topic.get("angle"), str) or not topic["angle"].strip():
        _report_failure("topic_validation", "missing_topic_or_trend_angle")
        log.info("  ✗ refusing topic without a name and a specific current-trend angle")
        return None
    if not trends.has_current_trend_evidence(topic, min_wiki_spike, trend_cfg):
        _report_failure("topic_validation", "current_trend_evidence_missing")
        log.info("  ✗ refusing topic %r without a qualifying live trend signal", topic.get("topic"))
        return None
    if viral_score < min_viral:
        _report_failure("topic_validation", "below_viral_potential_floor")
        log.info("  ✗ refusing topic %r below the viral-potential floor (%.1f/%.1f)",
                 topic.get("topic"), viral_score, min_viral)
        return None
    log.info("▶ [%d] topic=%r format=%s hook=%s voice=%s", idx, topic["topic"], plan["format"],
             plan["hook_style"], plan["voice"])
    assets = produce_assets(cfg, writer, topic, plan, idx, run_id)
    if not assets:
        return None
    r, report = render_with_qa(assets, cfg, writer)
    return package(assets, r, report, cfg)


def run(cfg: dict, count: int | None = None, upload: bool | None = None, keep_work: bool = False) -> list[dict]:
    run_id = now_utc().strftime("%Y%m%d-%H%M")
    n = count or daily_target(cfg)
    do_upload = cfg["upload"]["enabled"] if upload is None else upload
    budgets = RecoveryBudgets.from_config(cfg)
    receipt = run_report.current_receipt()
    if receipt:
        receipt.set_target(n)
        receipt.set_retry_budgets(budgets.report())
        receipt.set_stage("initialization")
    # Start the soft budget at pipeline entry so trend selection and initialization also
    # consume wall-clock. The outer Actions timeout remains the final hard ceiling.
    deadline = time.monotonic() + 60 * float(cfg["schedule"].get("max_run_minutes", 300))
    llm = LLM(cfg)
    strat = Strategy(cfg)
    writer = ScriptWriter(cfg, llm, strat)

    # 1. trends
    _report_stage("trends")
    recent = trends.recent_topics(cfg["content"]["dedupe_days"])
    cands = trends.collect(cfg)
    if not cands:
        _report_failure("trends", "no_live_trend_candidates")
    topic_hint = os.environ.get("AUTOTUBE_TOPIC_HINT", "").strip()
    if topic_hint:
        found = len(cands)
        cands = filter_topic_candidates_by_hint(cands, topic_hint)
        log.info("topic focus matched %d/%d freshly collected candidate(s)", len(cands), found)
        if not cands:
            _report_failure("topic_selection", "topic_hint_no_live_match")
    _report_stage("topic_selection")
    picks = writer.select_topics(cands, n, recent) if cands else []
    if cands and not picks:
        _report_failure("topic_selection", "no_eligible_trending_topics")
    tried: set[str] = set()
    rounds = 1
    plans = strat.plan(n)
    log.info("topic queue: %s", [p["topic"][:40] for p in picks[: n + 4]])

    # A configured key/model is not proof that image input works today. Probe one tiny
    # image before spending more text calls on scripts or synthesizing narration; this
    # is only a readiness check, never a substitute for per-shot and final-render QA.
    if picks and cfg.get("media", {}).get("require_llm_verdict", True):
        _report_stage("vision_preflight")
        try:
            route = llm.preflight_vision()
            log.info("  ✓ live vision preflight passed via %s; all visual QA gates remain mandatory", route)
        except Exception as e:  # noqa: BLE001 — strict mode stops before script/TTS investment
            _report_failure("vision_preflight", "vision_preflight_unavailable", e)
            log.error("  ✗ live vision preflight unavailable before scripting/TTS: %s", str(e)[:420])
            return []

    # 2. produce
    _report_stage("topic_production")
    series_mod.reset()          # episode numbers are per-run unique, see series._ISSUED
    results, used_titles, used_formats = [], set(), set()
    pi = 0
    vision_down = False

    def select_more_topics() -> bool:
        """Replenish the fallback queue within the existing bounded selection budget."""
        nonlocal rounds, picks
        if (rounds >= budgets.selection_rounds or not cands or time.monotonic() > deadline
                or not llm.has_available_provider()):
            return False
        current_receipt = run_report.current_receipt()
        if current_receipt:
            current_receipt.set_stage("topic_selection")
            current_receipt.note_retry("topic_selection", rounds, budgets.selection_rounds,
                                       "select_fresh_live_candidates")
        rounds += 1
        # Exclude both attempted topics and any picks already waiting in the queue. Early
        # replenishment must not enqueue duplicates of the still-untried first-round picks.
        exclude = tried | {str(row.get("topic_key") or "") for row in picks[pi:]}
        more = writer.select_topics(cands, n - len(results), recent, exclude=exclude)
        if not more:
            _report_failure("topic_selection", "no_more_eligible_trending_topics")
            return False
        log.info("selection round %d: %d more viral picks", rounds, len(more))
        picks += more
        return True

    for i in range(n):
        if vision_down:
            break
        plan = plans[i]
        slot_attempts = 0
        while True:
            if not llm.has_available_provider():
                _report_failure("script", "all_text_providers_circuit_open_or_unusable")
                break
            if slot_attempts >= budgets.topics_per_video:
                _report_failure("topic_selection", "topic_attempt_budget_exhausted", attempt=slot_attempts)
                log.warning("topic retry budget exhausted for slot %d (%d topics)", i + 1, slot_attempts)
                break
            if pi >= len(picks):
                if not select_more_topics():
                    if rounds >= budgets.selection_rounds or not cands or time.monotonic() > deadline:
                        _report_failure("topic_selection", "selection_round_budget_exhausted")
                    break
            reserve = 60 * float(cfg["schedule"].get("topic_reserve_minutes", 45))
            if time.monotonic() > deadline - reserve:
                _report_failure("topic_production", "soft_deadline_reserve_reached")
                log.warning("time budget reached (within %.0f min reserve) — stopping with %d videos",
                            reserve / 60, len(results))
                break
            topic = picks[pi]
            pi += 1
            slot_attempts += 1
            tried.add(topic.get("topic_key", ""))
            if receipt:
                receipt.set_topic(topic, i + 1)
            topic["category"] = topic.get("category") or random.choice(cfg["channel"]["categories"])
            fitted = strat.fit(plan, topic, used_formats)
            try:
                res = make_one(cfg, writer, topic, fitted, i + 1, run_id)
            except VisionUnavailable as e:
                _report_failure("visuals", "vision_unavailable", e, slot_attempts)
                if receipt:
                    receipt.note_topic("rejected", category="vision_unavailable")
                log.error("  ✗ %s — stopping this run (nothing unverified is published; the next run retries)", e)
                vision_down = True
                break
            except Exception as e:  # noqa: BLE001
                stage = receipt.current_stage if receipt else "topic_production"
                category = run_report.classify_failure(stage, str(e))
                _report_failure(stage, category, e, slot_attempts)
                log.exception("  ✗ failed on %r: %s", topic["topic"], e)
                res = None
            if res and res["source"]["title"] not in used_titles:
                used_titles.add(res["source"]["title"])
                used_formats.add(res["plan"]["format"])
                results.append(res)
                if receipt:
                    receipt.note_topic("accepted")
                    receipt.data["produced_count"] = len(results)
                break
            if res:
                _report_failure("topic_production", "duplicate_source_in_run", attempt=slot_attempts)
            if receipt:
                failure = receipt.last_failure or {}
                receipt.note_topic("rejected", category=failure.get("category", "topic_not_produced"))
            if (pi < len(picks) and rounds < budgets.selection_rounds
                    and llm.has_available_provider()):
                log.info("topic rejected; replenishing the fallback queue while existing picks remain")
                select_more_topics()
    log.info("produced %d/%d videos", len(results), n)

    # 3. upload (scheduled)
    _report_stage("upload")
    hist = read_json("history.json", [])
    slots = publish_slots(cfg, len(results), taken_slots(hist))
    for res, slot in zip(results, slots):
        entry = {
            "created_at": now_utc().isoformat(), "run_id": run_id, "title": res["meta"]["title"],
            "topic": res["topic"]["topic"], "topic_key": res["topic"].get("topic_key", res["topic"]["topic"].lower()),
            "wiki_title": res["source"]["title"],
            "choice": {"category": res["topic"].get("category"), **res["plan"],
                       "source": trends.primary_source(res["topic"].get("sources", []))},
            **_history_trend(res["topic"]),
            "review_score": res["review"].get("score"),
            # Optional reviewer telemetry: stored for later analysis, never part of the publish decision.
            "turn_strength": res["review"].get("turn_strength"),
            "turn_strength_status": res["review"].get("turn_strength_status", "missing"),
            "turn_strength_reason": res["review"].get("turn_strength_reason", ""),
            "early_turn": res.get("early_turn") or {},
            "trend_alignment": res["review"].get("trend_alignment"),
            "trend_alignment_reason": res["review"].get("trend_alignment_reason"),
            "reviewer": res["review"].get("reviewer"),
            "review_independent": res["review"].get("independent"),
            "series": res.get("series"), "episode": res.get("episode"),
            # Phrase sketch of what was actually said, so the next run can refuse to
            # publish a near-duplicate. Fixed width, so history cannot grow without bound.
            "sketch": originality.sketch(originality.narration_of(res["script"])),
            # Recorded now, not derived later: once this entry joins history nobody can
            # reconstruct what the back catalogue looked like when it was written.
            "max_similarity": _max_similarity(res["script"], hist),
            "commentary_ratio": round(originality.commentary_ratio(res["script"]), 3),
            "duration": res["duration"], "words": res.get("words"), "archetype": res.get("archetype"),
            "tts_engine": res["tts_engine"], "file": res["file"].name, "publish_at": slot.isoformat() if slot else None,
            "llm": llm.last_used,
        }
        if not _qa_pass_is_explicit(res):
            _report_failure("upload", "explicit_final_qa_pass_missing")
            log.error("refusing to upload %s: final QA did not pass", res["file"].name)
            continue
        if do_upload:
            from . import youtube
            try:
                entry["video_id"] = youtube.upload(res["file"], res["meta"], cfg, slot, res["thumb"])
            except Exception as e:  # noqa: BLE001
                _report_failure("upload", run_report.classify_failure("upload", str(e)), e)
                if receipt:
                    receipt.data["upload_failure_count"] = int(receipt.data.get("upload_failure_count", 0)) + 1
                log.error("upload failed: %s", e)
                entry["status"] = f"upload_failed: {str(e)[:200]}"
            else:
                try:
                    observed = youtube.verify_upload(entry["video_id"], slot, cfg)
                except Exception as e:  # noqa: BLE001 — do not claim a schedule without a YouTube status read
                    _report_failure("upload_verification", "youtube_status_unverified", e)
                    if receipt:
                        receipt.data["upload_failure_count"] = int(receipt.data.get("upload_failure_count", 0)) + 1
                    log.error("uploaded %s but YouTube publication state was not verified: %s",
                              entry["video_id"], str(e)[:200])
                    entry["status"] = f"upload_failed_verification: {str(e)[:160]}"
                else:
                    entry["youtube_status"] = {key: observed.get(key) for key in
                                               ("video_id", "privacy_status", "upload_status", "publish_at")}
                    entry["status"] = "scheduled"
                    if receipt:
                        receipt.data["scheduled_count"] = int(receipt.data.get("scheduled_count", 0)) + 1
        else:
            entry["status"] = "rendered"
        res["entry"] = entry
        if do_upload:
            # The appeal packet, written at the moment the evidence still exists. After a
            # strike the work directory is long gone and the model that wrote the script
            # has been deprecated; the only surviving artefact would be the video itself,
            # which is the thing under suspicion.
            #
            # Gated on video_id, not on do_upload: an upload that raised leaves a fully
            # rendered video that was never published, and recording it would have the
            # dossier claim episodes that are not on the channel. A reviewer who checks
            # one claimed episode and cannot find it discredits the whole archive, so
            # inflated evidence is worse here than missing evidence.
            if entry.get("video_id"):
                entry["provenance"] = provenance.record(res, entry, cfg)
            hist.append(entry)   # dry runs never touch history (no dedupe / slot side effects)
    if do_upload:
        write_json("history.json", hist[-int(cfg.get("analytics", {}).get("history_keep", 2000)):])
    _report_stage("cleanup")
    if not keep_work:
        shutil.rmtree(WORK_DIR / run_id, ignore_errors=True)
    return results
