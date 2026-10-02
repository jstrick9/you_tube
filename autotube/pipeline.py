"""Orchestrator: trends → plan → ground → script → review → TTS → visuals → render → upload.

One run produces `schedule.videos_per_day` videos and schedules each to go public
at the configured local time slots (YouTube handles publishing via publishAt), so
a single daily job is enough — no always-on server needed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import media, music, qa, render, trends
from .vision import VisionUnavailable
from .common import OUTPUT_DIR, WORK_DIR, now_utc, read_json, slugify, write_json
from .llm import LLM
from .scriptwriter import ScriptWriter, spoken_text
from .strategy import Strategy
from .tts import synthesize

log = logging.getLogger("autotube.pipeline")


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


def publish_slots(cfg: dict, n: int, taken: set[str] | None = None) -> list[datetime | None]:
    """Next n free publish slots (local schedule times), skipping past slots and ones already booked."""
    tz = ZoneInfo(cfg["channel"].get("timezone", "UTC"))
    now_local = now_utc().astimezone(tz)
    taken = taken or set()
    slots = []
    times = cfg["schedule"]["publish_times"]
    day = now_local.date()
    for _ in range(60):                                  # hard stop: never loop forever
        for t in times:
            hh, mm = map(int, t.split(":"))
            dt = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz)
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
    return max(0, int(cfg["schedule"]["videos_per_day"]) - done)


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
    catch = (cfg.get("persona") or {}).get("catchphrase")
    parts = [
        script.get("description", "").strip(),
        "",
        *([f"✅ {catch} Every claim in this video is checked against the source below.", ""] if catch else []),
        f"📚 Source: {source['title']} — {source['url']}",
        *[f"📚 Also: {a['title']} — {a['url']}" for a in source.get("also", [])[:4]],
    ]
    if cfg["compliance"].get("ai_disclosure_in_description", True):
        parts += ["", "ℹ️ This video was researched from the cited source and produced with AI-assisted "
                      "scripting and a synthetic narrator voice. Facts are checked against the source before publishing."]
    parts += ["", "🖼️ Image & video credits:", media.credits_text(visuals),
              "🎵 Music: original, procedurally generated for this video.", "",
              f"Follow {cfg['channel'].get('handle') or cfg['channel']['name']} for a surprising fact, with sources, every day!", "", f"{tags} #shorts"]
    return clean_description("\n".join(parts))


def pause_plan(script: dict, plan: dict, cfg: dict) -> list[float]:
    """Silence before each line: normal, except a beat before the reveal (longer for the 3-2-1 countdown)."""
    gaps = [0.0] * len(script["segments"])
    for i, s in enumerate(script["segments"]):
        if s.get("reveal") and i > 0:
            key = "countdown_pause" if plan.get("format") == "guess_reveal" else "reveal_pause"
            gaps[i] = float(cfg["content"].get(key, 0.55))
    return gaps


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


def make_one(cfg: dict, writer: ScriptWriter, topic: dict, plan: dict, idx: int, run_id: str) -> dict | None:
    log.info("▶ [%d] topic=%r format=%s hook=%s voice=%s", idx, topic["topic"], plan["format"],
             plan["hook_style"], plan["voice"])
    produced = writer.produce(topic, plan)
    if not produced:
        log.info("  ✗ could not produce a verified script for %r", topic["topic"])
        return None
    script, source, review = produced
    slug = slugify(source["title"])
    work = WORK_DIR / run_id / f"{idx:02d}-{slug}"
    work.mkdir(parents=True, exist_ok=True)
    seed = int(hashlib.md5(f"{run_id}{slug}".encode()).hexdigest()[:8], 16)

    gaps = pause_plan(script, plan, cfg)
    lo, hi = cfg["video"]["target_seconds"]
    tts = synthesize([spoken_text(s) for s in script["segments"]], plan["voice"],
                     cfg["video"]["speech_rate"], work, gaps=gaps)
    # Completion rate is the dominant Shorts ranking input and it falls off fast with length, so hold the
    # configured band. Trim by DROPPING the least load-bearing body line (never the hook, reveal or payoff)
    # and re-synthesizing — speeding the voice up past the configured rate sounds robotic and costs retention too.
    for _ in range(3):
        if tts["duration"] <= hi + 2.0:
            break
        i = weakest_segment(script["segments"])
        if i is None:
            break
        log.info("  narration %.1fs > %.1fs target → dropping body line %d: %r",
                 tts["duration"], hi + 2.0, i + 1, script["segments"][i]["text"][:60])
        script["segments"].pop(i)
        gaps = pause_plan(script, plan, cfg)
        tts = synthesize([spoken_text(s) for s in script["segments"]], plan["voice"],
                         cfg["video"]["speech_rate"], work, gaps=gaps)
    if tts["duration"] > 59.0:      # Shorts must stay < 60 s for safest classification
        log.info("  narration %.1fs still too long → speeding up", tts["duration"])
        faster = f"+{int(cfg['video']['speech_rate'].strip('+%')) + int((tts['duration'] / 57 - 1) * 100) + 4}%"
        tts = synthesize([spoken_text(s) for s in script["segments"]], plan["voice"], faster, work, gaps=gaps)
        if tts["duration"] > 59.0:
            log.info("  ✗ still too long (%.1fs), skipping", tts["duration"])
            return None
    lines = [spoken_text(s) for s in script["segments"]]

    tsegs = tts["segments"]
    starts = [x["start"] for x in tsegs] + [tts["duration"]]
    seg_durs = [starts[i + 1] - starts[i] for i in range(len(tsegs))]
    try:
        visuals = media.gather(source, script["segments"], cfg, work / "img", llm=writer.llm, seg_durs=seg_durs)
    except media.NoVisualMatch as e:
        log.info("  ✗ visuals: %s — skipping topic", e)
        return None
    mus = music.generate(tts["duration"] + 1, work / "music.wav", seed) if cfg["video"]["background_music"] else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{run_id}-{idx:02d}-{slug}.mp4"
    hook_card = script.get("thumbnail_text") or ""
    title = script["title"].strip()
    if "#shorts" not in title.lower() and len(title) <= 90:
        title = f"{title} #shorts"
    subject = source["title"]
    fx = effects_plan(script, plan, source, cfg)

    # render → final QA on the finished file → (repair a failed shot with a verified spare, re-render) → …
    repairs = int(cfg.get("qa", {}).get("max_repairs", 2))
    report: dict = {"passed": False}
    for attempt in range(repairs + 1):
        r = render.render(tts, visuals["shots"], mus, hook_card, cfg["channel"]["name"], cfg, work, out, seed, fx=fx)
        report = qa.verify(out, r["timeline"], visuals["shots"], tts, script, title, hook_card, subject,
                           writer.llm, cfg)
        report["attempt"] = attempt + 1
        if report["passed"]:
            log.info("  ✓ final QA passed: %d frames checked against the narration", len(report["frames"]))
            break
        log.info("  ✗ final QA attempt %d: %s", attempt + 1, "; ".join(report["issues"])[:500])
        if not report["failed"] or len(report["failed"]) < len(report["issues"]):
            break                    # a non-image problem (narration/title/hook) can't be fixed by swapping photos
        if not all(media.use_alternative(visuals, seg, path, work / "img") for seg, path in report["failed"]):
            break

    thumb = render.thumbnail(r["first_frame"], hook_card or source["title"], out.with_suffix(".jpg"), r["theme"])
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

    meta = {
        "title": title,
        "description": build_description(script, source, visuals["shots"], cfg, tts["engine"]),
        "tags": list(dict.fromkeys([t.strip("#") for t in script.get("tags", [])] + ["shorts", "facts"]))[:25],
    }
    record = {"meta": meta, "script": script, "review": review, "source": {k: source[k] for k in ("title", "url")},
              "plan": plan, "tts_engine": tts["engine"], "duration": r["duration"],
              "qa": {k: report.get(k) for k in ("passed", "attempt", "issues", "frames", "meta")},
              "shots": [{"seg": v["seg"], "kind": v.get("kind", "image"), "window": v.get("window"),
                         "score": v.get("score"), "judge": v.get("judge"), "shows": v.get("shows"),
                         "want": v.get("want"), "reused": v.get("reused", False), "image": v["credit"].get("title"),
                         "page": v["credit"].get("page")} for v in visuals["shots"]],
              "timeline": r["timeline"]}
    if not report["passed"]:
        # keep the evidence in the run artifact, clearly marked, and never hand it to the uploader
        rejected = out.with_name(out.stem + "-REJECTED.mp4")
        out.replace(rejected)
        rejected.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
        log.info("  ✗ NOT publishing %r — final QA failed", source["title"])
        return None
    out.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
    log.info("  ✓ rendered %s (%.1fs, review=%s)", out.name, r["duration"], review.get("score"))
    return {"file": out, "thumb": thumb, "meta": meta, "script": script, "source": source, "review": review,
            "topic": topic, "plan": plan, "duration": r["duration"], "tts_engine": tts["engine"], "qa": report}


def run(cfg: dict, count: int | None = None, upload: bool | None = None, keep_work: bool = False) -> list[dict]:
    run_id = now_utc().strftime("%Y%m%d-%H%M")
    n = count or cfg["schedule"]["videos_per_day"]
    do_upload = cfg["upload"]["enabled"] if upload is None else upload
    llm = LLM(cfg)
    strat = Strategy(cfg)
    writer = ScriptWriter(cfg, llm, strat)

    # 1. trends
    recent = trends.recent_topics(cfg["content"]["dedupe_days"])
    cands = trends.collect(cfg)
    picks = writer.select_topics(cands, n, recent) if cands else []
    tried: set[str] = set()
    rounds = 1
    if cfg["content"].get("allow_evergreen", False):
        picks += writer.evergreen(recent)     # optional safety net (off by default: every video rides a live trend)
    plans = strat.plan(n)
    log.info("topic queue: %s", [p["topic"][:40] for p in picks[: n + 4]])

    # 2. produce
    results, used_titles, used_formats = [], set(), set()
    pi = 0
    deadline = time.time() + 60 * float(cfg["schedule"].get("max_run_minutes", 120))
    vision_down = False
    for i in range(n):
        if vision_down:
            break
        plan = plans[i]
        while True:
            if pi >= len(picks):
                # every pick so far failed the accuracy/visual checks → pick again from the remaining live trends
                if rounds >= int(cfg["content"].get("selection_rounds", 3)) or not cands or time.time() > deadline:
                    break
                rounds += 1
                more = writer.select_topics(cands, n - len(results), recent, exclude=tried)
                if not more:
                    break
                log.info("selection round %d: %d more viral picks", rounds, len(more))
                picks += more
            if time.time() > deadline:
                log.warning("time budget reached — stopping with %d videos", len(results))
                break
            topic = picks[pi]
            pi += 1
            tried.add(topic.get("topic_key", ""))
            topic["category"] = topic.get("category") or random.choice(cfg["channel"]["categories"])
            fitted = strat.fit(plan, topic, used_formats)
            try:
                res = make_one(cfg, writer, topic, fitted, i + 1, run_id)
            except VisionUnavailable as e:
                log.error("  ✗ %s — stopping this run (nothing unverified is published; the next run retries)", e)
                vision_down = True
                break
            except Exception as e:  # noqa: BLE001
                log.exception("  ✗ failed on %r: %s", topic["topic"], e)
                res = None
            if res and res["source"]["title"] not in used_titles:
                used_titles.add(res["source"]["title"])
                used_formats.add(res["plan"]["format"])
                results.append(res)
                break
    log.info("produced %d/%d videos", len(results), n)

    # 3. upload (scheduled)
    hist = read_json("history.json", [])
    slots = publish_slots(cfg, len(results), taken_slots(hist))
    for res, slot in zip(results, slots):
        entry = {
            "created_at": now_utc().isoformat(), "run_id": run_id, "title": res["meta"]["title"],
            "topic": res["topic"]["topic"], "topic_key": res["topic"].get("topic_key", res["topic"]["topic"].lower()),
            "wiki_title": res["source"]["title"], "trend_sources": res["topic"].get("sources", []),
            "choice": {"category": res["topic"].get("category"), **res["plan"],
                       "source": trends.primary_source(res["topic"].get("sources", []))},
            "viral_score": res["topic"].get("viral_score"), "why_trending": res["topic"].get("why_trending"),
            "trend_evidence": (res["topic"].get("context") or [])[:2],
            "review_score": res["review"].get("score"),
            "reviewer": res["review"].get("reviewer"),
            "review_independent": res["review"].get("independent"),
            "duration": res["duration"],
            "tts_engine": res["tts_engine"], "file": res["file"].name, "publish_at": slot.isoformat() if slot else None,
            "llm": llm.last_used,
        }
        if not (res.get("qa") or {}).get("passed"):
            log.error("refusing to upload %s: final QA did not pass", res["file"].name)
            continue
        if do_upload:
            from . import youtube
            try:
                entry["video_id"] = youtube.upload(res["file"], res["meta"], cfg, slot, res["thumb"])
                entry["status"] = "scheduled"
            except Exception as e:  # noqa: BLE001
                log.error("upload failed: %s", e)
                entry["status"] = f"upload_failed: {str(e)[:200]}"
        else:
            entry["status"] = "rendered"
        res["entry"] = entry
        if do_upload:            # dry runs never touch history (no topic dedupe / slot booking side effects)
            hist.append(entry)
    if do_upload:
        write_json("history.json", hist[-2000:])
    if not keep_work:
        shutil.rmtree(WORK_DIR / run_id, ignore_errors=True)
    return results
