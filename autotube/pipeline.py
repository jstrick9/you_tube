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
import shutil
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from . import media, music, render, trends
from .common import OUTPUT_DIR, WORK_DIR, now_utc, read_json, slugify, write_json
from .llm import LLM
from .scriptwriter import ScriptWriter
from .strategy import Strategy
from .tts import synthesize

log = logging.getLogger("autotube.pipeline")


def _slot_key(ts) -> str:
    d = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
    return d.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M")


def taken_slots(hist: list[dict]) -> set[str]:
    """Publish slots already used by successfully uploaded videos that haven't gone live yet."""
    now = now_utc()
    out = set()
    for h in hist:
        if h.get("video_id") and h.get("publish_at"):
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


def build_description(script: dict, source: dict, visuals: list[dict], cfg: dict, tts_engine: str) -> str:
    tags = " ".join(h if h.startswith("#") else f"#{h}" for h in script.get("hashtags", [])[:3])
    parts = [
        script.get("description", "").strip(),
        "",
        f"📚 Source: {source['title']} — {source['url']}",
    ]
    if cfg["compliance"].get("ai_disclosure_in_description", True):
        parts += ["", "ℹ️ This video was researched from the cited source and produced with AI-assisted "
                      "scripting and a synthetic narrator voice. Facts are checked against the source before publishing."]
    parts += ["", "🖼️ Image credits:", media.credits_text(visuals),
              "🎵 Music: original, procedurally generated for this video.", "",
              f"Follow {cfg['channel'].get('handle') or cfg['channel']['name']} for a surprising fact, with sources, every day!", "", f"{tags} #shorts"]
    return "\n".join(parts)


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

    tts = synthesize([s["text"] for s in script["segments"]], plan["voice"], cfg["video"]["speech_rate"], work)
    lo, hi = cfg["video"]["target_seconds"]
    if tts["duration"] > 59.0:      # Shorts must stay < 60 s for safest classification
        log.info("  narration %.1fs too long → speeding up", tts["duration"])
        faster = f"+{int(cfg['video']['speech_rate'].strip('+%')) + int((tts['duration'] / 57 - 1) * 100) + 4}%"
        tts = synthesize([s["text"] for s in script["segments"]], plan["voice"], faster, work)
        if tts["duration"] > 59.0:
            log.info("  ✗ still too long (%.1fs), skipping", tts["duration"])
            return None

    visuals = media.gather(source, script["segments"], cfg, work / "img")
    mus = music.generate(tts["duration"] + 1, work / "music.wav", seed) if cfg["video"]["background_music"] else None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{run_id}-{idx:02d}-{slug}.mp4"
    hook_card = script.get("thumbnail_text") or ""
    r = render.render(tts, visuals, mus, hook_card, cfg["channel"]["name"], cfg, work, out, seed)
    thumb = render.thumbnail(r["first_frame"], hook_card or source["title"], out.with_suffix(".jpg"), r["theme"])

    title = script["title"].strip()
    if "#shorts" not in title.lower() and len(title) <= 90:
        title = f"{title} #shorts"
    meta = {
        "title": title,
        "description": build_description(script, source, visuals, cfg, tts["engine"]),
        "tags": list(dict.fromkeys([t.strip("#") for t in script.get("tags", [])] + ["shorts", "facts"]))[:25],
    }
    (out.with_suffix(".json")).write_text(json.dumps({"meta": meta, "script": script, "review": review,
                                                       "source": {k: source[k] for k in ("title", "url")},
                                                       "plan": plan, "tts_engine": tts["engine"],
                                                       "duration": r["duration"]}, indent=2))
    log.info("  ✓ rendered %s (%.1fs, review=%s)", out.name, r["duration"], review.get("score"))
    return {"file": out, "thumb": thumb, "meta": meta, "script": script, "source": source, "review": review,
            "topic": topic, "plan": plan, "duration": r["duration"], "tts_engine": tts["engine"]}


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
    picks += writer.evergreen(recent)     # safety net
    plans = strat.plan(n)
    log.info("topic queue: %s", [p["topic"][:40] for p in picks[: n + 4]])

    # 2. produce
    results, used_titles = [], set()
    pi = 0
    deadline = time.time() + 60 * float(cfg["schedule"].get("max_run_minutes", 120))
    for i in range(n):
        plan = plans[i]
        while pi < len(picks):
            if time.time() > deadline:
                log.warning("time budget reached — stopping with %d videos", len(results))
                break
            topic = picks[pi]
            pi += 1
            topic["category"] = topic.get("category") or random.choice(cfg["channel"]["categories"])
            try:
                res = make_one(cfg, writer, topic, plan, i + 1, run_id)
            except Exception as e:  # noqa: BLE001
                log.exception("  ✗ failed on %r: %s", topic["topic"], e)
                res = None
            if res and res["source"]["title"] not in used_titles:
                used_titles.add(res["source"]["title"])
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
            "choice": {"category": res["topic"].get("category"), **res["plan"]},
            "review_score": res["review"].get("score"), "duration": res["duration"],
            "tts_engine": res["tts_engine"], "file": res["file"].name, "publish_at": slot.isoformat() if slot else None,
            "llm": llm.last_used,
        }
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
        hist.append(entry)
    write_json("history.json", hist[-2000:])
    if not keep_work:
        shutil.rmtree(WORK_DIR / run_id, ignore_errors=True)
    return results
