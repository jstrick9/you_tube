"""The appeal packet.

Enforcement against faceless channels is automated, channel-wide and retroactive, and
honest creators get caught in the sweeps. The thing that distinguishes a successful
appeal from a terminated channel is documentary evidence of originality: drafts,
sources, project files - proof that a human-designed process produced each video rather
than a template emitting variations.

That evidence has to exist *before* it is needed. After a strike the work directory has
long since been cleaned up, the model that wrote the script has been deprecated, and the
only surviving artefact is the video itself, which is precisely the thing under
suspicion. So every published episode writes a small, permanent record of how it came to
exist, at the moment it exists.

Deliberately not stored here: the rendered video (it is already on YouTube and would
dwarf everything else) and the raw source page text (it is reachable from the URL, and
copying it is the one thing that would make the archive look like scraped content). What
is stored is the chain of decisions - why this topic, from which trend signal, grounded
in which sources, scored how, by which model, and how different it was from everything
published before it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .common import STATE_DIR

DIR = STATE_DIR / "provenance"


def _slug(text: str, n: int = 40) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in (text or "").lower()]
    return "".join(keep).strip("-")[:n] or "episode"


def record(res: dict, entry: dict, cfg: dict) -> str:
    """Write one episode's provenance file. Returns its path, or "" if it could not be written.

    Never raises. A failure to write the archive must not take down a run that has
    already rendered and uploaded a video - the record is insurance, and insurance that
    can destroy the thing it insures is worse than none.
    """
    try:
        DIR.mkdir(parents=True, exist_ok=True)
        script = res.get("script") or {}
        source = res.get("source") or {}
        topic = res.get("topic") or {}
        review = res.get("review") or {}
        doc = {
            "schema": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "video": {
                "title": entry.get("title"), "video_id": entry.get("video_id"),
                "series": entry.get("series"), "episode": entry.get("episode"),
                "duration": entry.get("duration"), "file": entry.get("file"),
                "published_at": entry.get("publish_at"),
            },
            # Why this topic rather than any other: the trend signal that surfaced it.
            # This is what demonstrates editorial selection instead of bulk enumeration.
            "selection": {
                "topic": topic.get("topic"), "category": topic.get("category"),
                "trend_sources": topic.get("sources"), "why_trending": topic.get("why_trending"),
                "viral_score": topic.get("viral_score"), "evidence": (topic.get("context") or [])[:3],
                "format": (res.get("plan") or {}).get("format"),
                "hook_style": (res.get("plan") or {}).get("hook_style"),
            },
            # Where every claim came from, claim by claim. The single most useful section
            # in an appeal, because it answers the accusation directly.
            "grounding": {
                "primary": {"title": source.get("title"), "url": source.get("url")},
                "corroboration": [{"title": a.get("title"), "url": a.get("url")}
                                  for a in (source.get("also") or [])],
                "claims": [{"says": s.get("text"), "evidence": s.get("evidence")}
                           for s in (script.get("segments") or [])],
            },
            # The script as published, plus the narrator's own commentary separated out,
            # since "no commentary or insight" is the specific allegation it rebuts.
            "script": {
                "title": script.get("title"),
                "segments": [s.get("text") for s in (script.get("segments") or [])],
                "asides": [s.get("aside") for s in (script.get("segments") or []) if (s.get("aside") or "").strip()],
                "commentary_ratio": entry.get("commentary_ratio"),
            },
            # Proof that each episode was checked and scored, and that the check was not
            # the same model marking its own homework.
            "review": {
                "score": review.get("score"), "reviewer": review.get("reviewer"),
                "independent": review.get("independent"), "issues": review.get("issues"),
            },
            # Which models, so a record written today is still interpretable after they
            # are deprecated and renamed.
            "pipeline": {
                "llm": entry.get("llm"), "tts_engine": entry.get("tts_engine"),
                "voice": (cfg.get("persona") or {}).get("voice"),
                "run_id": entry.get("run_id"),
            },
            # Evidence of difference, recorded at the time. After the fact nobody can
            # reconstruct what the back catalogue looked like when this was written.
            "originality": {
                "sketch": entry.get("sketch"),
                "max_similarity_to_prior": entry.get("max_similarity"),
            },
        }
        name = f"{(entry.get('created_at') or '')[:10]}-{_slug(entry.get('title') or '')}.json"
        path = DIR / name
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return str(path)
    except Exception:
        return ""


def dossier(limit: int = 0) -> dict:
    """Aggregate the archive into the summary an appeal actually opens with.

    An appeal reviewer will not read 300 episode files. They want the channel-level
    claim - every episode sourced, every episode independently reviewed, no episode
    a rewrite of another - with the per-episode files underneath as the proof.
    """
    files = sorted(DIR.glob("*.json")) if DIR.exists() else []
    if limit:
        files = files[-limit:]
    docs = []
    for f in files:
        try:
            docs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    sourced = sum(1 for d in docs if (d.get("grounding") or {}).get("primary", {}).get("url"))
    independent = sum(1 for d in docs if (d.get("review") or {}).get("independent"))
    with_commentary = sum(1 for d in docs if (d.get("script") or {}).get("asides"))
    sims = [s for s in ((d.get("originality") or {}).get("max_similarity_to_prior") for d in docs)
            if isinstance(s, (int, float))]
    return {
        "episodes": len(docs),
        "with_cited_primary_source": sourced,
        "independently_reviewed": independent,
        "with_narrator_commentary": with_commentary,
        "highest_similarity_between_any_episode_and_its_predecessors": round(max(sims), 3) if sims else None,
        "distinct_sources": len({(d.get("grounding") or {}).get("primary", {}).get("url") for d in docs} - {None}),
        "series": sorted({(d.get("video") or {}).get("series") for d in docs} - {None}),
        "earliest": (docs[0].get("recorded_at") if docs else None),
        "latest": (docs[-1].get("recorded_at") if docs else None),
    }
