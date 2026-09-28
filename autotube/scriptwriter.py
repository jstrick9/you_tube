"""Topic selection + grounded scriptwriting + independent quality/compliance review."""
from __future__ import annotations

import json
import logging
import random
import re

from .llm import LLM, LLMError
from .research import ground, unsupported_numbers
from .strategy import Strategy

log = logging.getLogger("autotube.script")

FORMAT_GUIDE = {
    "facts3": "Three surprising, escalating facts about the subject. Fact 3 must be the most mind-blowing.",
    "backstory": "The little-known origin story / how it came to be, told as a mini narrative with a twist at the end.",
    "myth_vs_fact": "Open with a common misconception, then debunk it with what the source actually says, then one bonus fact.",
    "timeline": "A rapid-fire timeline of 4-5 key moments, each with its year (years must come from the source).",
    "by_the_numbers": "Structure the video around 3-4 striking numbers from the source, explaining what each means in human terms.",
    "what_if": "Pose a vivid 'what would happen if' or 'imagine' framing grounded in real facts from the source, then reveal the real facts.",
}
HOOK_GUIDE = {
    "question": "Open with an intriguing question the viewer can't answer yet.",
    "bold_claim": "Open with a bold, surprising (but TRUE per the source) statement.",
    "number_first": "Open with a striking number from the source.",
    "you_statement": "Open by addressing the viewer directly ('You've probably...', 'You won't believe...').",
}

SELECT_SYSTEM = (
    "You are the content strategist for a faceless educational YouTube Shorts channel. "
    "You choose topics that are trending right now AND can be turned into a genuinely informative, "
    "factual, family-friendly 45-second video. You reject gossip, tragedies, politics, medical/financial "
    "advice, living-person biographies, and anything that could mislead. Reply with JSON only."
)

WRITER_SYSTEM = (
    "You are an expert short-form scriptwriter for an educational YouTube Shorts channel. "
    "You write punchy, high-retention scripts that are 100% factually grounded in the SOURCE TEXT provided. "
    "Hard rules: (1) Every factual claim, number, date and name must appear in or be directly implied by the SOURCE TEXT. "
    "(2) Never invent quotes, statistics or events. (3) No clickbait that the video doesn't deliver on. "
    "(4) Family-friendly, no profanity, no medical/financial/legal advice. (5) Write for the ear: short sentences, "
    "numbers written as digits, no emojis, no hashtags in narration, no stage directions. Reply with JSON only."
)

REVIEW_SYSTEM = (
    "You are a strict YouTube policy and quality reviewer. You check a Shorts script against its SOURCE TEXT for "
    "factual accuracy, misleading claims, clickbait mismatch, advertiser-friendliness, originality and viewer value. "
    "Reply with JSON only."
)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0]


class ScriptWriter:
    def __init__(self, cfg: dict, llm: LLM, strategy: Strategy):
        self.cfg = cfg
        self.llm = llm
        self.strategy = strategy
        self.lang = cfg["channel"].get("language", "en")
        self.src_chars = 2600 if llm.lite else 7000

    # ── 1. topic selection ───────────────────────────────────────────────────
    def select_topics(self, candidates: list[dict], n: int, recent: set[str]) -> list[dict]:
        cats = self.cfg["channel"]["categories"]
        pool = [c for c in candidates if c["topic_key"] not in recent and c["score"] >= self.cfg["content"]["min_trend_score"] * 0.5][:70]
        listing = "\n".join(
            f"{i}. {c['topic']} | sources={','.join(sorted(set(c['sources'])))} | score={c['score']:.2f}"
            + (f" | context: {_clip(' / '.join(str(x) for x in c.get('context', [])[:2]), 140)}" if c.get("context") else "")
            for i, c in enumerate(pool))
        weights = {c: round(self.strategy.category_weight(c), 2) for c in cats}
        user = f"""CHANNEL NICHE: {self.cfg['channel']['niche']}
CATEGORIES (with learned audience preference 0-1, favor higher): {json.dumps(weights)}
RECENTLY COVERED (avoid): {', '.join(list(recent)[:40]) or 'none'}

TRENDING CANDIDATES:
{listing}

Pick the {n + 3} best candidates for our channel. For each, give an educational ANGLE that makes it interesting
even to someone who doesn't follow the news, and the single best English Wikipedia article title to ground the facts
(e.g. for a trending sports star, prefer the sport, stadium, or record itself unless the person is historical; for a
TIL post, the underlying subject). Prefer topics with timeless educational depth that ride the trend.

Return JSON: {{"picks": [{{"index": <int>, "category": "<one of categories>", "angle": "<one sentence>",
"wiki_query": "<wikipedia article title>", "why_trending": "<short>", "risk": "none|low|high"}}]}}"""

        def validate(o):
            assert isinstance(o.get("picks"), list) and o["picks"], "no picks"

        try:
            res = self.llm.json(SELECT_SYSTEM, user, temperature=0.5, validate=validate)
            picks = []
            for p in res["picks"]:
                try:
                    idx = int(p["index"])
                    if not (0 <= idx < len(pool)) or p.get("risk") == "high":
                        continue
                    c = dict(pool[idx])
                    c.update({"category": p.get("category") if p.get("category") in cats else random.choice(cats),
                              "angle": p.get("angle", ""), "wiki_query": p.get("wiki_query") or c["topic"],
                              "why_trending": p.get("why_trending", "")})
                    picks.append(c)
                except (KeyError, ValueError, TypeError):
                    continue
            if picks:
                return picks
        except LLMError as e:
            log.warning("LLM topic selection failed (%s); falling back to heuristic", e)
        # heuristic fallback: fact-shaped + wikipedia items first
        fallback = [c for c in pool if c.get("is_fact") or c["source"] == "wikipedia"] or pool
        for c in fallback:
            c.setdefault("category", random.choice(cats))
            c.setdefault("wiki_query", c.get("wiki_title") or c["topic"])
            c.setdefault("angle", "")
        return fallback[: n + 3]

    def evergreen(self, recent: set[str]) -> list[dict]:
        seeds = [s for s in self.cfg["channel"]["evergreen_seeds"] if s.lower() not in recent]
        random.shuffle(seeds)
        return [{"topic": s, "topic_key": s.lower(), "wiki_query": s, "sources": ["evergreen"], "score": 0.3,
                 "category": random.choice(self.cfg["channel"]["categories"]), "angle": ""} for s in seeds]

    # ── 2. grounded script ───────────────────────────────────────────────────
    def write(self, topic: dict, plan: dict, source: dict) -> dict:
        lo, hi = self.cfg["video"]["target_seconds"]
        words_lo, words_hi = int(lo * 2.6), int(hi * 2.6)
        ev_len = "5-9" if self.llm.lite else "8-15"
        user = f"""TOPIC: {topic['topic']}
WHY IT'S TRENDING: {topic.get('why_trending') or ', '.join(topic.get('sources', []))}
ANGLE: {topic.get('angle') or 'most surprising educational angle'}
FORMAT: {plan['format']} — {FORMAT_GUIDE[plan['format']]}
HOOK STYLE: {plan['hook_style']} — {HOOK_GUIDE[plan['hook_style']]}
LENGTH: {words_lo}-{words_hi} words total narration ({lo}-{hi} seconds).
CHANNEL: {self.cfg['channel']['name']}

SOURCE TEXT (Wikipedia: "{source['title']}") — the ONLY allowed source of facts:
\"\"\"{source['text'][:self.src_chars]}\"\"\"

Write the script as 6-7 segments:
  - segment 1 = HOOK, 8-14 words, creates curiosity in the first 2 seconds;
  - segments 2..N-1 = BODY, each 18-28 words (1-2 sentences, a concrete fact with a specific detail);
  - last segment = payoff + natural call to follow (e.g. "Follow for more."), 8-14 words.
The TOPIC line is just a trend headline — do NOT repeat its claims or numbers unless the SOURCE TEXT states them.
TOTAL narration MUST be {words_lo}-{words_hi} words. Count them. Too short = rejected.
For each segment give:
  "visual": what the viewer SEES while that line is spoken — it must literally depict what the line is about:
     "shows": one concrete, photographable scene of the SPECIFIC thing the line is about — the named person, place,
              object, species, artwork or event itself (e.g. "close-up of a cat's face with long white whiskers",
              "Wesley Willis performing on stage", "Winamp player window on a 1990s PC"). NEVER a generic stand-in
              (random strangers, a generic office, street or computer) for a named subject, and never unrelated real
              people next to health, crime or other sensitive facts — show the subject or their actual work instead,
     "queries": 2-3 short photo-library searches (2-4 words each) that ALWAYS include the physical thing's noun
                (e.g. "cat whiskers close-up", "kitten face macro") — never a bare topic word that could also be
                a place, business, film or brand name. Never ask for text, charts, maps or abstract ideas; for an
                abstract line, show the video's main subject in a way that fits it. Hook/CTA: show the main subject.
  "evidence": a VERBATIM {ev_len} word quote copied from the SOURCE TEXT that supports the segment ("" only for hook/CTA).

Return JSON:
{{"title": "<= 70 chars, curiosity-driven but accurate, no emojis, no clickbait lies",
 "segments": [{{"text": "...", "visual": {{"shows": "...", "queries": ["...", "..."]}}, "evidence": "..."}}]}}"""

        def validate(o):
            segs = o.get("segments")
            assert isinstance(segs, list) and 4 <= len(segs) <= 10, "need 6-7 segments"
            assert all(isinstance(s.get("text"), str) and s["text"].strip() for s in segs), "empty segment"
            assert o.get("title"), "missing title"
            vis = [s.get("visual") for s in segs]
            bad = [i + 1 for i, v in enumerate(vis) if not (isinstance(v, dict) and str(v.get("shows", "")).strip()
                                                             and isinstance(v.get("queries"), list) and v["queries"])]
            assert len(bad) <= 1, f"segments {bad} are missing \"visual\": {{\"shows\": ..., \"queries\": [...]}}"
            words = sum(len(s["text"].split()) for s in segs)
            assert words_lo * 0.75 <= words <= words_hi * 1.3, (
                f"narration is {words} words but must be {words_lo}-{words_hi} words — "
                + ("add more concrete facts from the source to the body segments" if words < words_lo
                   else "shorten the body segments"))

        script = self.llm.json(WRITER_SYSTEM, user, validate=validate)
        script.update(self.metadata(script, source))
        return script

    def metadata(self, script: dict, source: dict) -> dict:
        narration = " ".join(s["text"] for s in script["segments"])
        user = f"""VIDEO TITLE: {script['title']}
NARRATION: {narration}
SOURCE ARTICLE: {source['title']}

Write YouTube metadata for this Short. Return JSON:
{{"description": "2-3 accurate sentences on what the viewer learns, no hashtags",
 "tags": ["8-12 relevant search tags"],
 "hashtags": ["#three", "#relevant", "#hashtags"],
 "thumbnail_text": "2-4 punchy words for the on-screen title card"}}"""
        try:
            m = self.llm.json(WRITER_SYSTEM, user, temperature=0.5,
                              validate=lambda o: o["description"] and o["thumbnail_text"])
            return {k: m.get(k) for k in ("description", "tags", "hashtags", "thumbnail_text") if m.get(k)}
        except LLMError:
            words = script["title"].split()
            return {"description": narration[:300], "tags": [source["title"]], "hashtags": [],
                    "thumbnail_text": " ".join(words[:4])}

    # ── 3. programmatic + LLM review ─────────────────────────────────────────
    def check(self, script: dict, source: dict) -> tuple[bool, dict]:
        narration = " ".join(s["text"] for s in script["segments"])
        issues = []
        bad_nums = unsupported_numbers(narration + " " + script.get("title", ""), source["text"])
        if bad_nums:
            issues.append(f"numbers not found in source: {bad_nums}")
        grounded = 0
        src_low = source["text"].lower()
        for s in script["segments"]:
            ev = (s.get("evidence") or "").strip().lower()
            if len(ev) > 25:
                words = re.findall(r"[a-z0-9']+", ev)
                hits = sum(w in src_low for w in words)
                if ev[:45] in src_low or (words and hits / len(words) >= 0.8):
                    grounded += 1
        if self.cfg["content"]["require_grounding"] and grounded < self.cfg["content"]["min_grounded_segments"]:
            issues.append(f"only {grounded} segments have verifiable evidence")
        blocked = [b for b in self.cfg["compliance"]["blocked_topics"]
                   if re.search(rf"\b{re.escape(b)}\b", (narration + ' ' + script['title']).lower())]
        if blocked:
            issues.append(f"blocked terms in script: {blocked}")

        review = {"score": 0, "issues": issues, "programmatic_ok": not issues}
        if issues:
            return False, review

        user = f"""SOURCE TEXT:\n\"\"\"{source['text'][:self.src_chars]}\"\"\"\n
SCRIPT TITLE: {script['title']}
SCRIPT DESCRIPTION: {script.get('description', '')}
SCRIPT NARRATION:
{chr(10).join(f'- {s["text"]}' for s in script['segments'])}

Evaluate. Return JSON: {{"factual_errors": ["..."], "misleading_title": true|false, "advertiser_friendly": true|false,
"policy_concerns": ["..."], "value_add": "<what the viewer learns>", "hook_strength": 0-10, "score": 0-10,
"fixes": ["..."]}}
Score 9-10 = accurate, engaging, clearly valuable; 7-8 = good; <7 = do not publish."""
        try:
            r = self.llm.json(REVIEW_SYSTEM, user, temperature=0.2,
                              validate=lambda o: float(o["score"]))
        except LLMError as e:
            log.warning("review LLM unavailable: %s", e)
            review["score"] = 7
            review["issues"].append("llm review unavailable; programmatic checks passed")
            return True, review
        score = float(r.get("score", 0))
        ok = (score >= self.cfg["compliance"]["quality_gate_min_score"] and not r.get("factual_errors")
              and not r.get("misleading_title") and r.get("advertiser_friendly", True)
              and not r.get("policy_concerns"))
        review.update(r)
        return ok, review

    def produce(self, topic: dict, plan: dict, max_attempts: int = 3) -> tuple[dict, dict, dict] | None:
        """Ground → write → check (rewrite with feedback). Returns (script, source, review) or None."""
        source = ground(topic["topic"], topic.get("wiki_query"), self.lang,
                        allow_living=self.cfg["content"]["allow_living_people"],
                        blocked=self.cfg["compliance"]["blocked_topics"])
        if not source:
            return None
        topic["wiki_title"] = source["title"]
        feedback = ""
        for attempt in range(max_attempts):
            try:
                t = dict(topic)
                if feedback:
                    t["angle"] = (t.get("angle") or "") + f" (PREVIOUS DRAFT REJECTED — fix: {feedback})"
                script = self.write(t, plan, source)
            except LLMError as e:
                log.warning("script generation failed: %s", e)
                return None
            ok, review = self.check(script, source)
            log.info("  draft %d for %r → ok=%s score=%s issues=%s", attempt + 1, source["title"], ok,
                     review.get("score"), (review.get("issues") or review.get("factual_errors") or [])[:2])
            if ok:
                return script, source, review
            feedback = "; ".join((review.get("issues") or []) + (review.get("factual_errors") or [])
                                 + (review.get("policy_concerns") or []) + (review.get("fixes") or []))[:600]
        return None
