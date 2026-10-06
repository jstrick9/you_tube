"""Topic selection + grounded scriptwriting + independent quality/compliance review."""
from __future__ import annotations

import json
import logging
import random
import re

from . import gates, originality, safety, series
from . import lanes
from .common import now_utc, read_json
from .llm import LLM, LLMError
from .research import ground, unsupported_numbers
from .strategy import Strategy
from .trends import primary_source

log = logging.getLogger("autotube.script")

FORMAT_GUIDE = {
    "facts3": "Three surprising, escalating facts about the subject. Fact 3 must be the most mind-blowing.",
    "backstory": "The little-known origin story / how it came to be, told as a mini narrative with a twist at the end.",
    "myth_vs_fact": "Open with a common misconception, then debunk it with what the source actually says, then one bonus fact.",
    "timeline": "A rapid-fire timeline of 4-5 key moments, each with its year (years must come from the source).",
    "by_the_numbers": "Structure the video around 3-4 striking numbers from the source, explaining what each means in human terms.",
    "what_if": "Pose a vivid 'what would happen if' or 'imagine' framing grounded in real facts from the source, then reveal the real facts.",
    # ── entertainment-first formats ──
    "sounds_fake": ("SOUNDS FAKE, BUT IT'S TRUE: open with the single most unbelievable true claim; the narrator can "
                    "barely believe it either; then prove it with 2-3 specific details from the source (who, when, where, "
                    "how we know), each crazier than the last. The reveal line is the detail that makes it even wilder."),
    "ranked_escalation": ("RANKED FROM WEIRD TO INSANE: 3 true facts in escalating order. Start each body line with its "
                          "rank word, e.g. 'Weird:', 'Weirder:', 'Completely unhinged:'. The last one (the reveal) must "
                          "be the most jaw-dropping."),
    "guess_reveal": ("GUESS BEFORE THE REVEAL: the hook challenges the viewer to guess something specific the source "
                     "answers ('Guess what this was actually used for.' / 'Guess how old this is.'); 1-2 clue lines make "
                     "them commit to a (probably wrong) guess; the ANSWER line is the reveal — the video shows an "
                     "on-screen 3-2-1 countdown right before it, so never say a countdown; then one bonus line."),
    "plot_twist": ("PLOT TWIST: tell it like an ordinary story the viewer thinks they understand, then flip it with a "
                   "true twist from the source they didn't see coming (the twist line is the reveal)."),
    "myth_buster": ("MYTHS YOU WERE TAUGHT: open with something most people were taught or assume, then show what is "
                    "actually true. ONLY use a myth, misconception, legend or popular belief that the SOURCE TEXT itself "
                    "describes as one — never invent a myth. The correction line is the reveal."),
    "would_you_survive": ("COULD YOU HANDLE IT: put the viewer inside the real situation in the second person ('You are "
                          "a sailor in 1850, and...'), escalate the real challenges from the source, then reveal what "
                          "actually happened or how people coped. Family-friendly: no gore, injuries or death details."),
    "dumbest_decision": ("THE MOST ABSURD DECISION: a real, documented decision, plan, rule or invention from the source "
                         "that sounds baffling, told with comic disbelief; give the reasoning at the time (from the "
                         "source) and the ridiculous result. Laugh at the situation, never at victims or real groups."),
    "scale_shock": ("SCALE SHOCK: make one true number from the source feel enormous (or tiny) by comparing it with "
                    "everyday things IN WORDS ('longer than a football field', 'heavier than a car') — never compute or "
                    "state numbers that are not in the source."),
    "creepy_true": ("CREEPY BUT TRUE (spooky season): an eerie, goosebump-level true story or fact — strange phenomena, "
                    "unexplained sounds, abandoned places, unsettling creatures, odd history. Build unease line by line "
                    "and end on the chilling detail. Spooky, never gory: no violence, injuries, bodies or death details, "
                    "and never claim the supernatural is real."),
}
# formats whose structure centres on one big reveal (gets the riser, flash and a beat of silence)
REVEAL_FORMATS = {"guess_reveal", "plot_twist", "myth_buster", "sounds_fake", "ranked_escalation", "creepy_true",
                  "dumbest_decision", "would_you_survive"}
# How to leave the viewer with something to say. Across the channel's first 22 measured videos:
# 233 likes and five comments. Comments are a heavy ranking input and the closest thing the
# algorithm has to a "worth replying to" signal, and the scripts were earning none of them -
# they were closed, complete and correct, which is exactly what nobody replies to. Learned as a
# bandit dimension, so the channel discovers which of these works instead of assuming.
def grounding_floor(n_segments: int, ccfg: dict) -> int:
    """How many segments must carry verifiable evidence, for a script of this length."""
    import math
    ratio = float(ccfg.get("min_grounded_ratio", 0.5))
    floor = int(ccfg.get("min_grounded_floor", 2))
    cap = int(ccfg.get("min_grounded_segments", 3))
    # clamped to n_segments: the floor must never demand more grounded segments than exist, or a
    # short script becomes unsatisfiable and every draft is rejected until the topic is abandoned
    return min(max(1, n_segments), max(floor, min(cap, math.ceil(ratio * max(1, n_segments)))))


COMMENT_GUIDE = {
    "none": "",
    "poll": "Finish by making the viewer pick between two concrete options from the story itself "
            "(\"the pipe or the tunnel - which would you have taken?\").",
    "hot_take": "State one defensible opinion about the facts as though it were obvious. People "
                "correct a confident opinion far more readily than they answer a question.",
    "unanswered_question": "Leave one genuinely open question that the source does not settle, and "
                           "say plainly that nobody knows.",
    "challenge": "Pit the story against what the viewer thinks they know (\"almost nobody gets the "
                 "next part right\") so they want to prove that they did.",
    "correction_bait": "Name the popular WRONG version of this story explicitly, so everyone who "
                       "believed it has something to say. Never assert anything false in your own voice.",
}


HOOK_GUIDE = {
    "question": "Open with a specific question a stranger can't answer but instantly wants to (not 'Did you know...?').",
    "bold_claim": "Open with the single most surprising TRUE claim from the source, stated flatly, no preamble.",
    "number_first": "Open with the most striking number from the source and what it means, in the first 3 words.",
    "you_statement": "Open by putting the viewer inside the fact ('Your...', 'You could...'), specific and surprising.",
    "disbelief": ("State the wildest true detail flatly, then undercut it with a 1-3 word deadpan reaction "
                  "(e.g. '<wild true fact>. Seriously.') — the fact itself comes first, never a vague 'this sounds fake'."),
}
HOOK_RULES = (
    "HOOK RULES (the first line decides if 70%+ of viewers stay): 6-12 words; lead with the most surprising, specific "
    "true detail (a number, a contradiction, an impossible-sounding fact); open a question that only the end of the "
    "video answers. BANNED openers: 'Did you know', 'Here are', 'In this video', 'Today we', 'Let's talk about', "
    "'Have you ever wondered', greetings, the channel name, the topic name alone.")

SELECT_SYSTEM = (
    "You are the content strategist for a faceless YouTube Shorts channel whose promise is 'Every file is real. That's the problem.' "
    "— a numbered archive of real, sourced things that should not be possible: unsolved disappearances, "
    "declassified records, deep-ocean and deep-space discoveries, and history nobody managed to explain. "
    "You choose topics that are trending right now AND can be turned into a genuinely entertaining, "
    "factual, family-friendly 15-20 second video. You reject gossip, tragedies, politics, medical/financial "
    "advice, living-person biographies, and anything that could mislead. Reply with JSON only."
)

WRITER_SYSTEM = (
    "You are an expert short-form comedy-documentary scriptwriter for a YouTube Shorts channel whose promise is "
    "'Every file is real. That's the problem.' You write punchy, funny, high-retention scripts that entertain first and teach along the way — "
    "scripts that are 100% factually grounded in the SOURCE TEXT provided. "
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


ASIDE_BAD = re.compile(r"\d|\b(hundred|thousand|million|billion|trillion|percent|dozen)s?\b", re.I)


def spoken_text(seg: dict) -> str:
    """What the narrator actually says for a segment: the factual line plus its optional humorous aside."""
    aside = (seg.get("aside") or "").strip()
    return f"{seg['text'].strip()} {aside}".strip() if aside else seg["text"].strip()


def tidy(script: dict, fmt: str = "", max_asides: int = 2) -> dict:
    """Normalise the entertainment fields the writer returns, dropping anything that could smuggle in a claim:
    asides must be short, number-free, not on the hook and at most `max_asides`; emphasis words must really be in
    the line; exactly one reveal (defaults to the last body line for reveal-centred formats)."""
    segs = script.get("segments") or []
    kept = 0
    for i, seg in enumerate(segs):
        a = re.sub(r"\s+", " ", str(seg.get("aside") or "")).strip().strip('"')
        if a and (i == 0 or kept >= max_asides or len(a.split()) > 10 or ASIDE_BAD.search(a)):
            a = ""
        if a and a[-1] not in ".!?":
            a += "."
        seg["aside"] = a
        kept += bool(a)
        low = seg["text"].lower()
        emph = seg.get("emphasis") if isinstance(seg.get("emphasis"), list) else []
        seg["emphasis"] = [e.strip() for e in emph if isinstance(e, str) and e.strip()
                           and e.strip().lower() in low][:2]
    reveals = [i for i, seg in enumerate(segs) if seg.get("reveal") is True and i > 0]
    for seg in segs:
        seg["reveal"] = False
    if reveals:
        segs[reveals[0]]["reveal"] = True
    elif fmt in REVEAL_FORMATS and len(segs) >= 3:
        segs[-2]["reveal"] = True
    return script


def echoes_hook(first: str, last: str) -> float:
    """How much of the closing line is just the hook again, 0..1.

    Prompting alone did not fix this. Three consecutive published scripts ended by
    restating their own first line - "a plane punched a hole in the sky" closing with
    "a plane just punched a hole in the sky" - which spends the final fifth of the
    runtime, the exact moment a viewer decides to replay or leave, delivering nothing.
    The prompt even offered "...and that is why" as a worked example, so the model was
    obeying it.

    Measured as containment of the closing line's meaningful words in the hook, not
    Jaccard: a short echo inside a longer hook should still score high. Stop-words and
    the connectives these endings lean on are ignored so "so yes, he is 193 years old"
    is judged on "193 years old".
    """
    from .originality import tokens

    drop = {"and", "that", "is", "why", "so", "the", "a", "an", "it", "its", "this",
            "was", "were", "just", "yes", "how", "of", "in", "on", "to", "still", "he",
            "she", "they", "then", "now", "at", "for", "with", "by", "from", "as"}
    f = {t for t in tokens(first) if t not in drop}
    l = [t for t in tokens(last) if t not in drop]
    if not l:
        return 1.0
    return sum(1 for t in l if t in f) / len(l)


class ScriptWriter:
    def __init__(self, cfg: dict, llm: LLM, strategy: Strategy):
        self.cfg = cfg
        self.llm = llm
        self.strategy = strategy
        self.lang = cfg["channel"].get("language", "en")
        self.src_chars = 2600 if llm.lite else 7000
        self._history: list[dict] | None = None      # lazily read, see check()

    # ── 1. topic selection ───────────────────────────────────────────────────
    def select_topics(self, candidates: list[dict], n: int, recent: set[str],
                      exclude: set[str] | None = None) -> list[dict]:
        """Pick the topics most likely to go viral, judged WITH their trend evidence; drop weak ones.

        Every candidate comes from a live trend signal. The LLM scores viral potential 0-10 on the factors that
        drive Shorts reach (scroll-stopping surprise, broad appeal, proven demand right now, emotional pull, and
        whether real photos can show it), and only picks scoring >= content.min_viral_score survive. Final order
        blends that score with the measured trend strength and the channel's learned category preferences.
        """
        cats = self.cfg["channel"]["categories"]
        ccfg = self.cfg["content"]
        exclude = exclude or set()
        min_viral = float(ccfg.get("min_viral_score", 7))
        pool = [c for c in candidates if c["topic_key"] not in recent and c["topic_key"] not in exclude
                and c["score"] >= ccfg["min_trend_score"] * 0.5][:80]
        if not pool:
            return []

        def evidence(c):
            bits = [f"sources={','.join(sorted(set(c['sources'])))}", f"trend={c['score']:.2f}"]
            if c.get("context"):
                bits.append("evidence: " + _clip(" / ".join(str(x) for x in c["context"][:2] if x), 170))
            return " | ".join(bits)

        listing = "\n".join(f"{i}. {c['topic'][:150]} | {evidence(c)}" for i, c in enumerate(pool))
        weights = {c: round(self.strategy.category_weight(c), 2) for c in cats}
        today = now_utc()
        want = min(len(pool), 3 * n + 4)
        in_season = getattr(self.strategy, "formats_in_season", None)
        formats = in_season() if in_season else list(ccfg.get("formats", []))
        fmt_list = "\n".join(f"  {f}: {_clip(FORMAT_GUIDE[f], 110)}" for f in formats if f in FORMAT_GUIDE)
        season = ""
        if "creepy_true" in (getattr(self.strategy, "seasonal_now", lambda: [])()):
            season = ("\nSPOOKY SEASON: it is October — eerie-but-true subjects (strange phenomena, abandoned places, "
                      "unsettling creatures, odd history, unexplained sounds) are extra viral now; favour them, never gore.")
        user = f"""TODAY: {today:%A %B %d, %Y}. (Consider seasonal interest in the coming week, e.g. holidays, events.){season}
CHANNEL NICHE: {self.cfg['channel']['niche']}
CATEGORIES (learned audience preference 0-1 from our own analytics, favor higher): {json.dumps(weights)}
RECENTLY COVERED (avoid): {', '.join(list(recent)[:40]) or 'none'}

LIVE TREND CANDIDATES (every one is trending now; evidence shows how strongly):
  youtube_outliers = a fact Short on this topic is going viral right now (proven demand — strongest signal)
  wikipedia + "Nx its normal" = sudden surge in people reading about it
  google_trends = search spike; reddit_* = top post today; on_this_day = anniversary today/tomorrow
{listing}

Pick the {want} candidates most likely to make a VIRAL, genuinely ENTERTAINING (not just "interesting") YouTube Short
for our channel. Score each one's viral_score 0-10:
  + a scroll-stopping, "wait, WHAT?" true fact or story a total stranger would stop for (surprise, disbelief, awe)
  + broad appeal — needs no prior knowledge; not niche insider news
  + proven demand right now (see evidence) — a viral Short on the same subject is the strongest proof
  + emotional pull — makes people laugh, gasp, cringe, feel creeped out or say "no way" out loud (absurd, funny,
    eerie, jaw-dropping), or awe/nostalgia — without being tragic or divisive; would people SEND it to a friend?
  + it can be shown with real photos of the actual thing (animals, places, objects, artworks, historic photos)
  - cap at 4: dry news, sports scores, award results, celebrity/relationship gossip, politics, anything a Wikipedia
    article can't support with surprising facts; cap at 2: tragedies, crime victims, living private people
FORMATS we can make (pick the 2-4 this topic can HONESTLY support from its Wikipedia article — e.g. myth_buster only
if the article describes a misconception, dumbest_decision only for a documented absurd decision):
{fmt_list}
For a viral Short topic, give our OWN angle (never copy the other video) and the English Wikipedia article that
grounds it (for a trending person or event, prefer the underlying record, place, object or phenomenon unless the
person is historical; for a TIL/viral post, the underlying subject).

Return JSON: {{"picks": [{{"index": <int>, "viral_score": <0-10>, "category": "<one of categories>",
"angle": "<one sentence: the single most surprising/funny true hook>", "wiki_query": "<wikipedia article title>",
"formats": ["<2-4 fitting formats from FORMATS>"],
"why_trending": "<short, cite the evidence>", "risk": "none|low|high"}}]}} — best first."""

        def validate(o):
            assert isinstance(o.get("picks"), list) and o["picks"], "no picks"

        try:
            res = self.llm.json(SELECT_SYSTEM, user, temperature=0.4, validate=validate)
        except LLMError as e:
            # fail closed on virality: no un-scored "maybe trending" guesses
            log.warning("topic selection unavailable (%s) — no topics this round", e)
            return []
        picks, dropped = [], []
        for p in res["picks"]:
            try:
                idx = int(p["index"])
                if not (0 <= idx < len(pool)) or p.get("risk") == "high":
                    continue
                vs = float(p.get("viral_score", 0))
                c = dict(pool[idx])
                if vs < min_viral:
                    dropped.append(f"{c['topic'][:40]} ({vs:.0f})")
                    continue
                cat = p.get("category") if p.get("category") in cats else random.choice(cats)
                fits = [f for f in (p.get("formats") or []) if isinstance(f, str) and f in formats]
                c.update({"category": cat, "angle": p.get("angle", ""), "wiki_query": p.get("wiki_query") or c["topic"],
                          "formats": fits,
                          "why_trending": p.get("why_trending", ""), "viral_score": vs})
                # blend: LLM viral potential (50%), measured trend strength (25%), and what our own analytics learned
                # about this category (12.5%) and this kind of trend source (12.5%)
                src_w = self.strategy.source_weight(primary_source(c.get("sources", [])))
                c["priority"] = round(0.50 * vs / 10 + 0.25 * c["score"] + 0.125 * weights.get(cat, 0.5)
                                      + 0.125 * src_w, 4)
                picks.append(c)
            except (KeyError, ValueError, TypeError):
                continue
        picks.sort(key=lambda c: -c["priority"])
        log.info("viral picks: %s", [f"{c['topic'][:35]} ({c['viral_score']:.0f})" for c in picks])
        if dropped:
            log.info("dropped as not viral enough (< %.0f): %s", min_viral, dropped[:10])
        return picks

    def evergreen(self, recent: set[str]) -> list[dict]:
        seeds = [s for s in self.cfg["channel"]["evergreen_seeds"] if s.lower() not in recent]
        random.shuffle(seeds)
        return [{"topic": s, "topic_key": s.lower(), "wiki_query": s, "sources": ["evergreen"], "score": 0.3,
                 "category": random.choice(self.cfg["channel"]["categories"]), "angle": ""} for s in seeds]

    # ── 2. grounded script ───────────────────────────────────────────────────
    def write(self, topic: dict, plan: dict, source: dict) -> dict:
        lo, hi = self.cfg["video"]["target_seconds"]
        words_lo, words_hi = int(lo * 2.6), int(hi * 2.6)
        # A global word count is something a model estimates badly; a per-segment budget is
        # something it can actually hold in mind while writing the line. Overshoot was the
        # single largest cause of abandoned scripts once the quality gates stopped being the
        # bottleneck - 58, 68 and 76 words against a 52-word ceiling in one run.
        per_seg = max(1, round((words_lo + words_hi) / 2 / 5.5))
        ev_len = "5-9" if self.llm.lite else "8-15"
        if self.cfg["content"].get("loop_ending", True):
            loop_rule = (
                "deliver the STRONGEST REMAINING FACT — new information the viewer has not heard yet. "
                "It must leave a thought the hook then completes on replay. "
                "NEVER restate, paraphrase or echo the hook: do not begin 'And that is why', 'So yes', "
                "'So that is how', and do not reuse the hook's subject and verb. A last line that repeats "
                "the first spends the most important seconds of the video saying nothing. "
                "Good: hook 'a waterfall that runs blood red' → last line 'so the ice down there is still bleeding'. "
                "Bad: hook 'a plane punched a hole in the sky' → last line 'a plane just punched a hole in the sky'. "
                "No 'follow for more', no goodbye.")
        else:
            loop_rule = "answer the hook's question with the final surprising fact + a natural 'Follow for more.'"
        persona = self.cfg.get("persona") or {}
        max_asides = int(persona.get("max_asides", 2)) if persona.get("asides", True) else 0
        # The series remit, so episodes conform to the show's promise rather than merely
        # carrying its label. Without this the numbering is a prefix stapled to whatever
        # the trend feed produced, which is the opposite of what a series is for.
        _skey = series.assign(topic.get("category"), self.cfg)
        _spec = series.definitions(self.cfg).get(_skey or "", {})
        series_rule = (f'This episode is {_spec.get("tag", "")} in a recurring series. Its remit: '
                       f'{_spec["remit"]}. Choose the angle that fits that remit.\n'
                       if _spec.get("remit") else "")
        fmt = plan["format"] if plan["format"] in FORMAT_GUIDE else "sounds_fake"
        reveal_rule = ('  "reveal": true on exactly ONE segment — the line with the biggest surprise/answer/twist (the video adds '
                       'a beat of silence, a riser and a flash right before it). Never the hook; usually the last BODY line.\n')
        aside_rule = (
            f'  "aside" (REQUIRED: at least 1, at most {max_asides} in the whole script, never on the hook): a 2-8 word deadpan\n'
            '     reaction the narrator says right AFTER that line — e.g. "Which is, frankly, rude.", "Nature, please.",\n'
            '     "Bold plan. Terrible plan." It is a JOKE/OPINION ONLY: no facts, numbers, names, dates or claims; never\n'
            '     mean about real people, groups or victims; family-friendly. A script with no aside anywhere is\n'
            '     rejected: narration that only recites sourced facts has no voice of its own, and a channel of\n'
            '     those reads as bulk-produced rather than as a show. Make it actually funny.\n'
        ) if max_asides else ""
        # A living person reached this point only because safety.check_person approved one specific
        # professional angle. The writer has to be told what that angle is, or the model will drift
        # back to whatever made the person trend — which is usually the thing we just refused.
        lane_rule = lanes.writer_rule(topic, self.cfg)
        person_rule = ""
        if source.get("person"):
            person_rule = (
                "REAL LIVING PERSON: stay strictly on their public professional work"
                + (f" — specifically: {source['safe_angle']}" if source.get("safe_angle") else "")
                + ". Never mention their health, death, family, relationships, appearance, money, "
                  "legal matters, accusations, feuds or politics. State nothing you cannot point to "
                  "in the SOURCE TEXT. No 'allegedly', 'reportedly' or 'sources say'.")
        persona_line = (f"NARRATOR PERSONA: {persona['character']}\nCHANNEL PROMISE: \"{persona.get('catchphrase', '')}\" "
                        "— every wild claim is proven by the source (a PROVEN stamp + source card closes each video).\n"
                        if persona.get("character") else "")
        _cd = plan.get("comment_device") or "none"
        _cg = COMMENT_GUIDE.get(_cd, "")
        # The device must never override the facts or the loop ending; it shapes the LAST line only.
        comment_line = (f"LEAVE THEM SOMETHING TO SAY ({_cd}): {_cg} Work this into the PAYOFF line's "
                        f"wording — do NOT add a segment, do not say \"comment below\", \"let me know\" or "
                        f"any call to action, and do not change a single fact to make it land.\n") if _cg else ""
        user = f"""TOPIC: {topic['topic']}
WHY IT'S TRENDING: {topic.get('why_trending') or ', '.join(topic.get('sources', []))}
ANGLE: {topic.get('angle') or 'most surprising educational angle'}
FORMAT: {fmt} — {FORMAT_GUIDE[fmt]}
HOOK STYLE: {plan['hook_style']} — {HOOK_GUIDE[plan['hook_style']]}
{comment_line}LENGTH: {words_lo}-{words_hi} words total narration ({lo}-{hi} seconds).
CHANNEL: {self.cfg['channel']['name']}
{persona_line}
SOURCE TEXT (Wikipedia: "{source['title']}") — the ONLY allowed source of facts:
\"\"\"{source['text'][:self.src_chars]}\"\"\"

{HOOK_RULES}
Write the script as 4-5 segments. Each beat has a DIFFERENT job — do not write interchangeable body lines:
  - segment 1 = HOOK (see rules), spoken in under 3 seconds;
  - segment 2 = THE TURN, 9-11 words. This single line decides whether the video is watched, and it is the
    one most often written wrong. It must make the hook BIGGER — a second surprise, or the consequence of the
    first ("which meant...", "except..."). It must NOT be setup, background, a definition, a birth date, a
    founding year, an origin story or any sentence beginning "In 1923..." / "Born in..." / "X is a Y that...".
    The viewer already decided the premise was interesting; explaining it to them is why they leave;
  - segments 3..N-1 = ESCALATION, each 9-11 words, one concrete new fact each, every line raising the stakes
    above the line before it. Use contrast and consequence ("so", "which meant", "except"). No filler, no repetition;
  - segment 1 (after the hook) must NOT resolve or deflate the hook. No "it's not aliens, it's just
    physics", no "experts say it was only a raccoon" this early. Deepen the puzzle or raise the stakes;
    the answer belongs in the payoff. Resolving it here ends the video at three seconds.
  - order the body so each line is more surprising than the last. The best fact goes LAST, never mid-script.
  - last segment = PAYOFF, 6-9 words: {loop_rule}
Leave ONE question deliberately open from the hook until the payoff — the viewer should be unable to stop
watching without learning the answer. Never answer it in segment 2.
The TOPIC line is just a trend headline — do NOT repeat its claims or numbers unless the SOURCE TEXT states them.
{person_rule}{series_rule}{lane_rule}ENTERTAIN: write it like a friend telling the most unbelievable true story they know — conversational, vivid, with
comic timing and personality (reactions, contrast, "and it gets worse"). The facts stay 100% exact; the humour comes
from HOW you tell them and from the asides, never from changing what happened.
TOTAL narration (text + asides) MUST be {words_lo}-{words_hi} words — that is about {per_seg} words per segment,
which is ONE short spoken sentence each, not two. Both ends are rejected: too short AND too long.
The ceiling is the hard one. Write the whole thing, then count, then cut until it fits before you answer.
For each segment give:
  "text": the spoken line (facts),
{aside_rule}  "emphasis": 1-2 key words copied exactly from "text" that flash on screen in colour (the surprising word/number),
{reveal_rule}
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
{{"title": "<= 60 chars: names the subject + a curiosity gap the video truly answers; no emojis, no clickbait lies",
 "segments": [{{"text": "...", "aside": "", "emphasis": ["..."], "reveal": false,
               "visual": {{"shows": "...", "queries": ["...", "..."]}}, "evidence": "..."}}]}}"""

        def validate(o):
            segs = o.get("segments")
            assert isinstance(segs, list) and 4 <= len(segs) <= 8, "need 5-6 segments"
            assert all(isinstance(s.get("text"), str) and s["text"].strip() for s in segs), "empty segment"
            assert o.get("title"), "missing title"
            vis = [s.get("visual") for s in segs]
            bad = [i + 1 for i, v in enumerate(vis) if not (isinstance(v, dict) and str(v.get("shows", "")).strip()
                                                             and isinstance(v.get("queries"), list) and v["queries"])]
            assert len(bad) <= 1, f"segments {bad} are missing \"visual\": {{\"shows\": ..., \"queries\": [...]}}"
            # A closing line that is the hook again is the single most common defect in
            # this channel's published output, and asking the model not to do it has a
            # 0-for-3 record. Reject it where rejection actually costs the model a retry.
            if len(segs) >= 3:
                echo = echoes_hook(spoken_text(segs[0]), spoken_text(segs[-1]))
                assert echo < 0.6, (
                    f"the last line just restates the hook ({echo:.0%} of its words are from it) — "
                    "end on the strongest fact the viewer has NOT heard yet, something the hook "
                    "completes on replay; never 'and that is why' or 'so yes'")
            words = sum(len(spoken_text(s).split()) for s in segs)
            # The upper tolerance used to be 1.3x, which accepted 120 words against a 93-word
            # target. At the measured ~3 words/second that is a 40-second narration before pauses
            # and trailing silence — the direct cause of the 44s and 46s videos in the history. A
            # script that long is cheaper to reject here, where the model can simply rewrite it,
            # than to fix later by deleting a fact it had already built the story around.
            assert words_lo * 0.75 <= words <= words_hi * 1.08, (
                f"narration is {words} words but must be {words_lo}-{words_hi} words — "
                + ("add more concrete facts from the source to the body segments" if words < words_lo
                   else "shorten the body segments"))

        script = self.llm.json(WRITER_SYSTEM, user, validate=validate)
        script["_writer_model"] = self.llm.last_used          # so the reviewer can be a different one
        tidy(script, fmt, max_asides)
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
        """Approve or reject a draft, remembering what was approved.

        The similarity gate reads history from disk, but history is only written at the
        END of a run and a run makes up to three videos. Without this, videos two and
        three are never compared against video one - and same-run videos are the most
        likely to collide, because they are drawn from the same day's trends, the same
        lane and the same format pool. Approved drafts are appended to the in-memory
        history so the rest of the run can see them.

        Only approved drafts are registered. A rejected draft is about to be rewritten
        on the same topic, and holding it against its own replacement would guarantee
        the retry fails too.
        """
        ok, review = self._check(script, source)
        if ok:
            if getattr(self, "_history", None) is None:
                self._history = read_json("history.json", [])
            self._history.append({
                "title": script.get("title", ""),
                "sketch": originality.sketch(originality.narration_of(script)),
                "_in_flight": True,
            })
        return ok, review

    def _check(self, script: dict, source: dict) -> tuple[bool, dict]:
        # numbers + blocked words are checked on EVERYTHING spoken (asides included); evidence only on facts
        narration = " ".join(spoken_text(s) for s in script["segments"])
        issues = []
        # Deterministic gates first: banned openers, CTA closers, clickbait titles, stage directions,
        # emoji/hashtags/URLs in speech, verbatim repeats. These are free, reliable and run before a
        # single review token is spent — see autotube/gates.py for why they don't belong to the LLM.
        issues += gates.run_all(script, spoken_text, self.cfg)
        # Repetition and absence-of-voice are the two things that get a faceless channel
        # demonetized in bulk, and both are only visible by comparing this draft against
        # what already shipped - so they cannot live in gates.py, which sees one script.
        # History is cached per writer: check() runs once per retry and the file is small,
        # but re-reading it inside a retry loop is pointless IO.
        # getattr rather than self._history: ScriptWriter is constructed without __init__
        # in several tests and by the retry path, and an originality check is not worth
        # an AttributeError in the middle of script validation.
        if getattr(self, "_history", None) is None:
            self._history = read_json("history.json", [])
        issues += originality.check(script, self.cfg, self._history)
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
        # Scaled to the script's length, not a flat count. A flat 3 was written when scripts were 5-6
        # segments; at the 15-20s band they are 4-5, so the same number silently became 3-of-4 - a 25%
        # tightening nobody asked for, and one that would have shown up as unexplained topic abandonment
        # rather than as a failing test. The ratio keeps 5- and 6-segment behaviour identical.
        need = grounding_floor(len(script["segments"]), self.cfg["content"])
        if self.cfg["content"]["require_grounding"] and grounded < need:
            issues.append(f"only {grounded} of {len(script['segments'])} segments have verifiable "
                          f"evidence (need {need})")
        # Context-aware (autotube/safety.py): titles stay strict, narration is only hard-blocked on
        # genuinely non-negotiable terms. The LLM reviewer still judges advertiser-friendliness in
        # context, which is the check that actually matches YouTube's guidelines.
        issues += safety.check_script(script.get("title", ""), narration, self.cfg,
                                      about_person=bool(source.get("person")))

        review = {"score": 0, "issues": issues, "programmatic_ok": not issues}
        if issues:
            return False, review

        rows = []
        for s in script["segments"]:
            rows.append(f"- {s['text']}")
            if s.get("aside"):
                rows.append(f"    [ASIDE - joke/opinion, not a factual claim] {s['aside']}")
        narration_list = "\n".join(rows)
        user = f"""SOURCE TEXT:\n\"\"\"{source['text'][:self.src_chars]}\"\"\"\n
SCRIPT TITLE: {script['title']}
SCRIPT DESCRIPTION: {script.get('description', '')}
SCRIPT NARRATION (lines marked ASIDE are the narrator's jokes, spoken right after the line above them):
{narration_list}

Evaluate. Return JSON: {{"factual_errors": ["..."], "misleading_title": true|false, "advertiser_friendly": true|false,
"policy_concerns": ["..."], "value_add": "<what the viewer learns>", "hook_strength": 0-10, "entertainment": 0-10,
"score": 0-10, "loops": true|false, "fixes": ["..."]}}
Score 9-10 = accurate, engaging, clearly valuable; 7-8 = good; <7 = do not publish.
"loops": does the LAST line flow naturally back into the FIRST when the Short replays? Judge it by meaning,
not by shared words - "so the ice keeps bleeding" loops cleanly into "a waterfall that runs blood red".
A replay counts as a view and pushes average view percentage past 100%, which is one of the strongest
satisfaction signals Shorts has. If it does not loop, say so in "fixes" and give the re-worded last line.
Read EVERY sentence literally, word by word: if its literal meaning is false or garbled (e.g. the wrong subject doing
the action — "rivers carved warnings" when people carved them; a date or place attached to the wrong thing), list
it in factual_errors even if the gist is right.
ASIDES are humour, not claims: never list an aside in factual_errors for being a joke, exaggerated opinion or
sarcasm. Put an aside in policy_concerns ONLY if it states a new specific fact as true, mocks real people, groups,
victims or a tragedy, or is not family-friendly. A joke must never change what the facts say.
entertainment — BE HARSH, this channel competes with comedians: a list of accurate facts read in a neutral tone is 5
at most, however interesting. 7 = has a story shape (setup, escalation, payoff) and personality. 8 = also at least one
genuinely funny or gasp-out-loud moment. 9-10 = you would send it to a friend. If < 8, put a concrete TRUE rewrite
idea in "fixes" (story tension, comic contrast, a sharper aside, a stronger reveal).
hook_strength: 9-10 = the first line alone would stop a stranger scrolling (specific, surprising, opens a question the
video answers); 7-8 = decent; <=6 = generic ('Did you know', 'Here are some facts', topic name, slow setup).
If hook_strength < 9, put a stronger TRUE first line in "fixes"."""
        try:
            # A writer grading its own homework agrees with itself. Review runs on a different
            # provider/model wherever one is reachable (llm.review_providers), and never on the exact
            # model that produced this draft.
            review_fn = getattr(self.llm, "review_json", None)
            if review_fn is not None:
                r = review_fn(REVIEW_SYSTEM, user, avoid=script.get("_writer_model"),
                              temperature=0.2, validate=lambda o: float(o["score"]))
            else:                                     # minimal LLM-like object (tests, custom routers)
                r = self.llm.json(REVIEW_SYSTEM, user, temperature=0.2, validate=lambda o: float(o["score"]))
            review["reviewer"] = getattr(self.llm, "last_used", "")
            review["independent"] = bool(review["reviewer"]) and review["reviewer"] != script.get("_writer_model")
            if review["reviewer"] and not review["independent"]:
                log.info("  review ran on the same model as the writer (%s) — no independent reviewer reachable",
                         review["reviewer"])
                if self.cfg["compliance"].get("require_independent_review", False):
                    review["issues"].append("no independent reviewer available — not publishing")
                    review["unavailable"] = True
                    return False, review
        except LLMError as e:
            log.warning("review LLM unavailable: %s", e)
            if self.cfg["compliance"].get("require_llm_review", True):
                # fail closed: a script nobody fact-checked is never published
                review["unavailable"] = True
                review["issues"].append("fact-check unavailable — not publishing an unchecked script")
                return False, review
            review["score"] = 7
            review["issues"].append("llm review unavailable; programmatic checks passed")
            return True, review
        score = float(r.get("score", 0))
        min_hook = float(self.cfg["content"].get("min_hook_score", 0))
        try:
            hook = float(r.get("hook_strength", 10))
        except (TypeError, ValueError):
            hook = 0.0
        if hook < min_hook:
            review["issues"].append(f"hook too weak ({hook:.0f}/10, need {min_hook:.0f}) — open with the single most "
                                    "surprising specific fact; no 'Did you know' or slow setup")
        min_ent = float(self.cfg["content"].get("min_entertainment", 0))
        try:
            ent = float(r.get("entertainment", 10))
        except (TypeError, ValueError):
            ent = 0.0
        if ent < min_ent:
            review["issues"].append(f"not entertaining enough ({ent:.0f}/10, need {min_ent:.0f}) — tell it as a story "
                                    "with tension and a payoff, add comic contrast or a sharper aside; facts unchanged")
        # The loop is judged semantically by the reviewer, because a lexical check rejects good loops
        # (see the note in gates.py). Enforced only when require_loop is on, and it defaults off: it
        # is a new criterion, and turning it into a hard gate before we know its false-positive rate
        # would start abandoning topics for a reason nobody is watching yet. The verdict is recorded
        # either way, so the rate can be read off history before anyone flips the switch.
        loops = r.get("loops")
        review["loops"] = loops
        if loops is False and self.cfg["content"].get("require_loop", False):
            review["issues"].append("the last line does not flow back into the first, so the Short will not "
                                    "loop — re-word the payoff to pick up the hook; a replay counts as a view")

        ok = (ent >= min_ent and hook >= min_hook and score >= self.cfg["compliance"]["quality_gate_min_score"] and not r.get("factual_errors")
              and not r.get("misleading_title") and r.get("advertiser_friendly", True)
              and not r.get("policy_concerns"))
        review.update(r)
        return ok, review

    def produce(self, topic: dict, plan: dict, max_attempts: int | None = None) -> tuple[dict, dict, dict] | None:
        """Ground → write → check (rewrite with feedback). Returns (script, source, review) or None."""
        source = ground(topic["topic"], topic.get("wiki_query"), self.lang,
                        allow_living=self.cfg["content"]["allow_living_people"],
                        blocked=self.cfg["compliance"].get("blocked_topics"),
                        cfg=self.cfg, llm=self.llm)
        if not source:
            return None
        topic["wiki_title"] = source["title"]
        feedback = ""
        max_attempts = max_attempts or int(self.cfg["content"].get("max_drafts", 4))
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
            if review.get("unavailable"):
                return None
            feedback = "; ".join((review.get("issues") or []) + (review.get("factual_errors") or [])
                                 + (review.get("policy_concerns") or []) + (review.get("fixes") or []))[:600]
        return None
