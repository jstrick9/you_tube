"""Topic selection + grounded scriptwriting + independent quality/compliance review."""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone

from . import gates, originality, safety, series, run_report
from . import lanes
from .common import now_utc, read_json
from .llm import LLM, LLMError
from .research import ground, unsupported_numbers
from .recovery import MAX_SCRIPT_DRAFTS, RecoveryBudgets, bounded_attempts
from .strategy import Strategy
from .trends import has_current_trend_evidence, lane_of, primary_source

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

# Explicit competition/league and team identifiers prevent a model from relabelling
# football or other sports trends as history or culture. Generic "match" and "field"
# words only count when tied to an explicit matchup or sport, avoiding unrelated science topics.
_SPORTS_TREND_MARKER = re.compile(
    r"\b(?:borussia|bundesliga|champions\s+league|uefa|fifa|premier\s+league|la\s+liga|serie\s+a|"
    r"football\s+club|soccer\s+club|football\s+team|soccer\s+team|sports?\s+club|"
    r"cricket\s+(?:history|match|team|club|league|score|player|innings|championship|world\s+cup)|"
    r"ligue\s+1|world\s+cup|super\s+bowl|nfl|nba|wnba|mlb|nhl|ncaaf|ncaab|ufc|wwe|"
    r"formula\s*1|f1|copa\s+(?:america|libertadores)|grand\s+slam|wimbledon|ipl|atp|wta|"
    r"real\s+madrid|bayern\s+munich|manchester\s+(?:united|city)|liverpool\s+(?:fc|football)|"
    r"arsenal\s+fc|chelsea\s+fc|juventus|fc\s+barcelona|barcelona\s+fc)\b", re.I)
_SPORTS_MATCHUP = re.compile(r"\b(?:vs\.?|versus)\b", re.I)
_SPORTS_FIXTURE_CONTEXT = re.compile(
    r"\b(?:game|match|matchup|fixture|how\s+to\s+watch|start\s+time|preview|playoffs?|"
    r"tournament|kick[ -]?off|score|standings|bracket|league|season|championship|finals?)\b", re.I)
_SPORTS_KIND = re.compile(
    r"\b(?:baseball|basketball|hockey|soccer|football|cricket|rugby|tennis|golf|boxing|"
    r"volleyball|softball|lacrosse|wrestling|motorsport)\b", re.I)


def _is_sports_trend(text: str) -> bool:
    """Catch explicit competitions and source-context-confirmed live fixtures."""
    if _SPORTS_TREND_MARKER.search(text):
        return True
    fixture = bool(_SPORTS_FIXTURE_CONTEXT.search(text))
    return fixture and bool(_SPORTS_MATCHUP.search(text) or _SPORTS_KIND.search(text))

# formats whose structure centres on one big reveal (gets the riser, flash and a beat of silence)
REVEAL_FORMATS = {"guess_reveal", "plot_twist", "myth_buster", "sounds_fake", "ranked_escalation", "creepy_true",
                  "dumbest_decision", "would_you_survive"}
# How to leave the viewer with something to say. One early sample had 233 likes and five comments;
# these optional devices invite relevant participation without claiming a fixed ranking benefit. They are
# learned as a local experiment rather than assumed to help every topic.
def grounding_floor(n_segments: int, ccfg: dict) -> int:
    """How many segments must carry verifiable evidence, for a script of this length."""
    import math
    ratio = float(ccfg.get("min_grounded_ratio", 0.5))
    floor = int(ccfg.get("min_grounded_floor", 2))
    cap = int(ccfg.get("min_grounded_segments", 3))
    # clamped to n_segments: the floor must never demand more grounded segments than exist, or a
    # short script becomes unsatisfiable and every draft is rejected until the topic is abandoned
    return min(max(1, n_segments), max(floor, min(cap, math.ceil(ratio * max(1, n_segments)))))


_EVIDENCE_STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from had has have he her hers
him his how i if in into is it its may might more most must my no nor not of on once only or other our
ours out over own same she should so some such than that the their theirs them then there these they
this those through to too under until up very was we were what when where which while who why will with
would you your yours
""".split())
_EVIDENCE_WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)
_EVIDENCE_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _evidence_terms(text: str) -> set[str]:
    """Normalize meaningful words for conservative script-to-source quote matching."""
    terms = set()
    for match in _EVIDENCE_WORD_RE.finditer(text or ""):
        raw = match.group(0).casefold().replace("’", "'")
        if raw.endswith("'s"):
            raw = raw[:-2]
        # A one-letter plural normalization helps ordinary paraphrases without stemming
        # unrelated words or turning broad topical overlap into apparent factual support.
        if len(raw) > 4 and raw.endswith("s") and not raw.endswith(("ss", "us", "is")):
            raw = raw[:-1]
        if raw not in _EVIDENCE_STOPWORDS and (len(raw) > 2 or raw.isdigit()
                                                or match.group(0).isupper()):
            terms.add(raw)
    return terms


def _source_evidence_quote(segment_text: str, source_text: str, min_words: int = 8,
                           max_words: int = 15, used: set[str] | None = None) -> str | None:
    """Find a short, verbatim source passage with strong lexical support for a script line.

    This only fills a missing metadata field: it never rewrites the narration or invents a
    quotation. Weak topical overlap, unsupported numerals, and reused quotes are rejected.
    """
    segment_terms = _evidence_terms(segment_text)
    if len(segment_terms) < 2 or not source_text:
        return None
    numbers = {m.group(0).replace(",", "").rstrip(".") for m in _EVIDENCE_NUMBER_RE.finditer(segment_text)}
    used_quotes = {q.strip().casefold() for q in (used or set()) if isinstance(q, str)}
    used_term_sets = [_evidence_terms(q) for q in used_quotes]
    best: tuple[float, str] | None = None
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", source_text):
        sentence = sentence.strip()
        matches = list(_EVIDENCE_WORD_RE.finditer(sentence))
        if len(matches) < min_words:
            continue
        for length in range(min(max_words, len(matches)), min_words - 1, -1):
            for start in range(0, len(matches) - length + 1):
                quote = sentence[matches[start].start():matches[start + length - 1].end()].strip()
                if len(quote) <= 25 or quote.casefold() in used_quotes:
                    continue
                quote_numbers = {m.group(0).replace(",", "").rstrip(".")
                                 for m in _EVIDENCE_NUMBER_RE.finditer(quote)}
                if not numbers.issubset(quote_numbers):
                    continue
                quote_terms = _evidence_terms(quote)
                if any(quote_terms and terms and len(quote_terms & terms) / min(len(quote_terms), len(terms)) >= 0.7
                       for terms in used_term_sets):
                    continue
                overlap = segment_terms & quote_terms
                if len(overlap) < 2:
                    continue
                coverage = len(overlap) / len(segment_terms)
                precision = len(overlap) / max(1, len(quote_terms))
                if coverage < 0.4 or precision < 0.28:
                    continue
                score = coverage * 0.7 + precision * 0.3
                if best is None or score > best[0]:
                    best = (score, quote)
    return best[1] if best else None


def _attach_missing_evidence(segments: list, source_text: str, min_words: int = 8,
                             max_words: int = 15) -> int:
    """Backfill omitted evidence fields from distinct, source-matched verbatim quotes.

    The hook may remain empty. A factual segment is filled only when its own words and any
    numerals match the quoted source passage; the existing grounding floor remains unchanged.
    """
    used = {s.get("evidence", "").strip() for s in segments
            if isinstance(s, dict) and isinstance(s.get("evidence"), str) and s.get("evidence", "").strip()}
    attached = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        if "evidence" not in segment:
            segment["evidence"] = ""
        evidence = segment.get("evidence")
        if not isinstance(evidence, str) or evidence.strip() or index == 0:
            continue
        quote = _source_evidence_quote(str(segment.get("text") or ""), source_text,
                                       min_words=min_words, max_words=max_words, used=used)
        if quote:
            segment["evidence"] = quote
            used.add(quote)
            attached += 1
    return attached


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
    "HOOK RULES (internal creative targets, not a platform threshold): {hook_min}-{hook_max} words; one short, self-contained line that "
    "makes sense without the title. Name the specific subject or concrete object/event and lead with its most surprising "
    "source-backed detail (a number, contradiction or impossible-sounding fact). Create curiosity without a vague pronoun, "
    "preamble or forced literal question; obey the assigned lane's hook style. BANNED openers: 'Did you know', 'Here are', "
    "'In this video', 'Today we', 'Let's talk about', 'Have you ever wondered', greetings, the channel name, or the topic name alone.")

TURN_VARIANT_GUIDE = {
    "consequence_first": (
        "Open segment 2's first clause with the most concrete source-backed consequence, cost, exception, "
        "or reversal of the hook. No lead-in or background before it; do not invent a consequence if the "
        "source does not support one."),
    "new_surprise_first": (
        "Open segment 2's first clause with the next distinct, most surprising source-backed fact, "
        "before adding explanation. Then connect it to the hook; do not repeat the hook or invent a fact."),
}

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
    "(4) Create an original treatment in our own words: never imitate, paraphrase, reconstruct, or reuse a viral Short, "
    "competitor script, or source publisher's distinctive wording. (5) Family-friendly, no profanity, no medical/financial/legal advice. "
    "(6) Write for the ear: short sentences, "
    "numbers written as digits, no emojis, no hashtags in narration, no stage directions. Reply with JSON only."
)

REVIEW_SYSTEM = (
    "You are a strict YouTube policy and quality reviewer. You check a Shorts script against its SOURCE TEXT for "
    "factual accuracy, misleading claims, clickbait mismatch, advertiser-friendliness, originality and viewer value. "
    "When current-trend context is supplied, also judge whether the script tells the specific story/angle that is "
    "trending, rather than merely mentioning the same broad subject. Trend evidence is not factual source text. "
    "Reply with JSON only."
)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0]


def _rewrite_draft_excerpt(script: dict, limit: int = 900) -> str:
    """Pass only the prior narration into a rewrite prompt; it is never factual evidence."""
    if not isinstance(script, dict) or not isinstance(script.get("segments"), list):
        return ""
    rows = []
    for index, segment in enumerate(script["segments"], 1):
        if not isinstance(segment, dict):
            continue
        text = re.sub(r"\s+", " ", str(segment.get("text") or "")).strip()
        aside = re.sub(r"\s+", " ", str(segment.get("aside") or "")).strip()
        if text:
            rows.append(f"{index}. {text}" + (f" [aside: {aside}]" if aside else ""))
    return _clip("\n".join(rows), limit)


def trend_evidence_summary(candidate: dict) -> str:
    """Describe measured, source-linked trend signals without trusting selector prose."""
    summaries = []
    for signal in candidate.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        source = str(signal.get("source") or "")
        evidence = signal.get("evidence") if isinstance(signal.get("evidence"), dict) else {}
        try:
            rank = int(evidence.get("source_rank") or 0)
        except (TypeError, ValueError):
            rank = 0
        if source == "youtube_outliers":
            try:
                views = int(evidence.get("views") or 0)
                hours = float(evidence.get("hours") or 0)
                vph = int(evidence.get("vph") or (views / hours if hours > 0 else 0))
                breakout = float(evidence.get("breakout") or 0)
            except (TypeError, ValueError):
                views, hours, vph, breakout = 0, 0.0, 0, 0.0
            if views > 0 and hours > 0:
                detail = f"YouTube Short: {views:,} views in {hours:g}h (~{vph:,} views/hour)"
                if breakout > 0:
                    detail += f", {breakout:g}x channel subscribers"
                summaries.append(detail)
        elif source == "google_trends":
            try:
                traffic = int(evidence.get("traffic") or 0)
            except (TypeError, ValueError):
                traffic = 0
            rank_text = f"rank #{rank}" if rank else "current feed"
            metric = f", about {traffic:,} searches" if traffic > 0 else ""
            summaries.append(f"Google Trends {rank_text}{metric}")
        elif source == "wikipedia":
            try:
                spike = float(evidence.get("spike") or 0)
                views = int(evidence.get("views") or 0)
            except (TypeError, ValueError):
                spike, views = 0.0, 0
            if spike > 0:
                summaries.append(f"Wikipedia readership {spike:g}x its 30-day baseline"
                                 + (f" ({views:,} pageviews/day)" if views > 0 else ""))
        elif source.startswith("reddit_"):
            community = source.removeprefix("reddit_")
            summaries.append(f"r/{community} top-of-day post" + (f", rank #{rank}" if rank else ""))
        elif source == "hackernews":
            try:
                points = int(evidence.get("points") or 0)
            except (TypeError, ValueError):
                points = 0
            detail = f"Hacker News front page" + (f", rank #{rank}" if rank else "")
            if points > 0:
                detail += f", {points:,} points"
            summaries.append(detail)
        elif source == "youtube_chart":
            try:
                views = int(evidence.get("view_count") or 0)
            except (TypeError, ValueError):
                views = 0
            detail = "YouTube most-popular chart" + (f", rank #{rank}" if rank else "")
            if views > 0:
                detail += f", {views:,} views"
            summaries.append(detail)
        if len(summaries) >= 3:
            break
    if summaries:
        return " | ".join(summaries)
    # Small fixtures and older cached rows can predate structured signals. Preserve their
    # actual feed context, but never substitute the selector's unsupported narrative.
    context = [str(x).strip() for x in (candidate.get("context") or []) if str(x).strip()]
    if context:
        return " | ".join(context[:3])
    sources = candidate.get("sources") or ([candidate.get("source")] if candidate.get("source") else [])
    return "Current feed signal(s): " + ", ".join(str(s) for s in sources if s)


_FORMAT_PULSE_MIN_SAMPLE = 8
_FORMAT_PULSE_MAX_AGE_HOURS = 72
_FORMAT_DURATION_LABELS = {
    "le_20s": "20 seconds or less", "21_40s": "21–40 seconds",
    "41_60s": "41–60 seconds", "over_60s": "over 60 seconds",
}


def outlier_structure_guidance(topic: dict, cfg: dict) -> str:
    """Return fresh, lane-specific aggregate metadata guidance, never creator examples."""
    trend_cfg = cfg.get("trends", {}) if isinstance(cfg, dict) else {}
    if isinstance(trend_cfg, dict) and trend_cfg.get("format_pulse", True) is False:
        return ""
    pulse = read_json("format_pulse.json", {})
    if not isinstance(pulse, dict):
        return ""
    raw_updated = pulse.get("updated")
    try:
        updated = datetime.fromisoformat(str(raw_updated).replace("Z", "+00:00"))
        if updated.tzinfo is None:
            return ""
        age = now_utc() - updated.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return ""
    if age < -timedelta(minutes=5) or age > timedelta(hours=_FORMAT_PULSE_MAX_AGE_HOURS):
        return ""

    lane = lanes.lane_for(topic, cfg)
    by_lane = pulse.get("by_lane")
    if isinstance(by_lane, dict):
        summary = by_lane.get(lane) if lane else None
        if not isinstance(summary, dict):
            return ""  # Do not let one approved lane's patterns stand in for the other.
        scope = f"the {lane} lane"
    else:
        # Backward-compatible, fresh pulse files have only a pooled hook distribution.
        summary = pulse
        scope = "a pooled all-lane sample (lane detail not yet available)"

    sample = summary.get("sample", pulse.get("sample"))
    if type(sample) is not int or sample < _FORMAT_PULSE_MIN_SAMPLE:
        return ""
    content_cfg = cfg.get("content", {}) if isinstance(cfg, dict) else {}
    allowed_styles = set(content_cfg.get("hook_styles") or []) if isinstance(content_cfg, dict) else set()
    raw_styles = summary.get("title_hook_style") or summary.get("hook_style")
    styles: dict[str, float] = {}
    if isinstance(raw_styles, dict):
        for key, value in raw_styles.items():
            if (not allowed_styles or key in allowed_styles) and _finite_in_range(value, 0, 1):
                styles[key] = float(value)
    total_style = sum(styles.values())
    details = []
    if total_style > 0:
        ranked = sorted(styles.items(), key=lambda item: (-item[1], item[0]))[:3]
        details.append("metadata title-pattern mix (views/hour weighted): " + ", ".join(
            f"{name} {value / total_style:.0%}" for name, value in ranked))

    duration_sample = summary.get("duration_sample", 0)
    raw_durations = summary.get("duration_bands")
    durations: dict[str, float] = {}
    if (type(duration_sample) is int and duration_sample >= _FORMAT_PULSE_MIN_SAMPLE
            and isinstance(raw_durations, dict)):
        for key, value in raw_durations.items():
            if key in _FORMAT_DURATION_LABELS and _finite_in_range(value, 0, 1):
                durations[key] = float(value)
    total_duration = sum(durations.values())
    if total_duration > 0:
        ranked = sorted(durations.items(), key=lambda item: (-item[1], item[0]))
        details.append("outlier clip-duration mix (metadata only): " + ", ".join(
            f"{_FORMAT_DURATION_LABELS[key]} {value / total_duration:.0%}" for key, value in ranked))
    if not details:
        return ""

    duration = cfg.get("video", {}).get("target_seconds", [15, 20])
    target = (f"{duration[0]}–{duration[1]} seconds"
              if isinstance(duration, (list, tuple)) and len(duration) == 2 else "the configured runtime")
    return (
        f"OUTLIER-DERIVED FORMAT GUIDANCE (aggregate metadata only; {scope}; n={sample}): "
        + "; ".join(details)
        + ". This is an observational association, not evidence of causation or viewer retention. "
          "Use a broad pattern only if it naturally fits the independently sourced topic and angle. "
          "Do not copy or imitate any creator's wording, sequence, captions, visuals, audio, or signature style; "
          "no creator examples are supplied to this prompt. Keep the assigned format, lane hook, required beat count, "
          f"and {target} target. The public video metadata cannot reveal the spoken hook, beat count, or caption density; do not infer them.\n"
    )


def retention_guidance(cfg: dict, summary: dict | None = None, now=None) -> str:
    """Turn fresh channel retention measurements into cautious writing guidance, not a platform rule."""
    data = read_json("analytics_summary.json", {}) if summary is None else summary
    if not isinstance(data, dict):
        return ""
    raw_updated = data.get("updated")
    try:
        updated = datetime.fromisoformat(str(raw_updated).replace("Z", "+00:00"))
        if updated.tzinfo is None:
            return ""
        reference = now or now_utc()
        if reference.tzinfo is None:
            return ""
        age = reference.astimezone(timezone.utc) - updated.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return ""
    if age < -timedelta(minutes=5) or age > timedelta(days=14):
        return ""

    target_summary = data.get("average_view_target")
    target_summary = target_summary if isinstance(target_summary, dict) else {}
    curve = data.get("retention")
    curve = curve if isinstance(curve, dict) else {}
    qualified = target_summary.get("learning_qualified")
    qualified = qualified if isinstance(qualified, dict) else None
    try:
        if qualified is not None:
            n_views = int(qualified.get("n", 0))
            median_value = qualified.get("median_avg_view_pct")
            minimum_sample = int(qualified.get("minimum_sample_views", 50))
            provisional_n = int(target_summary.get("provisional_n", 0))
            sample_detail = (f"sample-qualified at ≥{minimum_sample} engaged views (public views used only "
                             f"when engaged-view data is unavailable; {provisional_n} low- or unknown-sample values excluded)")
        else:  # older summaries have no sample-qualified cohort; retain the previous guarded fallback
            n_views = int(target_summary.get("n", 0))
            median_value = target_summary.get("median_avg_view_pct", curve.get("median_avg_view_pct"))
            sample_detail = "sample-count qualification not available in this older summary"
        median = float(median_value)
        target = float(target_summary.get("gate_pct", cfg.get("analytics", {}).get("target_avg_view_pct", 70)))
        n_curves = int(curve.get("videos", 0))
    except (TypeError, ValueError, OverflowError):
        return ""
    if (n_views < 10 or n_curves < 10 or not math.isfinite(median) or not math.isfinite(target)
            or not 0 <= median <= 400 or not 0 < target <= 400):
        return ""
    drop = curve.get("worst_drop")
    drop = drop if isinstance(drop, dict) else {}
    start, end = drop.get("from_pct"), drop.get("to_pct")
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= 100
            or curve.get("where") != "hook" or start >= 20 or end > 30):
        return ""

    ccfg = cfg.get("content", {}) if isinstance(cfg, dict) else {}
    hook_min, hook_max = ccfg.get("hook_words_min", 6), ccfg.get("hook_words_max", 9)
    details = (f"median average view percentage {median:.0f}% against the internal {target:.0f}% target "
               f"(n={n_views}; {sample_detail}) and the largest relative watch-ratio decline at "
               f"{start}-{end}% of runtime (n={n_curves} curves)")
    return (
        "CHANNEL RETENTION LEARNING (fresh internal observations, not a YouTube ranking rule): " + details + ". "
        f"Treat this only as a writing hypothesis: keep the hook {hook_min}-{hook_max} words, make it clear without "
        "the title, identify the specific subject by beat two, and let beat two deliver a distinct source-backed "
        "surprise or consequence immediately. Do not add a preamble, repeat the hook, invent stakes, or withhold "
        "basic context. Watch-ratio curves are not viewer-survival percentages or feed-swipe data; never state this "
        "analytics note in the video.\n"
    )


def actionable_review_fixes(value) -> list[str]:
    """Normalize the independent reviewer's remaining change requests for the publish gate."""
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif value is None:
        return ["reviewer omitted the required fixes list"]
    else:
        return ["reviewer returned a malformed fixes list"]
    no_change = {"", "none", "n/a", "na", "-", "no fixes", "no changes", "no changes needed",
                 "no edits", "no edits needed", "nothing to change", "nothing to fix"}
    out = []
    for item in items:
        text = str(item or "").strip()
        if text.lower().strip(" .!?") not in no_change:
            out.append(text[:300])
    return out


def _finite_in_range(number, low: float, high: float) -> bool:
    try:
        return (not isinstance(number, bool) and isinstance(number, (int, float))
                and math.isfinite(number) and low <= number <= high)
    except (OverflowError, TypeError, ValueError):
        return False


def _valid_review_score(number) -> bool:
    return _finite_in_range(number, 0, 10)


def metadata_schema_issues(value) -> list[str]:
    """Validate upload-facing metadata before it can be attached to an otherwise approved script."""
    if not isinstance(value, dict):
        return ["metadata response must be a JSON object"]
    issues = []
    description = value.get("description")
    if not isinstance(description, str) or len(description.strip()) < 40 or len(description) > 1_000 \
            or "#" in description or "\n" in description:
        issues.append("description must be a complete 40-1000 character paragraph with no hashtags")
    tags = value.get("tags")
    if (not isinstance(tags, list) or not 8 <= len(tags) <= 12
            or any(not isinstance(tag, str) or not tag.strip() or tag != tag.strip()
                   or tag.startswith("#") or len(tag) > 50 for tag in tags)
            or (isinstance(tags, list) and sum(len(tag) for tag in tags if isinstance(tag, str)) > 500)
            or (isinstance(tags, list)
                and len({tag.casefold() for tag in tags if isinstance(tag, str)}) != len(tags))):
        issues.append("tags must be an array of 8-12 short, non-hashtag strings")
    hashtags = value.get("hashtags")
    if (not isinstance(hashtags, list) or len(hashtags) != 3
            or any(not isinstance(tag, str) or not re.fullmatch(r"#[A-Za-z0-9_]{1,30}", tag) for tag in hashtags)
            or (isinstance(hashtags, list) and len(set(hashtags)) != len(hashtags))):
        issues.append("hashtags must be exactly three valid hashtag strings")
    thumbnail = value.get("thumbnail_text")
    if (not isinstance(thumbnail, str) or not 2 <= len(thumbnail.split()) <= 4 or len(thumbnail) > 60
            or "#" in thumbnail or "\n" in thumbnail):
        issues.append("thumbnail_text must be 2-4 words and at most 60 characters")
    return issues


def reviewer_schema_issues(value, has_trend: bool = False) -> list[str]:
    """Validate every decision used by the publish gate; never replace missing scores with a pass.

    Optional turn-strength telemetry is intentionally excluded: absent or malformed diagnostics must
    not affect approval, and are normalized to unavailable when the review is stored.
    """
    if not isinstance(value, dict):
        return ["response must be a JSON object"]
    issues = []
    for key in ("score", "hook_strength", "entertainment", "coherence"):
        number = value.get(key)
        if not _valid_review_score(number):
            issues.append(f"{key} must be a finite number from 0 to 10")
    value_add = value.get("value_add")
    if not isinstance(value_add, str) or not value_add.strip():
        issues.append("value_add must be a non-empty explanation of what the viewer learns")
    for key in ("misleading_title", "advertiser_friendly", "loops"):
        if not isinstance(value.get(key), bool):
            issues.append(f"{key} must be a boolean")
    for key in ("factual_errors", "policy_concerns", "fixes"):
        items = value.get(key)
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            issues.append(f"{key} must be an array of strings")
    if has_trend:
        number = value.get("trend_alignment")
        if not _valid_review_score(number):
            issues.append("trend_alignment must be a finite number from 0 to 10")
        reason = value.get("trend_alignment_reason")
        if not isinstance(reason, str) or not reason.strip():
            issues.append("trend_alignment_reason must be a non-empty string")
    if "issues" in value and (not isinstance(value["issues"], list)
                               or any(not isinstance(item, str) for item in value["issues"])):
        issues.append("issues must be an array of strings when present")
    return issues


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


def _subject_terms(subject: str, stop: set[str]) -> set[str]:
    """Distinctive subject words, excluding standalone calendar years.

    Years can anchor an event but are not depictable objects. Requiring a frame to
    literally name "2026" rejected valid astronomy visuals for titles such as
    "2026 in astronomy"; the actual subject word must still be shown and spoken.
    """
    from .originality import tokens

    base = re.sub(r"\(.*?\)", " ", subject or "")
    return {t for t in tokens(base) if len(t) > 3 and t not in stop
            and not re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})", t)}


def _subject_forms(term: str) -> set[str]:
    """A small plural/singular bridge; it must not turn the subject gate into fuzzy matching."""
    forms = {term}
    if len(term) > 4:
        if term.endswith("ing") and len(term) > 6:
            stem = term[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] in "bdgmnprt":
                stem = stem[:-1]
            forms.add(stem)
            forms.add(stem[:-1] + "ies" if stem.endswith("y") else stem + "s")
        elif term.endswith("ies"):
            forms.add(term[:-3] + "y")
        elif term.endswith("y"):
            forms.add(term[:-1] + "ies")
        elif term.endswith("s") and not term.endswith("ss"):
            forms.add(term[:-1])
        else:
            forms.add(term + "s")
    return forms


def _matched_subject_terms(key: set[str], observed: set[str]) -> set[str]:
    return {term for term in key if _subject_forms(term) & observed}


_VISUAL_SUBJECT_MODIFIERS = frozenset("""
a an after and are at ancient by crazy discovery discoveries discovered discover evolution evolving fast
for from giant hidden huge impossible in into insane jump jumps jumping lost loves love massive mystery mysteries
mysterious of old on or poison poisoned secret shocking survive survives survival terrifying the to toxic unearthed
venom weird with years year
""".split())
_VISUAL_GENERIC_SINGLETONS = frozenset("""
animal artifact cave city creature cycle cycles history island mountain mystery object river species sphere stone story temple
village town
""".split())


def visual_subject_for_topic(topic: dict | None, source: dict | None = None) -> str:
    """Resolve a trend label to the concrete name that must appear in a shot and narration.

    Trend titles often append a mechanism or click-driving adjective to the actual subject
    ("Mountain Goats Jumps", "Cleopatra Eyeliner Poison"). Those modifiers need not be
    pictured. If the trend label is only abstract framing, use the specific article selected
    to ground its angle; the trend-alignment gate still independently checks the live story.
    """
    topic = topic if isinstance(topic, dict) else {}
    source = source if isinstance(source, dict) else {}
    raw = str(topic.get("script_subject") or "").strip()
    source_title = str(source.get("title") or "").strip()

    def core(label: str) -> list[str]:
        result = []
        for word in re.findall(r"[\w’'-]+", label, re.UNICODE):
            term = word.casefold().replace("’", "'")
            term = term[:-2] if term.endswith("'s") else term
            if (len(term) > 2 and not term.isdigit()
                    and term not in _VISUAL_SUBJECT_MODIFIERS):
                result.append(word[:-2] if word.casefold().endswith("'s") else word)
        return result

    trend_core = core(raw)
    if trend_core and not (len(trend_core) == 1
                           and trend_core[0].casefold() in _VISUAL_GENERIC_SINGLETONS):
        return " ".join(trend_core)
    source_core = core(source_title)
    if source_core:
        return " ".join(source_core)
    return raw or source_title


def names_subject(script: dict, subject: str) -> bool:
    """Does any shot actually depict the thing the video is about?

    The Montauk Monster episode asked for "beach in Montauk New York with waves",
    "shoreline with driftwood and seaweed" and "waves washing onto sandy beach". The
    vision gate scored those 8, 10 and 10 - correctly, because they were delivered
    exactly. The viewer saw six interchangeable stock beaches and two photographs of
    healthy living raccoons, and never once saw the carcass the hook promised.

    Nothing in the system penalised that: easy requests score highest, so a writer
    optimising for a passing grade learns to ask for ambient scenery. This requires at
    least one shot to name the subject, which also means a topic with no free image of
    its own subject is abandoned rather than illustrated with filler - the right
    outcome, because a video that cannot show its subject has no reason to be made.
    """
    from .originality import tokens

    stop = {"the", "a", "an", "of", "and", "in", "on", "at", "for", "with", "to",
            "incident", "mystery", "case", "story", "history", "effect", "phenomenon", "signal"}
    key = _subject_terms(subject, stop)
    if not key:
        return True          # nothing distinctive to require; do not block the video
    # Every distinctive word, not any one of them. "Montauk Monster" reduced to a
    # match on "montauk" alone was satisfied by "beach in Montauk New York with
    # waves" - a shared place name is the setting, not the subject, and that single
    # token is what let an entire episode run without showing its own creature.
    need = len(key) if len(key) <= 3 else max(2, round(len(key) * 0.6))
    for seg in script.get("segments", []):
        vis = seg.get("visual") or {}
        blob = " ".join([str(vis.get("shows", ""))] + [str(q) for q in (vis.get("queries") or [])])
        if len(_matched_subject_terms(key, set(tokens(blob)))) >= need:
            return True
    return False


GENERIC_ASIDES = {
    "nature please", "rude", "which is frankly rude", "bold plan", "bold plan terrible plan",
    "no airbags", "office politics are brutal", "okay this one is unhinged",
    "internet zoology at its finest", "science is wild", "nature is metal",
}


def aside_is_generic(aside: str) -> bool:
    """Would this reaction fit any video on the channel?

    The prompt used to offer "Nature, please." as a worked example and the model
    returned it verbatim in two different scripts. Canned asides waste words and weaken
    the episode's distinct voice. This is a local writing-quality check, not a claim
    about a fixed YouTube policy threshold.
    """
    import re as _re

    norm = _re.sub(r"[^a-z ]", "", (aside or "").lower()).strip()
    norm = _re.sub(r"\s+", " ", norm)
    return norm in GENERIC_ASIDES


def subject_is_depictable(subject: str, cfg: dict, source: dict | None = None) -> bool:
    """Is there a free image of this subject before we write about it?

    First version asked commons_search for four results and treated an empty list as
    "cannot be pictured". It rejected Sea turtle, Tiger shark, Göbekli Tepe, Proboscis
    monkey and Mimic octopus - subjects with thousands of free photographs - because a
    four-result sample survives licence and width filtering only sometimes. Fifteen
    topics were discarded in one run and the run shipped nothing. A check that is wrong
    in the safe direction still stops the channel.

    The grounded article can provide a free, accurate image list when its title names
    the actual trend subject. A different article may explain the mechanism without
    depicting the thing that trended, so its images must not satisfy this gate. In that
    case, search for the trend subject itself before spending tokens on a draft.
    """
    from .originality import tokens

    if not subject:
        return True
    if source is not None and (source.get("images") or []):
        stop = {"the", "a", "an", "of", "and", "in", "on", "at", "for", "with", "to",
                "incident", "mystery", "case", "story", "history", "effect", "phenomenon", "signal"}
        key = _subject_terms(subject, stop)
        title_terms = set(tokens(str(source.get("title") or "")))
        if not key or key.issubset(_matched_subject_terms(key, title_terms)):
            return True
    from .media import commons_search

    # Search the full label first, then progressively remove trailing modifiers. Commons
    # metadata often indexes "iPhone Duo" but not the longer writer label "iPhone Duo
    # cameras". This is a discovery preflight only: the script's subject, the matched
    # assets, and final-render QA still have to pass their independent relevance gates.
    queries = [subject]
    terms = [term for term in tokens(subject) if len(term) > 2]
    for end in range(len(terms) - 1, 0, -1):
        query = " ".join(terms[:end])
        if query.casefold() not in {item.casefold() for item in queries}:
            queries.append(query)
    try:
        for query in queries:
            if commons_search(query, cfg.get("media", {}).get("licenses_allowed") or [],
                              200, limit=20):
                return True
    except Exception:  # noqa: BLE001
        return True
    return False


BANNED_OPENERS = (
    "did you know", "ever wonder", "have you ever", "here are", "here is",
    "in this video", "today we", "let's talk about", "lets talk about",
    "what if i told you", "you won't believe", "you wont believe",
    "imagine if", "this is the story of", "welcome back",
    # Seen in production: "Imagine a creature with a 33-foot wingspan." An instruction
    # to the viewer is not a fact, and it delays the specific by three words.
    "imagine a", "imagine the", "imagine you", "picture this", "picture a",
    "meet the", "let me tell you", "here's why", "heres why",
)


def banned_opener(hook: str) -> str | None:
    """Which banned opener this hook uses, if any.

    The prompt already lists these, and a hook still shipped as "Ever wonder why
    airplane toilets make that loud WHOOSH sound?" - the ban said "have you ever
    wondered" and the model wrote a variant. Enumerating variants in a prompt is a
    losing game; matching a prefix is not.
    """
    import re as _re

    norm = _re.sub(r"[^a-z' ]", " ", (hook or "").lower())
    norm = _re.sub(r"\s+", " ", norm).strip()
    for b in BANNED_OPENERS:
        if norm.startswith(b):
            return b
    return None


def payoff_is_a_question(last: str) -> bool:
    """A question is not a payoff.

    One script closed "The Hittite Empire or the Nile's floods - which won?", which
    passes an emptiness check because it carries content words, yet answers nothing
    and leaves the viewer with homework instead of a reason to replay. The opening creates
    a curiosity gap; the last line must resolve it with a fact.
    """
    return (last or "").strip().endswith("?")


def narration_names_subject(script: dict, subject: str) -> bool:
    """Does the script ever say what it is about?

    Two scripts shipped as "Some fish spend three quarters of life on land" and
    "Imagine a creature with a 33-foot wingspan" - neither ever names the mudskipper
    or the pterosaur. Coyness is not suspense: a viewer cannot search, recognise or
    remember "some fish", and the specific noun is the thing that makes a fact feel
    real rather than like filler.

    The visual gate already requires the subject on screen; this requires it in the
    words. Parenthetical qualifiers are stripped, so "Jonathan (tortoise)" is
    satisfied by either word.
    """
    import re as _re

    from .originality import tokens

    stop = {"the", "a", "an", "of", "and", "in", "on", "at", "for", "with", "to",
            "incident", "mystery", "case", "story", "history", "effect", "phenomenon", "signal",
            "accident", "escape", "project", "experiment"}
    key = _subject_terms(subject, stop)
    if not key:
        return True
    said = set()
    for seg in script.get("segments", []):
        said |= set(tokens(spoken_text(seg)))
    return bool(_matched_subject_terms(key, said))


def names_subject_in_opening(script: dict, subject: str, max_segments: int = 2) -> bool:
    """Is the actual subject named by the hook or the early turn, not introduced after viewers leave?"""
    from .originality import tokens

    stop = {"the", "a", "an", "of", "and", "in", "on", "at", "for", "with", "to",
            "incident", "mystery", "case", "story", "history", "effect", "phenomenon", "signal",
            "accident", "escape", "project", "experiment"}
    key = _subject_terms(subject, stop)
    if not key:
        return True
    segments = script.get("segments", []) if isinstance(script, dict) else []
    if not isinstance(segments, list):
        return False
    observed = set()
    for seg in segments[:max(1, int(max_segments))]:
        if isinstance(seg, dict):
            aside = seg.get("aside") if isinstance(seg.get("aside"), str) else ""
            observed.update(tokens(str(seg.get("text") or "") + " " + aside))
    return bool(_matched_subject_terms(key, observed))


ESCALATION_TEMPLATE = ("weird:", "weirder:", "weirdest:", "completely unhinged:",
                       "but wait:", "it gets worse:", "it gets better:", "plot twist:")


def uses_escalation_template(script: dict) -> int:
    """Count lines opening with the canned escalation ladder.

    "Weird: ... Weirder: ... Completely unhinged: ..." appeared in a cricket script and
    an Alcatraz script days apart. It is a list wearing the costume of a story - the
    labels assert escalation instead of the facts earning it - and repeated across
    uploads it is the near-identical structure that YouTube's repetitious-content
    policy is written about.
    """
    n = 0
    for seg in script.get("segments", []):
        low = spoken_text(seg).strip().lower()
        if any(low.startswith(t) for t in ESCALATION_TEMPLATE):
            n += 1
    return n


def _angle_terms(angle: str, subject: str = "") -> set[str]:
    """Distinctive angle words after removing the subject name and connective filler."""
    from .originality import tokens

    stop = {"the", "a", "an", "of", "and", "in", "on", "at", "for", "with", "to", "is",
            "was", "were", "that", "this", "it", "its", "but", "not", "they", "their",
            "from", "about", "into", "than", "then", "when", "how", "why", "what",
            "most", "more", "very", "just", "even", "only", "also", "some", "one",
            "first", "actually", "really", "still", "because", "which", "who", "been",
            "have", "has", "had", "are", "can", "could", "would", "will", "did", "does",
            # Rhetorical framing is not evidence that the specific factual angle appears in
            # the source (e.g. "explore the unsettling reality" or "sheer impossible nerve").
            "after", "ancient", "break", "baffled", "crazy", "down", "examine", "explore", "impossible",
            "insane", "leaving", "near", "perfectly", "required", "reality", "scary", "shocking", "sheer",
            "terrifying", "unsettling", "wild"}
    key = {t for t in tokens(angle or "") if len(t) > 3 and t not in stop}
    # The subject appears in every script about it; only the words describing the
    # surprising turn can prove the selected trend angle was delivered.
    key -= {t for t in tokens(subject or "") if len(t) > 3}
    return key


def delivers_angle(script: dict, angle: str, subject: str = "") -> bool:
    """Check that a script carries the distinctive point of the current trend.

    Topic-level matching used to pass when ONE generic word overlapped. That let a
    sales-figures script pass for a trend about an underpowered console faking 3D.
    Requiring multiple distinctive terms is still paraphrase-tolerant; the independent
    reviewer separately judges the meaning, rather than demanding an exact quote.
    A missing/empty angle fails closed: a current trend without a specific story is not
    enough reason to make an episode.
    """
    from .originality import tokens

    key = _angle_terms(angle, subject)
    if not key:
        return False
    said = {t for seg in script.get("segments", []) for t in tokens(spoken_text(seg))}
    needed = 1 if len(key) == 1 else max(2, (len(key) + 2) // 3)
    return len(key & said) >= needed


def source_supports_angle(angle: str, subject: str, source_text: str) -> bool:
    """Fail closed before drafting if the grounded research cannot substantiate the trend angle.

    The candidate selector sees trend metadata, not article text, so an angle can be a
    plausible-sounding guess that the selected source never supports. Requiring at least
    half of the angle's distinctive terms in the actual research text is a conservative
    preflight: it prevents spending a draft on a generic/adjacent article while leaving
    the existing script and independent-review gates in place for meaning and facts.
    """
    from .originality import tokens

    key = _angle_terms(angle, subject)
    if len(key) < 2 or not isinstance(source_text, str) or not source_text.strip():
        return False
    observed = set(tokens(source_text))
    supported = _matched_subject_terms(key, observed)
    required = max(2, math.ceil(len(key) * 0.5))
    return len(supported) >= required

UNREADABLE_SHOTS = ("infographic", "chart", "graph", "diagram", "schematic", "timeline",
                    "bar chart", "pie chart", "screenshot", "table of", "map showing",
                    "contour map", "sonar map", "bathymetry", "bathymetric", "plot of",
                    "statistics", "data visualisation", "data visualization")


def unreadable_shot_request(script: dict) -> str | None:
    """Is the writer asking for something no phone viewer can read?

    A shipped script requested "Mariana Trench depth infographic" and got exactly that:
    a bathymetric contour chart with axis labels and a depth profile reading -10100.
    The vision judge scored it 10/10 because the chart genuinely does show the trench.
    Correct and unreadable is the worst combination on a Short - the viewer sees noise
    and leaves - and the cheapest place to stop it is before the search runs.
    """
    for seg in script.get("segments", []):
        vis = seg.get("visual") or {}
        blob = " ".join([str(vis.get("shows", ""))] + [str(q) for q in (vis.get("queries") or [])]).lower()
        for bad in UNREADABLE_SHOTS:
            if bad in blob:
                return bad
    return None


def payoff_is_empty(hook: str, last: str) -> bool:
    """Does the closing line actually say anything new?

    Closing the "repeat the hook" exit opened another one. With echoing blocked, a
    script ended "And that's exactly why we have to ask..." - no fact, no number, no
    answer, just a rhetorical lead-in to nothing, with the video's best fact stranded
    mid-script. That is the same failure as the echo in a different costume: the last
    line carries no information.

    A payoff qualifies if it contributes something the hook did not: a number, or at
    least two content words the hook does not already have. Trailing teasers are
    rejected outright because they are the specific shape the model reaches for.
    """
    import re as _re

    from .originality import tokens

    low = last.lower().strip()
    teasers = ("we have to ask", "the question remains", "makes you wonder",
               "what happens next", "and that is the question", "so the question")
    if any(t in low for t in teasers) or low.endswith("..."):
        return True
    if _re.search(r"\d", last):
        return False                       # a concrete number is information
    stop = {"and", "that", "the", "a", "an", "is", "was", "are", "were", "to", "of",
            "in", "on", "at", "for", "with", "so", "it", "its", "this", "but", "why",
            "how", "we", "you", "they", "he", "she", "just", "still", "even", "all",
            "exactly", "have", "has", "had", "be", "been", "as", "by", "from", "not"}
    hook_t = set(tokens(hook))
    fresh = [t for t in tokens(last) if t not in stop and t not in hook_t]
    return len(fresh) < 2


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
        if type(n) is not int or n <= 0 or not isinstance(candidates, list):
            return []
        cats = self.cfg["channel"]["categories"]
        ccfg = self.cfg["content"]
        exclude = exclude if isinstance(exclude, set) else set()
        recent = recent if isinstance(recent, set) else set()
        min_viral_raw = ccfg.get("min_viral_score", 7)
        if not _finite_in_range(min_viral_raw, 7, 10):
            log.warning("minimum viral-score configuration must be finite and at least 7; no topics selected")
            return []
        min_viral = float(min_viral_raw)
        trend_cfg = self.cfg.get("trends", {}) or {}
        if not isinstance(trend_cfg, dict):
            log.warning("invalid trend configuration; no topics selected")
            return []
        min_wiki_raw = trend_cfg.get("min_wikipedia_spike", 3.0)
        if not _finite_in_range(min_wiki_raw, 0, float("inf")) or min_wiki_raw <= 0:
            log.warning("invalid Wikipedia trend threshold; no topics selected")
            return []
        min_wiki_spike = float(min_wiki_raw)
        int_thresholds = {
            "min_google_trend_traffic": trend_cfg.get("min_google_trend_traffic", 1_000),
            "min_independent_trend_families": trend_cfg.get("min_independent_trend_families", 2),
            "min_hackernews_points": trend_cfg.get("min_hackernews_points", 10),
            "outlier_min_views": trend_cfg.get("outlier_min_views", 30_000),
        }
        if any(not _finite_in_range(value, 1, float("inf")) or int(value) != value
               for value in int_thresholds.values()):
            log.warning("trend count thresholds must be positive integers; no topics selected")
            return []
        min_google_traffic = int(int_thresholds["min_google_trend_traffic"])
        min_families = int(int_thresholds["min_independent_trend_families"])
        min_hn_points = int(int_thresholds["min_hackernews_points"])
        outlier_min_views = int(int_thresholds["outlier_min_views"])
        outlier_window_raw = trend_cfg.get("outlier_window_hours", 72)
        if not _finite_in_range(outlier_window_raw, 0, float("inf")) or outlier_window_raw <= 0:
            log.warning("outlier-window threshold must be finite and positive; no topics selected")
            return []
        outlier_window_hours = float(outlier_window_raw)
        if min_families < 2:
            log.warning("at least two independent trend families are required; no topics selected")
            return []

        def valid_candidate(candidate):
            return (isinstance(candidate, dict) and isinstance(candidate.get("topic"), str)
                    and bool(candidate["topic"].strip()) and isinstance(candidate.get("topic_key"), str)
                    and bool(candidate["topic_key"].strip()) and isinstance(candidate.get("sources"), list)
                    and all(isinstance(source, str) and source for source in candidate["sources"])
                    and _finite_in_range(candidate.get("score"), 0, 1))

        lane_specs = trend_cfg.get("lanes") or {}
        if not isinstance(lane_specs, dict) or not lane_specs:
            log.warning("approved content lanes are missing or malformed; no topics selected")
            return []
        approved_lanes = set(lane_specs)

        def in_editorial_scope(candidate):
            source_lane = candidate.get("lane")
            if source_lane and source_lane not in approved_lanes:
                return False
            # Use the trend's own context, not a model-chosen Wikipedia category, to
            # identify obvious sports fixtures or clubs such as Borussia Dortmund.
            trend_context = [str(candidate.get("topic") or "")]
            raw_context = candidate.get("context")
            if isinstance(raw_context, list):
                trend_context.extend(str(item) for item in raw_context if item)
            raw_signals = candidate.get("signals")
            if isinstance(raw_signals, list):
                for item in raw_signals:
                    if not isinstance(item, dict):
                        continue
                    if item.get("topic"):
                        trend_context.append(str(item["topic"]))
                    if isinstance(item.get("context"), list):
                        trend_context.extend(str(value) for value in item["context"] if value)
            if _is_sports_trend(" ".join(trend_context)):
                return False
            # A YouTube-chart category is direct metadata, not a model interpretation.
            # Sports, gaming and news/politics trends cannot be relabelled into our lanes.
            for signal in candidate.get("signals") or []:
                if not isinstance(signal, dict) or signal.get("source") != "youtube_chart":
                    continue
                evidence = signal.get("evidence") or {}
                if isinstance(evidence, dict) and str(evidence.get("yt_category", "")) in {"17", "20", "25"}:
                    return False
            return True

        eligible = [c for c in candidates if valid_candidate(c)
                    and has_current_trend_evidence(c, min_wiki_spike, trend_cfg)]
        live_count = len(eligible)
        eligible = [c for c in eligible if in_editorial_scope(c)]
        if live_count != len(eligible):
            log.info("excluding %d live candidate(s) outside the approved history/mystery or science/nature scope",
                     live_count - len(eligible))
        if len(eligible) < len(candidates):
            log.info("excluding %d candidate(s) without qualifying live evidence or approved scope",
                     len(candidates) - len(eligible))
        # Keep the decision batch bounded for free-tier model context windows. Candidates
        # are already sorted by measured trend strength; 24 leaves ample choice and backups.
        pool = [c for c in eligible if c["topic_key"] not in recent and c["topic_key"] not in exclude
                and c["score"] >= ccfg["min_trend_score"] * 0.5][:24]
        if not pool:
            return []

        def evidence(c):
            bits = [f"sources={','.join(sorted(set(c['sources'])))}", f"trend={c['score']:.2f}"]
            if isinstance(c.get("lane"), str) and c["lane"] in approved_lanes:
                bits.append(f"source lane={c['lane']}")
            measured = trend_evidence_summary(c)
            if measured:
                bits.append("measured trend evidence: " + _clip(measured, 150))
            raw_context = c.get("context")
            context = [str(x).strip() for x in raw_context if str(x).strip()] if isinstance(raw_context, list) else []
            if context:
                # News/viral context distinguishes an ambiguous trend label (e.g. a
                # club from a city). It is discovery evidence only, never a script source.
                bits.append("trend context (discovery only): " + _clip(" / ".join(context[:2]), 140))
            return " | ".join(bits)

        listing = "\n".join(f"{i}. {c['topic'][:150]} | {evidence(c)}" for i, c in enumerate(pool))
        log.info("live-qualified selector shortlist: %s", [
            f"{i}: {c['topic'][:60]} | {evidence(c)}" for i, c in enumerate(pool)])
        weights = {c: round(self.strategy.category_weight(c), 2) for c in cats}
        today = now_utc()
        want = min(len(pool), 3 * n + 4)
        in_season = getattr(self.strategy, "formats_in_season", None)
        formats = in_season() if in_season else list(ccfg.get("formats", []))
        fmt_list = "\n".join(
            f"  {f}: {_clip(FORMAT_GUIDE[f].split('. ', 1)[0], 72)}"
            for f in formats if f in FORMAT_GUIDE)
        season = ""
        if "creepy_true" in (getattr(self.strategy, "seasonal_now", lambda: [])()):
            season = " October is spooky season; favour eerie-but-true subjects, never gore."
        user = f"""TODAY: {today:%A %B %d, %Y}.{season}
CHANNEL: {self.cfg['channel']['niche']}
HARD EDITORIAL SCOPE: ONLY history_mystery (documented history, archaeology, unresolved past events) and
science_nature (science, space, animals, natural phenomena, geography). Exclude sports, gaming, current news or
politics, celebrity gossip, living-person biographies and unrelated entertainment. Never relabel an off-scope trend.
CATEGORIES (learned audience preference 0-1; favour higher): {json.dumps(weights)}
RECENTLY COVERED — avoid: {', '.join(list(recent)[:40]) or 'none'}

LIVE TREND CANDIDATES (each passed the independent current-evidence gate; evidence below is for discovery, not facts):
{listing}

Return as many qualifying candidates as possible, up to {want}, as fallbacks for the requested video count.
Do not stop after one if other candidates independently score >= {min_viral:g}/10 and fit scope and safety; rank the best first.
Choose only original, genuinely ENTERTAINING, family-friendly 15-20 second Shorts with a scroll-stopping true surprise,
broad appeal, measured demand NOW, emotional pull or awe without tragedy/divisiveness, and relevant real visuals.
Do not choose an evergreen or Wikipedia-only idea. Return fewer picks or none when candidates do not genuinely fit.

FORMATS we can make (choose 2-4 that the specific article honestly supports; myth_buster requires an explicit misconception):
{fmt_list}

For every pick:
- `script_subject` is a concise concrete name (1-6 words) for the CURRENT TREND CANDIDATE — it must name and be
  depicted in the video, share a meaningful word with the candidate, and must NOT be replaced by a background article title.
- `angle` must be a specific, independently worded factual claim. Trend evidence is discovery, not proof; the selector
  lacks article text, so invent no details. Grounded research must support the angle before drafting.
- `wiki_query` must be the MOST SPECIFIC English Wikipedia article supporting it, not a broad category (e.g. Bou Craa,
  not “Conveyor belt”). If none supports it, do not select the candidate.
- Viral/outlier titles are discovery evidence only: do not copy the outlier headline, wording, script, sequence, captions,
  visuals, audio, or creator style. `why_trending` briefly cites the measured evidence shown above.
- Use a valid channel `category` and only fitting format names. Every pick MUST include an explicit risk verdict
  (`none`, `low`, or `high`); use `high` if any safety or scope concern cannot be ruled out.

Return JSON only, as an object with a `picks` array, best first. Each pick has exactly these fields:
{{"picks":[{{"index":<candidate index>,"viral_score":<0-10>,"category":"<one of {', '.join(cats)}>",
"script_subject":"<1-6 words>","angle":"<one specific sentence>","wiki_query":"<specific article or empty>",
"formats":["<2-4 fitting names>"],"why_trending":"<brief evidence>","risk":"none|low|high"}}]}}"""

        def validate(o):
            assert isinstance(o, dict), "selection response must be a JSON object"
            assert isinstance(o.get("picks"), list) and o["picks"], "no picks"
            assert all(isinstance(pick, dict) for pick in o["picks"]), "every pick must be a JSON object"
            required = ("index", "viral_score", "category", "script_subject", "angle",
                        "wiki_query", "formats", "risk")
            for i, pick in enumerate(o["picks"]):
                # A high-risk pick is dropped immediately; only its risk verdict is
                # needed. Every potentially usable pick must carry the full schema.
                risk = pick.get("risk")
                assert isinstance(risk, str) and risk in {"none", "low", "high"}, \
                    f"pick {i} needs an explicit risk verdict: none, low, or high"
                if risk == "high":
                    continue
                missing = [key for key in required if key != "risk" and key not in pick]
                assert not missing, f"pick {i} is missing required field(s): {', '.join(missing)}"
                index = pick.get("index")
                subject = pick.get("script_subject")
                if type(index) is int and 0 <= index < len(pool):
                    assert isinstance(subject, str) and subject.strip() and 1 <= len(subject.split()) <= 6, \
                        f"pick {i} script_subject must be a concise, concrete label"
                    subject_terms = set(originality.tokens(subject))
                    trend_terms = set(originality.tokens(pool[index]["topic"]))
                    assert subject_terms and trend_terms and subject_terms & trend_terms, \
                        f"pick {i} script_subject must name the trend candidate, not a background article"

        try:
            res = self.llm.json(SELECT_SYSTEM, user, temperature=0.4, validate=validate)
            validate(res)  # protect adapters that do not enforce the callback
        except Exception as e:  # noqa: BLE001 — invalid/unavailable selection yields no candidates
            # Fail closed on virality: no un-scored "maybe trending" guesses.
            log.warning("topic selection unavailable or malformed (%s) — no topics this round", str(e)[:300])
            return []
        picks, dropped = [], []
        seen_topics: set[str] = set()
        for p in res["picks"]:
            try:
                idx = p.get("index")
                if type(idx) is not int or not 0 <= idx < len(pool):
                    dropped.append("(invalid candidate index)")
                    continue
                c = dict(pool[idx])
                risk = p.get("risk")
                if risk not in {"none", "low", "high"}:
                    dropped.append(f"{c['topic'][:40]} (missing or invalid risk verdict)")
                    continue
                if risk == "high":
                    continue
                viral_raw = p.get("viral_score")
                if not _valid_review_score(viral_raw):
                    dropped.append(f"{c['topic'][:40]} (invalid viral score)")
                    continue
                vs = float(viral_raw)
                if vs < min_viral:
                    dropped.append(f"{c['topic'][:40]} ({vs:.0f})")
                    continue
                # A topic must have a specific article and a specific trend angle. Falling
                # back to the raw headline can resolve to a broad category page and erase
                # the event/detail that actually made people notice the topic.
                raw_query = p.get("wiki_query")
                if not isinstance(raw_query, str) or not raw_query.strip():
                    dropped.append(f"{c['topic'][:40]} (no specific article)")
                    continue
                wiki_query = raw_query.strip()
                raw_angle = p.get("angle")
                if not isinstance(raw_angle, str) or not raw_angle.strip():
                    dropped.append(f"{c['topic'][:40]} (no specific trend angle)")
                    continue
                angle = raw_angle.strip()
                if len(_angle_terms(angle, c["topic"])) < 2:
                    dropped.append(f"{c['topic'][:40]} (no distinctive trend angle)")
                    continue
                raw_subject = p.get("script_subject")
                if not isinstance(raw_subject, str) or not raw_subject.strip():
                    dropped.append(f"{c['topic'][:40]} (no independent script subject)")
                    continue
                script_subject = raw_subject.strip()
                subject_terms = set(originality.tokens(script_subject))
                trend_terms = set(originality.tokens(c["topic"]))
                if (len(script_subject.split()) > 6 or len(script_subject) > 80
                        or not subject_terms or not (subject_terms & trend_terms)):
                    dropped.append(f"{c['topic'][:40]} (script subject is not a concise name for the trend)")
                    continue
                cat = p.get("category")
                if not isinstance(cat, str) or cat not in cats:
                    dropped.append(f"{c['topic'][:40]} (invalid category)")
                    continue
                selected_lane = lane_of(self.cfg, cat)
                source_lane = c.get("lane")
                if not selected_lane or (source_lane and source_lane != selected_lane):
                    dropped.append(f"{c['topic'][:40]} (category does not match its approved source lane)")
                    continue
                raw_formats = p.get("formats")
                if not isinstance(raw_formats, list) or any(not isinstance(f, str) for f in raw_formats):
                    dropped.append(f"{c['topic'][:40]} (malformed format list)")
                    continue
                fits = [f for f in raw_formats if f in formats]
                if not fits:
                    dropped.append(f"{c['topic'][:40]} (no fitting format)")
                    continue
                c.update({"category": cat, "angle": angle, "wiki_query": wiki_query,
                          "script_subject": script_subject,
                          "formats": fits,
                          # Preserve reproducible observations, not an LLM's potentially invented trend claim.
                          "why_trending": trend_evidence_summary(c), "viral_score": vs})
                # blend: LLM viral potential (50%), measured trend strength (25%), and what our own analytics learned
                # about this category (12.5%) and this kind of trend source (12.5%)
                src_w = self.strategy.source_weight(primary_source(c.get("sources", [])))
                c["priority"] = round(0.50 * vs / 10 + 0.25 * c["score"] + 0.125 * weights.get(cat, 0.5)
                                      + 0.125 * src_w, 4)
                if c["topic_key"] in seen_topics:
                    dropped.append(f"{c['topic'][:40]} (duplicate selector result)")
                    continue
                seen_topics.add(c["topic_key"])
                picks.append(c)
            except (KeyError, ValueError, TypeError):
                continue
        picks.sort(key=lambda c: -c["priority"])
        log.info("viral picks: %s", [
            f"{c['topic'][:32]} → {c['script_subject'][:40]} [{c['category']}, {c['viral_score']:.0f}]"
            for c in picks])
        if dropped:
            log.info("dropped as not viral enough (< %.0f): %s", min_viral, dropped[:10])
        return picks

    # ── 2. grounded script ───────────────────────────────────────────────────
    def write(self, topic: dict, plan: dict, source: dict) -> dict:
        lo, hi = self.cfg["video"]["target_seconds"]
        words_lo, words_hi = int(lo * 2.6), int(hi * 2.6)
        ccfg = self.cfg.get("content", {}) or {}
        body_min, body_max = tuple(ccfg.get("body_beat_words", [9, 11]))
        payoff_min, payoff_max = tuple(ccfg.get("payoff_words", [6, 9]))
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
                "The last line must contain a FACT - a number, a name or an outcome, never a "
                "rhetorical lead-in like 'and that is exactly why we have to ask...', which ends "
                "the video on nothing. No 'follow for more', no goodbye.")
        else:
            loop_rule = (
                "finish on a new, surprising source-backed fact or outcome that completes the concrete curiosity gap; "
                "no call to action")
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
            '     reaction the narrator says right AFTER that line. It MUST react to something specific in THAT\n'
            '     line - name the thing, the number or the absurdity it just described. A reaction that would fit\n'
            '     any video at all ("Nature, please.", "Rude.", "Bold plan.", "No airbags.") is generic filler: it spends\n'
            '     words without adding a topic-specific reaction. Reusing canned reactions weakens the episode\'s\n'
            '     distinct voice; this is an internal writing standard, not a fixed YouTube policy threshold. It is a\n'
            '     JOKE/OPINION ONLY: no facts, numbers, names, dates or claims; never mean about real people, groups or\n'
            '     victims; family-friendly. Include a specific narrator reaction so the script sounds like Archive 13,\n'
            '     not a bare list of sourced facts. Make it actually funny.\n'
        ) if max_asides else ""
        # A living person reached this point only because safety.check_person approved one specific
        # professional angle. The writer has to be told what that angle is, or the model will drift
        # back to whatever made the person trend — which is usually the thing we just refused.
        lane_rule = lanes.writer_rule(topic, self.cfg)
        # One hook instruction, not two. The prompt used to state the bandit's
        # hook_style AND the lane's hook shape, which for a history episode read
        # "HOOK STYLE: question" immediately followed by "never open with a question".
        # The model followed the conflicting generic instruction in that sample, so the lane-specific
        # direction now takes precedence. This is an editorial consistency fix, not a monetization-policy claim.
        #
        # The lane is the more specific instruction and the one tied to the research,
        # so when a lane defines a hook shape it replaces the generic line entirely.
        _lane_hook = lanes.profile_for(topic, self.cfg).get("hook_style", "")
        if _lane_hook:
            hook_line = f"HOOK: {_lane_hook.strip()}"
        else:
            hook_line = f"HOOK STYLE: {plan['hook_style']} — {HOOK_GUIDE[plan['hook_style']]}"
        structure_line = outlier_structure_guidance(topic, self.cfg)
        originality_line = (
            "ORIGINALITY: Trend evidence is for choosing a currently interesting subject, not a template. "
            "Write a fresh, independently valuable treatment from the source text and angle; never reuse or imitate "
            "another creator's title wording, script, sequence, visuals, captions, audio or signature style.\n"
        )
        person_rule = ""
        if source.get("person"):
            person_rule = (
                "REAL LIVING PERSON: stay strictly on their public professional work"
                + (f" — specifically: {source['safe_angle']}" if source.get("safe_angle") else "")
                + ". Never mention their health, death, family, relationships, appearance, money, "
                  "legal matters, accusations, feuds or politics. State nothing you cannot point to "
                  "in the SOURCE TEXT. No 'allegedly', 'reportedly' or 'sources say'.")
        trend_subject = str(topic.get("script_subject") or source.get("title") or "").strip()
        subject_for_visuals = visual_subject_for_topic(topic, source) or trend_subject
        persona_line = (f"NARRATOR PERSONA: {persona['character']}\nCHANNEL PROMISE: \"{persona.get('catchphrase', '')}\" "
                        "— every wild claim is proven by the source (a PROVEN stamp + source card closes each video).\n"
                        if persona.get("character") else "")
        _cd = plan.get("comment_device") or "none"
        _cg = COMMENT_GUIDE.get(_cd, "")
        # The device must never override the facts or the loop ending; it shapes the LAST line only.
        comment_line = (f"LEAVE THEM SOMETHING TO SAY ({_cd}): {_cg} Work this into the PAYOFF line's "
                        f"wording — do NOT add a segment, do not say \"comment below\", \"let me know\" or "
                        f"any call to action, and do not change a single fact to make it land.\n") if _cg else ""
        turn_variant = plan.get("turn_variant")
        turn_rule = TURN_VARIANT_GUIDE.get(turn_variant)
        turn_line = (f"RANDOMIZED EARLY-TURN TREATMENT ({turn_variant}): {turn_rule} This is an ordering-only experiment across distinct current-trend topics. "
                     "Keep the script original; do not reuse another upload's topic, wording, or substance. Invent no facts and relax no factual, originality, safety, or approval standard. "
                     "Apply the treatment only when the source supports it.\n"
                     if turn_rule else "")
        draft_feedback = str(topic.get("_draft_feedback") or "")[:1000]
        feedback_lower = draft_feedback.lower()
        try:
            rewrite_attempt = max(1, int(topic.get("_rewrite_attempt", 1)))
        except (TypeError, ValueError, OverflowError):
            rewrite_attempt = 1
        feedback_guidance = []
        if "consequence_first" in feedback_lower or "concrete result or reversal" in feedback_lower:
            feedback_guidance.append(
                "ORDERING REPAIR (consequence_first): Segment 2 opens with a concrete, source-supported result or "
                "reversal, then its cause/actor. Keep the beat word limit; do not copy a reviewer example unless "
                "SOURCE TEXT supports it.")
        if "list of facts, not a story" in feedback_lower:
            feedback_guidance.append(
                "CAUSAL-CHAIN REPAIR: Keep one source-supported event or argument. Make each beat cause, explain, "
                "or reveal the meaning of the preceding beat; replace or remove disconnected facts. Never invent a "
                "causal link just to create a story.")
        if ("word-count repair required" in feedback_lower or "word count" in feedback_lower
                or ("words" in feedback_lower and any(term in feedback_lower
                                                       for term in ("hook", "turn", "escalation", "payoff")))):
            feedback_guidance.append(
                "WORD-COUNT REPAIR: obey the exact range in each reported failure; count whitespace-separated words "
                "in factual text only, not asides. Rewrite only the flagged beat(s), preserve the other already-valid "
                "beats, and never pad with an aside or an unsupported fact.")
        if "not entertaining enough" in feedback_lower:
            feedback_guidance.append(
                "ENTERTAINMENT REPAIR: use one genuinely surprising detail the SOURCE TEXT supports; make the story "
                "turn on a clear expectation-versus-reality contrast, build to a concrete payoff, and add a specific, "
                "fact-free narrator reaction. Make a meaningful change to the treatment, not just stronger adjectives; "
                "invent no stakes or claims.")
        if "overall review score too low" in feedback_lower:
            feedback_guidance.append(
                "OVERALL-QUALITY REPAIR: fix each named reviewer concern in the actual beats; do not chase the number "
                "or trade away factual accuracy, the trend angle, originality, safety, or beat limits.")
        if "hook too weak" in feedback_lower:
            feedback_guidance.append(
                "HOOK REPAIR: make the first line specific and immediately surprising without relying on the title; "
                "keep its exact configured word band and preserve the payoff for the end.")
        if "specific current-trend angle" in feedback_lower:
            feedback_guidance.append(
                "TREND-ANGLE REPAIR: make the selected live-trend detail the story's through-line, but use only "
                "details the SOURCE TEXT actually supports; if it cannot be supported, do not embellish it.")
        if rewrite_attempt >= 3 and draft_feedback:
            feedback_guidance.append(
                "LATER-REWRITE ESCALATION: earlier draft(s) have not cleared every gate. Do not merely synonym-swap "
                "or rephrase the prior version; materially improve the failing beat(s) while preserving all supported "
                "facts and every factual, trend, originality, safety, and quality gate.")
        feedback_line = (f"PREVIOUS DRAFT FEEDBACK (fix without changing supported facts or the selected trend angle): "
                         f"{draft_feedback}\n"
                         + ("REWRITE PRIORITIES: " + " ".join(feedback_guidance) + "\n"
                            if feedback_guidance else "")
                         if draft_feedback else "")
        prior_excerpt = _rewrite_draft_excerpt(topic.get("_prior_draft"))
        prior_line = ("PREVIOUS AUTO-TUBE ATTEMPT (our own draft, NOT factual evidence):\n" + prior_excerpt + "\n"
                      "Use this only to see what must change. Keep facts only when SOURCE TEXT supports them. "
                      "For a count-only repair, preserve passing beats; for a story/entertainment repair, change the "
                      "treatment enough to address the concern.\n"
                      if prior_excerpt else "")
        user = f"""CURRENT TREND SUBJECT (deliver its specific live-trend angle; do not copy the raw headline): {trend_subject}
CONCRETE SUBJECT TO NAME AND SHOW: {subject_for_visuals} — show this specific person, place, object, or species itself.
The trend angle remains the story; this concrete subject is its on-screen referent, not an excuse to make a generic article summary.
RESEARCH SOURCE SUBJECT (facts must come from this article; the trend angle still determines the story): {source.get('title') or trend_subject}
WHY IT'S TRENDING: {topic.get('why_trending') or ', '.join(topic.get('sources', []))}
{originality_line}ANGLE (this is the story to tell — not the article's summary, not the subject's general
history. If the angle names a specific surprise, that surprise IS the video):
{topic.get('angle') or 'most surprising educational angle'}
{feedback_line}{prior_line}FORMAT: {fmt} — {FORMAT_GUIDE[fmt]}
{hook_line}
{structure_line}{comment_line}{turn_line}LENGTH: {words_lo}-{words_hi} words total narration ({lo}-{hi} seconds).
CHANNEL: {self.cfg['channel']['name']}
{persona_line}
SOURCE TEXT (Wikipedia: "{source['title']}") — the ONLY allowed source of facts:
\"\"\"{source['text'][:self.src_chars]}\"\"\"

{HOOK_RULES.format(hook_min=ccfg.get('hook_words_min', 6), hook_max=ccfg.get('hook_words_max', 9))}
{retention_guidance(self.cfg)}
Write the script as exactly 4 or 5 segments. Each beat has a DIFFERENT job — do not write interchangeable body lines:
  - segment 1 = HOOK (see rules), spoken in under 3 seconds;
  - segment 2 = THE TURN, {body_min}-{body_max} factual words. This single line decides whether the video is watched, and it is the
    one most often written wrong. It must make the hook BIGGER — a second surprise, or the consequence of the
    first ("which meant...", "except..."). It must NOT be setup, background, a definition, a birth date, a
    founding year, an origin story or any sentence beginning "In 1923..." / "Born in..." / "X is a Y that...".
    The viewer already decided the premise was interesting; explaining it to them is why they leave;
  - segments 3..N-1 = ESCALATION, each {body_min}-{body_max} factual words, one concrete new fact each, every line raising the stakes
    above the line before it. Use contrast and consequence ("so", "which meant", "except"). No filler, no repetition;
  - AT LEAST ONE segment's "visual" must show the SUBJECT ITSELF, named explicitly — the object,
    creature, person, document or place the video is about. Ambient scenery ("waves on a beach",
    "a sunset", "a forest") is backdrop, not evidence: a viewer who is promised a nightmare and
    shown a sunset leaves. If the subject genuinely cannot be pictured, this is the wrong topic.
  - segment 2 (THE TURN) must NOT resolve or deflate the entire hook. Reward early attention with a new
    source-backed fact, consequence or reversal, but save the final answer/twist for the payoff. Do not bury
    the subject under background, and do not use a vague tease that only makes sense after the title.
  - order the body so each line is more surprising than the last. The best fact goes LAST, never mid-script.
  - ONE STORY, not a list of facts about a subject. Each line must follow from the one before it - because of
    it, despite it, or revealing what it really meant. A viewer should never be able to reorder your lines
    without the script breaking. If the only thing your lines share is the topic, you have written an
    encyclopedia entry and people leave at line two.
    GOOD (each line caused by the last): a prince is dying of a mystery illness / his pulse betrays him when
    one person enters / she is his stepmother / his father divorces her to save him / they went on to have
    five children.
    BAD (same subject, no chain): Egypt built pyramids without modern tools / they used the Nile's flooding /
    they signed the first peace treaty.
  - last segment = PAYOFF, {payoff_min}-{payoff_max} factual words: {loop_rule}
Create ONE concrete curiosity gap from the source-backed hook and close it in the payoff; a spoken question is
optional. Segment 2 must add new value, not repeat the hook or explain basic background. Do not withhold the
subject's identity just to manufacture suspense.
The trend subject and angle are discovery labels, not factual sources — do NOT repeat their claims or numbers unless the SOURCE TEXT states them.
{person_rule}{series_rule}{lane_rule}ENTERTAIN: write it like a friend telling the most unbelievable true story they know — conversational, vivid, with
comic timing and personality (reactions, contrast, "and it gets worse"). The facts stay 100% exact; the humour comes
from HOW you tell them and from the asides, never from changing what happened.
TOTAL narration (text + asides) MUST be {words_lo}-{words_hi} words — that is about {per_seg} words per segment,
which is ONE short spoken sentence each, not two. Both ends are rejected: too short AND too long.
The factual "text" in every turn/escalation must independently contain {body_min}-{body_max} words, and the payoff text
{payoff_min}-{payoff_max}; asides do not count toward those beat ranges. Every beat must be a complete, grammatical
sentence. Never end on a fragment such as "All 215 million years ago".
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

        # The live trend remains the required story; this concrete core is used for
        # depiction and opening-name checks so modifiers such as "poison" or
        # "unearthed" do not become impossible image targets.

        def validate(o):
            assert isinstance(o, dict), "script must be a JSON object"
            segs = o.get("segments")
            assert isinstance(segs, list) and 4 <= len(segs) <= 5, "need exactly 4 or 5 segments"
            assert all(isinstance(s, dict) for s in segs), "every segment must be a JSON object"
            assert all(isinstance(s.get("text"), str) and s["text"].strip() for s in segs), "empty segment"
            assert all("aside" not in s or isinstance(s["aside"], str) for s in segs), "aside must be text"
            evidence_min, evidence_max = (5, 9) if self.llm.lite else (8, 15)
            attached_evidence = _attach_missing_evidence(
                segs, str(source.get("text") or ""), evidence_min, evidence_max)
            if attached_evidence:
                log.info("  attached %d source-matched evidence quote(s) to omitted draft fields", attached_evidence)
            assert all(isinstance(s.get("evidence"), str) for s in segs), "every segment must include textual evidence"
            assert isinstance(o.get("title"), str) and o["title"].strip(), "missing title"
            bad = [i + 1 for i, s in enumerate(segs) if not (
                isinstance(s.get("visual"), dict)
                and isinstance(s["visual"].get("shows"), str) and s["visual"]["shows"].strip()
                and isinstance(s["visual"].get("queries"), list) and 2 <= len(s["visual"]["queries"]) <= 3
                and all(isinstance(q, str) and q.strip() for q in s["visual"]["queries"]))]
            assert not bad, f"segments {bad} need a \"visual\": {{\"shows\": ..., \"queries\": [2-3 strings]}}"
            # A closing line that is the hook again is the single most common defect in
            # this channel's published output, and asking the model not to do it has a
            # 0-for-3 record. Reject it where rejection actually costs the model a retry.
            assert names_subject(o, subject_for_visuals), (
                f"no shot shows {subject_for_visuals!r} itself — every visual is ambient scenery. "
                "Name the subject in at least one \"visual\": the object, creature, person or "
                "document the video is about, not the landscape around it")
            if len(segs) >= 3:
                echo = echoes_hook(spoken_text(segs[0]), spoken_text(segs[-1]))
                assert not payoff_is_empty(spoken_text(segs[0]), spoken_text(segs[-1])), (
                    "the last line delivers no information - it trails off instead of paying off. "
                    "End on the strongest FACT the viewer has not heard: a number, a name, an "
                    "outcome. Never 'and that is exactly why we have to ask...'")
                assert echo < 0.6, (
                    f"the last line just restates the hook ({echo:.0%} of its words are from it) — "
                    "end on the strongest fact the viewer has NOT heard yet, something the hook "
                    "completes on replay; never 'and that is why' or 'so yes'")
            _bad_shot = unreadable_shot_request(o)
            assert not _bad_shot, (
                f"a shot asks for a {_bad_shot!r} — nobody reads a chart on a phone in two "
                "seconds. Ask for the THING itself: the creature, the place, the object, the "
                "person. If the fact is a number, show what the number is about")
            if topic.get("angle"):
                assert delivers_angle(o, topic["angle"], subject_for_visuals), (
                    f"this is not the story that trended. The angle was: "
                    f"{topic['angle'][:160]!r} — tell THAT. The article's own "
                    "summary is the least surprising paragraph on the page, and leading with "
                    "it throws away the reason anyone clicked")
            assert narration_names_subject(o, subject_for_visuals), (
                f"the script never says {subject_for_visuals!r} out loud — it hides behind "
                "'some fish' or 'a creature'. Name it: the specific noun is what makes a "
                "fact land instead of sounding like filler")
            assert names_subject_in_opening(o, subject_for_visuals, max_segments=2), (
                f"the opening two beats never name {subject_for_visuals!r}; name the specific subject by beat two "
                "so viewers understand the file before the story escalates")
            _esc = uses_escalation_template(o)
            assert _esc < 2, (
                "stop using the 'Weird: / Weirder: / Completely unhinged:' ladder — it is a "
                "list pretending to be a story, it has appeared in several videos already, "
                "and near-identical structure across uploads is what gets a channel "
                "flagged as repetitious. Make the facts escalate without announcing it")
            _b = banned_opener(spoken_text(segs[0]))
            assert not _b, (
                f"the hook opens with {_b!r}, which is filler every channel uses. Open on the "
                "surprising fact itself - the detail, the number, the contradiction")
            assert not payoff_is_a_question(spoken_text(segs[-1])), (
                "the last line asks a question instead of delivering a payoff. Close the concrete curiosity gap "
                "with a new source-backed fact; do not leave homework for the viewer")
            canned = [a for sg in segs if (a := sg.get("aside")) and aside_is_generic(a)]
            assert not canned, (
                f"aside {canned[0]!r} is canned filler that would fit any video. React to "
                "something specific in that line - the thing, the number, the absurdity it "
                "just described")
            hook_lo = int(ccfg.get("hook_words_min", 6))
            hook_hi = int(ccfg.get("hook_words_max", 9))
            hook_words = len(segs[0]["text"].split())
            count_issues = []
            if not hook_lo <= hook_words <= hook_hi:
                count_issues.append(f"hook is {hook_words} words; it must be {hook_lo}-{hook_hi}")
            count_issues.extend(gates.check_segment_lengths(segs, self.cfg))
            assert not count_issues, "word-count repair required: " + "; ".join(count_issues)
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
        validate(script)  # trust-boundary check even for custom routers that ignore the callback
        script["_writer_model"] = getattr(self.llm, "last_used", "")
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
        def validate_metadata(value):
            schema_issues = metadata_schema_issues(value)
            if schema_issues:
                raise AssertionError("metadata schema invalid: " + "; ".join(schema_issues))

        try:
            m = self.llm.json(WRITER_SYSTEM, user, temperature=0.5, validate=validate_metadata)
            validate_metadata(m)  # adapters that skip validation still cannot inject malformed upload metadata
            return {k: m[k] for k in ("description", "tags", "hashtags", "thumbnail_text")}
        except Exception as e:  # noqa: BLE001 — use grounded deterministic metadata rather than publish bad output
            log.warning("metadata response unavailable or malformed; using source-grounded fallback: %s", str(e)[:200])
            words = script["title"].split()
            return {"description": narration[:300], "tags": [source["title"]], "hashtags": [],
                    "thumbnail_text": " ".join(words[:4])}

    # ── 3. programmatic + LLM review ─────────────────────────────────────────
    def check(self, script: dict, source: dict, topic: dict | None = None) -> tuple[bool, dict]:
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
        ok, review = self._check(script, source, topic)
        if ok:
            if getattr(self, "_history", None) is None:
                self._history = read_json("history.json", [])
            self._history.append({
                "title": script.get("title", ""),
                "sketch": originality.sketch(originality.narration_of(script)),
                "_in_flight": True,
            })
        return ok, review

    def _check(self, script: dict, source: dict, topic: dict | None = None) -> tuple[bool, dict]:
        # numbers + blocked words are checked on EVERYTHING spoken (asides included); evidence only on facts
        narration = " ".join(spoken_text(s) for s in script["segments"])
        issues = []
        trend_angle = str((topic or {}).get("angle") or "").strip()
        if topic is not None and not trend_angle:
            issues.append("no specific current-trend angle was supplied; do not publish a generic subject video")
        # Deterministic gates first: banned openers, CTA closers, clickbait titles, stage directions,
        # emoji/hashtags/URLs in speech, verbatim repeats. These are free, reliable and run before a
        # single review token is spent — see autotube/gates.py for why they don't belong to the LLM.
        issues += gates.run_all(script, spoken_text, self.cfg)
        # Similarity and narrator-voice checks need recent channel history; they are internal safeguards
        # against low-originality output, not published numerical monetization-policy thresholds.
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
        require_grounding = self.cfg["content"].get("require_grounding", True)
        if type(require_grounding) is not bool:
            issues.append("require_grounding configuration must be a Boolean; not publishing")
            require_grounding = True
        if require_grounding and grounded < need:
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
        for i, s in enumerate(script["segments"]):
            beat = ("HOOK" if i == 0 else "TURN" if i == 1
                    else "PAYOFF" if i == len(script["segments"]) - 1 else "ESCALATION")
            rows.append(f"- SEGMENT {i + 1} ({beat}): {s['text']}")
            if s.get("aside"):
                rows.append(f"    [ASIDE - joke/opinion, not a factual claim] {s['aside']}")
        narration_list = "\n".join(rows)
        trend_context = ""
        if topic is not None:
            signals = []
            for signal in (topic.get("signals") or [])[:4]:
                if not isinstance(signal, dict):
                    continue
                detail = signal.get("evidence") or {}
                source_name = str(signal.get("source") or "")
                context_text = "; ".join(str(x) for x in (signal.get("context") or [])[:2] if x)
                measured = ", ".join(f"{k}={v}" for k, v in detail.items()
                                     if isinstance(v, (str, int, float)) and k not in {"url", "video_url"})
                signals.append(" | ".join(x for x in (source_name, context_text, measured) if x))
            evidence_text = "\n".join(f"- {x[:280]}" for x in signals) or "; ".join(
                str(x) for x in (topic.get("context") or [])[:3] if x)
            approved_lane = lane_of(self.cfg, str(topic.get("category") or ""))
            lane_remit = {
                "history_mystery": "historical evidence, unresolved past events, archaeology, strange documented history",
                "science_nature": "science, space, animals, natural phenomena, geography",
            }.get(approved_lane, "no approved lane matched this category")
            trend_context = f"""APPROVED EDITORIAL LANE: {approved_lane or 'invalid'} — {lane_remit}.
This is a hard channel-scope boundary. Sports rules/results, gaming, current news/politics, celebrity gossip and unrelated
entertainment are not in scope; only a clear, source-backed history/mystery or science/nature subject fits. If this script
falls outside the approved lane, put a concrete scope correction in fixes so it cannot be approved.
CURRENT TREND CONTEXT (editorial alignment check; NOT factual source material):
Trend headline: {str(topic.get('topic') or '')[:200]}
Selected specific angle: {trend_angle[:400]}
Why this topic is rising: {str(topic.get('why_trending') or '')[:300]}
Observed live evidence:
{evidence_text or '- No detail provided'}
The script must deliver the selected angle, not just repeat the trend headline or discuss the broad subject.
\n"""
        trend_schema = ', "trend_alignment": 0-10, "trend_alignment_reason": "brief evidence-based reason"' if topic is not None else ""
        turn_variant = (topic or {}).get("_turn_variant") if isinstance(topic, dict) else None
        turn_rule = TURN_VARIANT_GUIDE.get(turn_variant)
        turn_context = (f"ASSIGNED RANDOMIZED BEAT-TWO TREATMENT ({turn_variant}): {turn_rule} "
                        "This is a structure test across distinct topics; preserve the script's originality and factual standards. "
                        "The turn score requested below is telemetry only: it must not alter approval, the overall score, or fixes.\n\n"
                        if turn_rule else "")
        content_cfg = self.cfg.get("content", {})
        compliance_cfg = self.cfg.get("compliance", {})
        if not isinstance(content_cfg, dict) or not isinstance(compliance_cfg, dict):
            review["issues"].append("review score configuration is malformed; not publishing")
            return False, review
        require_independent_review = compliance_cfg.get("require_independent_review", False)
        if type(require_independent_review) is not bool:
            review["issues"].append("independent-review requirement must be a Boolean; not publishing")
            return False, review
        thresholds = {
            "min_hook": content_cfg.get("min_hook_score", 7),
            "min_ent": content_cfg.get("min_entertainment", 6),
            "min_coh": content_cfg.get("min_coherence", 6),
            "min_alignment": content_cfg.get("min_trend_alignment", 8),
            "min_review": compliance_cfg.get("quality_gate_min_score", 7),
        }
        if any(not _finite_in_range(value, 0, 10) for value in thresholds.values()):
            review["issues"].append("review score thresholds must be finite numbers from 0 to 10; not publishing")
            return False, review
        min_hook = float(thresholds["min_hook"])
        min_ent = float(thresholds["min_ent"])
        min_coh = float(thresholds["min_coh"])
        min_trend_alignment = float(thresholds["min_alignment"])
        min_review_score = float(thresholds["min_review"])
        trend_rubric = (f"\ntrend_alignment — score whether the script tells the CURRENT TREND CONTEXT's specific selected angle, not merely its broad topic. "
                        f"10 = clearly delivers the distinctive viral/trending story; 8-9 = direct, strong coverage; 6-7 = partial or diluted; 0-5 = generic, tangential, or absent. "
                        f"Need at least {min_trend_alignment:g}/10. Explain the score briefly in trend_alignment_reason. The angle is not proof of facts; SOURCE TEXT remains the only fact source."
                        if topic is not None else "")
        require_loop = self.cfg.get("content", {}).get("require_loop", False)
        if type(require_loop) is not bool:
            review["issues"].append("loop requirement configuration is malformed; not publishing")
            return False, review
        loop_instruction = ("A loop is required: if loops=false, include a concrete reworded last line in fixes."
                           if require_loop else
                           "A loop is optional: record loops accurately, but do not put a missing loop in fixes by itself.")
        user = f"""{trend_context}{turn_context}SOURCE TEXT:\n\"\"\"{source['text'][:self.src_chars]}\"\"\"\n
SCRIPT TITLE: {script['title']}
SCRIPT DESCRIPTION: {script.get('description', '')}
SCRIPT NARRATION (lines marked ASIDE are the narrator's jokes, spoken right after the line above them):
{narration_list}

Evaluate. Verify that the script offers a specific viewer takeaway and an original treatment in its own words, not a bare restatement of the headline or a close copy of source phrasing. If you cannot identify a concrete takeaway, leave value_add empty; if originality is doubtful, say so in policy_concerns or fixes.
Return JSON: {{"factual_errors": ["..."], "misleading_title": true|false, "advertiser_friendly": true|false,
"policy_concerns": ["..."], "value_add": "<specific takeaway the viewer learns>", "hook_strength": 0-10, "entertainment": 0-10,
"score": 0-10, "loops": true|false, "coherence": 0-10{trend_schema},
"turn_strength": 0-10 or null, "turn_strength_reason": "one concise explanation or empty string", "fixes": ["..."]}}
Score 9-10 = accurate, engaging, clearly valuable; 7-8 = good; below {min_review_score:g} = do not publish.
turn_strength is optional TELEMETRY ONLY: score segment 2 (TURN) for immediacy, source-backed escalation and whether it follows its assigned treatment. 0-3 = setup/repetition/delayed, 4-6 = some progress but soft or padded, 7-8 = immediate and clearly escalates, 9-10 = unusually sharp and fully delivered. It must not change approval, the overall score, or "fixes"; do not add a fix solely for this diagnostic. Return null/empty if it cannot be judged. Missing or malformed turn fields are ignored.
"loops": does the LAST line flow naturally back into the FIRST when the Short replays? Judge it by meaning,
not by shared words - "so the ice keeps bleeding" loops cleanly into "a waterfall that runs blood red".
A natural loop is a creative choice, not a guaranteed increase in views or averageViewPercentage; do not cite a metric
benefit. {loop_instruction}
coherence — is this ONE story or a list of facts that happen to share a subject? Ask whether the lines could be
reordered without the script breaking: if they could, it is a list. 8-10 = each line follows from the one before
(a prince is dying / his pulse betrays him / she is his stepmother / his father divorces her). 4-6 = loosely
themed, some connective tissue. 0-3 = unrelated facts in a row (Egypt built pyramids / used Nile flooding /
signed a peace treaty). A list of true facts is accurate and still unwatchable, so score it on structure, not
truth, and put the re-ordering or the missing causal link in "fixes". Need at least {min_coh:g}/10.
Read EVERY sentence literally, word by word: if its literal meaning is false or garbled (e.g. the wrong subject doing
the action — "rivers carved warnings" when people carved them; a date or place attached to the wrong thing), list
it in factual_errors even if the gist is right.
ASIDES are humour, not claims: never list an aside in factual_errors for being a joke, exaggerated opinion or
sarcasm. Put an aside in policy_concerns ONLY if it states a new specific fact as true, mocks real people, groups,
victims or a tragedy, or is not family-friendly. A joke must never change what the facts say.
entertainment — BE HARSH, this channel competes with comedians: a list of accurate facts read in a neutral tone is 5
at most, however interesting. 7 = has a story shape (setup, escalation, payoff) and personality. 8 = also at least one
genuinely funny or gasp-out-loud moment. 9-10 = you would send it to a friend. If entertainment < {min_ent:g}, put a concrete TRUE rewrite
idea in "fixes" (story tension, comic contrast, a sharper aside, a stronger reveal).
hook_strength: 9-10 = the first line is concise, specific and surprising, and by beat two the viewer can name the
subject without relying on the title; 7-8 = decent; <=6 = vague, overlong, generic, or slow setup. A literal question
is optional; clarity and an immediate source-backed reason to keep watching are not.
If hook_strength < {min_hook:g}, put a stronger TRUE first line in "fixes".
Any non-empty "fixes" list means the draft still needs revision; return [] only when it is ready to publish.
Be internally consistent: if a fix says the last line must be rewritten to loop, "loops" must be false.""" + trend_rubric
        def validate_review(o):
            schema_issues = reviewer_schema_issues(o, has_trend=topic is not None)
            if schema_issues:
                raise AssertionError("reviewer schema invalid: " + "; ".join(schema_issues))

        try:
            # A writer grading its own homework agrees with itself. Review runs on a different
            # provider/model wherever one is reachable (llm.review_providers), and never on the exact
            # model that produced this draft.
            review_fn = getattr(self.llm, "review_json", None)
            if review_fn is not None:
                r = review_fn(REVIEW_SYSTEM, user, avoid=script.get("_writer_model"),
                              temperature=0.2, validate=validate_review)
            else:                                     # minimal LLM-like object (tests, custom routers)
                r = self.llm.json(REVIEW_SYSTEM, user, temperature=0.2, validate=validate_review)
            reviewer_id = getattr(self.llm, "last_used", "")
            reviewer_id = reviewer_id if isinstance(reviewer_id, str) else ""
            review["reviewer"] = reviewer_id
            writer_model = script.get("_writer_model")
            review["independent"] = (isinstance(writer_model, str) and bool(writer_model)
                                      and bool(reviewer_id) and reviewer_id != writer_model)
            if review["reviewer"] and not review["independent"]:
                reason = "same model" if writer_model else "writer identity unavailable"
                log.info("  review is not known to be independent (%s; reviewer=%s)", reason, review["reviewer"])
            if require_independent_review and not review["independent"]:
                review["issues"].append("no independent reviewer identity available — not publishing")
                review["unavailable"] = True
                return False, review
        except Exception as e:  # noqa: BLE001 — any provider/adapter failure blocks publication
            log.warning("review LLM unavailable or invalid: %s", str(e)[:300])
            # Fail closed unconditionally. A config switch must not turn a missing or
            # malformed compliance review into an approval.
            review["unavailable"] = True
            review["issues"].append("fact-check unavailable — not publishing an unchecked script")
            return False, review
        schema_issues = reviewer_schema_issues(r, has_trend=topic is not None)
        if schema_issues:
            log.warning("reviewer returned malformed output: %s", "; ".join(schema_issues)[:500])
            review["unavailable"] = True
            review["review_schema_issues"] = schema_issues
            if isinstance(r, dict):
                review["fixes"] = actionable_review_fixes(r.get("fixes"))
            else:
                review["fixes"] = actionable_review_fixes(None)
            review["issues"].append("reviewer output was incomplete or malformed — not publishing")
            review["issues"].extend("reviewer schema: " + issue for issue in schema_issues)
            return False, review
        score = float(r["score"])
        hook = float(r["hook_strength"])
        if hook < min_hook:
            review["issues"].append(f"hook too weak ({hook:.0f}/10, need {min_hook:.0f}) — open with the single most "
                                    "surprising specific fact; no 'Did you know' or slow setup")
        ent = float(r["entertainment"])
        if ent < min_ent:
            review["issues"].append(f"not entertaining enough ({ent:.0f}/10, need {min_ent:.0f}) — tell it as a story "
                                    "with tension and a payoff, add comic contrast or a sharper aside; facts unchanged")
        if score < min_review_score:
            review["issues"].append(
                f"overall review score too low ({score:.0f}/10, need {min_review_score:.0f}) — address the review's core concerns")
        alignment_ok = True
        alignment = None
        alignment_reason = ""
        if topic is not None:
            alignment = float(r["trend_alignment"])
            alignment_reason = r["trend_alignment_reason"].strip()[:300]
            alignment_ok = math.isfinite(alignment) and min_trend_alignment <= alignment <= 10
            review["trend_alignment"] = alignment
            review["trend_alignment_reason"] = alignment_reason
            if not alignment_ok:
                review["issues"].append(
                    f"script does not deliver the specific current-trend angle ({alignment:.0f}/10, need {min_trend_alignment:.0f})"
                    + (f": {alignment_reason}" if alignment_reason else ""))
        # The loop is judged semantically by the reviewer, because a lexical check rejects good loops
        # (see the note in gates.py). Enforced only when require_loop is on, and it defaults off: it
        # is a new criterion, and turning it into a hard gate before we know its false-positive rate
        # would start abandoning topics for a reason nobody is watching yet. The verdict is recorded
        # either way, so the rate can be read off history before anyone flips the switch.
        # Coherence is a real approval criterion, not telemetry only. A missing or malformed reviewer
        # score fails closed, and the configured min_coherence threshold is checked again in `ok` below.
        coh = float(r["coherence"])
        review["coherence"] = coh
        if coh < min_coh:
            review["issues"].append(
                f"this is a list of facts, not a story ({coh:.0f}/10, need {min_coh:.0f}) — "
                "make each line follow from the one before it: because of it, despite it, or "
                "revealing what it really meant")
        loops = r.get("loops")
        review["loops"] = loops
        if loops is False and self.cfg["content"].get("require_loop", False):
            review["issues"].append("the last line does not flow back into the first — re-word the payoff to pick up the hook; "
                                    "this is a stylistic criterion, not a guaranteed analytics gain")

        fixes = actionable_review_fixes(r["fixes"])
        fix_issue = ("independent reviewer still requests a rewrite: " + "; ".join(fixes)[:500]
                     if fixes else "")
        if fix_issue:
            review["issues"].append(fix_issue)
        llm_issues = r.get("issues", [])
        review_issues = list(review.get("issues") or []) + [x.strip() for x in llm_issues if x.strip()]
        ok = (ent >= min_ent and hook >= min_hook and coh >= min_coh and alignment_ok
              and score >= min_review_score and not r["factual_errors"]
              and not r["misleading_title"] and r["advertiser_friendly"]
              and not r["policy_concerns"] and not fixes and not review_issues)
        # Preserve only reviewer decisions we explicitly validate. Extra provider keys must not
        # overwrite orchestration metadata such as reviewer identity, independence, or availability.
        for key in ("score", "hook_strength", "entertainment", "factual_errors", "misleading_title",
                    "advertiser_friendly", "policy_concerns", "value_add", "loops"):
            review[key] = r[key]
        # Optional diagnostic only: deliberately normalized after `ok` is decided, so a missing,
        # string-valued, NaN, or out-of-range turn score cannot become a new publication veto.
        raw_turn_strength = r.get("turn_strength")
        if _valid_review_score(raw_turn_strength):
            review["turn_strength"] = float(raw_turn_strength)
            review["turn_strength_status"] = "scored"
        else:
            review["turn_strength"] = None
            review["turn_strength_status"] = (
                "missing" if "turn_strength" not in r or raw_turn_strength is None else "malformed")
        raw_turn_reason = r.get("turn_strength_reason")
        review["turn_strength_reason"] = raw_turn_reason.strip()[:300] if isinstance(raw_turn_reason, str) else ""
        review["turn_strength_reason_available"] = bool(review["turn_strength_reason"])
        review["issues"] = review_issues
        review["fixes"] = fixes
        review["coherence"] = coh
        if topic is not None:
            review["trend_alignment"] = alignment
            review["trend_alignment_reason"] = alignment_reason
        return ok, review

    def produce(self, topic: dict, plan: dict, max_attempts: int | None = None) -> tuple[dict, dict, dict] | None:
        """Ground → write → check with a hard-capped script-only rewrite budget."""
        receipt = run_report.current_receipt()
        if receipt:
            receipt.set_stage("script_research")
        if len(_angle_terms(str(topic.get("angle") or ""), str(topic.get("topic") or ""))) < 2:
            if receipt:
                receipt.note_failure("script_research", "trend_angle_insufficient")
            log.info("skip %r: no specific current-trend angle to deliver", topic.get("topic"))
            return None
        source = ground(topic["topic"], topic.get("wiki_query"), self.lang,
                        allow_living=self.cfg["content"]["allow_living_people"],
                        blocked=self.cfg["compliance"].get("blocked_topics"),
                        cfg=self.cfg, llm=self.llm)
        if not source:
            if receipt:
                receipt.note_failure("script_research", "source_grounding_failed")
            return None
        topic["wiki_title"] = source["title"]
        visual_subject = visual_subject_for_topic(topic, source)
        if not source_supports_angle(str(topic.get("angle") or ""), visual_subject,
                                     str(source.get("text") or "")):
            if receipt:
                receipt.note_failure("script_research", "trend_angle_unsupported_by_source")
            log.info("  ✗ grounded research does not substantiate the current-trend angle for %r — skipping before draft",
                     source["title"])
            return None
        if not subject_is_depictable(visual_subject, self.cfg, source):
            if receipt:
                receipt.note_failure("script_research", "no_depictable_source_asset")
            log.info("  ✗ no free image of trend subject %r exists — skipping before writing", visual_subject)
            return None
        configured = RecoveryBudgets.from_config(self.cfg).script_drafts
        max_attempts = configured if max_attempts is None else bounded_attempts(
            max_attempts, configured, MAX_SCRIPT_DRAFTS, 1)
        feedback = ""
        sticky_review_feedback = ""
        previous_draft = None
        for attempt in range(max_attempts):
            if receipt:
                receipt.set_stage("script_draft")
            try:
                t = dict(topic)
                t["_rewrite_attempt"] = attempt + 1
                if feedback:
                    t["_draft_feedback"] = feedback
                if previous_draft is not None:
                    t["_prior_draft"] = previous_draft
                script = self.write(t, plan, source)
            except LLMError as e:
                if receipt:
                    receipt.note_failure("script_draft", run_report.classify_failure("script", str(e)),
                                         exception_type=type(e).__name__, attempt=attempt + 1)
                log.warning("script generation failed: %s", e)
                return None
            review_topic = dict(topic)
            review_topic["_turn_variant"] = plan.get("turn_variant")
            review_topic["_turn_experiment_id"] = plan.get("turn_experiment_id")
            if receipt:
                receipt.set_stage("script_review")
            ok, review = self.check(script, source, review_topic)
            log.info("  draft %d for %r → ok=%s score=%s issues=%s", attempt + 1, source["title"], ok,
                     review.get("score"), (review.get("issues") or review.get("factual_errors") or [])[:2])
            if ok:
                return script, source, review
            if review.get("unavailable"):
                if receipt:
                    receipt.note_failure("script_review", "independent_review_unavailable", attempt=attempt + 1)
                return None

            items = []
            for key in ("issues", "factual_errors", "policy_concerns", "fixes"):
                value = review.get(key) or []
                if isinstance(value, str):
                    value = [value]
                if isinstance(value, (list, tuple)):
                    for item in value:
                        text = str(item or "").strip()
                        if text and text not in items:
                            items.append(text)
            current_feedback = "; ".join(items)[:900]
            # A programmatic-only failure (for example, a beat word-count miss) has no
            # reviewer to reconsider the earlier quality diagnosis. Keep that diagnosis
            # attached until a later independent review either clears it or replaces it.
            if review.get("programmatic_ok") is True:
                sticky_review_feedback = current_feedback
            feedback_parts = []
            for item in (sticky_review_feedback, current_feedback):
                if item and item not in feedback_parts:
                    feedback_parts.append(item)
            feedback = "; ".join(feedback_parts)[:1000]
            previous_draft = script
            if attempt + 1 < max_attempts and receipt:
                receipt.note_retry("script_review", attempt + 1, max_attempts, "rewrite_with_review_feedback")
            elif receipt:
                receipt.note_failure("script_review", "script_quality_rejected", attempt=attempt + 1)
        return None
