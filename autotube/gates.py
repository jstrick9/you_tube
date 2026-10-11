"""Deterministic script gates — the rules a regex can enforce, enforced by a regex.

Why this module exists: the LLM reviewer grades generously and inconsistently. The sample in
docs/samples/ opens with "Did you know whooping cranes once vanished to just 21 birds?" and closes
with "Follow Curious Minute for more wildlife wonders." That is the first banned opener in
HOOK_RULES and a hard CTA that breaks the loop ending — and the reviewer scored it 9/10 with
hook_strength 9.

Raising the LLM thresholds cannot fix that; it is a miscalibrated instrument, not a strict one. So
anything checkable is checked here, deterministically, for free, before a single review token is
spent. The reviewer is left to judge the things only a reader can judge: accuracy, nuance, humour.

Every function returns a list of plain-language issue strings. They flow into the existing rewrite
loop as feedback, so a failure tells the writer exactly what to change.
"""
from __future__ import annotations

import re

# ── openers that signal low-effort content and waste the only 3 seconds that matter ──
BANNED_OPENERS = [
    r"did you know", r"have you ever (wondered|thought|heard)", r"ever wondered",
    r"here (are|is)\b", r"in this (video|short)", r"today (we|i)\b", r"let'?s (talk|dive|explore|discuss)",
    r"welcome (back|to)\b", r"hey (guys|everyone|there|folks)", r"hello (everyone|there|guys|friends)",
    r"what'?s up\b", r"imagine if i told you", r"you won'?t believe",
    r"this (is|might be) (the )?(most|craziest|wildest)", r"buckle up", r"get ready",
    r"i'?m about to", r"we'?re going to (look|talk|explore)", r"stick around",
]

# ── closers that kill the loop: a Short that ends on a CTA cannot replay seamlessly ──
BANNED_CLOSERS = [
    # "follow", optionally with a channel name in between, then for/to/and — catches
    # "follow for more", "follow us", "follow Curious Minute for more wildlife wonders"
    r"\bfollow\b(?:[\w' ]{0,30}?\b(for|to|and)\b|\s*[.!]|\s+(me|us|along)\b)",
    r"subscribe", r"\blike and\b", r"hit the (like|bell|follow)", r"smash that",
    r"comment (below|down)", r"let me know (in the comments|below)", r"link in (bio|the description)",
    r"thanks for watching", r"see you (next|in the next|soon)", r"stay tuned",
    r"check out (my|our|the) (other|next)", r"drop a (like|comment)", r"share this (with|video)",
    r"(more|other) (videos|facts|shorts) like this", r"for more \w+ (facts|stories|wonders|insights|content)",
]

# ── clickbait the video cannot deliver on (title-level; YouTube's misleading-metadata surface) ──
BANNED_TITLE = [
    r"you won'?t believe", r"will shock you", r"shocking truth", r"doctors hate",
    r"gone wrong", r"\bmust see\b", r"\bgone sexual\b", r"number \d+ will", r"this one (trick|weird)",
    r"!!!", r"\?\?\?",
]

# ── things that leak the production process into the narration ──
# ── the turn (segment 2) ────────────────────────────────────────────────────────────────────────
# Averaged over the channel's first 12 videos with retention curves, audience retention runs
# 112% at the 10% mark and 71.5% at the 20% mark: a 41-point cliff between roughly second 3 and
# second 5. The hook is not the problem - it is over by then. What loses them is beat two, which
# the model reliably writes as background exposition: a birth date, a definition, a founding year.
# The viewer came for the surprise promised in the hook and got a history lesson instead. These
# are the openers that signal exposition rather than escalation, anchored at the start of the line.
EXPOSITION_OPENERS = [
    r"born in\b", r"it was born", r"which (?:is|was)(?: actually)?\b", r"the (story|history|origins?) of\b",
    r"in (the )?(17|18|19|20)\d\d\b", r"back in\b", r"(it|he|she|they) (was|were) (first )?(built|founded|created|established|discovered|invented|published|opened)\b",
    r"(was|is) (a|an|the) [a-z]+ (who|that|which)\b",
    r"(located|situated|found) in\b", r"((it|this|that|they|he|she) )?dates? back to\b",
    r"for (centuries|decades|years)\b",
    r"throughout history\b", r"(it|this|he|she|they) (is|was) (also )?known as\b",
    r"according to\b", r"to understand (this|it|that|why)\b", r"(first|originally) (built|made|created|used|appeared)\b",
    r"the (idea|concept|practice|tradition) of\b", r"historians\b", r"scientists have long\b",
]


def check_turn(second: str) -> list[str]:
    """Beat two must escalate, not explain. This is where the channel loses 41% of its audience."""
    second = (second or "").strip()
    if not second:
        return []
    low = second.lower().lstrip("\"'(- ")
    for pat in EXPOSITION_OPENERS:
        if re.match(pat, low):
            return [f"segment 2 opens with background exposition ({re.match(pat, low).group(0)!r}). This is "
                    "where most viewers leave. It must ESCALATE the hook - a second, bigger surprise, or the "
                    "consequence of the first - never a definition, a date, an origin or a history lesson"]
    return []


STAGE_DIRECTION = re.compile(
    r"(^|\s)(\[|\(\s*(cut|pause|beat|music|sfx|narrator|voice ?over|vo)\b)|^\s*(narrator|host|vo)\s*:",
    re.I)
EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
HASHTAG = re.compile(r"(^|\s)#\w+")
URL = re.compile(r"https?://|\bwww\.")


def _hits(patterns: list[str], text: str) -> list[str]:
    """Matched patterns, reported as the words actually found so feedback reads like English."""
    low = text.lower()
    out = []
    for p in patterns:
        m = re.search(p, low)
        if m:
            out.append(m.group(0).strip())
    return out


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def check_hook(hook: str, lo: int = 6, hi: int = 12) -> list[str]:
    """A specific opening can help earn attention; no fixed viewer-retention cutoff is assumed."""
    issues = []
    hook = (hook or "").strip()
    if not hook:
        return ["the hook is empty"]
    if _hits(BANNED_OPENERS, hook):
        issues.append(
            f"the hook opens with a banned phrase ({_hits(BANNED_OPENERS, hook)[0]!r}). Cut the preamble and open "
            "directly on the most surprising specific fact — a number, a contradiction, an impossible-sounding claim")
    n = len(_words(hook))
    if n > hi:
        issues.append(f"the hook is {n} words, must be {lo}-{hi} — cut it to the single surprising detail")
    elif n < lo:
        issues.append(f"the hook is only {n} word{'s' if n != 1 else ''}, must be {lo}-{hi} — "
                      "make it a complete, specific claim")
    if STAGE_DIRECTION.search(hook):
        issues.append("the hook contains a stage direction; write only what the narrator says")
    return issues


def check_closer(last: str, loop: bool = True) -> list[str]:
    """On Shorts a replay counts as watch time, so the last line must flow back into the first, not sign off."""
    issues = []
    hit = _hits(BANNED_CLOSERS, last or "")
    if hit:
        issues.append(
            f"the final line is a call-to-action ({hit[0]!r}). " +
            ("End on the fact that loops straight back into the first line — no sign-off, no 'follow for more'"
             if loop else "End on the payoff fact, not a plug"))
    return issues


# A deterministic loop check was tried here and removed. Lexical overlap between the payoff and the
# hook looked like a cheap proxy for "does this Short loop", but it rejects good loops: for
# "Antarctica has a waterfall that runs blood red" / "so the ice keeps bleeding, and nobody can stop
# it" there is no shared token, yet the loop is clean - "bleeding" answers "blood red" semantically.
# No stemming or stop-word tuning fixes that class of miss, and a false positive here burns a
# generation retry and can abandon the topic. Whether a payoff loops is a semantic judgement, so it
# belongs to the LLM reviewer's rubric (see scriptwriter REVIEW_RUBRIC), not to a regex.


def check_title(title: str, max_len: int = 100) -> list[str]:
    issues = []
    title = (title or "").strip()
    if not title:
        return ["missing title"]
    hit = _hits(BANNED_TITLE, title)
    if hit:
        issues.append(f"the title uses clickbait the video cannot deliver ({hit[0]!r}) — state the real hook instead")
    bare = re.sub(r"#\w+", "", title).strip()
    if len(bare) > max_len:
        issues.append(f"the title is {len(bare)} characters, keep it under {max_len}")
    if bare.isupper() and len(bare) > 12:
        issues.append("the title is in all caps")
    if EMOJI.search(title):
        issues.append("the title contains emoji")
    return issues


def check_narration(segments: list[dict], spoken) -> list[str]:
    """Rules that apply to every spoken line, not just the first and last."""
    issues = []
    for i, seg in enumerate(segments):
        line = spoken(seg)
        if STAGE_DIRECTION.search(line):
            issues.append(f"line {i + 1} contains a stage direction or speaker label; write only spoken words")
        if EMOJI.search(line):
            issues.append(f"line {i + 1} contains emoji; this text is read aloud")
        if HASHTAG.search(line):
            issues.append(f"line {i + 1} contains a hashtag; hashtags belong in the description")
        if URL.search(line):
            issues.append(f"line {i + 1} contains a URL; it cannot be read aloud")
    # an identical sentence twice is the clearest "mass-produced" tell there is
    seen: dict[str, int] = {}
    for i, seg in enumerate(segments):
        key = re.sub(r"[^a-z0-9 ]", "", spoken(seg).lower()).strip()
        if key and key in seen:
            issues.append(f"line {i + 1} repeats line {seen[key] + 1} verbatim")
        seen[key] = i
    return issues


def _word_bounds(ccfg: dict, key: str, default: tuple[int, int]) -> tuple[int, int]:
    value = ccfg.get(key, default)
    try:
        lo, hi = int(value[0]), int(value[1])
        return (lo, hi) if 0 <= lo <= hi else default
    except (TypeError, ValueError, IndexError, KeyError):
        return default


def check_segment_lengths(segs: list[dict], cfg: dict) -> list[str]:
    """Enforce the beat lengths specified in the writer prompt, counting factual text only.

    Asides are optional spoken jokes, not a substitute for the story beat. The global TTS word
    budget separately counts both text and asides.
    """
    if len(segs) < 4:
        return []
    ccfg = (cfg.get("content", {}) or {})
    body_lo, body_hi = _word_bounds(ccfg, "body_beat_words", (9, 11))
    payoff_lo, payoff_hi = _word_bounds(ccfg, "payoff_words", (6, 9))
    issues = []
    for i, seg in enumerate(segs[1:-1], start=1):
        text = str(seg.get("text") or "") if isinstance(seg, dict) else ""
        count = len(_words(text))
        role = "turn" if i == 1 else "escalation"
        if not body_lo <= count <= body_hi:
            issues.append(f"segment {i + 1} {role} is {count} words; it must be {body_lo}-{body_hi} factual words. "
                          "Rewrite the beat as one complete, specific sentence; do not pad it with an aside.")
    last_text = str((segs[-1] or {}).get("text") or "") if isinstance(segs[-1], dict) else ""
    last_count = len(_words(last_text))
    if not payoff_lo <= last_count <= payoff_hi:
        issues.append(f"final payoff is {last_count} words; it must be {payoff_lo}-{payoff_hi} factual words. "
                      "End on a complete, concrete answer—not a fragment or a date dangling by itself.")
    return issues


def run_all(script: dict, spoken, cfg: dict) -> list[str]:
    """Every deterministic gate, in one call. Empty list = passed."""
    ccfg = cfg.get("content", {}) or {}
    segs = script.get("segments") or []
    if not segs:
        return ["script has no segments"]
    lo = int(ccfg.get("hook_words_min", 6))
    hi = int(ccfg.get("hook_words_max", 12))
    issues = check_hook(spoken(segs[0]), lo, hi)
    if len(segs) > 2:
        issues += check_turn(spoken(segs[1]))
    issues += check_segment_lengths(segs, cfg)
    issues += check_closer(spoken(segs[-1]), bool(ccfg.get("loop_ending", True)))
    issues += check_title(script.get("title", ""))
    issues += check_narration(segs, spoken)
    return issues
