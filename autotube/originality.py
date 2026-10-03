"""Defences against the two enforcement triggers this channel is actually exposed to.

The 2026 policy language is specific about what gets a faceless channel demonetized,
and the named pattern is "generic TTS narration over stock footage with AI-written
scripts, no commentary or insight". Two of those five clauses are permitted outright:
synthetic narration over stock and AI-written scripts are explicitly allowed, and a
consistent recurring narrator is an asset rather than a liability. The clauses that
actually bite are the last one - no commentary or insight - and the repetition rule
underneath it: five or more videos sharing a template with under twenty percent script
variation is treated as bulk-produced and demonetized as a batch.

So this module measures the two things that distinguish a show from a content farm,
and it measures them per draft, before publication, because enforcement is channel-wide
and retroactive. By the time a strike lands, the back catalogue is already the evidence.

The governing principle in the policy is that format may repeat but substance may not.
A fixed series structure is fine - that is what a programme is. What cannot repeat is
the actual language and the actual claims. That distinction is exactly what the
similarity check below is shaped to enforce: it compares what was *said*, not how the
episode was *built*, so tightening the format costs nothing while recycling a script
is caught.
"""
from __future__ import annotations

import hashlib
import re

# Words too common to carry evidence of copying. Two scripts both containing "the" and
# "was" are not related; two both containing "lighthouse" and "vanished" probably are.
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "there", "here", "then", "than", "so", "not", "no",
    "you", "your", "they", "them", "their", "he", "she", "his", "her", "we", "our", "i",
    "has", "have", "had", "do", "does", "did", "can", "could", "would", "should", "will",
    "just", "about", "into", "over", "out", "up", "down", "more", "most", "one", "two",
}

SKETCH_SIZE = 48          # bottom-k sketch width; see sketch() for why this is bounded


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w not in _STOP]


def shingles(text: str, n: int = 3) -> set[str]:
    """Overlapping n-word phrases.

    Single words are too weak a signal - any two space videos share "orbit" and
    "pressure" without either being a rewrite of the other. Phrases of three catch
    reused *sentence construction*, which is what low script variation actually looks
    like when a model is re-prompted on a similar topic.
    """
    ws = tokens(text)
    if len(ws) < n:
        return {" ".join(ws)} if ws else set()
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def sketch(text: str, k: int = SKETCH_SIZE) -> list[str]:
    """A bottom-k hash sketch of the script's phrases.

    History is read and rewritten on every run and gets copied into diagnostics, so
    storing every shingle of every past script would grow it without bound for a signal
    that only needs to be approximate. Taking the k numerically smallest hashes gives a
    fixed-size sample that still estimates Jaccard overlap honestly, because the same
    phrase always hashes to the same value and so is either in both sketches or neither.
    """
    hs = sorted(hashlib.blake2b(s.encode(), digest_size=8).hexdigest() for s in shingles(text))
    return hs[:k]


def similarity(a: list[str], b: list[str], k: int = SKETCH_SIZE) -> float:
    """Estimated Jaccard overlap between two sketches, in 0..1.

    Computed over the bottom-k of the *combined* hash space rather than by intersecting
    the two lists directly. Intersecting directly biases the estimate downward whenever
    the scripts differ in length, because the longer one's sketch covers a narrower slice
    of the hash range and genuinely shared phrases fall outside it.
    """
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union = sorted(sa | sb)[:k]
    if not union:
        return 0.0
    return sum(1 for h in union if h in sa and h in sb) / len(union)


def commentary_ratio(script: dict) -> float:
    """Share of spoken words that are the narrator's own reaction rather than recited fact.

    Asides are the only part of a script that is reliably commentary: they are generated
    as opinion, labelled as opinion for the fact-checker, and deliberately carry no
    claims. Everything else is sourced assertion. Measuring only asides therefore
    understates insight slightly, which is the right direction for a safety signal - it
    can tell us a video has no voice of its own, and it never flatters one that doesn't.
    """
    segs = script.get("segments") or []
    total = aside = 0
    for s in segs:
        total += len(re.findall(r"\S+", s.get("text") or ""))
        n = len(re.findall(r"\S+", s.get("aside") or ""))
        total += n
        aside += n
    return aside / total if total else 0.0


def narration_of(script: dict) -> str:
    segs = script.get("segments") or []
    return " ".join(f"{s.get('text', '')} {s.get('aside', '')}" for s in segs)


def check(script: dict, cfg: dict, history: list[dict] | None = None) -> list[str]:
    """Issues that would make this draft look bulk-produced. Empty list = passed."""
    ccfg = (cfg.get("content") or {})
    issues: list[str] = []

    # Enforced as a count, not as a ratio. The first version of this compared
    # commentary_ratio against a 0.04 floor and rejected a script sitting exactly on it:
    # two aside words in fifty is 0.04, which is not representable in binary and landed
    # one ULP low. Worse than the bug was what it revealed - a percentage floor also
    # makes the requirement depend on how wordy the rest of the script is, so the same
    # aside passes in a short episode and fails in a long one. What we actually want is
    # the thing the policy names: narration that is not purely recited fact. That is
    # "at least one aside", which is exact, explainable in the rejection message, and
    # impossible to land on the wrong side of by rounding. The ratio is still computed
    # and recorded per video, because it is useful telemetry - it is just not a gate.
    need = int(ccfg.get("min_asides", 0))
    if need > 0:
        have = sum(1 for s_ in (script.get("segments") or []) if (s_.get("aside") or "").strip())
        if have < need:
            issues.append(
                f"no narrator voice: {have} asides, need {need} — narration that only recites "
                "sourced facts is the exact 'generic TTS narration, no commentary or insight' "
                "pattern that gets faceless channels demonetized")

    limit = float(ccfg.get("max_script_similarity", 1.0))
    window = int(ccfg.get("similarity_window", 25))
    if limit < 1.0 and history:
        mine = sketch(narration_of(script))
        worst, worst_title = 0.0, ""
        for h in list(history)[-window:]:
            past = h.get("sketch") or []
            sim = similarity(mine, past)
            if sim > worst:
                worst, worst_title = sim, (h.get("title") or "")[:60]
        if worst > limit:
            issues.append(
                f"too close to a published episode ({worst:.0%} phrase overlap with {worst_title!r}, "
                f"limit {limit:.0%}) — five videos with under 20% script variation is treated as "
                "bulk production and demonetized as a batch")
    return issues
