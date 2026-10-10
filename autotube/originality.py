"""Internal originality safeguards for distinct scripts and meaningful narrator voice.

YouTube's monetization policy describes inauthentic and reused-content categories, but does not publish
numerical similarity or upload-count enforcement triggers. This module therefore uses local, configurable
checks to encourage substantive originality; it does not predict a policy decision or guarantee monetization.

A channel may keep recognizable Archive 13 formats and series, while each episode still needs distinct facts,
explanation, commentary and visual treatment. The lexical similarity score compares what was said, not the
creative format, and is only one narrow safeguard—not a substitute for editorial judgment or policy review.
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
    """Issues against local originality standards; this is not a YouTube policy verdict."""
    ccfg = (cfg.get("content") or {})
    issues: list[str] = []

    # Enforced as a count, not as a ratio. The first version of this compared
    # commentary_ratio against a 0.04 floor and rejected a script sitting exactly on it:
    # two aside words in fifty is 0.04, which is not representable in binary and landed
    # one ULP low. Worse than the bug was what it revealed - a percentage floor also
    # makes the requirement depend on how wordy the rest of the script is, so the same
    # aside passes in a short episode and fails in a long one. The configured aside count
    # is an internal voice/value-add choice, not a YouTube policy threshold. The ratio is
    # still computed and recorded per video as telemetry, but is not used as a policy claim.
    need = int(ccfg.get("min_asides", 0))
    if need > 0:
        have = sum(1 for s_ in (script.get("segments") or []) if (s_.get("aside") or "").strip())
        if have < need:
            issues.append(
                f"no narrator voice for this channel's internal target: {have} asides, need {need} — "
                "add a brief original reaction or revise the channel configuration")

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
                f"internal limit {limit:.0%}) — develop a distinct original treatment, not a near-duplicate")
    return issues


def channel_audit(history: list[dict], cfg: dict) -> dict:
    """What a policy reviewer sees when they look at the channel rather than a video.

    The per-draft check in check() is pairwise: it refuses a script that is too close to
    any single predecessor. That is an internal originality safeguard, not a published
    YouTube numerical enforcement rule. Pairwise checks alone can miss broad creative
    sameness, so this report summarizes patterns across the back catalogue without
    asserting a fixed episode-count trigger or similarity cutoff from YouTube.

    Reported, never enforced. These are slow-moving properties of a back catalogue, and
    a run that refuses to publish because the last twenty episodes skewed toward one
    format would be punishing today's video for last month's decisions - while also
    leaving the channel with nothing new, which helps nobody.
    """
    ccfg = (cfg.get("content") or {})
    window = int(ccfg.get("audit_window", 25))
    recent = [h for h in list(history)[-window:] if h.get("sketch")]
    n = len(recent)
    out: dict = {"episodes_examined": n, "window": window}
    if n < 2:
        out["verdict"] = "not enough published episodes with a recorded script fingerprint yet"
        return out

    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(similarity(recent[i]["sketch"], recent[j]["sketch"]))
    limit = float(ccfg.get("max_script_similarity", 1.0))
    out["mean_pairwise_similarity"] = round(sum(sims) / len(sims), 3)
    out["max_pairwise_similarity"] = round(max(sims), 3)
    # Count pairs above this repo's internal similarity limit as well as the mean; an average can hide a tight cluster.
    near = sum(1 for s in sims if s > limit)
    out["pairs_over_similarity_limit"] = near

    def concentration(key: str) -> tuple[str, float]:
        vals = [((h.get("choice") or {}).get(key)) for h in recent]
        vals = [v for v in vals if v]
        if not vals:
            return "", 0.0
        top = max(set(vals), key=vals.count)
        return top, round(vals.count(top) / len(vals), 3)

    fmt, fmt_share = concentration("format")
    cat, cat_share = concentration("category")
    out["most_common_format"] = {"name": fmt, "share": fmt_share}
    out["most_common_category"] = {"name": cat, "share": cat_share}
    voiced = sum(1 for h in recent if (h.get("commentary_ratio") or 0) > 0)
    out["share_with_narrator_commentary"] = round(voiced / n, 3)

    flags = []
    if near:
        flags.append(f"{near} pairs of episodes exceed the {limit:.0%} phrase-overlap limit")
    if fmt_share > 0.5 and n >= 5:
        flags.append(f"{fmt_share:.0%} of recent episodes use the '{fmt}' format — reads as one template")
    if out["share_with_narrator_commentary"] < 0.8:
        flags.append(f"only {out['share_with_narrator_commentary']:.0%} of episodes have any narrator commentary")
    out["flags"] = flags
    out["verdict"] = "looks like a show" if not flags else "drifting toward a template: " + "; ".join(flags)
    return out
