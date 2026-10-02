"""Context-aware topic safety, replacing a flat keyword blocklist.

The old design rejected any topic whose text matched one of 57 words — including `war`, `attack`,
`crash`, `leak`, `trial`, `cure`, `odds`, `poll`, `flood`, `earthquake`, `hurricane`. Those words
appear constantly in completely safe science, engineering and history content: a *crash* test, a
vacuum *leak*, a clinical *trial*, the Thirty Years' *War*, a *flood* basalt. The blocklist was
removing large, high-performing, advertiser-friendly territory — volcanoes, deep-sea, aviation
engineering, archaeology, the history of medicine — and it is a big part of why the topic funnel
kept starving and falling through to `selection_rounds: 3`.

YouTube's advertiser-friendly guidelines are about **context and graphicness**, not vocabulary.
Educational, non-graphic references to difficult subjects are explicitly fine.

So screening is tiered, and the expensive tier is the rare one:

  1. HARD_BLOCKED  — genuinely non-negotiable. Rejected instantly, never reviewed, no LLM call.
  2. no match      — the overwhelming majority of topics. Passes instantly, no LLM call.
  3. SENSITIVE     — a word that *can* indicate a problem. Only these go to the LLM, which judges the
                     actual treatment: graphic/tragic/exploitative, or educational and non-graphic?
  4. LLM down      — falls back to the old reject-on-keyword behaviour, so this is never *less* safe
                     than what it replaces.

Titles are held to a stricter standard than narration: a word that is fine mid-sentence in an
educational line can still read as shock-bait in a title.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("autotube.safety")

# Never produced, never reviewed, no context makes these OK for this channel.
HARD_BLOCKED = [
    "rape", "incest", "child abuse", "csam", "porn", "pornography", "onlyfans", "nude", "nudity",
    "gore", "beheading", "mutilation", "torture", "suicide", "self-harm", "overdose",
    "serial killer", "massacre", "genocide", "terror attack", "mass shooting", "school shooting",
    "xxx", "escort", "brothel",
]

# Words that merely *might* indicate a problem. Presence triggers a context judgement, not a rejection.
SENSITIVE = [
    "death", "dies", "died", "dead", "killed", "killing", "killer", "fatal", "casualties",
    "crime", "criminal", "convicted", "executed", "execution", "arrested", "charged", "trial",
    "lawsuit", "scandal", "kidnapping", "assault", "abuse", "murder", "shooting",
    "war", "attack", "battle", "invasion", "hostage", "bomb", "weapon",
    "crash", "disaster", "hurricane", "earthquake", "wildfire", "flood", "tornado", "famine",
    "election", "candidate", "vote", "poll", "politician", "abortion", "immigration",
    "vaccine", "cancer", "cure", "disease", "outbreak", "diet pill", "drug",
    "crypto", "stock tip", "lottery", "casino", "betting", "odds", "gambling",
    "sex", "sexual", "courtesan", "leak", "obituary", "funeral", "slavery", "slave",
]

# Never allowed in a TITLE regardless of how educational the video is — titles are the shock-bait surface.
TITLE_BLOCKED = [
    "murder", "killer", "massacre", "execution", "suicide", "rape", "porn", "nude", "gore",
    "dead body", "corpse", "tragedy", "victims",
]

CLASSIFY_SYSTEM = (
    "You are a YouTube advertiser-friendliness reviewer. You judge whether a proposed educational "
    "Short about a subject can be made in a way that is advertiser-friendly, family-friendly and "
    "respectful. You judge the TREATMENT and CONTEXT, not individual words: non-graphic, educational "
    "references to difficult history, science or engineering are acceptable; graphic detail, "
    "exploitation of real victims, recent tragedy, and divisive political or medical advocacy are not. "
    "Reply with JSON only."
)


def _match(terms: list[str], text: str) -> str | None:
    low = f" {(text or '').lower()} "
    for t in terms:
        if re.search(rf"\b{re.escape(t.lower())}\b", low):
            return t
    return None


def config(cfg: dict) -> dict:
    """Safety settings, with backward compatibility for the old compliance.blocked_topics list."""
    raw = dict((cfg.get("safety") or {}))
    comp = cfg.get("compliance") or {}
    # `null`/absent in YAML means "use the module default"; an explicit list overrides it.
    s = {
        "hard_blocked": raw.get("hard_blocked") or HARD_BLOCKED,
        "title_blocked": raw.get("title_blocked") or TITLE_BLOCKED,
        "context_review": raw.get("context_review", True),
    }
    if raw.get("sensitive"):
        s["sensitive"] = raw["sensitive"]
    elif cfg.get("safety") is not None:
        s["sensitive"] = SENSITIVE
    else:                                    # pre-safety config: the old flat list, as-is
        s["sensitive"] = comp.get("blocked_topics") or SENSITIVE
    return s


def screen(text: str, cfg: dict) -> tuple[str, str | None]:
    """Fast, free triage. Returns ('block'|'review'|'ok', matched_term)."""
    s = config(cfg)
    hit = _match(s["hard_blocked"], text)
    if hit:
        return "block", hit
    hit = _match(s["sensitive"], text)
    if hit:
        return ("review" if s.get("context_review", True) else "block"), hit
    return "ok", None


def screen_title(title: str, cfg: dict) -> str | None:
    """Titles are stricter than narration. Returns the offending term, or None."""
    s = config(cfg)
    return _match(s["hard_blocked"] + s["title_blocked"], title)


def judge(subject: str, context: str, term: str, llm, cfg: dict) -> tuple[bool, str]:
    """Ask a model whether this subject can be covered safely. Returns (allowed, reason).

    Only ever called for the minority of topics that tripped a SENSITIVE word, so the cost is small.
    If the model is unreachable we fall back to rejecting, which is exactly what the old keyword
    blocklist did — this path is never less safe than the behaviour it replaces.
    """
    from .llm import LLMError
    if llm is None:
        return False, f"no classifier available to judge {term!r}"
    user = f"""SUBJECT: {subject}
CONTEXT: {context[:1200]}
FLAGGED WORD: "{term}"

We want to make a 30-second, factual, family-friendly educational Short about this subject.
Can it be done in an advertiser-friendly, non-graphic, respectful way?

Say yes for: non-graphic historical or scientific references (e.g. an engineering crash test, a
vacuum leak, a clinical trial, a flood basalt, the Thirty Years' War as a date reference), natural
phenomena and geology, the history of medicine or technology, animal biology, archaeology.
Say no for: graphic injury or death detail, real identifiable victims, recent tragedy, true crime,
divisive politics or elections, medical/financial advice or claims, gambling or crypto promotion,
anything sexual.

Return JSON: {{"allowed": true|false, "reason": "<one short sentence>",
"angle": "<if allowed: the safe, non-graphic angle to take>"}}"""
    try:
        r = llm.json(CLASSIFY_SYSTEM, user, temperature=0.1,
                     validate=lambda o: isinstance(o.get("allowed"), bool))
    except (LLMError, Exception) as e:  # noqa: BLE001 — classifier failure must fail closed
        log.info("safety classifier unavailable (%s) → rejecting %r on the keyword", str(e)[:100], subject)
        return False, f"classifier unavailable; rejected on {term!r}"
    allowed = bool(r.get("allowed"))
    reason = str(r.get("reason", ""))[:200]
    log.info("  safety: %r flagged on %r → %s (%s)", subject[:50], term, "ALLOW" if allowed else "BLOCK", reason)
    return allowed, reason


# ── living people ─────────────────────────────────────────────────────────────
# Angles that are never worth the risk on an automated channel. Each is either an
# advertiser-friendliness problem, a YouTube harassment/privacy problem, or a defamation problem,
# and no amount of trend heat makes any of them pay.
PERSON_BLOCKED_ANGLES = [
    # legal / accusation — defamation exposure, and "alleged" is not a defence when a bot said it
    "allegation", "allegations", "accused", "accuser", "accusation", "lawsuit", "sued", "suing",
    "arrest", "arrested", "indicted", "indictment", "charged with", "convicted", "conviction",
    "trial of", "verdict", "sentencing", "prison", "jail", "fraud", "scandal", "controversy",
    "backlash", "cancelled", "canceled", "feud", "diss", "beef with", "slammed", "blasted",
    "clapped back", "shades", "throwing shade",
    # health, death, body — "sensitive events" under advertiser guidelines
    "cancer", "diagnosis", "diagnosed", "illness", "hospitalised", "hospitalized", "overdose",
    "rehab", "addiction", "mental breakdown", "suicide", "died", "death of", "dead at", "funeral",
    "obituary", "weight loss", "weight gain", "plastic surgery", "body transformation",
    "before and after",
    # private life
    "divorce", "divorced", "breakup", "break up", "split from", "dating", "girlfriend", "boyfriend",
    "affair", "cheating", "pregnant", "pregnancy", "baby bump", "custody", "net worth", "salary",
    "mansion", "house tour", "sexuality", "came out as",
    # speculation
    "rumour", "rumor", "rumoured", "rumored", "speculation", "conspiracy",
]

# A factual channel must never repeat a claim it cannot stand behind. These phrases are how an
# automated script launders a rumour into an assertion, so they are banned outright rather than
# judged — "Sounds fake. It's proven." is the whole premise.
SPECULATION = [
    "allegedly", "reportedly", "rumour has it", "rumor has it", "sources say", "insiders say",
    "it is claimed", "it's claimed", "apparently", "supposedly", "some say", "people are saying",
    "fans think", "fans believe", "we think", "could be", "might have", "may have secretly",
]

PERSON_SYSTEM = (
    "You are a YouTube Trust & Safety reviewer assessing whether a short video about a real, living "
    "person can be published by a fully automated channel with no human editor. You apply YouTube's "
    "harassment, privacy and advertiser-friendly guidelines strictly, and you err toward refusing. "
    "Reply with JSON only."
)


def speculation_hits(text: str) -> list[str]:
    low = f" {(text or '').lower()} "
    return [t for t in SPECULATION if re.search(rf"\b{re.escape(t)}\b", low)]


def person_angle_hits(text: str) -> list[str]:
    low = f" {(text or '').lower()} "
    return [t for t in PERSON_BLOCKED_ANGLES if re.search(rf"\b{re.escape(t)}\b", low)]


def person_config(cfg: dict) -> dict:
    raw = ((cfg.get("safety") or {}).get("person") or {})
    return {
        "require_public_figure": bool(raw.get("require_public_figure", True)),
        "blocked_angles": raw.get("blocked_angles") or PERSON_BLOCKED_ANGLES,
        "allow_minors": bool(raw.get("allow_minors", False)),
    }


def check_person(subject: str, context: str, cfg: dict, llm=None) -> tuple[bool, str, str]:
    """May we make a video about this living person? Returns (allowed, reason, safe_angle).

    Two gates, cheap one first. A deterministic pass rejects anything already framed as a blocked
    angle — legal trouble, health, death, relationships, money, rumour — without spending a model
    call. Whatever survives goes to a classifier that answers the question the keywords cannot:
    is this a PUBLIC FIGURE being discussed in their professional capacity, or a private individual
    who happens to be trending?

    That distinction is the one that matters. A private person who went viral has no public role to
    discuss, is protected by YouTube's privacy and harassment policies, and can demand removal —
    and they are precisely who a trend scanner will surface. Fails closed: no classifier, no video.
    """
    pcfg = person_config(cfg)
    hits = [t for t in pcfg["blocked_angles"]
            if re.search(rf"\b{re.escape(t.lower())}\b", f" {subject} {context}".lower())]
    if hits:
        return False, f"living person framed around a blocked angle ({', '.join(hits[:3])})", ""
    if llm is None:
        return False, "no classifier available to judge a living person", ""

    from .llm import LLMError
    user = f"""PERSON / TOPIC: {subject}
CONTEXT: {context[:1200]}

A fully automated channel wants to make a 30-60 second factual video involving this living person.
There is no human editor and no legal review.

Allow ONLY if ALL of these hold:
  - they are a genuine PUBLIC FIGURE (a notable professional, performer, athlete, executive,
    scientist, creator), not a private individual who merely went viral or appeared in a news story
  - they are an adult
  - the angle is their PUBLIC, PROFESSIONAL work: a release, performance, record, discovery,
    appointment, award, product, retirement from a role, or a public statement about their work
  - the facts needed are uncontroversial and already widely reported

Refuse if the person is private or a minor, or the angle touches: legal matters, accusations,
arrest, scandal, feuds, health, death, appearance or body, relationships, family, children,
finances or net worth, religion, sexuality, politics, or anything rumoured, alleged or speculative.
Refuse if the only reason this is trending is that something bad happened to them.

Return JSON: {{"is_public_figure": true|false, "is_adult": true|false,
"angle_is_professional": true|false, "allowed": true|false,
"reason": "<one short sentence>", "safe_angle": "<if allowed: the professional angle to take>"}}"""
    try:
        r = llm.json(PERSON_SYSTEM, user, temperature=0.1,
                     validate=lambda o: isinstance(o.get("allowed"), bool))
    except (LLMError, Exception) as e:  # noqa: BLE001 — must fail closed
        log.info("person classifier unavailable (%s) → rejecting %r", str(e)[:100], subject)
        return False, "classifier unavailable; living person rejected", ""

    allowed = bool(r.get("allowed"))
    reason = str(r.get("reason", ""))[:200]
    if allowed and pcfg["require_public_figure"] and not r.get("is_public_figure"):
        return False, "private individual, not a public figure", ""
    if allowed and not pcfg["allow_minors"] and r.get("is_adult") is False:
        return False, "subject is a minor", ""
    if allowed and not r.get("angle_is_professional"):
        return False, "angle is not about their professional work", ""
    log.info("  person: %r → %s (%s)", subject[:50], "ALLOW" if allowed else "BLOCK", reason)
    return allowed, reason, str(r.get("safe_angle", ""))[:300]


def check_subject(subject: str, context: str, cfg: dict, llm=None) -> tuple[bool, str]:
    """Full screen for a topic/source. Returns (allowed, reason)."""
    verdict, term = screen(f"{subject} {context}", cfg)
    if verdict == "ok":
        return True, ""
    if verdict == "block":
        return False, f"hard-blocked subject ({term})"
    allowed, reason = judge(subject, context, term, llm, cfg)
    return allowed, reason if allowed else f"sensitive subject ({term}): {reason}"


def check_script(title: str, narration: str, cfg: dict, llm=None,
                 about_person: bool = False) -> list[str]:
    """Script-level screen. Titles are strict; narration is judged in context."""
    issues = []
    bad_title = screen_title(title, cfg)
    if bad_title:
        issues.append(f"the title contains {bad_title!r}, which is never allowed in a title — "
                      "rephrase it around the subject itself")
    hit = _match(config(cfg)["hard_blocked"], narration)
    if hit:
        issues.append(f"the narration contains {hit!r}, which is never allowed")
    for s_hit in speculation_hits(f"{title} {narration}")[:3]:
        issues.append(f"{s_hit!r} states something we cannot prove — every claim must be "
                      "sourced and stated plainly, or cut")
    if about_person:
        for a_hit in person_angle_hits(f"{title} {narration}")[:3]:
            issues.append(f"{a_hit!r} is off-limits when covering a living person — stay on their "
                          "public professional work")
    return issues
