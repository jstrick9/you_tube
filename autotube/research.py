"""Grounding: resolve a topic to a verifiable source (Wikipedia) and fetch its text.

Every script is written ONLY from this fetched text, and the numbers in the final
script are checked against it programmatically. This is the core defense against
AI hallucination / misinformation, which YouTube's policies (and viewers) punish.
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from urllib.parse import quote

from . import news, safety, sources
from .common import get_json
from .trends import is_living_person, wiki_summary

log = logging.getLogger("autotube.research")

API = "https://{lang}.wikipedia.org/w/api.php"


def search_title(query: str, lang: str = "en") -> str | None:
    try:
        data = get_json(API.format(lang=lang), params={
            "action": "query", "list": "search", "srsearch": query, "srlimit": 1,
            "format": "json", "srnamespace": 0})
        hits = data.get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None
    except Exception as e:  # noqa: BLE001
        log.warning("wiki search failed for %r: %s", query, e)
        return None


_ARTICLE_FOCUS_STOPWORDS = frozenset("""
    a an and are as at be been being but by can could did do does doing for from had has have he her here hers
    him his how i if in into is it its may might more most my of on or our out over she should so than that the
    their theirs them then there these they this those through to under up us was we were what when where which
    while who why will with would you your actually amazing new viral video short shorts today trend trending
""".split())

_ARTICLE_FOCUS_EQUIVALENTS = {
    "feces": frozenset({"feces", "faeces", "fecal", "faecal", "pap", "poop", "popo"}),
    "faeces": frozenset({"feces", "faeces", "fecal", "faecal", "pap", "poop", "popo"}),
    "fecal": frozenset({"feces", "faeces", "fecal", "faecal", "pap", "poop", "popo"}),
    "faecal": frozenset({"feces", "faeces", "fecal", "faecal", "pap", "poop", "popo"}),
    "poop": frozenset({"feces", "faeces", "fecal", "faecal", "pap", "poop", "popo"}),
    "popo": frozenset({"feces", "faeces", "fecal", "faecal", "pap", "poop", "popo"}),
    "codebreaker": frozenset({"codebreaker", "codebreakers", "codebreaking", "cryptanalyst", "cryptanalysts", "cryptanalysis"}),
    "codebreakers": frozenset({"codebreaker", "codebreakers", "codebreaking", "cryptanalyst", "cryptanalysts", "cryptanalysis"}),
    "cryptanalyst": frozenset({"codebreaker", "codebreakers", "codebreaking", "cryptanalyst", "cryptanalysts", "cryptanalysis"}),
    "cryptanalysts": frozenset({"codebreaker", "codebreakers", "codebreaking", "cryptanalyst", "cryptanalysts", "cryptanalysis"}),
}


def _article_focus_word_forms(word: str) -> set[str]:
    """Small retrieval-only inflection bridge; source validation remains unchanged."""
    forms = {word}
    if word.endswith("ies") and len(word) > 4:
        forms.add(word[:-3] + "y")
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        forms.add(word[:-1])
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        forms.update({stem, stem + "e"})
    elif word.endswith("ed") and len(word) > 4:
        forms.add(word[:-2])
    return forms


def _article_focus_tokens(value: str) -> list[str]:
    ascii_text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", ascii_text.casefold())


def _article_focus_groups(query: str) -> list[frozenset[str]]:
    groups: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for word in _article_focus_tokens(query):
        if len(word) < 3 or word in _ARTICLE_FOCUS_STOPWORDS:
            continue
        equivalents = _ARTICLE_FOCUS_EQUIVALENTS.get(word, frozenset({word}))
        forms = set().union(*(_article_focus_word_forms(term) for term in equivalents))
        group = frozenset(forms)
        if group not in seen:
            groups.append(group)
            seen.add(group)
    return groups


def _focused_article_excerpt(text: str, focus_query: str, max_chars: int) -> str:
    """Keep the lead plus source paragraphs relevant to a query, even when they occur deep in the article.

    This only improves retrieval: the selected text is copied verbatim from the fetched article and
    every existing source, safety, originality, and script QA gate still runs against it.
    """
    if not focus_query or max_chars <= 0:
        return text[:max_chars]
    groups = _article_focus_groups(focus_query)
    if not groups:
        return text[:max_chars]

    paragraphs = [re.sub(r"\s+", " ", part).strip()
                  for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs:
        return text[:max_chars]

    def is_heading(paragraph: str) -> bool:
        return (len(paragraph) <= 120 and len(paragraph.split()) <= 14
                and not re.search(r"[.!?;:]", paragraph)
                and not paragraph.lower().startswith(("from ", "list of ")))

    heading_at: list[int | None] = []
    current_heading: int | None = None
    for i, paragraph in enumerate(paragraphs):
        if is_heading(paragraph):
            current_heading = i
        heading_at.append(current_heading)

    first_heading = next((i for i, p in enumerate(paragraphs) if is_heading(p)), min(2, len(paragraphs)))
    lead_parts: list[str] = []
    lead_size = 0
    lead_budget = min(3500, max_chars // 2)
    for paragraph in paragraphs[:first_heading]:
        if lead_size + len(paragraph) + 2 > lead_budget:
            if not lead_parts and max_chars > 0:
                # Keep at least a bounded beginning of a long lead instead of losing it entirely.
                clipped = paragraph[:lead_budget]
                lead_parts.append(clipped)
                lead_size += len(clipped) + 2
            break
        lead_parts.append(paragraph)
        lead_size += len(paragraph) + 2

    token_sets = [set(_article_focus_tokens(paragraph)) for paragraph in paragraphs]
    section_tokens = [
        (tokens | (token_sets[heading_at[i]] if heading_at[i] is not None else set()))
        for i, tokens in enumerate(token_sets)
    ]
    frequencies = [sum(bool(group & tokens) for tokens in section_tokens) for group in groups]
    weights = [1.0 + math.log((len(paragraphs) + 1) / (frequency + 1))
               for frequency in frequencies]
    ranked: list[tuple[float, int, int]] = []
    for i, paragraph in enumerate(paragraphs):
        if is_heading(paragraph):
            continue
        matched = [j for j, group in enumerate(groups) if group & section_tokens[i]]
        if matched:
            ranked.append((sum(weights[j] for j in matched), len(matched), i))
    if not ranked:
        return text[:max_chars]

    selected = set(range(len(lead_parts)))
    used = lead_size
    budget = max_chars - used
    for _score, _matches, i in sorted(ranked, key=lambda item: (-item[0], -item[1], item[2])):
        additions = []
        heading = heading_at[i]
        if heading is not None and heading not in selected:
            additions.append(heading)
        if i not in selected:
            additions.append(i)
        extra = sum(len(paragraphs[j]) + 2 for j in additions)
        if extra <= budget:
            selected.update(additions)
            used += extra
            budget -= extra
        if budget < 240:
            break

    if not any(i >= first_heading for i in selected):
        return text[:max_chars]
    excerpt = "\n\n".join(paragraphs[i] for i in sorted(selected))
    return excerpt[:max_chars]


def article_text(title: str, lang: str = "en", max_chars: int = 9000,
                 focus_query: str | None = None) -> str:
    data = get_json(API.format(lang=lang), params={
        "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
        "titles": title, "format": "json", "redirects": 1})
    pages = data.get("query", {}).get("pages", {})
    text = next(iter(pages.values()), {}).get("extract", "") if pages else ""
    # Drop reference-ish tail sections while preserving the article's paragraph boundaries.
    text = re.split(r"\n(See also|References|External links|Notes|Further reading|Bibliography)\n", text)[0]
    text = re.sub(r"\n[ \t]*\n{2,}", "\n\n", text).strip()
    return _focused_article_excerpt(text, focus_query or "", max_chars)


def mentions(title: str, lang: str = "en", max_articles: int = 6, max_chars: int = 5000) -> tuple[str, list[str]]:
    """Paragraphs from OTHER Wikipedia articles that talk about `title` (for short but viral subjects).

    Only paragraphs that actually mention the subject are kept, each labelled with its article, so every fact
    in the script still comes from Wikipedia and still goes through the same evidence/number/LLM fact-checks.
    """
    subject = re.sub(r"\s*\(.*?\)", "", title).strip()
    stem = re.escape(subject.lower().rstrip("s"))
    forms = [subject, subject[:-1] if subject.endswith("s") else subject + "s"]     # singular + plural
    hits: list[str] = []
    for f in forms:
        try:
            data = get_json(API.format(lang=lang), params={
                "action": "query", "list": "search", "srsearch": f'"{f}"', "srlimit": max_articles + 2,
                "format": "json", "srnamespace": 0})
            hits += [h["title"] for h in data.get("query", {}).get("search", [])
                     if h["title"] != title and h["title"] not in hits]
        except Exception as e:  # noqa: BLE001
            log.warning("wiki mention search failed for %r: %s", f, e)
    max_articles = max_articles * 2
    parts, used = [], []
    total = 0
    for h in hits[:max_articles]:
        if re.match(r"^(List|Lists|Timeline|Index|Outline|Glossary) of ", h):
            continue
        try:
            txt = article_text(h, lang, max_chars=200000)
        except Exception:  # noqa: BLE001
            continue
        paras = [x.strip() for x in txt.split("\n") if len(x.strip()) > 80 and re.search(stem, x.lower())][:3]
        if not paras:
            continue
        block = f'From the Wikipedia article "{h}": ' + " ".join(paras)
        parts.append(block[:1600])
        used.append(h)
        total += len(parts[-1])
        if total >= max_chars:
            break
    return "\n".join(parts), used


def article_images(title: str, lang: str = "en", limit: int = 25) -> list[str]:
    """File titles used on the article (for on-topic visuals)."""
    try:
        data = get_json(API.format(lang=lang), params={
            "action": "query", "prop": "images", "titles": title, "imlimit": limit,
            "format": "json", "redirects": 1})
        pages = data.get("query", {}).get("pages", {})
        imgs = next(iter(pages.values()), {}).get("images", [])
        return [i["title"] for i in imgs
                if re.search(r"\.(jpe?g|png|webp|tiff?)$", i["title"], re.I)
                and not re.search(r"(logo|icon|flag|symbol|map|signature|commons-|wiki|edit|question|ambox|portal)",
                                  i["title"], re.I)]
    except Exception:  # noqa: BLE001
        return []



def ground_news(topic: str, search_query: str | None, lang: str = "en",
                cfg: dict | None = None, llm=None) -> dict | None:
    """Ground a trend that has no Wikipedia article in the open news record.

    Two rules keep this from becoming a rumour mill. First, a story must be carried by at least
    `research.min_outlets` INDEPENDENT publishers before it can be used at all, so one outlet's
    scoop — or one wire story syndicated five times — is not treated as established fact. Second,
    only headlines are taken: reproducing publishers' article bodies creates copyright and originality
    concerns and would not meet AutoTube's original-value requirement.

    Headlines alone cannot carry a 60-second script, so the explainer half is anchored on a related
    encyclopedic subject (Sora 2 → OpenAI). If no such backbone resolves, the topic is dropped
    rather than written from headlines alone.
    """
    rcfg = ((cfg or {}).get("research") or {})
    if not rcfg.get("news_fallback", True):
        return None
    subject = (search_query or topic or "").strip()
    region = ((cfg or {}).get("channel") or {}).get("region", "US")
    brief = news.brief(subject, lang, region, int(rcfg.get("min_outlets", 2)))
    if not brief:
        return None
    person_angle = ""
    if cfg is not None:
        allowed, reason = safety.check_subject(subject, brief["text"][:1200], cfg, llm)
        if not allowed:
            log.info("skip %r: %s", subject, reason)
            return None
        # A breaking story about a person is the single riskiest thing a trend scanner can hand us,
        # and here there is no Wikipedia description to test, so the classifier decides from the
        # headlines whether a real person is the subject at all.
        if (cfg.get("content") or {}).get("allow_living_people"):
            ok, reason, person_angle = safety.check_person(subject, brief["text"][:1200], cfg, llm)
            if not ok and "not about a person" not in reason.lower():
                log.info("skip %r: %s", subject, reason)
                return None

    # anchor the background on the first backbone entity that actually has an article
    anchor_title, anchor_text = None, ""
    for cand in brief["backbone"]:
        t = search_title(cand, lang)
        if not t or re.match(r"^(List|Lists|Timeline|Index|Outline|Glossary) of ", t):
            continue
        body = article_text(t, lang)
        if len(body) >= 1500:
            anchor_title, anchor_text = t, body
            break
    if not anchor_title:
        log.info("skip %r: %d outlets agree it is happening, but no encyclopedic backbone to "
                 "explain it from", subject, brief["n_outlets"])
        return None

    log.info("news-grounded %r: %d independent outlets, background from %r",
             subject, brief["n_outlets"], anchor_title)
    text = brief["text"] + "\n\n" + f"Background on {anchor_title}:\n" + anchor_text
    summ = wiki_summary(anchor_title, lang) or {}
    return {
        "title": topic,
        "url": brief["also"][0]["url"] if brief["also"] else "",
        "description": f"trending news topic, corroborated by {brief['n_outlets']} outlets",
        "summary": "; ".join(brief["consensus"][:6]),
        "text": text,
        "also": brief["also"] + [{"title": anchor_title,
                                  "url": f"https://{lang}.wikipedia.org/wiki/{quote(anchor_title.replace(' ', '_'))}",
                                  "publisher": "Wikipedia"}],
        "images": article_images(anchor_title, lang),
        "lead_image": (summ.get("originalimage") or {}).get("source"),
        "news": {"outlets": brief["outlets"], "consensus": brief["consensus"], "anchor": anchor_title},
        "safe_angle": person_angle,
    }


def _words(x: str) -> list:
    import re as _re
    return [w for w in _re.findall(r"[\w\u00c0-\u024f]+", (x or "").lower()) if len(w) > 2]


_GENERIC_NARROWING_TERMS = frozenset({
    "origin", "history", "overview", "background", "introduction", "meaning", "guide", "summary",
})


def widened(query: str, title: str) -> bool:
    """Did Wikipedia search drop the distinguishing part of the request?

    Search returns the best *broad* match, so a descriptive query collapses onto its
    category: "Bou Craa conveyor belt" resolves to "Conveyor belt" even though the
    article "Bou Craa" exists and is the thing that actually trended. The writer may
    only cite the article it is given, so the detail that made the topic worth
    watching is deleted before a word is drafted.

    Detected structurally rather than lexically: the resolved title is a strict subset
    of the words asked for. That is precisely the shape of a generalisation, and it
    does not fire on a near-miss like "Caliga" -> "Caligae", which shares no whole word.
    """
    q, t = set(_words(query)), set(_words(title))
    return bool(t) and t < q


def narrow_title(query: str, lang: str = "en"):
    """Resolve a query, and if search widened it, try again on the distinctive part.

    Returns the most specific article that still answers the request, or the widened
    one when nothing better exists - a general article is worse than a specific one
    but better than no video.
    """
    title = search_title(query, lang)
    if not title or not widened(query, title):
        return title
    extra = [w for w in _words(query)
             if w not in set(_words(title)) and w not in _GENERIC_NARROWING_TERMS]
    if not extra:
        return title
    retry = search_title(" ".join(extra), lang)
    if retry and not widened(" ".join(extra), retry):
        log.info("narrowed %r: %r -> %r", query[:48], title, retry)
        return retry
    return title


def ground(topic: str, search_query: str | None, lang: str = "en", allow_living: bool = False,
           blocked: list[str] | None = None, cfg: dict | None = None, llm=None,
           research_query: str | None = None, focus_query: str | None = None) -> dict | None:
    """Return {'title','url','summary','text','images'} or None if unusable.

    Wikipedia resolves and anchors the subject; `research.extra_sources` then widens what we can
    verify against (see autotube/sources.py). A narrow `research_query` may prioritize a matching
    PubMed abstract and place it in the writer's bounded excerpt without replacing Wikipedia or changing
    any evidence checks. `focus_query` selects relevant verbatim paragraphs from the full
    article extract without changing or weakening later evidence checks. Safety screening is
    context-aware (autotube/safety.py).
    """
    title = None
    for q in [search_query, topic]:
        if q:
            title = narrow_title(q, lang)
            if title:
                break
    if not title:
        # No encyclopedia article. Most things that genuinely trend are like this — a launch, a
        # controversy, a meme — and discarding them all is what kept the channel off live trends.
        return ground_news(topic, search_query, lang, cfg, llm)
    if re.match(r"^(List|Lists|Timeline|Index|Outline|Glossary) of ", title):
        log.info("skip %r: list-style article (thin narrative)", title)
        return None
    summ = wiki_summary(title, lang) or {}
    if summ.get("type") == "disambiguation":
        log.info("skip %r: disambiguation page", title)
        return None
    probe = f"{summ.get('description', '')} {(summ.get('extract') or '')[:600]}"
    if cfg is not None:
        # Context-aware: a word like "war", "crash" or "trial" sends the subject to a classifier
        # instead of deleting it. Only genuinely non-negotiable subjects are rejected outright.
        allowed, reason = safety.check_subject(title, probe, cfg, llm)
        if not allowed:
            log.info("skip %r: %s", title, reason)
            return None
    elif blocked:                                   # legacy keyword path (callers without cfg)
        hit = next((b for b in blocked if re.search(r"\b" + re.escape(b.lower()) + r"\b",
                                                   f" {title} {probe} ".lower())), None)
        if hit:
            log.info("skip %r: source is about a blocked subject (%s)", title, hit)
            return None
    person = is_living_person(summ)
    if person and not allow_living:
        log.info("skip %r: looks like a living person (%s)", title, summ.get("description"))
        return None
    safe_angle = ""
    if person:
        # Living people are allowed only as public figures discussed in their professional
        # capacity (autotube/safety.check_person). Fails closed.
        ok, reason, safe_angle = safety.check_person(title, probe, cfg or {}, llm)
        if not ok:
            log.info("skip %r: %s", title, reason)
            return None
    text = article_text(title, lang, focus_query=focus_query)
    also: list[str] = []
    extra_credits: list[dict] = []
    rcfg = ((cfg or {}).get("research") or {})
    enabled = list(rcfg.get("extra_sources") or [])
    specific_query = research_query.strip()[:180] if isinstance(research_query, str) else ""
    specific_pubmed = bool(specific_query and "pubmed" in enabled)
    if enabled:
        # Always reach past Wikipedia, not only when the article is thin. A long article is exactly
        # the case where a NASA caption or a peer-reviewed abstract adds something Wikipedia lacks —
        # a primary source and hard, unit-bearing numbers — and citing more than one publisher is
        # itself part of not looking mass-produced. Long articles just get a smaller budget.
        full = int(rcfg.get("enrich_max_chars", 4000))
        budget = full if len(text) < int(rcfg.get("enrich_below_chars", 7000)) else max(1200, full // 3)
        if specific_pubmed:
            # If the exact trend has a known primary paper, fetch it first with a tight query,
            # then let the other configured corpora use only the remaining character budget.
            # This avoids a broad "Dog" PubMed search winning the budget with unrelated abstracts.
            block, extra_credits = sources.enrich(specific_query, ["pubmed"], budget)
            remainder = max(0, budget - len(block) - (1 if block else 0))
            other_sources = [name for name in enabled if name != "pubmed"]
            if remainder and other_sources:
                other_block, other_credits = sources.enrich(title, other_sources, remainder)
                if other_block:
                    block = f"{block}\n{other_block}" if block else other_block
                extra_credits.extend(other_credits)
        else:
            block, extra_credits = sources.enrich(title, enabled, budget)
        if block:
            log.info("  %r → +%d chars from %s", title, len(block),
                     ", ".join(sorted({c["publisher"] for c in extra_credits})))
            # A specific primary abstract must not disappear beyond the writer's bounded source excerpt.
            # Place it first for this narrow query; it remains labeled and still passes the same source checks.
            text = f"{block}\n{text}" if specific_pubmed else text + "\n" + block
    if len(text) < 3000:
        # short article (common for viral oddities like "Hunger stone"): add what other articles say about it
        extra, also = mentions(title, lang)
        if extra:
            log.info("  %r is short (%d chars) → +%d chars from %s", title, len(text), len(extra), also)
            text = text + "\n" + extra
    if len(text) < 1500:
        log.info("skip %r: source text too thin (%d chars)", title, len(text))
        return None
    return {
        "title": title,
        "url": f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "description": summ.get("description", ""),
        "summary": summ.get("extract", ""),
        "text": text,
        "also": [{"title": a, "url": f"https://{lang}.wikipedia.org/wiki/{quote(a.replace(' ', '_'))}"}
                 for a in also] + extra_credits,
        "images": article_images(title, lang),
        "lead_image": (summ.get("originalimage") or {}).get("source"),
        "person": person,
        "safe_angle": safe_angle,
    }


# ── programmatic fact check ───────────────────────────────────────────────────
_WORDNUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9,
}
_NUMBER_WORDS = set(_WORDNUM) | {"and"}
_NUMBER_WORD_ALT = "|".join(sorted((re.escape(word) for word in _NUMBER_WORDS), key=len, reverse=True))
_NUMBER_WORD_RE = re.compile(r"\b(?:" + _NUMBER_WORD_ALT + r")(?:[ -]+(?:" + _NUMBER_WORD_ALT + r"))*\b", re.I)
_NUMBER_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def _numbers(text: str) -> set[str]:
    nums = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        n = m.group(0).replace(",", "").rstrip(".")
        if n:
            nums.add(n.lstrip("0") or "0")
    return nums


def _number_key(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return format(value, ".12g")
    return str(int(value))


def _scaled_number_values(text: str) -> set[str]:
    """Expand digit-plus-scale phrases (e.g. ``15 billion``) to their full values."""
    values = set()
    pattern = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s+(thousand|million|billion)s?\b", re.I)
    for match in pattern.finditer(text or ""):
        try:
            amount = float(match.group(1).replace(",", ""))
            scale = _NUMBER_SCALES[match.group(2).lower()]
            values.add(_number_key(amount * scale))
        except (KeyError, ValueError, OverflowError):
            continue
    return values


def _word_number_values(text: str) -> set[str]:
    """Parse spelled-out cardinal phrases, including tens and large scales."""
    values = set()
    for match in _NUMBER_WORD_RE.finditer(text or ""):
        words = re.findall(r"[a-z]+", match.group(0).lower())
        total = current = 0
        saw_number = False
        for word in words:
            if word == "and":
                continue
            value = _WORDNUM[word]
            if value < 100:
                current += value
                saw_number = True
            elif value == 100:
                current = max(1, current) * value
                saw_number = True
            else:
                if not saw_number:
                    prefix = (text or "")[:match.start()]
                    if re.search(r"\d[\d,]*(?:\.\d+)?\s*$", prefix):
                        continue  # a digit-plus-scale phrase is normalized by _scaled_number_values
                    previous = re.search(r"([a-z]+)\s*$", prefix, re.I)
                    if previous and previous.group(1).casefold() in {
                            "few", "many", "multiple", "numerous", "several", "various"}:
                        continue  # do not turn vague quantities into exact counts
                    current = 1  # idioms such as "a million" are explicit cardinal quantities
                    saw_number = True
                total += max(1, current) * value
                current = 0
        if saw_number:
            values.add(str(total + current))
    return values


def _number_mentions(text: str) -> set[str]:
    """Collect digit, scaled-digit, and spelled-out cardinal values."""
    text = text or ""
    return _numbers(text) | _scaled_number_values(text) | _word_number_values(text)


def unsupported_numbers(script_text: str, source_text: str) -> list[str]:
    """Numbers in the script that never appear in the source. Small counts (<=12)
    are ignored because they're usually structural ('3 facts', 'part 2'). Digit and
    spelled-out values are normalized, including phrases such as ``twenty million``."""
    src = _number_mentions(source_text)
    # Also accept modest rounding/conversion forms already present in the source:
    # 4.6 → 4.6, 4,600 → 4600, and 4,600,000 → 4.6 million.
    src_alt = set(src)
    for s in src:
        try:
            f = float(s)
            src_alt.add(str(int(round(f))))
            src_alt.add(f"{f:.1f}".rstrip("0").rstrip("."))
            for scale in (1e3, 1e6, 1e9):
                if f >= scale:
                    v = f / scale
                    src_alt.add(f"{v:.1f}".rstrip("0").rstrip("."))
                    src_alt.add(str(int(round(v))))
                    src_alt.add(str(int(round(v) * scale)))
        except ValueError:
            pass
    bad = []
    for n in _number_mentions(script_text):
        try:
            if float(n) <= 12:
                continue
        except ValueError:
            continue
        if n not in src_alt:
            bad.append(n)
    return sorted(bad, key=lambda n: (float(n), n))
