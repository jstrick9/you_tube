"""Grounding: resolve a topic to a verifiable source (Wikipedia) and fetch its text.

Every script is written ONLY from this fetched text, and the numbers in the final
script are checked against it programmatically. This is the core defense against
AI hallucination / misinformation, which YouTube's policies (and viewers) punish.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote

from . import safety, sources
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


def article_text(title: str, lang: str = "en", max_chars: int = 9000) -> str:
    data = get_json(API.format(lang=lang), params={
        "action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
        "titles": title, "format": "json", "redirects": 1})
    pages = data.get("query", {}).get("pages", {})
    text = next(iter(pages.values()), {}).get("extract", "") if pages else ""
    # drop reference-ish tail sections
    text = re.split(r"\n(See also|References|External links|Notes|Further reading|Bibliography)\n", text)[0]
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text[:max_chars]


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


def ground(topic: str, search_query: str | None, lang: str = "en", allow_living: bool = False,
           blocked: list[str] | None = None, cfg: dict | None = None, llm=None) -> dict | None:
    """Return {'title','url','summary','text','images'} or None if unusable.

    Wikipedia resolves and anchors the subject; `research.extra_sources` then widens what we can
    verify against (see autotube/sources.py). Safety screening is context-aware (autotube/safety.py).
    """
    title = None
    for q in [search_query, topic]:
        if q:
            title = search_title(q, lang)
            if title:
                break
    if not title:
        return None
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
    if not allow_living and is_living_person(summ):
        log.info("skip %r: looks like a living person (%s)", title, summ.get("description"))
        return None
    text = article_text(title, lang)
    also: list[str] = []
    extra_credits: list[dict] = []
    rcfg = ((cfg or {}).get("research") or {})
    enabled = list(rcfg.get("extra_sources") or [])
    if enabled:
        # Always reach past Wikipedia, not only when the article is thin. A long article is exactly
        # the case where a NASA caption or a peer-reviewed abstract adds something Wikipedia lacks —
        # a primary source and hard, unit-bearing numbers — and citing more than one publisher is
        # itself part of not looking mass-produced. Long articles just get a smaller budget.
        full = int(rcfg.get("enrich_max_chars", 4000))
        budget = full if len(text) < int(rcfg.get("enrich_below_chars", 7000)) else max(1200, full // 3)
        block, extra_credits = sources.enrich(title, enabled, budget)
        if block:
            log.info("  %r → +%d chars from %s", title, len(block),
                     ", ".join(sorted({c["publisher"] for c in extra_credits})))
            text = text + "\n" + block
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
    }


# ── programmatic fact check ───────────────────────────────────────────────────
_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100,
            "thousand": 1000, "million": 10**6, "billion": 10**9}


def _numbers(text: str) -> set[str]:
    nums = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        n = m.group(0).replace(",", "").rstrip(".")
        if n:
            nums.add(n.lstrip("0") or "0")
    return nums


def unsupported_numbers(script_text: str, source_text: str) -> list[str]:
    """Numbers in the script that never appear in the source. Small counts (<=12)
    are ignored because they're usually structural ('3 facts', 'part 2')."""
    src = _numbers(source_text)
    # also accept rounded forms: 4.6 → 4.6, 4,600 → 4600, "4.5 billion" etc.
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
        except ValueError:
            pass
    bad = []
    for n in _numbers(script_text):
        try:
            if float(n) <= 12:
                continue
        except ValueError:
            continue
        if n not in src_alt:
            bad.append(n)
    return bad
