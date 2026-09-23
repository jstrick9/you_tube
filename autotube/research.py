"""Grounding: resolve a topic to a verifiable source (Wikipedia) and fetch its text.

Every script is written ONLY from this fetched text, and the numbers in the final
script are checked against it programmatically. This is the core defense against
AI hallucination / misinformation, which YouTube's policies (and viewers) punish.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote

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
           blocked: list[str] | None = None) -> dict | None:
    """Return {'title','url','summary','text','images'} or None if unusable."""
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
    if blocked:
        probe = f" {title} {summ.get('description', '')} {(summ.get('extract') or '')[:400]} ".lower()
        hit = next((b for b in blocked if re.search(r"\b" + re.escape(b.lower()) + r"\b", probe)), None)
        if hit:
            log.info("skip %r: source is about a blocked subject (%s)", title, hit)
            return None
    if not allow_living and is_living_person(summ):
        log.info("skip %r: looks like a living person (%s)", title, summ.get("description"))
        return None
    text = article_text(title, lang)
    if len(text) < 1500:
        log.info("skip %r: source text too thin (%d chars)", title, len(text))
        return None
    return {
        "title": title,
        "url": f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "description": summ.get("description", ""),
        "summary": summ.get("extract", ""),
        "text": text,
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
