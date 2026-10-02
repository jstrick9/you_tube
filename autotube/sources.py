"""Additional grounding corpora beyond Wikipedia — all keyless, all free, all quotable prose.

Why: `require_grounding` + Wikipedia-only is excellent for accuracy and a hard ceiling on virality.
The most shareable facts are often ones Wikipedia covers in a single dry clause, or not at all, so
the topic funnel kept rejecting picks until only the safe, flat ones survived (hence the three
selection rounds). We were optimising for *provable* at the cost of *shareable*.

The fix is not to loosen verification — it is to widen what we can verify against. Every corpus here
returns **verbatim prose from a high-credibility publisher**, so the existing machinery is untouched:
evidence quotes are still matched against the source text, numbers are still checked against it, and
every passage is labelled and credited in the description.

Corpora (chosen because they are keyless, stable, and write in plain prose):
  • NASA Image & Video Library — public domain, rich descriptive captions on space/earth science
  • arXiv                      — abstracts across physics, astronomy, CS, biology
  • PubMed / Europe PMC        — abstracts across medicine and life sciences
  • Wikidata                   — structured facts rendered as sentences (dates, sizes, records)

Everything fails soft: a corpus that is slow, rate-limited or down simply contributes nothing.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

from .common import get_json, http

log = logging.getLogger("autotube.sources")

MIN_PASSAGE = 120          # ignore fragments too short to support a claim
MAX_PER_SOURCE = 2         # keep the prompt focused; 2 good passages beat 6 thin ones


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _mentions(text: str, subject: str) -> list[int]:
    """Positions where `subject` is named. Multi-word subjects must match as a phrase."""
    low = (text or "").lower()
    subj = re.sub(r"\s*\(.*?\)", "", subject or "").strip().lower()
    if not subj:
        return []
    words = subj.split()
    if len(words) > 1:
        pat = (r"\s+".join(re.escape(w) for w in words[:-1]) + r"\s+"
               + re.escape(words[-1].rstrip("s")) + r"(s|'s)?\b")
    else:
        pat = rf"\b{re.escape(words[0].rstrip('s'))}(s|es|'s)?\b"
    return [m.start() for m in re.finditer(pat, low)]


def _relevant(title: str, text: str, subject: str) -> bool:
    """Is this passage ABOUT the subject, or does it merely mention it in passing?

    The distinction matters more than it looks. Searching NASA for "whooping crane" returns captions
    about *sandhill* cranes that happen to say "like their endangered relatives the whooping cranes,
    sandhills live to be older than most birds... some sandhill cranes live up to 20 years." Appended
    to the source text, that becomes quotable evidence that whooping cranes live 20 years — and
    because the figure genuinely appears in the source, the number check waves it straight through.

    So a passage qualifies only if the subject is its actual topic: named in the title, or named
    early AND more than once.
    """
    if _mentions(title, subject):
        return True
    hits = _mentions(text, subject)
    return len(hits) >= 2 and hits[0] < 300


# ── NASA Image & Video Library (public domain, keyless) ───────────────────────
def nasa(subject: str, limit: int = 6) -> list[dict]:
    out = []
    try:
        data = get_json("https://images-api.nasa.gov/search",
                        params={"q": subject, "media_type": "image", "page_size": limit})
        for item in (data.get("collection", {}).get("items") or [])[:limit]:
            d = (item.get("data") or [{}])[0]
            text = _clean(d.get("description") or d.get("description_508") or "")
            title = _clean(d.get("title") or "")
            if len(text) < MIN_PASSAGE or not _relevant(title, text, subject):
                continue
            nid = d.get("nasa_id", "")
            out.append({
                "kind": "nasa", "title": title or "NASA",
                "url": f"https://images.nasa.gov/details/{nid}" if nid else "https://images.nasa.gov",
                "text": text[:1500], "publisher": "NASA",
            })
            if len(out) >= MAX_PER_SOURCE:
                break
    except Exception as e:  # noqa: BLE001
        log.debug("nasa lookup failed for %r: %s", subject, str(e)[:120])
    return out


# ── arXiv abstracts (keyless Atom API) ────────────────────────────────────────
_ATOM = "{http://www.w3.org/2005/Atom}"


def arxiv(subject: str, limit: int = 5) -> list[dict]:
    out = []
    try:
        r = http().get("https://export.arxiv.org/api/query",
                       params={"search_query": f'all:"{subject}"', "max_results": limit,
                               "sortBy": "relevance"}, timeout=25)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for entry in root.findall(f"{_ATOM}entry")[:limit]:
            title = _clean((entry.findtext(f"{_ATOM}title") or ""))
            text = _clean(entry.findtext(f"{_ATOM}summary") or "")
            link = entry.findtext(f"{_ATOM}id") or ""
            if len(text) < MIN_PASSAGE or not _relevant(title, text, subject):
                continue
            out.append({"kind": "arxiv", "title": title[:160], "url": link,
                        "text": text[:1500], "publisher": "arXiv"})
            if len(out) >= MAX_PER_SOURCE:
                break
    except Exception as e:  # noqa: BLE001
        log.debug("arxiv lookup failed for %r: %s", subject, str(e)[:120])
    return out


# ── Europe PMC (PubMed abstracts, keyless) ────────────────────────────────────
def pubmed(subject: str, limit: int = 5) -> list[dict]:
    out = []
    try:
        data = get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                        params={"query": f'"{subject}" AND (HAS_ABSTRACT:Y)', "format": "json",
                                "pageSize": limit, "resultType": "core"})
        for res in (data.get("resultList", {}).get("result") or [])[:limit]:
            text = _clean(res.get("abstractText") or "")
            title = _clean(res.get("title") or "")
            if len(text) < MIN_PASSAGE or not _relevant(title, text, subject):
                continue
            pmid, doi = res.get("pmid"), res.get("doi")
            url = (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid
                   else f"https://doi.org/{doi}" if doi else "https://europepmc.org")
            out.append({"kind": "pubmed", "title": title[:160], "url": url,
                        "text": text[:1500], "publisher": res.get("journalTitle") or "Europe PMC"})
            if len(out) >= MAX_PER_SOURCE:
                break
    except Exception as e:  # noqa: BLE001
        log.debug("pubmed lookup failed for %r: %s", subject, str(e)[:120])
    return out


# ── Wikidata (structured facts rendered as sentences) ─────────────────────────
WD_PROPS = {
    "P2044": "elevation above sea level", "P2048": "height", "P2049": "width", "P2067": "mass",
    "P2046": "area", "P1082": "population", "P571": "inception date", "P576": "dissolved or abolished",
    "P2043": "length", "P2386": "diameter", "P2050": "wingspan", "P2047": "duration",
    "P1083": "maximum capacity", "P2052": "speed", "P2120": "radius", "P2102": "depth",
}


_UNIT_CACHE: dict[str, str] = {"1": ""}       # Q-id → English label ("1" = dimensionless count)


def _unit_labels(qids: list[str]) -> dict[str, str]:
    """Resolve unit Q-ids to words. A bare number with no unit is worse than no number at all:
    'the height of Olympus Mons is 21.9' invites the script to invent the unit, and because the
    figure really is in the source text our number check would happily wave it through."""
    todo = [q for q in dict.fromkeys(qids) if q and q not in _UNIT_CACHE]
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        try:
            ent = get_json("https://www.wikidata.org/w/api.php",
                           params={"action": "wbgetentities", "ids": "|".join(batch),
                                   "format": "json", "props": "labels", "languages": "en"})
            for q, e in (ent.get("entities") or {}).items():
                _UNIT_CACHE[q] = ((e.get("labels") or {}).get("en") or {}).get("value") or ""
        except Exception as e:  # noqa: BLE001
            log.debug("unit label lookup failed: %s", str(e)[:100])
            for q in batch:
                _UNIT_CACHE.setdefault(q, "")
    return _UNIT_CACHE


def _year(dv: dict) -> str | None:
    """Render a Wikidata time value, or None if it cannot be stated plainly.

    Wikidata encodes deep time as e.g. '-3830000000-00-00T00:00:00Z' with precision 2 (billion
    years). Reading the first four digits off that string yields 'the year 3830' — which is how an
    automated pipeline states, with total confidence, something absurd. Anything coarser than a
    year, or outside a plausible range, is dropped rather than guessed at.
    """
    t = str(dv.get("time") or "")
    if int(dv.get("precision", 0)) < 9:            # 9 = year; coarser means decade/century/eon
        return None
    m = re.match(r"([+-])(\d+)-", t)
    if not m:
        return None
    sign, digits = m.group(1), m.group(2)
    year = int(digits)
    if year == 0 or year > 2400:
        return None
    return f"{year} BC" if sign == "-" else str(year)


def wikidata(subject: str, limit: int = 1) -> list[dict]:
    """Quantitative facts (height, mass, population, inception) as plain, UNIT-BEARING sentences.

    These are exactly the numbers a 'scale_shock' or 'by_the_numbers' script is built on, and they
    are frequently absent from the prose of the Wikipedia article even when Wikidata holds them.
    """
    try:
        s = get_json("https://www.wikidata.org/w/api.php",
                     params={"action": "wbsearchentities", "search": subject, "language": "en",
                             "format": "json", "limit": limit, "type": "item"})
        hits = s.get("search") or []
        if not hits:
            return []
        qid = hits[0]["id"]
        ent = get_json("https://www.wikidata.org/w/api.php",
                       params={"action": "wbgetentities", "ids": qid, "format": "json",
                               "props": "claims|labels", "languages": "en"})
        e = (ent.get("entities") or {}).get(qid) or {}
        label = ((e.get("labels") or {}).get("en") or {}).get("value") or subject
        claims = e.get("claims") or {}

        wanted = []
        for pid, name in WD_PROPS.items():
            for claim in claims.get(pid, [])[:1]:
                dv = (((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value"))
                if isinstance(dv, dict):
                    wanted.append((name, dv))
        units = _unit_labels([str(dv.get("unit", "")).rsplit("/", 1)[-1]
                              for _, dv in wanted if "amount" in dv])

        lines = []
        for name, dv in wanted:
            if "amount" in dv:
                amount = str(dv["amount"]).lstrip("+")
                uq = str(dv.get("unit", "")).rsplit("/", 1)[-1]
                unit = units.get(uq, "")
                if uq not in ("1", "") and not unit:
                    continue                       # unresolved unit → say nothing rather than a bare number
                if uq in ("1", "") and name not in ("population", "maximum capacity"):
                    continue                       # dimensionless where a dimension is expected
                lines.append(f"The {name} of {label} is {amount}{' ' + unit if unit else ''}.")
            elif "time" in dv:
                y = _year(dv)
                if y:
                    lines.append(f"The {name} of {label} is {y}.")
        if not lines:
            return []
        return [{"kind": "wikidata", "title": f"Wikidata: {label}",
                 "url": f"https://www.wikidata.org/wiki/{qid}",
                 "text": " ".join(lines)[:1200], "publisher": "Wikidata"}]
    except Exception as e:  # noqa: BLE001
        log.debug("wikidata lookup failed for %r: %s", subject, str(e)[:120])
    return []


FETCHERS = {"nasa": nasa, "arxiv": arxiv, "pubmed": pubmed, "wikidata": wikidata}


def enrich(subject: str, enabled: list[str], max_chars: int = 4000) -> tuple[str, list[dict]]:
    """Labelled passages about `subject` from the enabled corpora.

    Returns (text_block, credits). The block is appended to the Wikipedia text with each passage
    attributed inline, so the writer can quote from it and every existing check still applies.
    """
    parts, credits, total = [], [], 0
    for name in enabled:
        fn = FETCHERS.get(name)
        if not fn:
            continue
        for item in fn(subject):
            block = f'From {item["publisher"]} ("{item["title"]}"): {item["text"]}'
            if total + len(block) > max_chars:
                continue
            parts.append(block)
            credits.append({"title": f'{item["publisher"]} — {item["title"]}', "url": item["url"],
                            "publisher": item["publisher"]})
            total += len(block)
    log.debug("enrich(%r): +%d chars, %d passages", subject, total, len(parts))
    return "\n".join(parts), credits
