"""Grounding for trends that have no Wikipedia article.

Most things that actually trend — a product launch, a sudden controversy, a meme, a game patch —
will never have an encyclopedia article, so `research.ground()` used to discard every one of them.
This module is the fallback: it establishes WHAT happened from the open news record, and refuses to
state anything that only a single outlet is claiming.

A deliberate limitation drives the design. Google News RSS returns a headline, an outlet and a date
— its <description> is just the headline wrapped in a link, not a snippet — and fetching and
reproducing publishers' article bodies would create copyright and originality concerns and would
not meet AutoTube's value-add goal. So headlines are used for a narrow purpose: evidence that an event
is receiving current coverage, not as a substitute for original scripting. The depth
of the script has to come from somewhere safer, which is why `backbone()` hands back a related
encyclopedic subject (Sora 2 → OpenAI, text-to-video) for the explainer half of the video.

Everything here is keyless and fails soft.
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

from collections import Counter
from urllib.parse import quote, urlparse

from .common import http, now_utc

log = logging.getLogger("autotube.news")

GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl={lang}-{region}&gl={region}&ceid={region}:{lang}"
GDELT = ("https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist"
         "&maxrecords=40&format=json&sort=hybridrel")

# Outlets whose "coverage" is aggregation or syndication of someone else's reporting. Counting them
# as independent corroboration would let one story masquerade as a consensus of five.
AGGREGATORS = {
    "news.google.com", "msn.com", "yahoo.com", "news.yahoo.com", "flipboard.com", "smartnews.com",
    "newsbreak.com", "biztoc.com", "headtopics.com", "newslookup.com", "inshorts.com", "reddit.com",
}

STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "at", "by", "from",
    "as", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "new", "now", "how", "why", "what", "who", "when", "will", "can", "says", "said", "after",
    "before", "over", "into", "about", "more", "most", "best", "top", "first", "last", "has", "have",
    "not", "you", "your", "his", "her", "their", "our", "out", "up", "down", "off", "than", "then",
    "news", "report", "reports", "update", "updates", "video", "watch", "live", "here", "all",
}


def _outlet(name: str = "", url: str = "") -> str:
    """A comparable publisher identity. Falls back to the registered domain."""
    host = (urlparse(url).netloc or "").lower().lstrip("www.")
    if name and name.strip():
        return name.strip()
    return host


def _domain(url: str) -> str:
    return (urlparse(url).netloc or "").lower().removeprefix("www.")


def _clean(title: str, outlet: str = "") -> str:
    """Strip the ' - Outlet Name' suffix Google News appends to every headline.

    The outlet is passed in and removed by name because a generic 'last dash onwards' rule fails on
    outlets that contain a hyphen ('tech-insider.org') and mangles headlines that legitimately end
    in a dashed clause.
    """
    orig = (title or "").strip()
    t = orig
    if outlet:
        t = re.sub(r"\s*[-–—|]\s*" + re.escape(outlet.strip()) + r"\s*$", "", t, flags=re.I).strip()
    t = re.sub(r"\s+[-–—|]\s+[\w .']{2,30}$", "", t).strip()
    # a headline that is *only* an outlet name must not be reduced to nothing
    return t or orig


def google_news(subject: str, lang: str = "en", region: str = "US", limit: int = 40) -> list[dict]:
    try:
        r = http().get(GOOGLE_NEWS.format(q=quote(subject), lang=lang, region=region), timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        out = []
        for it in root.findall(".//item")[:limit]:
            title = (it.findtext("title") or "").strip()
            src = it.find("source")
            oname = (src.text if src is not None else "") or ""
            out.append({
                "title": _clean(title, oname),
                "outlet": _outlet((src.text if src is not None else ""), (src.get("url") if src is not None else "") or ""),
                "url": (it.findtext("link") or "").strip(),
                "published": (it.findtext("pubDate") or "").strip(),
                "via": "google_news",
            })
        return [a for a in out if a["title"] and a["outlet"]]
    except Exception as e:  # noqa: BLE001
        log.debug("google news failed for %r: %s", subject, str(e)[:120])
        return []


def gdelt(subject: str, limit: int = 40) -> list[dict]:
    """Secondary corroboration source. Frequently rate-limits (429); that is fine, it is optional."""
    try:
        # single attempt, no retry: GDELT rate-limits constantly and it is only a bonus source
        r = http().get(GDELT.format(q=quote(f'"{subject}"')), timeout=15)
        if r.status_code != 200:
            return []
        d = r.json()
        return [{
            "title": _clean(a.get("title") or ""),
            "outlet": _outlet("", a.get("url") or "") or (a.get("domain") or ""),
            "url": a.get("url") or "",
            "published": a.get("seendate") or "",
            "via": "gdelt",
        } for a in (d.get("articles") or [])[:limit]]
    except Exception as e:  # noqa: BLE001
        log.debug("gdelt failed for %r: %s", subject, str(e)[:120])
        return []


def independent(articles: list[dict]) -> list[dict]:
    """One article per publisher, aggregators removed.

    Five syndications of one wire story are one source, not five, and treating them as five is how
    an automated pipeline convinces itself a rumour is established fact.
    """
    seen: set[str] = set()
    out = []
    for a in articles:
        # Google News link URLs are always news.google.com redirects, so the URL says nothing about
        # who published the story — only the <source> element does. Judging those by domain would
        # classify the entire feed as an aggregator and throw all of it away.
        dom = "" if a.get("via") == "google_news" else _domain(a.get("url", ""))
        name = (a.get("outlet") or dom).strip().lower()
        if not name or name in seen:
            continue
        if dom in AGGREGATORS or name in AGGREGATORS or name.replace(" ", "") + ".com" in AGGREGATORS:
            continue
        seen.add(name)
        out.append(a)
    return out


def consensus(articles: list[dict], min_outlets: int = 2) -> list[str]:
    """Phrases that several independent outlets are using about this story.

    This is the only thing headlines can honestly support: not detail, but agreement. A phrase
    carried by one outlet is that outlet's claim; a phrase carried by four is the story.
    """
    counts: Counter = Counter()
    for a in articles:
        words = re.findall(r"[A-Za-z0-9][\w'’-]*", a.get("title", "").lower())
        grams = set()
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                g = words[i:i + n]
                if g[0] in STOP or g[-1] in STOP:
                    continue
                if all(len(w) < 3 for w in g):
                    continue
                grams.add(" ".join(g))
        counts.update(grams)          # set() per article: one outlet cannot vote twice
    return [g for g, n in counts.most_common(40) if n >= min_outlets]


def _title_cased(title: str) -> bool:
    """Is this headline written in Title Case (where capitalisation carries no information)?"""
    # The first word is capitalised in every headline ever written, so it carries no signal and is
    # dropped before measuring. Without that, any short headline containing two proper nouns
    # ("Users mourn OpenAI and Sora") looks like Title Case and gets thrown out of the evidence.
    words = re.findall(r"[A-Za-z][\w'’-]*", title or "")[1:]
    words = [w for w in words if len(w) >= 4]
    if len(words) < 3:
        return False
    return sum(1 for w in words if w[:1].isupper()) / len(words) >= 0.7


def backbone(subject: str, phrases: list[str], articles: list[dict]) -> list[str]:
    """Encyclopedic subjects to carry the explainer half of the video.

    "Sora 2" has no article, but "OpenAI" and "Disney" do. The news half says what just happened;
    this is where the background that makes it a *video* rather than a headline read comes from.

    A phrase qualifies only if independent outlets write it capitalised AWAY from the start of a
    sentence. That is a cheap and surprisingly reliable proper-noun test, and the mid-sentence part
    matters: judging on capitalisation alone promotes any word that happens to open a headline
    ("Slop", "Farewell"), which sends us looking up encyclopedia articles that do not exist.
    Surface forms are read from the real titles rather than reconstructed, because title-casing a
    lowercased phrase turns "openai" into "Openai" and silently discards the main entity.
    """
    # Title-cased headlines capitalise every word, so they are evidence of nothing and would
    # promote "Slop" and "Dead" alongside "Disney". Drop them from the evidence pool entirely.
    titles = [t for t in (a.get("title", "") for a in articles) if not _title_cased(t)]
    scored: list[tuple[int, str]] = []
    for p in phrases:
        if len(p) < 4 or p.lower() == subject.lower() or "'" in p or "\u2019" in p:
            continue
        forms: dict[str, int] = {}
        good = total = 0
        for t in titles:
            for m in re.finditer(r"\b" + re.escape(p).replace(r"\ ", r"\s+") + r"\b", t, re.I):
                surface = m.group(0)
                # An occurrence that opens the headline or follows a full stop is capitalised for
                # grammatical reasons and says nothing either way, so it is neutral evidence rather
                # than evidence against — otherwise entities that habitually lead a headline
                # ("OpenAI pulls the plug on...") get penalised for being the subject of the story.
                if m.start() == 0 or t[max(0, m.start() - 2):m.start()].strip() in (".", "!", "?", ":"):
                    continue
                total += 1
                if surface[:1].isupper():
                    good += 1
                    forms[surface] = forms.get(surface, 0) + 1
        # One mid-sentence capitalised use is enough. Demanding two loses entities that almost
        # always lead their own headline ("OpenAI pulls the plug on..."), which tend to be exactly
        # the subject of the story; noise is already held down by the consensus threshold and by
        # discarding title-cased headlines.
        if total >= 1 and good / total >= 0.6 and forms:
            scored.append((good, max(forms, key=forms.get)))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, form in scored:
        if form.lower() not in seen:
            seen.add(form.lower())
            out.append(form)
    return out[:6]


def brief(subject: str, lang: str = "en", region: str = "US", min_outlets: int = 2) -> dict | None:
    """What independent newsrooms agree is happening with `subject`, or None if too few agree."""
    arts = independent(google_news(subject, lang, region) + gdelt(subject))
    if len(arts) < min_outlets:
        log.info("news: %r carried by only %d independent outlet(s), need %d — skipping",
                 subject, len(arts), min_outlets)
        return None
    phrases = consensus(arts, min_outlets)
    if not phrases:
        log.info("news: %r has %d outlets but no agreed framing — skipping", subject, len(arts))
        return None
    lines = [f'{a["outlet"]}: "{a["title"]}"' for a in arts[:12]]
    text = (f"Recent coverage of {subject} from {len(arts)} independent outlets "
            f"(headlines only; each is that outlet's own wording):\n" + "\n".join(lines))
    log.info("news: %r corroborated by %d outlets; agreed framing: %s",
             subject, len(arts), ", ".join(phrases[:5]))
    return {
        "subject": subject,
        "outlets": [a["outlet"] for a in arts],
        "n_outlets": len(arts),
        "consensus": phrases,
        "backbone": backbone(subject, phrases, arts),
        "text": text,
        "also": [{"title": f'{a["outlet"]} — {a["title"]}', "url": a["url"], "publisher": a["outlet"]}
                 for a in arts[:8]],
        "fetched": now_utc().isoformat(timespec="seconds"),
    }
