"""Licensed visuals sourcing with full attribution tracking.

Sources (all free, commercially usable):
  • Wikimedia Commons – the grounding article's own images first (most on-topic),
                         then keyword search. Only PD / CC0 / CC-BY / CC-BY-SA accepted.
  • Openverse         – CC-licensed image search (keyless).
  • Pexels            – stock photos/videos (free key; Pexels license, no attribution required
                         but we credit anyway).
Every asset's author + license + URL is recorded and printed in the video
description → CC-BY compliance and a clean copyright trail.
"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import re
from pathlib import Path

from PIL import Image

from .common import get_json, http

log = logging.getLogger("autotube.media")

COMMONS = "https://commons.wikimedia.org/w/api.php"


def _license_ok(lic: str, allowed: list[str]) -> bool:
    lic = (lic or "").lower().replace("-", " ")
    if not lic:
        return False
    if any(x in lic for x in ("nc", "nd", "non commercial", "fair use", "copyrighted", "all rights")):
        # NC/ND not allowed for monetizable, transformed content
        if not ("public domain" in lic or lic.startswith("pd")):
            return False
    allowed_n = [a.lower().replace("-", " ") for a in allowed]
    return any(a in lic for a in allowed_n) or lic.startswith("pd") or "public domain" in lic or lic.startswith("cc0")


def _strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()[:80]


def _commons_info(titles: list[str], allowed: list[str], min_w: int) -> list[dict]:
    if not titles:
        return []
    out = []
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        try:
            data = get_json(COMMONS, params={
                "action": "query", "titles": "|".join(batch), "prop": "imageinfo",
                "iiprop": "url|extmetadata|size|mime", "iiurlwidth": 1600, "format": "json"})
        except Exception as e:  # noqa: BLE001
            log.debug("commons info failed: %s", e)
            continue
        for p in data.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if not ii or ii.get("mime") not in ("image/jpeg", "image/png", "image/webp", "image/tiff"):
                continue
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if ii.get("width", 0) < min_w or not _license_ok(lic, allowed):
                continue
            out.append({
                "url": ii.get("thumburl") or ii["url"], "page": ii.get("descriptionurl"),
                "title": p["title"].replace("File:", ""), "license": lic,
                "author": _strip_html(meta.get("Artist", {}).get("value", "")) or "Wikimedia Commons",
                "source": "Wikimedia Commons", "w": ii.get("width"), "h": ii.get("height"),
            })
    return out


def commons_search(query: str, allowed: list[str], min_w: int, limit: int = 12) -> list[dict]:
    try:
        data = get_json(COMMONS, params={
            "action": "query", "generator": "search", "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime", "iiurlwidth": 1600, "format": "json"})
    except Exception as e:  # noqa: BLE001
        log.debug("commons search failed: %s", e)
        return []
    titles = [p["title"] for p in data.get("query", {}).get("pages", {}).values()]
    return _commons_info(titles, allowed, min_w)


def openverse_search(query: str, min_w: int, limit: int = 10) -> list[dict]:
    try:
        data = get_json("https://api.openverse.org/v1/images/", params={
            "q": query, "license": "cc0,pdm,by,by-sa", "page_size": limit, "mature": "false"})
    except Exception as e:  # noqa: BLE001
        log.debug("openverse failed: %s", e)
        return []
    out = []
    for r in data.get("results", []):
        if (r.get("width") or 0) < min_w:
            continue
        out.append({"url": r["url"], "page": r.get("foreign_landing_url"), "title": r.get("title", "")[:80],
                    "license": f"CC {r.get('license', '').upper()} {r.get('license_version', '')}".strip(),
                    "author": r.get("creator") or "Unknown", "source": "Openverse/" + (r.get("provider") or ""),
                    "w": r.get("width"), "h": r.get("height")})
    return out


def pexels_search(query: str, limit: int = 8) -> list[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    try:
        data = get_json("https://api.pexels.com/v1/search", params={
            "query": query, "per_page": limit, "orientation": "portrait"}, headers={"Authorization": key})
    except Exception as e:  # noqa: BLE001
        log.debug("pexels failed: %s", e)
        return []
    return [{"url": p["src"]["large2x"], "page": p["url"], "title": p.get("alt") or query,
             "license": "Pexels License", "author": p.get("photographer", "Pexels"), "source": "Pexels",
             "w": p["width"], "h": p["height"]} for p in data.get("photos", [])]


STOP = set("the a an of and in on for to with from by at is are was were its it this that as or de la le".split())


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in STOP}


DOC_WORDS = {"screenshot", "document", "register", "federal", "pdf", "page", "report", "chart", "graph",
             "diagram", "table", "establishment", "nonessential", "letter", "form", "certificate", "scan",
             "text", "title", "cover", "brochure", "leaflet", "infographic", "slide", "website",
             "icon", "icons", "clipart", "clip", "svg", "vector", "pictogram", "emoji", "silhouette", "cartoon"}
PLACE_WORDS = {"wharf", "street", "station", "stadium", "hotel", "building", "tower", "bridge", "road",
               "avenue", "square", "city", "town", "village", "airport", "mall", "church", "school",
               "district", "skyline", "london", "downtown", "harbour", "harbor", "sign", "logo", "poster"}


def relevant(asset: dict, subject: str, keyword: str = "", context: str = "") -> bool:
    """Title-based relevance: must match the subject's key words (or the keyword + subject),
    and must not look like a same-name place/building unless the subject is one."""
    t = _tokens(asset.get("title", ""))
    subj = _tokens(subject)
    kw = _tokens(keyword)
    ctx = _tokens(context)
    if (t & PLACE_WORDS) and not ((subj | kw | ctx) & PLACE_WORDS):
        return False
    if (t & DOC_WORDS) and not ((subj | kw) & DOC_WORDS):
        return False
    if not subj:
        return bool(t & kw)
    head = sorted(subj, key=lambda w: subject.lower().rfind(w))[-1]   # last word ≈ head noun
    if head not in t:
        return len(subj) > 1 and len(t & subj) >= 2
    if len(subj) == 1:
        return True
    return bool(t & ((subj | ctx | kw) - {head}))


def _looks_like_document(im: Image.Image) -> bool:
    """Reject screenshots/scans/text pages: lots of flat near-white area + low colour variance."""
    small = im.resize((96, 96)).convert("RGB")
    px = list(small.getdata())
    white = sum(1 for r, g, b in px if r > 225 and g > 225 and b > 225) / len(px)
    flat = sum(1 for r, g, b in px if max(r, g, b) - min(r, g, b) < 18) / len(px)
    colours = len(set((r // 16, g // 16, b // 16) for r, g, b in px))
    return (white > 0.38 and flat > 0.55) or colours < 40      # text pages / flat clip-art


def _download(asset: dict, dest_dir: Path) -> Path | None:
    h = hashlib.md5(asset["url"].encode()).hexdigest()[:12]
    dest = dest_dir / f"img_{h}.jpg"
    if dest.exists():
        return dest
    try:
        r = http().get(asset["url"], timeout=40)
        r.raise_for_status()
        tmp = dest_dir / f"tmp_{h}"
        tmp.write_bytes(r.content)
        with Image.open(tmp) as im:
            im = im.convert("RGB")
            if min(im.size) < 400 or _looks_like_document(im):
                tmp.unlink(missing_ok=True)
                return None
            # cap size for speed
            im.thumbnail((2400, 2400))
            im.save(dest, "JPEG", quality=90)
        tmp.unlink(missing_ok=True)
        return dest
    except Exception as e:  # noqa: BLE001
        log.debug("download failed %s: %s", asset["url"][:80], e)
        return None


def gather(source: dict, segments: list[dict], cfg: dict, work: Path) -> list[dict]:
    """Return one visual per segment: [{'path','credit'...}], reusing images if needed."""
    allowed = cfg["compliance"]["allowed_licenses"]
    min_w = cfg["media"]["min_image_width"]
    sources = cfg["media"]["sources"]
    work.mkdir(parents=True, exist_ok=True)

    pool: list[dict] = []
    seen: set[str] = set()

    sigs: list[set[str]] = []

    def add(items):
        for a in items:
            if a["url"] in seen:
                continue
            sig = _tokens(a.get("title", "")) - {"jpg", "jpeg", "png", "tif", "tiff", "webp", "file"}
            # same photo mirrored on several sites (Commons + Flickr) → near-identical titles
            if sig and any(len(sig & o) / max(1, len(sig | o)) >= 0.75 for o in sigs):
                continue
            seen.add(a["url"])
            sigs.append(sig)
            pool.append(a)

    # 1) article images (most relevant)
    if "wikimedia" in sources:
        add(_commons_info(source.get("images", [])[:20], allowed, min_w))

    # 2) per-segment keywords — searched WITH the subject for relevance; results filtered
    subject = re.sub(r"\s*\(.*?\)", "", source["title"])
    context = source.get("description", "")
    per_seg: list[list[dict]] = []
    for seg in segments:
        found = []
        for kw in (seg.get("keywords") or [])[:2]:
            if "pexels" in sources:
                found += pexels_search(f"{kw}", 4)          # stock: generic b-roll is acceptable
            if "wikimedia" in sources and len(found) < 3:
                q = f'"{subject}" {kw}' if len(subject.split()) > 1 else f"{subject} {kw}"
                found += [a for a in commons_search(q, allowed, min_w, 8) if relevant(a, subject, kw, context)]
            if "openverse" in sources and len(found) < 2:
                q = f"{subject} {kw}"
                found += [a for a in openverse_search(q, min_w, 6) if relevant(a, subject, kw, context)]
        add(found)
        per_seg.append([a for a in found if a in pool])

    # 3) generic subject search as a safety net
    if len(pool) < len(segments):
        add([a for a in commons_search(f'"{subject}"', allowed, min_w, 20) if relevant(a, subject, "", context)])
    if len(pool) < len(segments):
        add([a for a in openverse_search(subject, min_w, 12) if relevant(a, subject, "", context)])

    # assign: prefer segment-specific, then article pool; download lazily
    visuals, used = [], set()
    article_pool = [a for a in pool]
    for i, seg in enumerate(segments):
        choice_path, choice = None, None
        candidates = per_seg[i] + article_pool
        if i == 0:  # hook: article lead image is usually the most recognizable
            candidates = article_pool[:3] + per_seg[i] + article_pool
        for a in candidates:
            if a["url"] in used:
                continue
            p = _download(a, work)
            if p:
                choice_path, choice = p, a
                used.add(a["url"])
                break
        if not choice_path and visuals:     # reuse with different motion
            prev = visuals[i % len(visuals)]
            choice_path, choice = prev["path"], prev["credit"]
        if choice_path:
            visuals.append({"path": choice_path, "credit": choice})
    if not visuals:
        raise RuntimeError("no licensed visuals found")
    while len(visuals) < len(segments):
        visuals.append(visuals[len(visuals) % max(1, len(visuals))])
    return visuals


def credits_text(visuals: list[dict]) -> str:
    seen, lines = set(), []
    for v in visuals:
        c = v["credit"]
        if c["url"] in seen:
            continue
        seen.add(c["url"])
        lines.append(f"• \"{c['title'][:60]}\" by {c['author']} — {c['license']} ({c['source']}) {c.get('page') or ''}".strip())
    return "\n".join(lines)
