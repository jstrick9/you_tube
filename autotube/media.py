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
        out.append({"url": r["url"], "thumb": r.get("thumbnail"), "page": r.get("foreign_landing_url"),
                    "title": r.get("title", "")[:80],
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
    return [{"url": p["src"]["large2x"], "thumb": p["src"].get("medium"), "page": p["url"], "title": p.get("alt") or query,
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
    raw = small.tobytes()
    px = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
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
        if len(r.content) > 40_000_000:
            return None
        tmp = dest_dir / f"tmp_{h}"
        tmp.write_bytes(r.content)
        with Image.open(tmp) as im:
            if im.width * im.height > 80_000_000:
                raise ValueError(f"image too large {im.size}")
            im.draft("RGB", (2400, 2400))
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


def _candidates_for(queries: list[str], sources: list[str], allowed: list[str], min_w: int) -> list[dict]:
    found: list[dict] = []
    for qi, q in enumerate(queries[:3]):
        if "pexels" in sources:
            found += pexels_search(q, 6)
        if "wikimedia" in sources:
            found += commons_search(q, allowed, min_w, 10)
        if "openverse" in sources and qi < 2:
            found += openverse_search(q, min_w, 8)
    return found


def _usable(a: dict, work: Path) -> Path | None:
    p = _download(a, work)
    if p:
        with Image.open(p) as im:
            if min(im.size) < 480:
                return None
    return p


def gather(source: dict, segments: list[dict], cfg: dict, work: Path, llm=None,
           seg_durs: list[float] | None = None) -> list[dict]:
    """Pick visually VERIFIED shots for every segment.

    Returns shots in order: [{'path','credit','seg','score','shows','judge','want'}]. A segment may get 2 shots
    when it is long. Every image has been looked at (vision LLM, or CLIP as fallback) and scored against that
    segment's narration; nothing below `media.min_match_score` is ever used.
    """
    from . import vision

    mcfg = cfg["media"]
    allowed = cfg["compliance"]["allowed_licenses"]
    min_w = mcfg["min_image_width"]
    sources = mcfg["sources"]
    min_score = float(mcfg.get("min_match_score", 7))
    multi_after = float(mcfg.get("multi_shot_seconds", 4.5))
    work.mkdir(parents=True, exist_ok=True)

    judge = vision.Judge(cfg, llm)
    if not judge.available():
        if mcfg.get("require_vision_check", True):
            raise RuntimeError("no vision judge available (need GEMINI_API_KEY or CLIP) — refusing unverified images")
        log.warning("no vision judge — falling back to unverified title matching")

    subject = re.sub(r"\s*\(.*?\)", "", source["title"])
    context = source.get("description", "")
    by_url: dict[str, dict] = {}

    def canon(items):
        out = []
        for a in items:
            if a["url"] not in by_url:
                by_url[a["url"]] = a
            out.append(by_url[a["url"]])
        return list({id(a): a for a in out}.values())

    article = canon(_commons_info(source.get("images", [])[:20], allowed, min_w)) if "wikimedia" in sources else []

    def score(cands, want, narration):
        cands = vision.fetch_thumbs(cands)                     # thumbs cached on the shared originals
        copies = [{k: v for k, v in c.items() if k not in ("vscore", "vshows", "vjudge", "clip", "clip_cos")}
                  for c in cands]
        return judge.score(copies, want, narration, subject, context)

    used_urls: set[str] = set()
    used_hashes: list[int] = []
    shots: list[dict] = []

    def take(c, i, want) -> bool:
        if c["url"] in used_urls or any(vision.same_image(c["_hash"], h) for h in used_hashes):
            return False
        p = _usable(c, work)
        if not p:
            return False
        used_urls.add(c["url"])
        used_hashes.append(c["_hash"])
        credit = {k: v for k, v in c.items() if not k.startswith("_") and k not in
                  ("vscore", "vshows", "vjudge", "clip", "clip_cos", "thumb")}
        shots.append({"path": p, "credit": credit, "seg": i, "score": c["vscore"], "shows": c.get("vshows", ""),
                      "judge": c.get("vjudge", ""), "want": want})
        log.info("    seg %d ← %.1f/10 [%s] %s  (%s)", i, c["vscore"], c.get("vjudge"), c.get("vshows", "")[:70],
                 c["title"][:50])
        return True

    subject_pool: list[dict] | None = None

    def subject_level() -> list[dict]:
        nonlocal subject_pool
        if subject_pool is None:
            q = f'"{subject}"' if len(subject.split()) > 1 else subject
            extra = canon(commons_search(q, allowed, min_w, 15) + openverse_search(subject, min_w, 10))
            subject_pool = score(article + extra, f"a clear, recognisable photo of {subject}",
                                 f"This video is about {subject}. {context}")
        return subject_pool

    for i, seg in enumerate(segments):
        vis = seg.get("visual") if isinstance(seg.get("visual"), dict) else {}
        kws = [k for k in (seg.get("keywords") or []) if isinstance(k, str)]
        want = (vis.get("shows") or "").strip() or (f"{subject}: " + ", ".join(kws) if kws else f"{subject}")
        queries = [q for q in (vis.get("queries") or []) if isinstance(q, str) and q.strip()]
        if not queries:
            queries = [f"{subject} {k}" for k in kws] or [subject]
        cands = canon(_candidates_for(queries, sources, allowed, min_w)) + article
        judged = score(list({id(c): c for c in cands}.values()), want, seg["text"])
        good = [c for c in judged if c["vscore"] >= min_score]
        want_n = 2 if seg_durs and i < len(seg_durs) and seg_durs[i] > multi_after else 1
        got = 0
        for c in good:
            if got >= want_n:
                break
            got += take(c, i, want)
        if not got:                          # nothing fits this exact line → a verified photo of the subject
            for c in subject_level():
                if c["vscore"] >= min_score and take(c, i, f"a clear photo of {subject}"):
                    got = 1
                    break
        if not got:
            log.info("    seg %d: no verified match yet (best %s) — will reuse a verified shot", i,
                     f"{judged[0]['vscore']:.1f}" if judged else "n/a")

    if not shots:
        raise RuntimeError(f"no images passed the visual relevance check for {subject!r}")
    distinct = len({s["path"] for s in shots})
    need = int(mcfg.get("min_distinct_images", 3))
    if distinct < need:
        raise RuntimeError(f"only {distinct} verified images for {subject!r} (need {need}) — skipping topic")

    # segments without their own shot reuse the best verified shots (renderer varies the motion)
    covered = {s["seg"] for s in shots}
    best = sorted({s["path"]: s for s in shots}.values(), key=lambda s: -s["score"])
    k = 0
    for i in range(len(segments)):
        if i not in covered:
            src = best[k % len(best)]
            k += 1
            shots.append({**src, "seg": i, "reused": True})
    shots.sort(key=lambda s: (s["seg"], s.get("reused", False)))
    log.info("  visuals: %d shots, %d distinct images, %d vision-LLM calls, judge=%s", len(shots), distinct,
             judge.calls, "llm" if judge.calls and not judge.llm_failed else "clip")
    return shots


def credits_text(visuals: list[dict]) -> str:
    seen, lines = set(), []
    for v in visuals:
        c = v["credit"]
        if c["url"] in seen:
            continue
        seen.add(c["url"])
        lines.append(f"• \"{c['title'][:60]}\" by {c['author']} — {c['license']} ({c['source']}) {c.get('page') or ''}".strip())
    return "\n".join(lines)
