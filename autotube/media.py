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
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for qi, q in enumerate(queries[:3]):
        if "pexels" in sources and os.environ.get("PEXELS_API_KEY"):
            jobs.append((pexels_search, (q, 6)))
        if "wikimedia" in sources:
            jobs.append((commons_search, (q, allowed, min_w, 10)))
        if "openverse" in sources and qi < 2:
            jobs.append((openverse_search, (q, min_w, 8)))
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda j: j[0](*j[1]), jobs))
    return [a for r in results for a in r]


def _usable(a: dict, work: Path) -> Path | None:
    p = _download(a, work)
    if p:
        with Image.open(p) as im:
            if min(im.size) < 480:
                return None
    return p


class NoVisualMatch(RuntimeError):
    """A narration line has no image that verifiably shows it → the topic is skipped."""


REQUERY_SYSTEM = ("You help an educational video editor find photos in free libraries (Wikimedia Commons, Openverse). "
                  "Return strict JSON only.")


def _requery(llm, subject: str, line: str, want: str, rejected: list[str]) -> tuple[str, list[str]]:
    """Ask for a different, findable shot + searches for a line whose first searches found nothing suitable."""
    user = f"""VIDEO SUBJECT: {subject}
NARRATION LINE: "{line}"
FIRST SHOT IDEA: {want}
WHAT THE SEARCHES FOUND (all rejected as not matching): {'; '.join(rejected[:8]) or 'nothing usable'}

Suggest ONE different real, photographable shot that literally shows what this line is about (the specific subject,
its actual parts, place, specimen, artwork, or a real photo/illustration of it) and is LIKELY to exist on Wikimedia
Commons — e.g. museum specimens, historical photos, scientific illustrations, photos of the actual place/object.
Return JSON: {{"shows": "...", "queries": ["3 short searches, 2-4 words, each containing the physical noun"]}}"""
    o = llm.json(REQUERY_SYSTEM, user, temperature=0.4,
                 validate=lambda o: o["shows"] and isinstance(o["queries"], list) and o["queries"])
    return str(o["shows"]), [str(q) for q in o["queries"]][:3]


def gather(source: dict, segments: list[dict], cfg: dict, work: Path, llm=None,
           seg_durs: list[float] | None = None) -> dict:
    """Pick visually VERIFIED shots for every narration line.

    Strict rules (media.require_llm_verdict: true, the default):
      • every image is judged by a vision model against THAT line's narration; CLIP only pre-ranks;
      • every line gets its own image scoring ≥ media.min_match_score for that line (a repeat is allowed only if it
        also scored ≥ threshold for that line); no blind fill-ins;
      • a line with no match gets a second search round with new queries; still nothing → NoVisualMatch (skip topic);
      • vision model unreachable → vision.VisionUnavailable (nothing unverified is ever used).
    Returns {"shots": [...], "alts": {seg: [verified spare candidates]}, "calls": n}.
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
        raise vision.VisionUnavailable("no vision model available (need GEMINI_API_KEY) — refusing unverified images")

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

    # images about the subject itself: candidates for EVERY line (still judged per line)
    common: list[dict] = []
    if "wikimedia" in sources:
        common += _commons_info(source.get("images", [])[:20], allowed, min_w)
        common += commons_search(f'"{subject}"' if len(subject.split()) > 1 else subject, allowed, min_w, 12)
    if "openverse" in sources:
        common += openverse_search(subject, min_w, 8)
    common = canon(common)

    def judge_line(cands, want, line):
        cands = vision.fetch_thumbs(list({id(c): c for c in cands}.values()))
        fresh = lambda: [{k: v for k, v in c.items() if k not in ("vscore", "vshows", "vjudge", "clip", "clip_cos")}
                         for c in cands]
        judged = judge.score(fresh(), want, line, subject, context)
        if not any(c["vscore"] >= min_score for c in judged):          # look at the next sheet too
            judged += judge.score(fresh(), want, line, subject, context, page=1)
        return sorted(judged, key=lambda c: -c["vscore"])

    used: list[dict] = []          # {"url","hash","emb","title","seg"}
    shots: list[dict] = []
    alts: dict[int, list[dict]] = {}

    def is_used(c, allow_seg_gap: int | None = None):
        for u in used:
            dup = (c["url"] == u["url"] or vision.same_image(c["_hash"], u["hash"])
                   or vision.near_duplicate(c.get("_emb"), u["emb"], t1=c.get("title", ""), t2=u["title"]))
            if dup:
                return u
        return None

    def add_shot(c, i, want, reused=False) -> bool:
        p = _usable(c, work)
        if not p:
            return False
        if not reused:
            used.append({"url": c["url"], "hash": c["_hash"], "emb": c.get("_emb"), "title": c.get("title", ""), "seg": i})
        credit = {k: v for k, v in c.items() if not k.startswith("_") and k not in
                  ("vscore", "vshows", "vjudge", "clip", "clip_cos", "thumb")}
        shots.append({"path": p, "credit": credit, "seg": i, "score": c["vscore"], "shows": c.get("vshows", ""),
                      "judge": c.get("vjudge", ""), "want": want, "reused": reused})
        log.info("    seg %d ← %.1f/10 [%s]%s %s  (%s)", i, c["vscore"], c.get("vjudge"), " (repeat)" if reused else "",
                 c.get("vshows", "")[:70], c["title"][:45])
        return True

    for i, seg in enumerate(segments):
        vis = seg.get("visual") if isinstance(seg.get("visual"), dict) else {}
        kws = [k for k in (seg.get("keywords") or []) if isinstance(k, str)]
        want = (vis.get("shows") or "").strip() or (f"{subject}: " + ", ".join(kws) if kws else subject)
        queries = [q for q in (vis.get("queries") or []) if isinstance(q, str) and q.strip()]
        if not queries:
            queries = [f"{subject} {k}" for k in kws] or [subject]
        judged = judge_line(canon(_candidates_for(queries, sources, allowed, min_w)) + common, want, seg["text"])
        good = [c for c in judged if c["vscore"] >= min_score]
        if not good and llm is not None and mcfg.get("requery", True):
            try:
                want2, q2 = _requery(llm, subject, seg["text"], want, [c.get("vshows", "") for c in judged])
                log.info("    seg %d: no match for %r → retrying with %r %s", i, want[:50], want2[:50], q2)
                judged2 = judge_line(canon(_candidates_for(q2, sources, allowed, min_w)), want2, seg["text"])
                good = [c for c in judged2 if c["vscore"] >= min_score]
                want = want2 if good else want
            except vision.VisionUnavailable:
                raise
            except Exception as e:  # noqa: BLE001
                log.debug("requery failed: %s", e)
        want_n = 2 if seg_durs and i < len(seg_durs) and seg_durs[i] > multi_after else 1
        got = 0
        spare = []
        for c in good:
            if got < want_n and not is_used(c) and add_shot(c, i, want):
                got += 1
            elif not is_used(c):
                spare.append(c)
        if not got:
            # every verified match for this line is already on screen elsewhere → repeat the best one,
            # but never right after the line that already shows it
            for c in good:
                u = is_used(c)
                if u and abs(u["seg"] - i) > 1 and add_shot(c, i, want, reused=True):
                    got = 1
                    break
        if not got:
            best = f"{judged[0]['vscore']:.0f}/10 ({judged[0].get('vshows', '')[:60]})" if judged else "none"
            raise NoVisualMatch(f"line {i + 1} has no image that shows it (best: {best}): {seg['text'][:80]!r}")
        alts[i] = spare

    distinct = len({s["path"] for s in shots})
    need = int(mcfg.get("min_distinct_images", 3))
    if distinct < need:
        raise NoVisualMatch(f"only {distinct} distinct verified images for {subject!r} (need {need})")
    shots.sort(key=lambda s: (s["seg"], s.get("reused", False)))
    log.info("  visuals: %d shots, %d distinct images, every line verified, %d vision calls", len(shots), distinct,
             judge.calls)
    return {"shots": shots, "alts": alts, "calls": judge.calls}


def use_alternative(visuals: dict, seg: int, bad_path: str, work: Path) -> bool:
    """Replace a shot that failed final QA with the next verified spare for that line."""
    from . import vision  # noqa: F401
    for c in list(visuals["alts"].get(seg, [])):
        visuals["alts"][seg].remove(c)
        p = _usable(c, work)
        if not p or str(p) == str(bad_path) or any(str(s["path"]) == str(p) for s in visuals["shots"]):
            continue
        for s in visuals["shots"]:
            if s["seg"] == seg and str(s["path"]) == str(bad_path):
                credit = {k: v for k, v in c.items() if not k.startswith("_") and k not in
                          ("vscore", "vshows", "vjudge", "clip", "clip_cos", "thumb")}
                s.update({"path": p, "credit": credit, "score": c["vscore"], "shows": c.get("vshows", ""),
                          "judge": c.get("vjudge", ""), "reused": False})
                log.info("    QA repair: seg %d → %s", seg, c.get("vshows", "")[:70])
                return True
    return False


def credits_text(visuals: list[dict]) -> str:
    seen, lines = set(), []
    for v in visuals:
        c = v["credit"]
        if c["url"] in seen:
            continue
        seen.add(c["url"])
        lines.append(f"• \"{c['title'][:60]}\" by {c['author']} — {c['license']} ({c['source']}) {c.get('page') or ''}".strip())
    return "\n".join(lines)
