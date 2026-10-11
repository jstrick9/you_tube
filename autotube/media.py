"""Licensed visuals sourcing with full attribution tracking.

Sources (all free, commercially usable):
  • Wikimedia Commons – the grounding article's own images first (most on-topic),
                         then keyword search. Only PD / CC0 / CC-BY / CC-BY-SA accepted.
  • Openverse         – CC-licensed image search (keyless).
  • Pexels            – stock photos/videos (free key; Pexels license, no attribution required
                         but we credit anyway).
  • Moving footage   – Wikimedia Commons videos (PD/CC0/CC-BY/CC-BY-SA), NASA's image & video library (public
                         domain, keyless) and Pexels videos (only with a free PEXELS_API_KEY). Clips are judged by the
                         same vision model as photos, then cut to a scene-cut-free window around the exact frame the
                         judge approved (see prepare_clip). Their audio is always dropped.
Every asset's author + license + URL is recorded and printed in the video
description → CC-BY compliance and a clean copyright trail.
"""
from __future__ import annotations

import hashlib
import time
import html
import logging
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from .common import ffmpeg_bin, get_json, http

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


# ── 9:16 fitness ──────────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 1080, 1920
FRAME_AR = FRAME_W / FRAME_H          # 0.5625


def crop_px(w: int | None, h: int | None) -> int:
    """Real horizontal pixels that survive a 9:16 cover-crop of a w x h asset.

    Since every shot became full-bleed, this - not raw width - is the number that decides whether a
    frame looks sharp. Cover-cropping a landscape asset throws the sides away and then upscales
    what is left to 1080 wide, so a 1920x1080 clip contributes only 1080*(9/16) = 608 real pixels
    and gets blown up 1.8x; a 1280x720 one contributes 405 and is blown up 2.7x. A raw width floor
    cannot see any of that, which is why a "1920px" asset could still render as mush.
    """
    try:
        w, h = int(w or 0), int(h or 0)
    except (TypeError, ValueError):
        return 0
    if w <= 0 or h <= 0:
        return 0
    return int(min(w, h * FRAME_AR))


def crop_quality(w: int | None, h: int | None) -> float:
    """crop_px as a fraction of the 1080 the frame needs. >=1.0 means no upscaling."""
    return crop_px(w, h) / FRAME_W


def shape_bonus(a: dict, weight: float = 1.0) -> float:
    """Ranking nudge toward assets that fill a 9:16 frame without being upscaled.

    Deliberately a bonus and not a filter. Hard-filtering to portrait shrinks the candidate pool so
    far that the vision judge starts approving weak matches, and a sharp picture of the wrong thing
    is worse than a slightly soft picture of the right one. Relevance still leads; this only breaks
    ties between candidates the judge already approved.
    """
    q = crop_quality(a.get("w"), a.get("h"))
    if q <= 0:
        return 0.0                      # unknown dimensions: no opinion either way
    if q >= 1.0:
        return weight                   # fills the frame at native resolution or better
    return weight * max(-1.0, (q - 1.0) * 1.5)   # soft penalty, scaled by how much upscaling it forces


def pexels_search(query: str, limit: int = 8) -> list[dict]:
    """Pexels photos, asking for portrait first but NOT restricting to it.

    This used to pass orientation=portrait as a hard filter, which made sense when the renderer
    letterboxed and a landscape photo was unusable. Now that every shot is cover-cropped full-bleed,
    that filter only starves the pool: Pexels' portrait subset is small, so a narrow subject
    returned almost nothing and the vision judge was left choosing among weak matches. Both
    orientations are fetched and shape_bonus() prefers the ones that fill 9:16 natively.
    """
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    out, seen = [], set()
    for orientation in ("portrait", None):
        params = {"query": query, "per_page": limit}
        if orientation:
            params["orientation"] = orientation
        try:
            data = get_json("https://api.pexels.com/v1/search", params=params,
                            headers={"Authorization": key})
        except Exception as e:  # noqa: BLE001
            log.debug("pexels failed: %s", e)
            continue
        for p in data.get("photos", []):
            if p["src"]["large2x"] in seen:
                continue
            seen.add(p["src"]["large2x"])
            out.append({"url": p["src"]["large2x"], "thumb": p["src"].get("medium"), "page": p["url"],
                        "title": p.get("alt") or query, "license": "Pexels License",
                        "author": p.get("photographer", "Pexels"), "source": "Pexels",
                        "w": p["width"], "h": p["height"]})
    return out


# ── moving footage ────────────────────────────────────────────────────────────
VIDEO_MIN_S, VIDEO_MAX_S = 3.0, 1800.0


def _seek_thumb(thumb: str, t: float) -> str:
    """Commons video thumbnail at a chosen second (…/500px--X.webm.jpg → …/500px-seek=12.3-X.webm.jpg)."""
    return re.sub(r"/(\d+)px-[^/]*?-([^/]+\.jpg)", lambda m: f"/{m.group(1)}px-seek={t:.1f}-{m.group(2)}", thumb, count=1)


def _pick_derivative(vi: dict) -> str | None:
    """Smallest transcode that is still sharp enough (720p preferred, then 480p/1080p), else the original."""
    ders = [d for d in vi.get("derivatives") or [] if d.get("src") and d.get("height")
            and any(x in (d.get("type") or "") for x in ("webm", "mp4", "ogg"))]
    ok = [d for d in ders if 480 <= int(d["height"]) <= 1080]
    if ok:
        return sorted(ok, key=lambda d: (abs(int(d["height"]) - 720), int(d["height"])))[0]["src"]
    return vi.get("url")


def commons_video_search(query: str, allowed: list[str], limit: int = 6) -> list[dict]:
    try:
        data = get_json(COMMONS, params={
            "action": "query", "generator": "search", "gsrsearch": f"filetype:video {query}", "gsrnamespace": 6,
            "gsrlimit": limit, "prop": "videoinfo", "viprop": "url|size|mime|extmetadata|derivatives",
            "viurlwidth": 500, "format": "json"})
    except Exception as e:  # noqa: BLE001
        log.debug("commons video search failed: %s", e)
        return []
    out = []
    for p in data.get("query", {}).get("pages", {}).values():
        vi = (p.get("videoinfo") or [None])[0]
        if not vi or not str(vi.get("mime", "")).startswith(("video/", "application/ogg")):
            continue
        dur = float(vi.get("duration") or 0)
        meta = vi.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if not (VIDEO_MIN_S <= dur <= VIDEO_MAX_S) or (vi.get("width") or 0) < 480 or not _license_ok(lic, allowed):
            continue
        src, thumb = _pick_derivative(vi), vi.get("thumburl")
        if not src or not thumb:
            continue
        t_ref = round(min(max(dur * 0.4, 1.0), dur - 1.0), 1)
        out.append({"kind": "video", "url": src, "thumb": _seek_thumb(thumb, t_ref), "t_ref": t_ref,
                    "page": vi.get("descriptionurl"), "title": p["title"].replace("File:", "") + " (video)",
                    "license": lic, "duration": dur, "source": "Wikimedia Commons",
                    "author": _strip_html(meta.get("Artist", {}).get("value", "")) or "Wikimedia Commons",
                    "w": vi.get("width"), "h": vi.get("height")})
    return out


NASA_ASSETS = "https://images-assets.nasa.gov/video/{i}/{i}~{v}.mp4"


def nasa_video_search(query: str, limit: int = 5) -> list[dict]:
    """NASA Image and Video Library (keyless; NASA media is generally public domain in the US)."""
    try:
        data = get_json("https://images-api.nasa.gov/search", params={"q": query, "media_type": "video",
                                                                       "page_size": limit})
    except Exception as e:  # noqa: BLE001
        log.debug("nasa search failed: %s", e)
        return []
    out = []
    for it in (data.get("collection", {}).get("items") or [])[:limit]:
        d = (it.get("data") or [{}])[0]
        nid = d.get("nasa_id")
        thumb = next((ln.get("href") for ln in it.get("links") or [] if ln.get("render") == "image"), None)
        if not nid or not thumb:
            continue
        q = quote(nid)
        out.append({"kind": "video", "url": NASA_ASSETS.format(i=q, v="medium"), "thumb": thumb.replace(" ", "%20"),
                    "variants": [NASA_ASSETS.format(i=q, v=v) for v in ("medium", "small", "mobile")],
                    "page": f"https://images.nasa.gov/details/{q}", "title": (d.get("title") or nid)[:80] + " (video)",
                    "license": "Public domain (NASA)", "author": f"NASA{('/' + d['center']) if d.get('center') else ''}",
                    "source": "NASA Image and Video Library", "w": 1280, "h": 720})
    return out


# Collections whose contents are public domain as a matter of the collection's own
# policy. archive.org also hosts plenty of in-copyright material, so an unfiltered
# search there would be a licensing problem, not a free footage win. Every result is
# additionally required to carry a public-domain or CC licence field below.
ARCHIVE_PD_COLLECTIONS = ("prelinger", "publicmovies", "government_films", "newsandpublicaffairs")


def archive_video_search(query: str, limit: int = 5) -> list[dict]:
    """Internet Archive public-domain film (keyless).

    Real archival motion footage, which is the one visual upgrade that strengthens
    rather than undermines the channel's own claim that every file is real. The
    Prelinger collection alone holds ~10,000 public-domain films.

    Two deliberate restrictions. The search is confined to collections that are public
    domain by policy, and each item must *also* declare a public-domain or Creative
    Commons licence - archive.org is a host, not a rights clearinghouse, and an
    unfiltered query there would pull in copyrighted uploads. Footage is typically
    640x480 and 4:3, so it is reported honestly here and treated as insert b-roll
    rather than full-frame by the renderer's existing shape scoring.
    """
    coll = " OR ".join(f"collection:{c}" for c in ARCHIVE_PD_COLLECTIONS)
    q = f'({query}) AND mediatype:movies AND ({coll})'
    try:
        data = get_json("https://archive.org/advancedsearch.php",
                        params={"q": q, "fl[]": ["identifier", "title", "licenseurl"],
                                "rows": limit, "page": 1, "output": "json"})
    except Exception as e:  # noqa: BLE001
        log.debug("archive search failed: %s", e)
        return []
    out = []
    for doc in ((data.get("response") or {}).get("docs") or [])[:limit]:
        ident = doc.get("identifier")
        if not ident:
            continue
        lic = (doc.get("licenseurl") or "").lower()
        try:
            meta = get_json(f"https://archive.org/metadata/{quote(ident)}")
        except Exception as e:  # noqa: BLE001
            log.debug("archive metadata failed for %s: %s", ident, e)
            continue
        md = meta.get("metadata") or {}
        lic = lic or str(md.get("licenseurl") or "").lower()
        # Belt and braces: the collection is PD by policy, but refuse anything whose own
        # licence field contradicts that or carries a non-commercial restriction.
        if lic and not ("publicdomain" in lic or "creativecommons" in lic or "cc0" in lic):
            continue
        # Splitting the URL on "/" missed by-nc entirely, because the restriction lives
        # inside the licence code rather than as its own path segment. Match it as a
        # token instead. ND is refused for the same practical reason as NC: this
        # pipeline crops, trims and overlays, which makes every use a derivative work.
        if re.search(r"(^|[-/])n[cd]([-/]|$)", lic):
            log.debug("archive: refusing %s, licence forbids commercial or derivative use", ident)
            continue
        # Relevance guard. archive.org's keyword ranking over this corpus is weak - a
        # search for "deep sea ocean" happily returns "COLORADO PLATEAU" - and every
        # candidate that reaches the vision gate spends the scarcest quota we have.
        # Requiring a real query word in the title or description is crude, but it is
        # free and it keeps obvious mismatches from costing a vision call.
        hay = f"{md.get('title') or ''} {md.get('description') or ''}".lower()
        terms = [w for w in re.findall(r"[a-z]{4,}", query.lower())]
        if terms and not any(w in hay for w in terms):
            log.debug("archive: dropping %s, no query term in title/description", ident)
            continue
        vids = [f for f in (meta.get("files") or [])
                if str(f.get("name", "")).lower().endswith(".mp4") and int(f.get("size") or 0) > 0]
        if not vids:
            continue
        vids.sort(key=lambda f: int(f.get("size") or 0), reverse=True)
        best = vids[0]
        base = f"https://archive.org/download/{quote(ident)}/"
        try:
            w, h = int(best.get("width") or 640), int(best.get("height") or 480)
        except (TypeError, ValueError):
            w, h = 640, 480
        out.append({"kind": "video", "url": base + quote(best["name"]),
                    "variants": [base + quote(f["name"]) for f in vids[:3]],
                    "thumb": f"https://archive.org/services/img/{quote(ident)}",
                    "page": f"https://archive.org/details/{quote(ident)}",
                    "title": (str(md.get("title") or ident))[:80] + " (archive film)",
                    "license": "Public domain (Internet Archive)",
                    "author": str(md.get("creator") or "Internet Archive"),
                    "source": "Internet Archive", "w": w, "h": h})
    return out


def pixazo_image(query: str, limit: int = 1) -> list[dict]:
    """Generate an image via Pixazo's free tier (Flux Schnell), 1024x1024, unwatermarked.

    Deliberately last in the source order, and deliberately labelled.

    This channel's claim is "Every file is real", and a generated picture of a real
    event is a fabrication standing where evidence should be. So this exists for the
    case where no archive holds a usable shot and the alternative is abandoning the
    video: abstract texture, mood, a scene nobody is asserting is a photograph. The
    asset records itself as AI-generated so the provenance dossier stays truthful about
    which frames were synthetic - that record is the thing a monetisation appeal rests
    on, and quietly mixing generated frames into it would destroy its value.

    It stays behind the vision gate like every other source: if it does not show what
    the line says, it is rejected.
    """
    key = os.environ.get("PIXAZO_API_KEY")
    if not key:
        return []
    try:
        r = http().post("https://gateway.pixazo.ai/flux/text-to-image",
                        headers={"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": key},
                        json={"prompt": query}, timeout=90)
        if r.status_code != 200:
            log.debug("pixazo HTTP %s: %s", r.status_code, r.text[:200])
            return []
        data = r.json()
    except Exception as e:  # noqa: BLE001
        log.debug("pixazo failed: %s", e)
        return []
    url = None
    if isinstance(data, dict):
        for k in ("url", "image_url", "output", "image"):
            v = data.get(k)
            if isinstance(v, str) and v.startswith("http"):
                url = v
                break
            if isinstance(v, list) and v and isinstance(v[0], str) and v[0].startswith("http"):
                url = v[0]
                break
    if not url:
        log.debug("pixazo: no image url in response keys=%s", list(data)[:8] if isinstance(data, dict) else "?")
        return []
    return [{"kind": "image", "url": url, "thumb": url, "page": "https://pixazo.ai",
             "title": f"AI-generated: {query[:60]}",
             "license": "AI-generated (Pixazo/Flux Schnell)", "author": "AI-generated",
             "source": "Pixazo (AI-generated)", "synthetic": True, "w": 1024, "h": 1024}]


def pexels_video_search(query: str, limit: int = 5) -> list[dict]:
    """Pexels clips, fetched in both orientations and chosen by post-crop sharpness."""
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    out, seen = [], set()
    for orientation in ("portrait", None):
        params = {"query": query, "per_page": limit}
        if orientation:
            params["orientation"] = orientation
        try:
            data = get_json("https://api.pexels.com/videos/search", params=params,
                            headers={"Authorization": key})
        except Exception as e:  # noqa: BLE001
            log.debug("pexels video failed: %s", e)
            continue
        for v in data.get("videos", []):
            if v.get("id") in seen:
                continue
            files = [f for f in v.get("video_files", []) if f.get("link") and (f.get("height") or 0) >= 720]
            if not files or not (VIDEO_MIN_S <= float(v.get("duration") or 0) <= VIDEO_MAX_S):
                continue
            f = _best_video_file(files)
            seen.add(v.get("id"))
            pics = v.get("video_pictures") or []
            thumb = pics[len(pics) // 2]["picture"] if pics else v.get("image")
            out.append({"kind": "video", "url": f["link"], "thumb": thumb, "page": v.get("url"),
                        "title": (v.get("url") or query).rstrip("/").rsplit("/", 1)[-1][:70] + " (video)",
                        "license": "Pexels License", "author": (v.get("user") or {}).get("name", "Pexels"),
                        "source": "Pexels", "duration": float(v.get("duration") or 0), "w": f.get("width"),
                        "h": f.get("height")})
    return out


def _best_video_file(files: list[dict]) -> dict:
    """Cheapest rendition that still fills a 9:16 frame at native resolution.

    The old rule was "smallest file at or above 720p", which is the wrong axis entirely now that
    clips are cover-cropped: a 1280x720 landscape rendition keeps only 405 of the 1080 pixels the
    frame needs and gets upscaled 2.7x, so the bandwidth it saved bought a visibly soft shot. We
    now pick the smallest rendition whose POST-CROP width clears 1080, and fall back to the
    sharpest available when nothing does.
    """
    by_size = sorted(files, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))
    good = [f for f in by_size if crop_px(f.get("width"), f.get("height")) >= FRAME_W]
    if good:
        return good[0]
    return max(by_size, key=lambda f: crop_px(f.get("width"), f.get("height")))


def _probe_duration(path: Path) -> float:
    err = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", err)
    return int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0


def _gray_samples(path: Path, fps: float = 4.0, w: int = 32, h: int = 18):
    """Tiny grayscale frames at `fps` → (numpy array [n, h*w], fps). Cheap: a 2-minute clip is ~500 x 576 bytes."""
    import numpy as np
    raw = subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-i", str(path), "-an",
                          "-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True).stdout
    n = len(raw) // (w * h)
    return np.frombuffer(raw[:n * w * h], dtype=np.uint8).reshape(n, w * h).astype(float), fps


def _norm_vec(v):
    import numpy as np
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    return v / (np.linalg.norm(v) or 1.0)


def choose_window(samples, fps: float, need: float, ref=None, t_ref: float | None = None,
                  cut_thr: float = 30.0, min_corr: float = 0.55):
    """Pick [start, end) seconds of a clip for one shot.

    • finds scene cuts (big frame-to-frame change) so a window never spans two different shots;
    • centres the window on the frame that best matches `ref` (the thumbnail the vision judge approved), so what
      is shown is what was verified — if nothing in the clip matches it well enough, returns None;
    • a window shorter than `need` is allowed down to need/1.6 (played in gentle slow motion).
    Returns (start, end, speed) or None."""
    import numpy as np
    n = len(samples)
    if n < 3:
        return None
    diffs = np.abs(np.diff(samples, axis=0)).mean(axis=1)
    cuts = [0] + [i + 1 for i, d in enumerate(diffs) if d > cut_thr] + [n]
    if ref is not None:
        r = _norm_vec(ref)
        corr = np.array([float(_norm_vec(s) @ r) for s in samples])
        idx = int(corr.argmax())
        if corr[idx] < min_corr:
            return None
    else:
        idx = int(min(n - 1, max(0, round((t_ref or n / fps * 0.4) * fps))))
    lo = max(c for c in cuts if c <= idx)
    hi = min(c for c in cuts if c > idx)
    edge = 1 if lo > 0 else 0                     # stay clear of the cut itself
    s0, s1 = (lo + edge) / fps, (hi - (1 if hi < n else 0)) / fps
    avail = s1 - s0
    if avail < need / 1.6 or avail <= 0.5:
        return None
    if avail >= need:
        mid = idx / fps
        start = min(max(mid - need / 2, s0), s1 - need)
        return round(start, 2), round(start + need, 2), 1.0
    return round(s0, 2), round(s1, 2), round(avail / need, 3)


def _download_video(asset: dict, dest_dir: Path, max_mb: float) -> Path | None:
    for url in asset.get("variants") or [asset["url"]]:
        h = hashlib.md5(url.encode()).hexdigest()[:12]
        dest = dest_dir / f"src_{h}.video"
        if dest.exists():
            return dest
        try:
            for attempt in range(3):                  # Wikimedia rate-limits bursts (429): back off politely
                r = http().get(url, timeout=60, stream=True)
                if r.status_code != 429 or attempt == 2:
                    break
                wait = min(20.0, float(r.headers.get("retry-after") or 0) or 4.0 * (attempt + 1))
                r.close()
                time.sleep(wait)
            with r:
                r.raise_for_status()
                size = int(r.headers.get("content-length") or 0)
                if size > max_mb * 1e6:
                    log.info("      video too large (%.0f MB), trying a smaller version: %s", size / 1e6,
                             asset.get("title", "")[:50])
                    continue
                got = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        got += len(chunk)
                        if got > max_mb * 1e6:
                            raise ValueError("video too large")
                        f.write(chunk)
            return dest
        except Exception as e:  # noqa: BLE001
            dest.unlink(missing_ok=True)
            log.info("      video download failed %s: %s", asset.get("title", url)[:50], str(e)[:80])
    return None


def prepare_clip(asset: dict, work: Path, need: float, max_mb: float = 90) -> Path | None:
    """Download a judged video, cut a scene-cut-free window of `need` seconds around the approved frame, drop the
    audio, and write a poster frame (asset['_poster']) used for the thumbnail and the contact sheet."""
    import numpy as np
    key = hashlib.md5(f"{asset['url']}|{need:.2f}".encode()).hexdigest()[:12]
    out = work / f"clip_{key}.mp4"
    poster = work / f"clip_{key}.jpg"
    if out.exists() and poster.exists():
        asset["_poster"] = poster
        return out
    src = _download_video(asset, work, max_mb)
    if not src:
        return None
    try:
        samples, fps = _gray_samples(src)
        ref = None
        if asset.get("_img") is not None:
            im = asset["_img"].convert("L").resize((32, 18))
            ref = np.asarray(im, dtype=float).reshape(-1)
        win = choose_window(samples, fps, need, ref=ref, t_ref=asset.get("t_ref"))
        if not win:
            log.info("      clip rejected: no cut-free %.1fs window matching the approved frame (%s)", need,
                     asset.get("title", "")[:50])
            return None
        start, end, speed = win
        vf = (f"setpts=PTS/{speed}," if speed < 1 else "") + "fps=30,scale='min(1920,iw)':-2,format=yuv420p"
        res = subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-t",
                              f"{end - start:.2f}", "-i", str(src), "-an", "-vf", vf, "-c:v", "libx264", "-preset",
                              "veryfast", "-crf", "18", "-movflags", "+faststart", str(out)],
                             capture_output=True, text=True)
        if res.returncode != 0 or not out.exists() or _probe_duration(out) < need * 0.9:
            log.info("      clip cut failed (%s): %s", asset.get("title", "")[:40], res.stderr[-200:])
            out.unlink(missing_ok=True)
            return None
        mid = _probe_duration(out) / 2
        subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-y", "-ss", f"{mid:.2f}", "-i", str(out),
                        "-frames:v", "1", "-q:v", "2", str(poster)], capture_output=True)
        if not poster.exists():
            return None
        asset["_poster"] = poster
        asset["window"] = [start, end, speed]
        log.info("      clip ready: %.1f-%.1fs%s of %s", start, end, f" at {speed:.2f}x" if speed < 1 else "",
                 asset.get("title", "")[:50])
        return out
    finally:
        src.unlink(missing_ok=True)


def _looks_like_document(im: Image.Image) -> bool:
    """Reject screenshots/scans/text pages: lots of flat near-white area + low colour variance."""
    small = im.resize((96, 96)).convert("RGB")
    raw = small.tobytes()
    px = [tuple(raw[i:i + 3]) for i in range(0, len(raw), 3)]
    white = sum(1 for r, g, b in px if r > 225 and g > 225 and b > 225) / len(px)
    flat = sum(1 for r, g, b in px if max(r, g, b) - min(r, g, b) < 18) / len(px)
    colours = len(set((r // 16, g // 16, b // 16) for r, g, b in px))
    greys = len(set((r + g + b) // 24 for r, g, b in px))       # tonal range, works for B&W photos too
    # text pages / flat clip-art. NOTE: colour count alone must not decide — black-and-white photos and
    # engravings have < 40 colour bins but a full tonal range, and they're often exactly the right image.
    return (white > 0.38 and flat > 0.55) or (colours < 40 and greys < 12)


def _download(asset: dict, dest_dir: Path, doc_filter: bool = True) -> Path | None:
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
            if min(im.size) < 400 or (doc_filter and _looks_like_document(im)):
                log.info("      image dropped (%s): %s", "too small" if min(im.size) < 400 else "looks like a document",
                         asset.get("title", "")[:60])
                tmp.unlink(missing_ok=True)
                return None
            # cap size for speed
            im.thumbnail((2400, 2400))
            im.save(dest, "JPEG", quality=90)
        tmp.unlink(missing_ok=True)
        return dest
    except Exception as e:  # noqa: BLE001
        log.info("      download failed %s: %s", asset.get("title", asset["url"])[:60], str(e)[:80])
        return None


def _candidates_for(queries: list[str], sources: list[str], allowed: list[str], min_w: int) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for qi, q in enumerate(queries[:3]):
        if qi < 2 and "commons_video" in sources:
            jobs.append((commons_video_search, (q, allowed, 5)))
        if qi < 1 and "nasa_video" in sources:
            jobs.append((nasa_video_search, (q, 4)))
        if qi < 2 and "archive_video" in sources:
            jobs.append((archive_video_search, (q, 4)))
        if qi < 2 and "pexels_video" in sources and os.environ.get("PEXELS_API_KEY"):
            jobs.append((pexels_video_search, (q, 4)))
        if "pexels" in sources and os.environ.get("PEXELS_API_KEY"):
            jobs.append((pexels_search, (q, 6)))
        if "wikimedia" in sources:
            jobs.append((commons_search, (q, allowed, min_w, 10)))
        if "openverse" in sources and qi < 2:
            jobs.append((openverse_search, (q, min_w, 8)))
        # Last resort only: generated imagery is a fabrication where evidence should be,
        # so it is reached for when the archives have produced nothing usable.
        if qi < 1 and "pixazo" in sources and os.environ.get("PIXAZO_API_KEY"):
            jobs.append((pixazo_image, (q, 1)))
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(lambda j: j[0](*j[1]), jobs))
    return [a for r in results for a in r]


def _usable(a: dict, work: Path, need: float = 4.0, max_mb: float = 90) -> Path | None:
    if a.get("kind") == "video":
        return prepare_clip(a, work, need, max_mb)
    return _usable_image(a, work)


def _usable_image(a: dict, work: Path) -> Path | None:
    # An image a vision model has already approved for the line is not second-guessed by the crude
    # "document" heuristic (the judge rejects text/scans itself, and final QA re-checks the rendered frame).
    p = _download(a, work, doc_filter="vscore" not in a)
    if p:
        with Image.open(p) as im:
            if min(im.size) < 480:
                return None
    return p


CTA_RE = re.compile(r"\b(follow (?:for|us|me|along)|subscribe|like (?:and|for)|see you (?:next|tomorrow))\b", re.I)


class NoVisualMatch(RuntimeError):
    """A narration line has no image that verifiably shows it → the topic is skipped."""


REQUERY_SYSTEM = ("You help an educational video editor find photos in free libraries (Wikimedia Commons, Openverse). "
                  "Return strict JSON only.")


def _validate_requery_output(value, allow_empty: bool = False) -> dict:
    if not isinstance(value, dict):
        raise ValueError("visual requery response must be an object")
    shows, queries = value.get("shows"), value.get("queries")
    if not isinstance(shows, str) or len(shows) > 300 or not isinstance(queries, list) or len(queries) > 3:
        raise ValueError("visual requery requires concise text and at most three search strings")
    if allow_empty and not shows.strip() and not queries:
        return value
    if not shows.strip() or not queries:
        raise ValueError("a revised visual request needs at least one search query")
    if any(not isinstance(query, str) or not query.strip() or len(query) > 120 for query in queries):
        raise ValueError("visual requery search queries must be non-empty short strings")
    return value


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
    o = llm.json(REQUERY_SYSTEM, user, temperature=0.4, validate=_validate_requery_output)
    _validate_requery_output(o)  # a custom adapter must not skip the schema callback
    return o["shows"].strip(), [query.strip() for query in o["queries"]]


def ahash(path: Path) -> int | None:
    """64-bit average hash of an image: what the viewer sees, not what it is called.

    Distinctness was counted by file path, so two different files showing the same
    thing both counted. In one render frames 3 and 4 were the same underwater shot and
    frames 7 and 8 the same flat horizon; in another, three of eight frames were
    near-identical grey satellite views. The viewer experiences that as one picture
    held for five seconds, which is the opposite of the every-one-to-two-seconds
    change that keeps people watching.
    """
    try:
        with Image.open(path) as im:
            g = im.convert("L").resize((8, 8))
            px = list(g.getdata())
    except Exception:  # noqa: BLE001
        return None
    avg = sum(px) / len(px)
    bits = 0
    for i, v in enumerate(px):
        if v > avg:
            bits |= 1 << i
    return bits


def too_similar(a: int | None, b: int | None, max_distance: int = 8) -> bool:
    """Hamming distance on the hashes. 8 of 64 bits is a deliberately loose bar:
    it catches "the same photo again" and crops of one frame, while leaving two
    genuinely different pictures of the same subject alone."""
    if a is None or b is None:
        return False
    return bin(a ^ b).count("1") <= max_distance


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
    compliance = cfg.get("compliance", {}) or {}
    allowed = compliance.get("allowed_licenses", [])
    min_w = mcfg["min_image_width"]
    configured_sources = list(mcfg.get("sources", []))
    synthetic_disclosure = bool(compliance.get("contains_synthetic_media", False))
    sources = configured_sources if synthetic_disclosure else [s for s in configured_sources if s != "pixazo"]
    if "pixazo" in configured_sources and not synthetic_disclosure:
        log.warning("Pixazo candidate generation disabled: compliance.contains_synthetic_media is false")
    min_score = float(mcfg.get("min_match_score", 7))
    multi_after = float(mcfg.get("multi_shot_seconds", 4.5))
    video_bonus = float(mcfg.get("video_bonus", 1.0)) if mcfg.get("prefer_video", True) else 0.0
    shape_w = float(mcfg.get("shape_bonus", 1.0))
    max_mb = float(mcfg.get("max_video_mb", 90))
    need_by_seg: dict[int, float] = {}
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
    if "commons_video" in sources:
        common += commons_video_search(f'"{subject}"' if len(subject.split()) > 1 else subject, allowed, 6)
    if "nasa_video" in sources:
        common += nasa_video_search(subject, 4)
    common = canon(common)

    def judge_line(cands, want, line):
        cands = vision.fetch_thumbs(list({id(c): c for c in cands}.values()))
        fresh = lambda: [{k: v for k, v in c.items() if k not in ("vscore", "vshows", "vjudge", "clip", "clip_cos")}
                         for c in cands]
        judged = judge.score(fresh(), want, line, subject, context)
        if not any(c["vscore"] >= min_score for c in judged):          # look at the next sheet too
            judged += judge.score(fresh(), want, line, subject, context, page=1)
        return sorted(judged, key=lambda c: -c["vscore"])

    def rank(judged):
        """Approved candidates, best first.

        Three things decide the order: how well the vision judge thinks the asset shows the line,
        whether it is real moving footage, and whether it survives the 9:16 cover-crop without
        being upscaled. Relevance leads - shape only separates candidates the judge already
        approved, because a sharp picture of the wrong thing is worse than a soft picture of the
        right one.
        """
        good = [c for c in judged if c["vscore"] >= min_score]
        return sorted(good, key=lambda c: -(c["vscore"]
                                            + (video_bonus if c.get("kind") == "video" else 0)
                                            + shape_bonus(c, shape_w)))

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
        p = _usable(c, work, need_by_seg.get(i, 4.0), max_mb)
        if not p:
            return False
        if not reused:
            used.append({"url": c["url"], "hash": c["_hash"], "emb": c.get("_emb"), "title": c.get("title", ""), "seg": i})
        credit = {k: v for k, v in c.items() if not k.startswith("_") and k not in
                  ("vscore", "vshows", "vjudge", "clip", "clip_cos", "thumb", "variants", "t_ref", "window")}
        shots.append({"path": p, "credit": credit, "seg": i, "score": c["vscore"], "shows": c.get("vshows", ""),
                      "judge": c.get("vjudge", ""), "want": want, "reused": reused,
                      "kind": c.get("kind", "image"), "poster": c.get("_poster"), "window": c.get("window"),
                      "crop_q": round(crop_quality(c.get("w"), c.get("h")), 2)})
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
        if i == len(segments) - 1 and CTA_RE.search(seg["text"]):
            # sign-off ("Follow for more …"): show the video's own subject, never an arbitrary picture
            want = f"a clear, striking photo of {subject} itself"
            queries = [subject] + queries[:1]
        want_n = 2 if seg_durs and i < len(seg_durs) and seg_durs[i] > multi_after else 1
        seg_len = seg_durs[i] if seg_durs and i < len(seg_durs) else 4.0
        need_by_seg[i] = round(seg_len / want_n + 0.45, 2)        # + crossfade overlap and a little slack

        def place(good, want):
            """Put up to want_n approved, usable, not-yet-shown images on this line; return (got, spares)."""
            got, spare, dup, bad = 0, [], 0, 0
            for c in good:
                if is_used(c):
                    dup += 1
                elif got < want_n:
                    if add_shot(c, i, want):
                        got += 1
                    else:
                        bad += 1
                else:
                    spare.append(c)
            if not got:
                # every approved match for this line is already on screen elsewhere → repeat the best one,
                # but never right after the line that already shows it
                for c in good:
                    u = is_used(c)
                    if u and abs(u["seg"] - i) > 1 and add_shot(c, i, want, reused=True):
                        got = 1
                        break
            if good and not got:
                log.info("    seg %d: %d approved image(s) but none placeable (%d already shown, %d unusable)",
                         i, len(good), dup, bad)
            return got, spare

        judged = judge_line(canon(_candidates_for(queries, sources, allowed, min_w)) + common, want, seg["text"])
        good = rank(judged)
        got, spare = place(good, want)
        if not got and llm is not None and mcfg.get("requery", True):
            try:
                want2, q2 = _requery(llm, subject, seg["text"], want, [c.get("vshows", "") for c in judged])
                log.info("    seg %d: no usable match for %r → retrying with %r %s", i, want[:50], want2[:50], q2)
                judged2 = judge_line(canon(_candidates_for(q2, sources, allowed, min_w)), want2, seg["text"])
                good2 = rank(judged2)
                got, spare = place(good2, want2)
                judged = sorted(judged + judged2, key=lambda c: -c["vscore"])
            except vision.VisionUnavailable:
                raise
            except Exception as e:  # noqa: BLE001
                log.info("    seg %d: requery failed: %s", i, str(e)[:120])
        if not got:
            best = f"{judged[0]['vscore']:.0f}/10 ({judged[0].get('vshows', '')[:60]})" if judged else "none"
            raise NoVisualMatch(f"line {i + 1} has no image that shows it (best: {best}): {seg['text'][:80]!r}")
        alts[i] = spare

    # Count what the viewer sees. Distinctness used to be len(set of file paths), so two
    # different files showing the same thing both counted and a video could hold one
    # picture across several segments without tripping it.
    hashes = [ahash(s["path"]) for s in shots]
    seen: list[int] = []
    for h in hashes:
        if h is None or not any(too_similar(h, k) for k in seen):
            seen.append(h if h is not None else -len(seen))
    distinct = len(seen)
    need = int(mcfg.get("min_distinct_images", 3))
    if distinct < need:
        raise NoVisualMatch(
            f"only {distinct} visually distinct images for {subject!r} (need {need}) — "
            f"{len(shots)} shots collapsed to {distinct} once near-duplicates were merged")
    # Consecutive repeats are worse than a low total: the picture simply stops changing.
    for i in range(1, len(shots)):
        if too_similar(hashes[i], hashes[i - 1]):
            log.info("  visuals: shots %d and %d look identical, reordering", i - 1, i)
            for j in range(i + 1, len(shots)):
                if not too_similar(hashes[j], hashes[i - 1]):
                    shots[i], shots[j] = shots[j], shots[i]
                    hashes[i], hashes[j] = hashes[j], hashes[i]
                    break
    shots.sort(key=lambda s: (s["seg"], s.get("reused", False)))
    n_vid = sum(1 for s in shots if s.get("kind") == "video")
    # How many shots actually fill the 9:16 frame at native resolution. Since every shot became
    # full-bleed this is the number that decides whether the video looks sharp or upscaled, and it
    # is invisible in any other metric — log it so the next tuning pass has evidence, not a hunch.
    qs = [s["crop_q"] for s in shots if s.get("crop_q")]
    if qs:
        native = sum(1 for q in qs if q >= 1.0)
        log.info("  framing: %d/%d shots fill 9:16 natively (median crop quality %.2f%s)",
                 native, len(qs), sorted(qs)[len(qs) // 2],
                 "" if native == len(qs) else f", worst {min(qs):.2f} = {1 / max(min(qs), 0.01):.1f}x upscale")
    log.info("  visuals: %d shots (%d video clips), %d distinct, every line verified, %d vision calls", len(shots),
             n_vid, distinct, judge.calls)
    return {"shots": shots, "alts": alts, "calls": judge.calls, "need": need_by_seg, "max_mb": max_mb}


QA_REQUERY_SYSTEM = (
    "You repair a failed visual search for a factual educational Short. Keep the narration unchanged. "
    "Return strict JSON only."
)


def _requery_after_final_qa(llm, subject: str, line: str, want: str, failed_asset: str,
                            qa_feedback: str) -> tuple[str, list[str]]:
    """Ask for a genuinely different, evidence-faithful search after rendered pixels fail QA."""
    user = f"""VIDEO SUBJECT: {subject}
EXACT NARRATION: "{line}"
PREVIOUS SHOT REQUEST: {want}
FAILED ASSET: {failed_asset}
FINAL RENDERED-FRAME QA: {qa_feedback}

The final pixels were inspected and rejected; do not repeat the failed idea or merely change wording.
Suggest one physically findable visual that accurately supports this exact narration. Preserve the identity,
place, object, denomination, species, and historical period. Never substitute a generic look-alike, unrelated
person/object, fabricated reenactment, or symbolic stock photo. For a past event that cannot be photographed,
use a documented depiction or a directly relevant real place/object only if it remains honest about what the image
shows. Do not add unnecessary staging (for example, a hand holding an object) when a clear close-up is better.

Return JSON: {{"shows": "revised, literal shot request", "queries": ["up to 3 short searches, 2-4 words each"]}}.
If no defensible visual can be suggested, return {{"shows": "", "queries": []}} so this shot can remain rejected."""

    def validate(o):
        _validate_requery_output(o, allow_empty=True)

    result = llm.json(QA_REQUERY_SYSTEM, user, temperature=0.25, validate=validate)
    _validate_requery_output(result, allow_empty=True)  # revalidate custom adapters at the trust boundary
    shows = result["shows"].strip()
    queries = [query.strip() for query in result["queries"]]
    return (shows, queries) if shows and queries else ("", [])


def _activate_qa_candidate(visuals: dict, seg: int, bad_path: str, work: Path,
                           candidate: dict, want: str | None = None) -> bool:
    """Install one candidate without duplicating another shot or weakening final-frame QA."""
    shot = next((s for s in visuals["shots"]
                 if s["seg"] == seg and str(s["path"]) == str(bad_path)), None)
    if shot is None:
        return False
    p = _usable(candidate, work, visuals.get("need", {}).get(seg, 4.0), visuals.get("max_mb", 90))
    if not p or str(p) == str(bad_path):
        return False
    new_hash = ahash(p)
    for other in visuals["shots"]:
        if other is shot:
            continue
        if str(other["path"]) == str(p):
            return False
        if new_hash is not None and too_similar(new_hash, ahash(other["path"])):
            return False
    credit = {k: v for k, v in candidate.items() if not k.startswith("_") and k not in
              ("vscore", "vshows", "vjudge", "clip", "clip_cos", "thumb", "variants", "t_ref", "window")}
    shot.update({"path": p, "credit": credit, "score": candidate["vscore"],
                 "shows": candidate.get("vshows", ""), "judge": candidate.get("vjudge", ""),
                 "reused": False, "kind": candidate.get("kind", "image"),
                 "poster": candidate.get("_poster"), "window": candidate.get("window")})
    if want:
        shot["want"] = want
    log.info("    QA repair: seg %d → %s", seg, candidate.get("vshows", "")[:70])
    return True


def _find_final_qa_alternative(visuals: dict, seg: int, bad_path: str, work: Path,
                               source: dict, segment: dict, cfg: dict, llm, feedback: str) -> bool:
    """Re-search only after frame QA rejects all pre-verified spares; every new shot is re-judged."""
    from . import vision

    mcfg = cfg.get("media", {})
    if not mcfg.get("requery", True) or not llm or not feedback:
        return False
    shot = next((s for s in visuals["shots"]
                 if s["seg"] == seg and str(s["path"]) == str(bad_path)), None)
    if shot is None:
        return False
    credit = shot.get("credit", {})
    failed_asset = "; ".join(filter(None, (str(credit.get("title", "")), str(shot.get("shows", "")))))
    line = str(segment.get("text", ""))
    if not line:
        return False
    try:
        want, queries = _requery_after_final_qa(
            llm, str(source.get("title", "")), line, str(shot.get("want", "")), failed_asset, feedback)
    except Exception as e:  # noqa: BLE001 — fail closed; the rendered shot remains rejected
        log.info("    QA re-search prompt failed for seg %d: %s", seg, str(e)[:120])
        return False
    if not want or not queries:
        log.info("    QA re-search found no defensible new shot for seg %d", seg)
        return False

    allowed = cfg["compliance"]["allowed_licenses"]
    min_w = int(mcfg.get("min_image_width", 0))
    # AI-generated images are deliberately excluded here: factual historical/scientific evidence must not be
    # replaced by a fabricated scene when a first-party or openly licensed real asset is unavailable.
    sources = [s for s in mcfg.get("sources", []) if s != "pixazo"]
    try:
        candidates = _candidates_for(queries, sources, allowed, min_w)
    except Exception as e:  # noqa: BLE001 — providers are best-effort, not a reason to publish the old frame
        log.info("    QA re-search failed for seg %d: %s", seg, str(e)[:120])
        return False
    used_urls = {str(s.get("credit", {}).get("url", "")) for s in visuals["shots"]}
    for pool in visuals.get("alts", {}).values():
        used_urls.update(str(c.get("url", "")) for c in pool)
    candidates = [c for c in candidates if c.get("url") and str(c["url"]) not in used_urls]
    if not candidates:
        log.info("    QA re-search returned no new assets for seg %d", seg)
        return False

    judge = vision.Judge(cfg, llm)
    if not judge.available():
        raise vision.VisionUnavailable("no vision model available — refusing unverified QA-repair images")
    subject = str(source.get("title", ""))
    context = str(source.get("description", ""))
    judged = judge.score(candidates, want, line, subject, context)
    floor = max(8.0, float(cfg.get("qa", {}).get("repair_min_match_score", 8)))
    good = [c for c in judged if float(c.get("vscore", 0)) >= floor]
    if not good:
        judged += judge.score(candidates, want, line, subject, context, page=1)
        good = [c for c in judged if float(c.get("vscore", 0)) >= floor]
    video_bonus = float(mcfg.get("video_bonus", 1.0)) if mcfg.get("prefer_video", True) else 0.0
    shape_w = float(mcfg.get("shape_bonus", 1.0))
    good.sort(key=lambda c: -(float(c.get("vscore", 0)) +
                             (video_bonus if c.get("kind") == "video" else 0.0) + shape_bonus(c, shape_w)))
    for candidate in good:
        if _activate_qa_candidate(visuals, seg, bad_path, work, candidate, want):
            if isinstance(segment.get("visual"), dict):
                segment["visual"].update({"shows": want, "queries": queries})
            pool = visuals.setdefault("alts", {}).setdefault(seg, [])
            for spare in good:
                if spare is not candidate and len(pool) < 6:
                    pool.append(spare)
            log.info("    QA re-search: seg %d → %s (%.1f/10; %s)",
                     seg, candidate.get("vshows", "")[:55], float(candidate["vscore"]), ", ".join(queries))
            return True
    log.info("    QA re-search had no distinct usable image scoring ≥ %.1f for seg %d", floor, seg)
    return False


def use_alternative(visuals: dict, seg: int, bad_path: str, work: Path, *,
                    source: dict | None = None, segment: dict | None = None, cfg: dict | None = None,
                    llm=None, feedback: str = "") -> bool:
    """Use a verified spare, then re-search from final-frame QA evidence; never waive that final gate."""
    for candidate in list(visuals["alts"].get(seg, [])):
        visuals["alts"][seg].remove(candidate)
        if _activate_qa_candidate(visuals, seg, bad_path, work, candidate):
            return True
    if source and segment and cfg:
        return _find_final_qa_alternative(visuals, seg, bad_path, work, source, segment, cfg, llm, feedback)
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
