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


def pexels_video_search(query: str, limit: int = 5) -> list[dict]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        return []
    try:
        data = get_json("https://api.pexels.com/videos/search", params={"query": query, "per_page": limit},
                        headers={"Authorization": key})
    except Exception as e:  # noqa: BLE001
        log.debug("pexels video failed: %s", e)
        return []
    out = []
    for v in data.get("videos", []):
        files = [f for f in v.get("video_files", []) if f.get("link") and (f.get("height") or 0) >= 720]
        if not files or not (VIDEO_MIN_S <= float(v.get("duration") or 0) <= VIDEO_MAX_S):
            continue
        f = sorted(files, key=lambda f: f.get("width", 0) * f.get("height", 0))[0]
        pics = v.get("video_pictures") or []
        thumb = pics[len(pics) // 2]["picture"] if pics else v.get("image")
        out.append({"kind": "video", "url": f["link"], "thumb": thumb, "page": v.get("url"),
                    "title": (v.get("url") or query).rstrip("/").rsplit("/", 1)[-1][:70] + " (video)",
                    "license": "Pexels License", "author": (v.get("user") or {}).get("name", "Pexels"),
                    "source": "Pexels", "duration": float(v.get("duration") or 0), "w": f.get("width"),
                    "h": f.get("height")})
    return out


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
            with http().get(url, timeout=60, stream=True) as r:
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
        if qi < 2 and "pexels_video" in sources and os.environ.get("PEXELS_API_KEY"):
            jobs.append((pexels_video_search, (q, 4)))
        if "pexels" in sources and os.environ.get("PEXELS_API_KEY"):
            jobs.append((pexels_search, (q, 6)))
        if "wikimedia" in sources:
            jobs.append((commons_search, (q, allowed, min_w, 10)))
        if "openverse" in sources and qi < 2:
            jobs.append((openverse_search, (q, min_w, 8)))
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
    video_bonus = float(mcfg.get("video_bonus", 1.0)) if mcfg.get("prefer_video", True) else 0.0
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
        """Approved candidates, best first; real moving footage gets a bonus over an equally good still."""
        good = [c for c in judged if c["vscore"] >= min_score]
        return sorted(good, key=lambda c: -(c["vscore"] + (video_bonus if c.get("kind") == "video" else 0)))

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
                      "kind": c.get("kind", "image"), "poster": c.get("_poster"), "window": c.get("window")})
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

    distinct = len({s["path"] for s in shots})
    need = int(mcfg.get("min_distinct_images", 3))
    if distinct < need:
        raise NoVisualMatch(f"only {distinct} distinct verified images for {subject!r} (need {need})")
    shots.sort(key=lambda s: (s["seg"], s.get("reused", False)))
    n_vid = sum(1 for s in shots if s.get("kind") == "video")
    log.info("  visuals: %d shots (%d video clips), %d distinct, every line verified, %d vision calls", len(shots),
             n_vid, distinct, judge.calls)
    return {"shots": shots, "alts": alts, "calls": judge.calls, "need": need_by_seg, "max_mb": max_mb}


def use_alternative(visuals: dict, seg: int, bad_path: str, work: Path) -> bool:
    """Replace a shot that failed final QA with the next verified spare for that line."""
    from . import vision  # noqa: F401
    for c in list(visuals["alts"].get(seg, [])):
        visuals["alts"][seg].remove(c)
        p = _usable(c, work, visuals.get("need", {}).get(seg, 4.0), visuals.get("max_mb", 90))
        if not p or str(p) == str(bad_path) or any(str(s["path"]) == str(p) for s in visuals["shots"]):
            continue
        for s in visuals["shots"]:
            if s["seg"] == seg and str(s["path"]) == str(bad_path):
                credit = {k: v for k, v in c.items() if not k.startswith("_") and k not in
                          ("vscore", "vshows", "vjudge", "clip", "clip_cos", "thumb", "variants", "t_ref", "window")}
                s.update({"path": p, "credit": credit, "score": c["vscore"], "shows": c.get("vshows", ""),
                          "judge": c.get("vjudge", ""), "reused": False, "kind": c.get("kind", "image"),
                          "poster": c.get("_poster"), "window": c.get("window")})
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
