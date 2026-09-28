"""Pixel-level relevance checks for candidate visuals.

Why: file titles lie. "Whiskers" finds a bar called *Satan's Whiskers* and a street
called *Whiskers Lane*. Every image that reaches a video must be *looked at* and
confirmed to show what the narration is talking about.

Two judges, both free:
  1. CLIP (open_clip ViT-B-32, runs locally on the CPU) – fast image↔text similarity.
     Used to pre-rank dozens of candidates, and as the fallback judge.
  2. A vision LLM (Gemini free tier via GEMINI_API_KEY; Groq/OpenRouter if keyed) –
     sees a numbered contact sheet of the best candidates and scores each 0-10
     against the exact narration line, rejecting same-name buildings, signs, logos,
     text/maps/diagrams, watermarks, etc.

`Judge.score()` sets on each candidate: vscore (0-10), vshows (what it depicts), vjudge.
"""
from __future__ import annotations

import io
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageDraw, ImageFont

from .common import font_path, http

log = logging.getLogger("autotube.vision")

NEGATIVES = [
    "a building", "a house", "a bar or pub", "a shop front", "a street", "a road sign", "a sign with text",
    "a logo", "a page of text", "a document", "a map", "a diagram", "a chart", "a screenshot",
    "a crowd of people", "a portrait of a man", "a portrait of a woman", "a car", "a painting",
    "a landscape", "food on a plate", "a room interior", "a cartoon", "a product package", "a poster",
]

# ── thumbnails ────────────────────────────────────────────────────────────────
_COMMONS_RE = re.compile(r"^(https://upload\.wikimedia\.org/wikipedia/commons)/(thumb/)?([0-9a-f])/([0-9a-f]{2})/([^/]+)")


def thumb_url(c: dict, px: int = 500) -> str:
    if c.get("thumb"):
        return c["thumb"]
    u = c["url"]
    m = _COMMONS_RE.match(u)
    if m:
        base, _, a, ab, name = m.groups()
        ext = name.rsplit(".", 1)[-1].lower()
        suffix = ".jpg" if ext in ("tif", "tiff") else ""
        return f"{base}/thumb/{a}/{ab}/{name}/{px}px-{name}{suffix}"
    return u


def open_small(data: bytes, px: int) -> Image.Image:
    """Decode an image without ever materialising a huge bitmap (JPEG draft mode = DCT-domain downscale)."""
    im = Image.open(io.BytesIO(data))
    if im.width * im.height > 60_000_000:
        raise ValueError(f"image too large {im.size}")
    im.draft("RGB", (px, px))
    im = im.convert("RGB")
    im.thumbnail((px, px))
    return im


def _fetch_thumb(c: dict) -> Image.Image | None:
    if c.get("_img") is not None:
        return c["_img"]
    for url in dict.fromkeys([thumb_url(c), c["url"]]):
        try:
            r = http().get(url, timeout=25)
            r.raise_for_status()
            if len(r.content) > 25_000_000:
                continue
            im = open_small(r.content, 512)
            c["_img"] = im
            return im
        except Exception as e:  # noqa: BLE001
            log.debug("thumb failed %s: %s", url[:90], e)
    c["_img"] = None
    return None


def fetch_thumbs(cands: list[dict]) -> list[dict]:
    todo = [c for c in cands if "_img" not in c]
    if todo:
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(_fetch_thumb, todo))
    return [c for c in cands if c.get("_img") is not None]


def ahash(im: Image.Image) -> int:
    g = im.convert("L").resize((8, 8))
    px = list(g.tobytes())
    avg = sum(px) / 64
    return sum(1 << i for i, v in enumerate(px) if v > avg)


def same_image(h1: int, h2: int) -> bool:
    return bin(h1 ^ h2).count("1") <= 6


# ── contact sheet ─────────────────────────────────────────────────────────────
def contact_sheet(imgs: list[Image.Image], tile: int = 340, cols: int = 3) -> bytes:
    rows = math.ceil(len(imgs) / cols)
    gap = 6
    sheet = Image.new("RGB", (cols * tile + (cols + 1) * gap, rows * tile + (rows + 1) * gap), (40, 40, 40))
    font = ImageFont.truetype(font_path(), 46)
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(imgs):
        t = im.copy()
        t.thumbnail((tile, tile))
        x = gap + (i % cols) * (tile + gap)
        y = gap + (i // cols) * (tile + gap)
        sheet.paste(t, (x + (tile - t.width) // 2, y + (tile - t.height) // 2))
        d.rectangle((x, y, x + 58, y + 60), fill=(255, 215, 0))
        d.text((x + 29, y + 30), str(i + 1), font=font, fill=(0, 0, 0), anchor="mm")
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=85)
    return buf.getvalue()


# ── CLIP ──────────────────────────────────────────────────────────────────────
class _Clip:
    _inst = None

    def __init__(self, model: str, pretrained: str):
        import open_clip  # noqa: WPS433 (optional heavy dependency)
        import torch
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.torch = torch
        self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.eval()
        self.tok = open_clip.get_tokenizer(model)
        self._tcache: dict[str, object] = {}

    @classmethod
    def get(cls, model="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        if cls._inst is None:
            try:
                cls._inst = cls(model, pretrained)
                log.info("CLIP %s/%s loaded", model, pretrained)
            except Exception as e:  # noqa: BLE001
                log.warning("CLIP unavailable (%s) — vision LLM only", e)
                cls._inst = False
        return cls._inst or None

    def text(self, texts: list[str]):
        new = [t for t in texts if t not in self._tcache]
        if new:
            with self.torch.no_grad():
                f = self.model.encode_text(self.tok(new))
                f = f / f.norm(dim=-1, keepdim=True)
            for t, v in zip(new, f):
                self._tcache[t] = v
        return self.torch.stack([self._tcache[t] for t in texts])

    def images(self, imgs: list[Image.Image]):
        with self.torch.no_grad():
            f = self.model.encode_image(self.torch.stack([self.pre(i) for i in imgs]))
            return f / f.norm(dim=-1, keepdim=True)

    def score(self, cands: list[dict], want: str, subject: str) -> None:
        """clip = 0-10: probability the image is `want`/`subject` rather than a generic distractor."""
        pos = [f"a photo of {want}", f"a photo of {subject}"]
        tf_pos = self.text(pos)
        tf_neg_all = self.text(NEGATIVES)
        # drop distractors that describe the wanted thing itself (e.g. "a building" for a dam video)
        sim = (tf_neg_all @ tf_pos.T).max(dim=1).values
        keep = [i for i, s in enumerate(sim.tolist()) if s < 0.80]
        tf_neg = tf_neg_all[keep]
        for i in range(0, len(cands), 32):
            batch = cands[i:i + 32]
            imf = self.images([c["_img"] for c in batch])
            raw_pos = imf @ tf_pos.T                     # cosine
            raw_neg = imf @ tf_neg.T
            for j, c in enumerate(batch):
                probs = []
                for k in range(2):
                    logits = self.torch.cat([raw_pos[j, k:k + 1], raw_neg[j]]) * 100
                    probs.append(float(logits.softmax(0)[0]))
                p = max(probs[0], probs[1] * 0.9)
                cos = float(raw_pos[j].max())
                c["clip"] = round(10 * p * min(1.0, max(0.0, (cos - 0.15) / 0.10)), 2)
                c["clip_cos"] = round(cos, 3)


# ── the judge ─────────────────────────────────────────────────────────────────
VISION_SYSTEM = ("You are a strict photo editor for an educational short-video channel. You check whether candidate "
                 "images really show what the narration is talking about. Judge ONLY the pixels, never file names. "
                 "Return strict JSON only.")


class Judge:
    _down_until = 0.0          # shared across videos in a run: back off after repeated outages

    def __init__(self, cfg: dict, llm=None):
        m = cfg.get("media", {})
        self.llm = llm
        self.use_clip = m.get("clip", True)
        self.sheet_size = int(m.get("vision_sheet_size", 9))
        self.clip_model = m.get("clip_model", "ViT-B-32")
        self.clip_pretrained = m.get("clip_pretrained", "laion2b_s34b_b79k")
        self.calls = 0
        self.fail_streak = 0
        self.llm_failed = False

    @property
    def clip(self):
        return _Clip.get(self.clip_model, self.clip_pretrained) if self.use_clip else None

    def available(self) -> bool:
        return bool((self.llm and self.llm.vision_available()) or self.clip)

    def _llm_score(self, cands: list[dict], want: str, narration: str, subject: str, context: str) -> bool:
        if not self.llm or self.llm_failed or not self.llm.vision_available() or time.time() < Judge._down_until:
            return False
        n = len(cands)
        user = f"""VIDEO TOPIC: {subject}{f' — {context}' if context else ''}
NARRATION LINE: "{narration}"
SHOT NEEDED: {want}

The attached contact sheet has {n} candidate images, numbered 1-{n} (black number in the yellow box at each tile's top-left).
For EACH image: say what it literally shows, then score 0-10 how well it works as the on-screen visual while this narration line plays.
  10-9 = clearly and recognisably shows exactly the thing being narrated
  8-7  = clearly shows the video's topic in a way that fits this line
  6-4  = loosely related, generic, or the subject is tiny/unclear
  3-0  = wrong thing: a building, bar, street, sign, shop, logo, product, person or place that merely SHARES THE NAME;
         mostly text, a map, diagram, chart, document or screenshot; heavy watermark; very blurry; graphic or disturbing
Return JSON: {{"images": [{{"n": 1, "shows": "<= 12 words", "score": 0}}]}} with one entry per image."""

        def validate(o):
            items = o.get("images")
            assert isinstance(items, list) and len(items) >= max(1, n - 1), f"need {n} entries in 'images'"
            for it in items:
                int(it["n"]), float(it["score"])

        try:
            self.calls += 1
            out = self.llm.vision_json(VISION_SYSTEM, user, [contact_sheet([c["_img"] for c in cands])], validate)
        except Exception as e:  # noqa: BLE001
            codes = re.findall(r"^\s*(\S+?):vision: .*?(HTTP \d{3}|no \w+_API_KEY|\w+Error)", str(e), re.M)
            summary = ", ".join(f"{m} {c}" for m, c in dict(codes).items()) or str(e)[:300]
            self.fail_streak += 1
            if self.fail_streak >= 2:
                self.llm_failed = True
                Judge._down_until = time.time() + 15 * 60
                log.warning("vision LLM failed twice in a row (%s) — CLIP-only for the next 15 min", summary)
            else:
                log.warning("vision LLM call failed (%s) — CLIP for this line, will retry on the next", summary)
            return False
        self.fail_streak = 0
        by_n = {int(it["n"]): it for it in out["images"]}
        for i, c in enumerate(cands, 1):
            it = by_n.get(i)
            if it:
                c["vscore"] = max(0.0, min(10.0, float(it["score"])))
                c["vshows"] = str(it.get("shows", ""))[:120]
                c["vjudge"] = self.llm.last_used
        return True

    def score(self, cands: list[dict], want: str, narration: str, subject: str, context: str = "",
              keep: int | None = None) -> list[dict]:
        """Score candidates; returns them sorted best-first (only those with a verdict)."""
        cands = fetch_thumbs(cands)
        if not cands:
            return []
        for c in cands:
            c.setdefault("_hash", ahash(c["_img"]))
        clip = self.clip
        if clip:
            clip.score(cands, want, subject)
            cands.sort(key=lambda c: -c["clip"])
            cands = [c for c in cands if c["clip"] >= 1.0] or cands[:3]   # obvious junk never reaches the LLM
        top = cands[: keep or self.sheet_size]
        pending = [c for c in top if "vscore" not in c]
        if pending and not self._llm_score(pending, want, narration, subject, context) and clip:
            for c in pending:          # CLIP-only verdict: stricter mapping (needs clip ≥ 9.0 to reach 7)
                c["vscore"] = round(max(0.0, (c["clip"] - 3.6) * 1.3), 1)
                c["vshows"] = f"clip match {c['clip']:.1f}/10 (cos {c['clip_cos']})"
                c["vjudge"] = "clip"
        judged = [c for c in top if "vscore" in c]
        judged.sort(key=lambda c: (-c["vscore"], -c.get("clip", 0)))
        return judged
