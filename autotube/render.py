"""Video renderer: 1080x1920 Short with Ken Burns motion, crossfades, karaoke
captions, hook title card, progress bar, ducked procedural music, loudness norm.

Everything is ONE ffmpeg filter graph → fast, reproducible, zero paid tools.
"""
from __future__ import annotations

import logging
import random
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import motion, sfx
from .common import FONTS_DIR, ffmpeg_bin, font_path

log = logging.getLogger("autotube.render")

XF = 0.30          # crossfade seconds
TAIL = 0.7         # silence after last word
SS = 1.5           # supersampling factor for smooth zoompan
CAPTION_Y = 0.62   # caption baseline as a fraction of H (keeps clear of the Shorts UI chrome)
HOOK_CARD_SECONDS = 1.6   # hook title card: long enough to read, short enough not to fight the captions

# Visual archetypes. YouTube's Generic or Repetitive Content policy is assessed channel-level and
# explicitly targets "near-identical videos with minimal variation", so varying only the caption
# colour is not enough — every video had the same typeface, the same caption height, the same hook
# card timing and the same end card. These are the three bundled fonts (family names verified
# against the TTFs; a wrong name makes libass fall back silently to DejaVu and the variation
# disappears with no error anywhere — "Montserrat" does exactly that, it must be "Montserrat Black".
# tests/test_archetypes.py asserts every one of these resolves to a bundled file.)
ARCHETYPES = [
    {"name": "bold",      "font": "Anton",      "caption_y": 0.62, "cap_scale": 1.00,
     "hook_seconds": 1.6, "stamp": True},
    {"name": "condensed", "font": "Bebas Neue", "caption_y": 0.55, "cap_scale": 1.12,
     "hook_seconds": 1.2, "stamp": False},
    {"name": "heavy",     "font": "Montserrat Black", "caption_y": 0.66, "cap_scale": 0.88,
     "hook_seconds": 2.0, "stamp": True},
]

# caption colour themes (ASS colours are &HAABBGGRR) — rotated per video for variety
THEMES = [
    {"hi": "&H0000F0FF", "box": "&H0000D7FF", "boxtxt": "&H00101010"},   # yellow
    {"hi": "&H0066FF33", "box": "&H0066FF33", "boxtxt": "&H00101010"},   # green
    {"hi": "&H00FFD000", "box": "&H00FFC400", "boxtxt": "&H00101010"},   # cyan
    {"hi": "&H003C8CFF", "box": "&H003C8CFF", "boxtxt": "&H00FFFFFF"},   # orange
    {"hi": "&H00FF66E0", "box": "&H00FF66E0", "boxtxt": "&H00FFFFFF"},   # pink
]


# ── image prep ────────────────────────────────────────────────────────────────
# Full-bleed framing. A letterboxed picture floating on a blurred copy of itself is the single
# clearest "automated channel" tell on Shorts, and it throws away ~60% of the screen. Every shot now
# COVERS the 9:16 canvas; the crop window is chosen by edge energy so the subject survives the crop.
MAX_COVER_ASPECT = 2.60     # wider than this (panoramas) would lose too much → fall back to the inset
COVER_BIAS_Y = 0.42         # when cropping a tall source, keep slightly above centre (faces/subjects sit high)


def saliency_offset(im: Image.Image, cw: int, ch: int) -> tuple[int, int]:
    """Top-left of the cover-crop window, chosen by edge energy (cheap saliency, no extra deps).

    `im` is already resized to at least (cw, ch). Slides the crop window over a tiny edge map and
    keeps the position with the most detail, so a centre-crop never decapitates the subject.
    """
    ex, ey = im.width - cw, im.height - ch
    if ex <= 0 and ey <= 0:
        return max(0, ex // 2), max(0, ey // 2)
    try:
        SM = 64                                   # work on a ~64px-wide edge map: microseconds, good enough
        sw = SM
        sh = max(1, int(im.height * SM / im.width))
        small = im.convert("L").resize((sw, sh), Image.BILINEAR).filter(ImageFilter.FIND_EDGES)
        px = small.load()
        # integral image of edge energy
        acc = [[0] * (sw + 1) for _ in range(sh + 1)]
        for y in range(sh):
            row_sum = 0
            for x in range(sw):
                row_sum += px[x, y]
                acc[y + 1][x + 1] = acc[y][x + 1] + row_sum
        kx = max(1, round(cw * sw / im.width))
        ky = max(1, round(ch * sh / im.height))
        best, bx, by = -1.0, 0, 0
        for sy in range(0, max(1, sh - ky + 1)):
            for sx in range(0, max(1, sw - kx + 1)):
                e = (acc[sy + ky][sx + kx] - acc[sy][sx + kx] - acc[sy + ky][sx] + acc[sy][sx])
                # mild pull toward the composition sweet spot so flat images don't crop off-centre
                dx = abs((sx + kx / 2) / sw - 0.5)
                dy = abs((sy + ky / 2) / sh - COVER_BIAS_Y)
                e *= (1.0 - 0.18 * dx - 0.18 * dy)
                if e > best:
                    best, bx, by = e, sx, sy
        x = int(round(bx * im.width / sw))
        y = int(round(by * im.height / sh))
        return max(0, min(x, max(ex, 0))), max(0, min(y, max(ey, 0)))
    except Exception as e:  # noqa: BLE001 — never let framing break a render
        log.debug("saliency crop failed (%s) → centre crop", e)
        return max(0, ex // 2), max(0, int(ey * COVER_BIAS_Y))


def cover(im: Image.Image, cw: int, ch: int) -> Image.Image:
    """Resize + saliency-crop `im` so it exactly fills (cw, ch) with no bars."""
    r = max(cw / im.width, ch / im.height)
    full = im.resize((max(cw, int(im.width * r) + 2), max(ch, int(im.height * r) + 2)), Image.LANCZOS)
    l, t = saliency_offset(full, cw, ch)
    return full.crop((l, t, l + cw, t + ch))


def compose_frame(src: Path, dst: Path, W: int, H: int) -> Path:
    """Portrait canvas, full-bleed: the shot fills the whole 9:16 frame (saliency-aware cover crop).

    Only an extreme panorama falls back to an inset over a blurred bed, and even then the inset is
    large and the bed is zoomed, so there is no dead letterbox band.
    """
    cw, ch = int(W * SS), int(H * SS)
    with Image.open(src) as im:
        im = im.convert("RGB")
        aspect = im.width / im.height
        if aspect <= MAX_COVER_ASPECT:
            cover(im, cw, ch).save(dst, "JPEG", quality=92)
            return dst
        # panorama: showing it whole IS the point — inset it big over a zoomed, blurred bed
        bg = cover(im, cw, ch).filter(ImageFilter.GaussianBlur(40))
        bg = Image.eval(bg, lambda v: int(v * 0.40))
        fw = cw
        fh = max(2, int(fw / aspect))
        fg = im.resize((fw, fh), Image.LANCZOS)
        pos = ((cw - fw) // 2, int(ch * 0.44 - fh / 2))
        sh = Image.new("RGBA", (fw + 80, fh + 80), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rectangle((40, 40, fw + 40, fh + 40), fill=(0, 0, 0, 170))
        sh = sh.filter(ImageFilter.GaussianBlur(25))
        bg.paste(sh, (pos[0] - 40, pos[1] - 25), sh)
        bg.paste(fg, pos)
        bg.save(dst, "JPEG", quality=92)
    return dst


# ── captions (ASS) ────────────────────────────────────────────────────────────
def _ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(s: str) -> str:
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


EMPH_FOR = {0: 2, 1: 4, 2: 0, 3: 2, 4: 1}   # yellow→cyan, green→pink, cyan→yellow, orange→cyan, pink→green
NUM_RE = re.compile(r"^(\$?)(\d[\d,]*(?:\.\d+)?)(%?)$")
SCALE_WORDS = {"thousand", "million", "billion", "trillion", "percent"}
STAMP_GREEN = "&H0040D83A"       # ASS BGR
ASIDE_COL = "&H00C8F0FF"         # soft cream for the narrator's asides


def _norm_word(w: str) -> str:
    return re.sub(r"[^\w%$']", "", w.lower())


def _number(word: str, scaled: bool = False) -> tuple[str, float, int, str] | None:
    """'$1,500' → ('$', 1500.0, decimals, '%'|'') ; years (1000-2100 without comma) and < 10 are not counted up."""
    m = NUM_RE.match(word.strip(".,!?;:\"'()"))
    if not m:
        return None
    pre, num, pct = m.groups()
    try:
        v = float(num.replace(",", ""))
    except ValueError:
        return None
    dec = len(num.split(".")[1]) if "." in num else 0
    if (v < 10 and not (scaled and v > 0)) or ("," not in num and not pre and not pct and 1000 <= v <= 2100 and dec == 0):
        return None
    return pre, v, dec, pct


def _fmt_num(v: float, dec: int, commas: bool) -> str:
    if dec:
        return f"{v:,.{dec}f}" if commas else f"{v:.{dec}f}"
    return f"{int(round(v)):,}" if commas else str(int(round(v)))


def fx_lines(segments: list[dict], fx: dict, theme: dict, emph_col: str, W: int, H: int, total: float
             ) -> tuple[list[str], list[float], float | None, float | None]:
    """Kinetic extras: number count-ups, reveal flash, 3-2-1 countdown, PROVEN stamp.
    Returns (ASS dialogue lines, times numbers are spoken, reveal time, stamp time)."""
    lines: list[str] = []
    num_times: list[float] = []
    kinetic = fx.get("kinetic", True)
    asides = fx.get("asides") or []
    last_count = -9.0
    for si, seg in enumerate(segments):
        words = [w for w in seg["words"] if w["word"].strip()]
        k = len((asides[si] if si < len(asides) else "").split())
        facts = words[:len(words) - k] if k else words
        for wi, w in enumerate(facts):
            nxt0 = _norm_word(facts[wi + 1]["word"]) if wi + 1 < len(facts) else ""
            n = _number(w["word"], scaled=nxt0 in SCALE_WORDS - {"percent"})     # "1.7 billion" counts up too
            if re.search(r"\d", w["word"]) and (not num_times or w["start"] - num_times[-1] > 1.2):
                num_times.append(w["start"])
            if not (n and kinetic) or w["start"] - last_count < 2.5 or len([x for x in lines if "Count" in x]) >= 36:
                continue
            last_count = w["start"]
            pre, v, dec, pct = n
            nxt = _norm_word(facts[wi + 1]["word"]) if wi + 1 < len(facts) else ""
            suffix = pct + (" " + nxt.upper() if nxt in SCALE_WORDS else "")
            commas = "," in w["word"]
            st, steps, dur = w["start"], 12, 0.55
            for j in range(steps + 1):
                val = v * (1 - (1 - j / steps) ** 3)
                a = st + dur * j / steps
                b = st + dur * (j + 1) / steps if j < steps else min(total, st + dur + 1.1)
                pop = "\\t(0,120,\\fscx118\\fscy118)\\t(120,240,\\fscx100\\fscy100)" if j == steps else ""
                lines.append(f"Dialogue: 3,{_ts(a)},{_ts(b)},Count,,0,0,0,,{{\\pos({W // 2},{int(H * 0.30)})"
                             f"\\c{emph_col}{pop}}}{pre}{_fmt_num(val, dec, commas)}{_esc(suffix)}")
    reveal_at = None
    ri = fx.get("reveal")
    if ri is not None and 0 < ri < len(segments):
        reveal_at = segments[ri]["start"]
        if kinetic:
            # white flash on the reveal
            lines.append(f"Dialogue: 4,{_ts(reveal_at)},{_ts(reveal_at + 0.32)},Fx,,0,0,0,,"
                         f"{{\\an7\\pos(0,0)\\alpha&H78&\\fad(20,240)\\p1}}m 0 0 l {W} 0 l {W} {H} l 0 {H}{{\\p0}}")
        if fx.get("countdown"):
            a0 = segments[ri - 1]["end"]
            span = max(0.6, reveal_at - a0)
            lines.append(f"Dialogue: 3,{_ts(a0)},{_ts(reveal_at)},Label,,0,0,0,,{{\\pos({W // 2},{int(H * 0.25)})"
                         f"\\fad(80,0)}}LOCK IN YOUR GUESS")
            for j, d in enumerate("321"):
                a = a0 + span * j / 3
                lines.append(f"Dialogue: 3,{_ts(a)},{_ts(a + span / 3)},Countdown,,0,0,0,,{{\\pos({W // 2},{int(H * 0.42)})"
                             f"\\c{theme['hi']}\\fscx160\\fscy160\\t(0,140,\\fscx100\\fscy100)}}{d}")
    stamp_at = None
    st_cfg = fx.get("stamp")
    if st_cfg and total > 8:
        stamp_at = max(total - float(st_cfg.get("seconds", 2.0)), total * 0.6)
        cx, cy = W // 2, int(H * 0.40)
        bw, bh, th = 780, 220, 14
        org = f"\\org({cx},{cy})"
        slam = "\\fscx190\\fscy190\\alpha&H60&\\t(0,130,\\fscx100\\fscy100\\alpha&H00&)"
        frame = (f"m 0 0 l {bw} 0 l {bw} {bh} l 0 {bh} l 0 0 "
                 f"m {th} {th} l {th} {bh - th} l {bw - th} {bh - th} l {bw - th} {th} l {th} {th} ")
        check = "m 48 118 l 84 84 l 118 118 l 190 44 l 226 80 l 118 188 l 48 118"
        lines.append(f"Dialogue: 5,{_ts(stamp_at)},{_ts(total)},Stamp,,0,0,0,,{{\\an5\\pos({cx},{cy}){org}\\frz8"
                     f"{slam}\\p1}}{frame}{check}{{\\p0}}")
        lines.append(f"Dialogue: 5,{_ts(stamp_at)},{_ts(total)},StampTxt,,0,0,0,,{{\\an5\\pos({cx + 95},{cy + 2}){org}"
                     f"\\frz8{slam}}}PROVEN")
        src = _esc(str(st_cfg.get("source", "")))[:60]
        if src:
            lines.append(f"Dialogue: 5,{_ts(stamp_at + 0.15)},{_ts(total)},Source,,0,0,0,,{{\\an5\\pos({cx},{cy + 190})"
                         f"\\fad(120,0)}}{src}")
    return lines, num_times, reveal_at, stamp_at


HOOK_FS = 96
HOOK_MAX_LINES = 2        # a 3-4 line hook card grows down into the Count/Label zone at 0.25-0.30 H


def _hook_lines(text: str, fs: int = HOOK_FS, usable: int = 900) -> int:
    """Lines the hook card will wrap to. Condensed display faces average ~0.47 em per glyph."""
    cpl = max(8, int(usable / (fs * 0.47)))
    n, cur = 1, 0
    for w in text.split():
        if cur and cur + 1 + len(w) > cpl:
            n, cur = n + 1, len(w)
        else:
            cur += (1 if cur else 0) + len(w)
    return n


def fit_hook(text: str, max_lines: int = HOOK_MAX_LINES) -> str:
    """Trim the hook card to `max_lines`, on a word boundary.

    The card is top-anchored and grows downwards, so a long hook silently overlaps the big number
    counter and the "LOCK IN YOUR GUESS" label, both of which can fire inside the card's first
    1.6 s. Trimming is the right trade: the hook card is a glance-and-read title, not the script.
    """
    words = text.split()
    if not words:
        return ""
    trimmed = False
    while len(words) > 1 and _hook_lines(" ".join(words)) > max_lines:
        words.pop()
        trimmed = True
    if trimmed:   # never end a trimmed card on "FROM THE"
        while len(words) > 1 and _norm_word(words[-1]) in DANGLERS:
            words.pop()
    return " ".join(words).rstrip(",;:-") or words[0]


# Words that must not be left dangling at the end of a caption card. A chunk ending on "FOR" or
# "THE" reads as a dropped sentence: the eye finishes the card and the thought is not there yet.
DANGLERS = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "from",
            "with", "by", "as", "is", "was", "are", "were", "that", "this", "it", "its", "into",
            "over", "under", "than", "then", "so", "if", "when", "while", "because", "about"}
CHUNK_MAX_WORDS = 4
CHUNK_MAX_CHARS = 24


def _chunk_words(words: list[dict], max_words: int = CHUNK_MAX_WORDS,
                 max_chars: int = CHUNK_MAX_CHARS) -> list[list[dict]]:
    """Group timed words into caption cards of 2-4 words.

    Greedy filling alone produces two bad cards. It strands a single word at the end of a segment -
    usually the payoff, flashed alone for a fraction of a second - and it breaks after function
    words, so a card ends on "FOR" and the sentence appears to stop mid-thought. Both are fixed by
    a rebalancing pass rather than by changing the fill, because the fill is what keeps cards
    readable at speaking pace.
    """
    def chars(ch):
        return len(" ".join(x["word"] for x in ch))

    chunks, cur = [], []
    for wi_, w in enumerate(words):
        cur.append(w)
        nxt_aside = words[wi_ + 1].get("_aside") if wi_ + 1 < len(words) else None
        txt_len = chars(cur)
        hard_stop = w["word"][-1:] in ".!?"          # only sentence ends break a card; commas read badly
        if ((nxt_aside is not None and nxt_aside != w.get("_aside"))
                or len(cur) >= max_words or (len(cur) >= 2 and txt_len >= 18)
                or (hard_stop and len(cur) >= 2) or txt_len > max_chars):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)

    # ── push a dangling function word onto the card that completes the thought ──
    for i in range(len(chunks) - 1):
        a, b = chunks[i], chunks[i + 1]
        if len(a) < 2 or a[-1].get("_aside") != b[0].get("_aside"):
            continue
        # a few characters of overflow is cheaper than a card that ends on "for"
        if _norm_word(a[-1]["word"]) in DANGLERS and len(b) < max_words and chars(b) + len(a[-1]["word"]) + 1 <= max_chars + 4:
            b.insert(0, a.pop())

    # ── never leave a single word alone on the last card of a segment ──
    if len(chunks) >= 2 and len(chunks[-1]) == 1:
        prev, last = chunks[-2], chunks[-1]
        if prev[-1].get("_aside") == last[0].get("_aside"):
            if len(prev) + 1 <= max_words and chars(prev) + chars(last) + 1 <= max_chars + 6:
                prev.extend(last)            # short enough to read as one card
                chunks.pop()
            elif len(prev) >= 3:
                last.insert(0, prev.pop())   # otherwise split 4+1 into 3+2
    return [c for c in chunks if c]


def build_ass(segments: list[dict], hook_text: str, channel: str, W: int, H: int, theme: dict,
              font_name: str, out: Path, total: float, fx: dict | None = None, arch: dict | None = None,
              emph_col: str = "&H003C8CFF") -> Path:
    fx = fx or {}
    arch = arch or ARCHETYPES[0]
    fs = int(112 * float(arch.get("cap_scale", 1.0)))
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font_name},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,5,150,150,0,1
Style: Hook,{font_name},96,{theme['boxtxt']},&H00FFFFFF,{theme['box']},{theme['box']},-1,0,0,0,100,100,1,0,3,18,0,8,90,90,210,1
Style: Brand,{font_name},46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,2,0,1,3,0,8,40,40,90,1
Style: Count,{font_name},200,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,2,0,1,11,4,5,40,40,0,1
Style: Countdown,{font_name},420,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,14,6,5,40,40,0,1
Style: Label,{font_name},78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,3,0,1,7,3,5,40,40,0,1
Style: Fx,{font_name},20,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,&H00FFFFFF,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Stamp,{font_name},20,{STAMP_GREEN},{STAMP_GREEN},&H00101010,&H00000000,0,0,0,0,100,100,0,0,1,3,0,5,0,0,0,1
Style: StampTxt,{font_name},165,{STAMP_GREEN},{STAMP_GREEN},&H00101010,&H00000000,-1,0,0,0,100,100,6,0,1,3,0,5,0,0,0,1
Style: Source,{font_name},46,&H00FFFFFF,&H00FFFFFF,&H00000000,&HB4000000,0,0,0,0,100,100,1,0,3,14,0,5,60,60,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    cy = int(H * float(arch.get("caption_y", CAPTION_Y)))   # Shorts UI overlays the bottom ~18% and right ~15%

    extra, num_times, reveal_at, _ = fx_lines(segments, fx, theme, emph_col, W, H, total) if fx else ([], [], None, None)

    # hook title card, with pop-in. Both the card and the number counter are anchored near the top,
    # so the card has to be gone before the first counter or countdown label appears.
    if hook_text:
        hook_end = min(float(arch.get("hook_seconds", HOOK_CARD_SECONDS)), total)
        top_fx = [t for t in list(num_times or []) if t is not None]
        if fx.get("countdown") and reveal_at is not None:
            top_fx.append(reveal_at)
        if top_fx:
            hook_end = min(hook_end, max(0.5, min(top_fx) - 0.15))
        lines.append(f"Dialogue: 2,{_ts(0)},{_ts(hook_end)},Hook,,0,0,0,,"
                     f"{{\\fad(0,250)\\t(0,180,\\fscx112\\fscy112)\\t(180,320,\\fscx100\\fscy100)}}"
                     f"{_esc(fit_hook(hook_text.upper()))}")
        # Pattern interrupt on the hook-card handoff. A visual disruption inside the first five
        # seconds is worth roughly 23% retention against a static opening, and this is the moment
        # the opening is most static: the card is fading and the first shot has not moved yet. It
        # reuses the reveal flash primitive, shorter and fainter - this is a beat, not the payoff.
        if fx.get("kinetic", False) and hook_end + 0.3 < total:
            lines.append(f"Dialogue: 4,{_ts(hook_end)},{_ts(hook_end + 0.22)},Fx,,0,0,0,,"
                         f"{{\\an7\\pos(0,0)\\alpha&HA0&\\fad(0,180)\\p1}}"
                         f"m 0 0 l {W} 0 l {W} {H} l 0 {H}{{\\p0}}")
    lines.append(f"Dialogue: 1,{_ts(0)},{_ts(total)},Brand,,0,0,0,,{{\\alpha&H40&}}{_esc(channel)}")
    lines += extra
    asides = fx.get("asides") or []
    emphasis = fx.get("emphasis") or []
    kinetic = fx.get("kinetic", False)

    # karaoke word chunks
    for si, seg in enumerate(segments):
        words = [w for w in seg["words"] if w["word"].strip()]
        k_aside = len((asides[si] if si < len(asides) else "").split())
        aside_from = len(words) - k_aside if k_aside else len(words) + 1
        emph = {_norm_word(t) for e in (emphasis[si] if si < len(emphasis) else []) for t in str(e).split()}
        for wi_, w in enumerate(words):
            w["_aside"] = wi_ >= aside_from
            w["_emph"] = kinetic and not w["_aside"] and (_norm_word(w["word"]) in emph or bool(re.search(r"\d", w["word"])))
        chunks = _chunk_words(words)
        for ci, ch in enumerate(chunks):
            c_end = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else seg["end"] + 0.12
            for wi, w in enumerate(ch):
                st = w["start"]
                en = ch[wi + 1]["start"] if wi + 1 < len(ch) else c_end
                parts = []
                for wj, w2 in enumerate(ch):
                    aside = kinetic and w2.get("_aside")
                    txt = _esc(w2["word"].lower() if aside else w2["word"].upper())
                    base = ASIDE_COL if aside else (emph_col if w2.get("_emph") else "&H00FFFFFF&")
                    ital = "\\i1" if aside else ""
                    reset = f"{{\\c{base}\\fscx100\\fscy100\\i0}}" if (aside or w2.get("_emph")) else ""
                    if wj == wi:
                        big = 124 if w2.get("_emph") else 108
                        parts.append(f"{{\\c{emph_col if w2.get('_emph') else theme['hi']}{ital}\\fscx{big}\\fscy{big}}}"
                                     f"{txt}{{\\c&H00FFFFFF&\\fscx100\\fscy100\\i0}}")
                    elif reset:
                        parts.append(f"{{\\c{base}{ital}}}{txt}{{\\c&H00FFFFFF&\\i0}}")
                    else:
                        parts.append(txt)
                pop = "\\t(0,90,\\fscx104\\fscy104)\\t(90,160,\\fscx100\\fscy100)" if wi == 0 else ""
                lines.append(f"Dialogue: 0,{_ts(st)},{_ts(en)},Cap,,0,0,0,,{{\\pos({W // 2},{cy}){pop}}}" + " ".join(parts))
    out.write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    return out


# ── thumbnail ─────────────────────────────────────────────────────────────────
def thumbnail(frame: Path, text: str, out: Path, theme_idx: int) -> Path:
    colours = [(255, 215, 0), (51, 255, 102), (0, 208, 255), (255, 140, 60), (224, 102, 255)]
    col = colours[theme_idx % len(colours)]
    with Image.open(frame) as im:
        im = im.convert("RGB").resize((1080, 1920))
        d = ImageDraw.Draw(im)
        font = ImageFont.truetype(font_path(), 150)
        words = text.upper().split()
        lines, cur = [], ""
        for w in words:
            test = (cur + " " + w).strip()
            if d.textlength(test, font=font) > 940 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = test
        lines.append(cur)
        y = 1250
        for ln in lines[:3]:
            tw = d.textlength(ln, font=font)
            d.text(((1080 - tw) / 2, y), ln, font=font, fill=col, stroke_width=10, stroke_fill=(0, 0, 0))
            y += 170
        im.save(out, "JPEG", quality=88)
    return out


def is_video(v: dict) -> bool:
    return v.get("kind") == "video" or str(v.get("path", "")).lower().endswith((".mp4", ".webm", ".mov"))


def poster_frame(clip: Path, out: Path) -> Path:
    subprocess.run([ffmpeg_bin(), "-loglevel", "error", "-y", "-ss", "0.5", "-i", str(clip), "-frames:v", "1",
                    "-q:v", "2", str(out)], check=True, capture_output=True)
    return out


def video_chain(i: int, aspect: float, W: int, H: int, fps: int) -> str:
    """Same framing rule as a still: the clip fills the whole 9:16 frame (cover crop, biased slightly high).

    Only an extreme panorama keeps the inset-over-blurred-bed layout.
    """
    if aspect <= MAX_COVER_ASPECT:
        # crop toward the upper third of a tall source: subjects/horizons almost never sit dead centre
        return (f"[{i}:v]fps={fps},setpts=PTS-STARTPTS,scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H}:(iw-{W})/2:(ih-{H})*{COVER_BIAS_Y:.2f},setsar=1")
    fw, fh = W, max(2, int(W / aspect) // 2 * 2)
    y = int(H * 0.44 - fh / 2)
    return (f"[{i}:v]fps={fps},setpts=PTS-STARTPTS,setsar=1,split=2[va{i}][vb{i}];"
            f"[va{i}]scale={W // 4}:{H // 4}:force_original_aspect_ratio=increase,crop={W // 4}:{H // 4},"
            f"boxblur=10:2,colorchannelmixer=rr=0.40:gg=0.40:bb=0.40,scale={W}:{H},setsar=1[vbg{i}];"
            f"[vb{i}]scale={fw}:{fh},setsar=1[vfg{i}];[vbg{i}][vfg{i}]overlay=(W-w)/2:{y}")


def shot_plan(visuals: list[dict], n_segs: int) -> list[dict]:
    """Normalise visuals to an ordered shot list covering every segment (old format = one per segment)."""
    if visuals and all("seg" in v for v in visuals):
        shots = sorted([v for v in visuals if v["seg"] < n_segs], key=lambda v: v["seg"])
    else:
        shots = [dict(v, seg=i) for i, v in enumerate(visuals[:n_segs])]
    have = {v["seg"] for v in shots}
    for i in range(n_segs):
        if i not in have and shots:
            shots.append(dict(shots[i % len(shots)], seg=i))
    return sorted(shots, key=lambda v: v["seg"])


def _ff(cmd: list[str], what: str) -> None:
    res = subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error"] + cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({what}):\n" + res.stderr[-2000:])


def _kenburns(mo: str, nf: int) -> tuple[str, str, str]:
    z0, z1 = (1.0, 1.14) if mo != "out" else (1.14, 1.0)
    z = f"{z0}+({z1 - z0})*on/{nf}"
    if mo == "left":
        return z, f"(iw-iw/zoom)*(1-on/{nf})", "(ih-ih/zoom)/2"
    if mo == "right":
        return z, f"(iw-iw/zoom)*on/{nf}", "(ih-ih/zoom)/2"
    if mo == "up":
        return z, "(iw-iw/zoom)/2", f"(ih-ih/zoom)*(1-on/{nf})"
    return z, "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"


def build_video_track(frames: list[Path], clip_aspect: dict, clip_src: dict, durs: list[float], motions: list[str],
                      transitions: list[str], total: float, W: int, H: int, fps: int, work: Path) -> Path:
    """Silent picture track, built piece by piece so memory stays flat however many clips there are.

    1. every shot is rendered on its own at exactly its frame count (+ the crossfade overlap);
    2. piece i = shot i's solo part + the crossfade into shot i+1 (only two small files open at once);
    3. pieces are joined with the concat demuxer.
    Boundaries are quantised to whole frames, so pictures stay locked to the narration timeline."""
    n = len(frames)
    NF = int(round(total * fps))
    offs = [int(round(sum(durs[:i]) * fps)) for i in range(n)] + [NF]
    X = int(round(XF * fps))
    enc = ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(fps)]
    cw, ch = int(W * SS), int(H * SS)
    shot_files = []
    for i in range(n):
        nf = max(2, offs[i + 1] - offs[i] + (X if i < n - 1 else 0))
        out = work / f"shot{i:02d}.mp4"
        if i in clip_aspect:
            chain = video_chain(0, clip_aspect[i], W, H, fps)
            _ff(["-stream_loop", "-1", "-i", str(clip_src[i]), "-filter_complex",
                 chain + f",format=yuv420p[v]", "-map", "[v]", "-frames:v", str(nf)] + enc + [str(out)], f"shot {i}")
        else:
            z, x, y = _kenburns(motions[i % len(motions)], nf)
            _ff(["-i", str(frames[i]), "-vf", f"scale={cw}:{ch},setsar=1,zoompan=z='{z}':x='{x}':y='{y}':d={nf}:"
                 f"s={W}x{H}:fps={fps},format=yuv420p", "-frames:v", str(nf)] + enc + [str(out)], f"shot {i}")
        shot_files.append(out)
    pieces = []
    for i in range(n):
        a = X if i > 0 else 0
        b = offs[i + 1] - offs[i]
        out = work / f"piece{i:02d}.mp4"
        body = f"trim=start_frame={a}:end_frame={max(b, a + 1)},setpts=PTS-STARTPTS,fps={fps},settb=1/{fps}"
        if i < n - 1:
            fc = (f"[0:v]split=2[s0][s1];[s0]{body}[bo];"
                  f"[s1]trim=start_frame={b}:end_frame={b + X},setpts=PTS-STARTPTS,fps={fps},settb=1/{fps}[ta];"
                  f"[1:v]trim=end_frame={X},setpts=PTS-STARTPTS,fps={fps},settb=1/{fps}[hd];"
                  f"[ta][hd]xfade=transition={transitions[i]}:duration={X / fps:.4f}:offset=0[tr];"
                  f"[bo][tr]concat=n=2:v=1:a=0,format=yuv420p[v]")
            _ff(["-i", str(shot_files[i]), "-i", str(shot_files[i + 1]), "-filter_complex", fc, "-map", "[v]"]
                + enc + [str(out)], f"piece {i}")
        else:
            _ff(["-i", str(shot_files[i]), "-vf", body + ",format=yuv420p"] + enc + [str(out)], f"piece {i}")
        pieces.append(out)
    lst = work / "pieces.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in pieces))
    track = work / "track.mp4"
    _ff(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(track)], "concat")
    return track


# ── main render ───────────────────────────────────────────────────────────────
def render(tts: dict, visuals: list[dict], music_wav: Path | None, hook_text: str, channel: str,
           cfg: dict, work: Path, out_mp4: Path, seed: int, fx: dict | None = None) -> dict:
    rng = random.Random(seed)
    W, H, fps = cfg["video"]["width"], cfg["video"]["height"], cfg["video"]["fps"]
    theme_idx = rng.randrange(len(THEMES))
    theme = THEMES[theme_idx]
    arch = ARCHETYPES[rng.randrange(len(ARCHETYPES))]
    segs = tts["segments"]
    total = tts["duration"] + TAIL

    # segment display windows (each segment runs from its start to the next segment's start);
    # a segment with k shots splits its window into k equal cuts
    bounds = [s["start"] for s in segs] + [total]
    seg_durs = [bounds[i + 1] - bounds[i] for i in range(len(segs))]
    shots = shot_plan(visuals, len(segs))
    per_seg: dict[int, int] = {}
    for v in shots:
        per_seg[v["seg"]] = per_seg.get(v["seg"], 0) + 1
    durs = [seg_durs[v["seg"]] / per_seg[v["seg"]] for v in shots]

    frames, clip_aspect, clip_src = [], {}, {}
    animate = bool(cfg["video"].get("animate_stills", True)) and motion.available()
    moves: list[str] = []
    for i, v in enumerate(shots):
        still = Path(v["path"])
        if is_video(v):                    # moving footage: the poster frame stands in for thumbnails only
            still = Path(v.get("poster") or poster_frame(Path(v["path"]), work / f"poster{i:02d}.jpg"))
            with Image.open(still) as im:
                clip_aspect[i] = im.width / im.height
            clip_src[i] = Path(v["path"])
        elif animate:                      # verified still → 2.5D camera move (same pixels, real depth motion)
            d = durs[i] + (XF if i < len(shots) - 1 else 0) + 0.1
            mv = rng.choice([m for m in motion.MOVES if not moves or m != moves[-1]])
            moves.append(mv)
            key = f"{abs(hash((str(still), round(d, 2), mv))) % 10**10}"
            clip = work / f"anim_{key}.mp4"
            try:
                if not clip.exists():
                    motion.animate_still(still, clip, d, fps, seed + i, move=mv)
                with Image.open(still) as im:
                    clip_aspect[i] = im.width / im.height
                clip_src[i] = clip
            except Exception as e:  # noqa: BLE001
                log.warning("2.5D motion failed for shot %d (%s) — Ken Burns instead", i, str(e)[:120])
        frames.append(compose_frame(still, work / f"frame{i:02d}.jpg", W, H))

    emph_col = THEMES[EMPH_FOR[theme_idx]]["hi"]      # keyword colour clearly different from the karaoke highlight
    # an archetype that drops the end stamp must also drop its sound effect, or the audio keeps
    # punctuating a stamp that is no longer on screen
    if fx and not arch.get("stamp", True):
        fx = {**fx, "stamp": None}
    ass = build_ass(segs, hook_text, channel, W, H, theme, arch["font"], work / "captions.ass", total,
                    fx=fx, arch=arch, emph_col=emph_col)
    sfx_wav = None
    if fx and fx.get("sfx"):
        _, num_times, reveal_at, stamp_at = fx_lines(segs, fx, theme, emph_col, W, H, total)
        starts, t = [], 0.0
        for d in durs:
            starts.append({"start": t})
            t += d
        events = sfx.plan_events(starts, num_times, reveal_at, stamp_at, total)
        sfx_wav = sfx.build_track(events, total, work / "sfx.wav", seed)

    n = len(frames)
    motions: list[str] = []
    for _ in range(n):             # never the same camera move twice in a row
        motions.append(rng.choice([m for m in ("in", "out", "left", "right", "up") if not motions or m != motions[-1]]))
    trans = ["fade", "smoothleft", "smoothup", "circleopen", "fadeblack", "slideleft", "zoomin"]
    track = build_video_track(frames, clip_aspect, clip_src, durs, motions, [rng.choice(trans) for _ in range(n)],
                              total, W, H, fps, work)

    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-stats", "-i", str(track)]
    cmd += ["-i", str(tts["audio"])]
    a_idx = 1
    if music_wav:
        cmd += ["-i", str(music_wav)]
    m_idx = 2
    s_idx = 3 if music_wav else 2
    if sfx_wav:
        cmd += ["-i", str(sfx_wav)]
    fg = [f"[0:v]fps={fps},settb=1/{fps},setsar=1[x0]"]
    last = "x0"
    # progress bar + captions
    fg.append(f"color=c=white@0.85:s={W}x12:r={fps},format=rgba[bar]")
    fg.append(f"[{last}][bar]overlay=x='-W+W*t/{total:.3f}':y=H-12:shortest=1[vb]")
    fontsdir = str(FONTS_DIR).replace(":", r"\:")
    assp = str(ass).replace(":", r"\:")
    fg.append(f"[vb]ass='{assp}':fontsdir='{fontsdir}',trim=duration={total:.3f}[vout]")

    # audio: narration (+ ducked, low-passed music) → loudness normalize
    vol = cfg["video"].get("music_volume_db", -26)
    fg.append(f"[{a_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,apad=pad_dur={TAIL},asplit=2[nar][sc]")
    if music_wav:
        fg.append(f"[{m_idx}:a]lowpass=f=3200,volume={vol + 12}dB,atrim=duration={total:.3f}[mus]")
        fg.append("[mus][sc]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[duck]")
        fg.append("[nar][duck]amix=inputs=2:duration=first:normalize=0[mix0]")
    else:
        fg.append("[sc]anullsink")
        fg.append("[nar]anull[mix0]")
    if sfx_wav:
        sv = cfg["video"].get("sfx_volume_db", -12)
        fg.append(f"[{s_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo,volume={sv + 12}dB,"
                  f"atrim=duration={total:.3f}[sfx]")
        fg.append("[mix0][sfx]amix=inputs=2:duration=first:normalize=0[mix]")
    else:
        fg.append("[mix0]anull[mix]")
    lufs = cfg["video"].get("loudness_lufs", -14)
    fg.append(f"[mix]loudnorm=I={lufs}:TP=-1.5:LRA=11,atrim=duration={total:.3f}[aout]")

    cmd += ["-filter_complex", ";".join(fg), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-movflags", "+faststart",
            "-t", f"{total:.3f}", str(out_mp4)]
    log.info("final pass: captions, sound, %d scenes, %.1fs …", n, total)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + res.stderr[-3000:])
    timeline, t = [], 0.0
    for v, d in zip(shots, durs):
        timeline.append({"seg": v["seg"], "path": str(v["path"]), "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    return {"duration": total, "theme": theme_idx, "archetype": arch["name"],
            "first_frame": frames[0], "timeline": timeline}
