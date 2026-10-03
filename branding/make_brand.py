"""Archive 13 — channel art generator.

Regenerates every YouTube brand asset from one source emblem so the set can never
drift apart: avatar, transparent video watermark, channel banner and a preview mockup.

Run: python branding/make_brand.py

Two things this does deliberately.

The palette is *sampled from the emblem* rather than retyped as hex constants. The
previous version of this file carried its own copy of the brand colours, which is how
a banner ends up a slightly different yellow from the avatar nobody notices for a year.

The watermark's transparency comes from a flood fill seeded at the border, not from
"make dark pixels transparent". The emblem contains a deliberate solid-black redaction
bar *inside* the folder, and a naive colour key would punch a hole straight through it.
A border-seeded fill only removes background that is actually connected to the edge.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "branding"
FONTS = ROOT / "assets" / "fonts"
ANTON = FONTS / "Anton-Regular.ttf"
BEBAS = FONTS / "BebasNeue-Regular.ttf"
MBLACK = FONTS / "Montserrat-Black.ttf"
SRC = OUT / "src" / "emblem-source.png"

NAME = "ARCHIVE 13"
HANDLE = "@archive13"
TAGLINE = "Every file is real. That's the problem."
SERIES = "CASE  ·  unsolved history      FIELD  ·  the natural world"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


# ── palette, sampled from the artwork ───────────────────────────────────────
def palette(em: Image.Image) -> dict:
    rgb = em.convert("RGB")
    w, h = rgb.size
    ink = rgb.getpixel((4, 4))                       # background charcoal
    # The amber folder: most saturated colour in the middle band.
    best, amber = -1, (198, 140, 38)
    for y in range(int(h * 0.25), int(h * 0.85), 7):
        for x in range(int(w * 0.12), int(w * 0.88), 7):
            r, g, b = rgb.getpixel((x, y))
            sat = max(r, g, b) - min(r, g, b)
            if sat > best and r > 90:
                best, amber = sat, (r, g, b)
    return {
        "ink": ink,
        "ink2": tuple(min(255, c + 10) for c in ink),
        "amber": amber,
        "paper": (242, 238, 228),
        "grey": (138, 136, 132),
    }


# ── geometry ────────────────────────────────────────────────────────────────
def folder_bbox(em: Image.Image, tol: int = 26) -> tuple[int, int, int, int]:
    """Tight box around the emblem, ignoring the flat background."""
    rgb = em.convert("RGB")
    bg = rgb.getpixel((4, 4))
    xs, ys = [], []
    w, h = rgb.size
    for y in range(0, h, 3):
        for x in range(0, w, 3):
            p = rgb.getpixel((x, y))
            if sum(abs(a - b) for a, b in zip(p, bg)) > tol:
                xs.append(x)
                ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, w, h)


def cut_background(em: Image.Image, tol: int = 110) -> Image.Image:
    """Alpha from a border-seeded flood fill.

    Only background connected to the edge is removed, so the solid-black redaction
    bar inside the folder survives. A plain colour key would delete it.
    """
    rgb = em.convert("RGB")
    w, h = rgb.size
    px = rgb.load()
    bg = px[4, 4]
    alpha = Image.new("L", (w, h), 255)
    a = alpha.load()
    seen = bytearray(w * h)
    q: deque[tuple[int, int]] = deque()

    def like_bg(x: int, y: int) -> bool:
        p = px[x, y]
        return sum(abs(c - d) for c, d in zip(p, bg)) <= tol

    for x in range(w):
        for y in (0, h - 1):
            if not seen[y * w + x] and like_bg(x, y):
                seen[y * w + x] = 1
                q.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if not seen[y * w + x] and like_bg(x, y):
                seen[y * w + x] = 1
                q.append((x, y))
    while q:
        x, y = q.popleft()
        a[x, y] = 0
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not seen[ny * w + nx] and like_bg(nx, ny):
                seen[ny * w + nx] = 1
                q.append((nx, ny))
    out = em.convert("RGBA")
    out.putalpha(alpha.filter(ImageFilter.GaussianBlur(0.6)))
    return out


# ── assets ──────────────────────────────────────────────────────────────────
def avatar(em: Image.Image, pal: dict, size: int = 800) -> Image.Image:
    """Square avatar, cropped tighter than the source.

    YouTube renders this at 24px next to comments, where the generous margins of the
    original artwork waste most of the pixels. The preview sheet renders the avatar at
    96/64/48/32/24px precisely so this is checked rather than assumed - at 84% fill the
    numeral was already soft at 24px. It is not pushed to the edge either, because
    YouTube masks avatars to a CIRCLE: a square mark filling the frame loses its
    corners. 80% keeps the folder inside the inscribed circle while still being much
    tighter than the source artwork, and the preview sheet renders the circular crop
    so this is checked rather than trusted.
    """
    x0, y0, x1, y1 = folder_bbox(em)
    mark = em.crop((x0, y0, x1, y1))
    canvas = Image.new("RGB", (size, size), pal["ink"])
    target = int(size * 0.80)
    sc = min(target / mark.width, target / mark.height)
    mark = mark.resize((max(1, int(mark.width * sc)), max(1, int(mark.height * sc))), Image.LANCZOS)
    canvas.paste(mark, ((size - mark.width) // 2, (size - mark.height) // 2))
    return canvas


def watermark(em: Image.Image, size: int = 150) -> Image.Image:
    """Transparent branding watermark burned into the video player corner."""
    cut = cut_background(em)
    x0, y0, x1, y1 = folder_bbox(em)
    mark = cut.crop((x0, y0, x1, y1))
    sc = min(size / mark.width, size / mark.height)
    mark = mark.resize((max(1, int(mark.width * sc)), max(1, int(mark.height * sc))), Image.LANCZOS)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(mark, ((size - mark.width) // 2, (size - mark.height) // 2), mark)
    return out


def _texture(d: ImageDraw.ImageDraw, w: int, h: int, pal: dict) -> None:
    """Faint index rules and redaction bars — an archive page, well out of the way."""
    for i, y in enumerate(range(140, h - 100, 86)):
        d.line([(0, y), (w, y)], fill=pal["ink2"], width=2)
        if i % 3 == 1:
            bar_w = 180 + (i * 97) % 420
            x = 90 + (i * 317) % (w - 600)
            d.rectangle([x, y - 26, x + bar_w, y - 6], fill=pal["ink2"])


def banner(em: Image.Image, pal: dict, w: int = 2560, h: int = 1440) -> Image.Image:
    """Channel banner.

    YouTube crops this hard: 2560x1440 is the upload, but only the central
    1546x423 is guaranteed visible on every device. Everything that must be read
    lives inside that box; the texture is what survives into the wide desktop crop.
    """
    img = Image.new("RGB", (w, h), pal["ink"])
    d = ImageDraw.Draw(img)
    _texture(d, w, h, pal)

    sx, sy = (w - 1546) // 2, (h - 423) // 2        # safe area origin

    # Emblem, background keyed out. Pasting the opaque crop leaves the artwork's own
    # dark vignette as a visible lighter box floating on the banner.
    cut = cut_background(em)
    x0, y0, x1, y1 = folder_bbox(em)
    mark = cut.crop((x0, y0, x1, y1))
    mh = 236
    mark = mark.resize((int(mark.width * mh / mark.height), mh), Image.LANCZOS)
    img.paste(mark, (sx + 10, sy + (423 - mh) // 2), mark)

    tx = sx + 10 + mark.width + 70
    f_name = font(ANTON, 136)
    f_tag = font(BEBAS, 56)
    f_small = font(MBLACK, 26)

    # Measured, not guessed: the first version hardcoded the rule's y and drew it
    # through the middle of the wordmark, which reads as a strikethrough.
    ny = sy + 66
    d.text((tx, ny), NAME, font=f_name, fill=pal["paper"])
    nb = d.textbbox((tx, ny), NAME, font=f_name)
    rule_y = nb[3] + 22
    d.line([(tx + 3, rule_y), (nb[2], rule_y)], fill=pal["amber"], width=6)

    ty = rule_y + 20
    d.text((tx, ty), TAGLINE, font=f_tag, fill=pal["amber"])
    tb = d.textbbox((tx, ty), TAGLINE, font=f_tag)
    d.text((tx + 3, tb[3] + 20), f"{HANDLE}      {SERIES}", font=f_small, fill=pal["grey"])
    return img


def mockup(av: Image.Image, pal: dict, w: int = 1280, h: int = 720) -> Image.Image:
    """A rough 'how it looks on the channel page' sheet for eyeballing the set."""
    img = Image.new("RGB", (w, h), (18, 18, 20))
    d = ImageDraw.Draw(img)
    bn = Image.open(OUT / "banner-2560x1440.png").resize((w - 80, int((w - 80) * 1440 / 2560)), Image.LANCZOS)
    bn = bn.crop((0, bn.height // 2 - 92, bn.width, bn.height // 2 + 92))
    img.paste(bn, (40, 40))
    a = av.resize((104, 104), Image.LANCZOS)
    img.paste(a, (72, 40 + bn.height + 18))
    d.text((196, 40 + bn.height + 28), NAME, font=font(MBLACK, 34), fill=(244, 244, 244))
    d.text((198, 40 + bn.height + 74), f"{HANDLE}  ·  Shorts", font=font(MBLACK, 20), fill=pal["grey"])
    # Tiny-size legibility strip, in the circular mask YouTube actually applies.
    # An avatar that only works as a square is an avatar that does not work.
    d.text((72, 418), "AS YOUTUBE RENDERS IT  —  CIRCULAR CROP", font=font(MBLACK, 18), fill=pal["grey"])
    x = 72
    for s in (96, 64, 48, 32, 24):
        a2 = av.resize((s, s), Image.LANCZOS).convert("RGBA")
        mask = Image.new("L", (s * 4, s * 4), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, s * 4 - 1, s * 4 - 1], fill=255)
        a2.putalpha(mask.resize((s, s), Image.LANCZOS))
        img.paste(a2, (x, 458), a2)
        d.text((x, 458 + s + 8), f"{s}px", font=font(MBLACK, 13), fill=pal["grey"])
        x += s + 34

    d.text((72, 592), "SQUARE SOURCE", font=font(MBLACK, 18), fill=pal["grey"])
    x = 72
    for s in (64, 48, 32):
        img.paste(av.resize((s, s), Image.LANCZOS), (x, 626))
        x += s + 28
    return img


def main() -> None:
    em = Image.open(SRC).convert("RGB")
    pal = palette(em)
    OUT.mkdir(parents=True, exist_ok=True)

    av = avatar(em, pal)
    av.save(OUT / "profile-picture-800.png")
    watermark(em).save(OUT / "video-watermark-150.png")
    banner(em, pal).save(OUT / "banner-2560x1440.png")
    mockup(av, pal).save(OUT / "preview-mockup.png")
    print("palette:", pal)
    for f in ("profile-picture-800.png", "video-watermark-150.png",
              "banner-2560x1440.png", "preview-mockup.png"):
        im = Image.open(OUT / f)
        print(f"  {f:28} {im.size} {im.mode}")


if __name__ == "__main__":
    main()
