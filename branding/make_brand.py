"""Proof in a Minute: channel art generator (profile picture, banner, watermark, preview mockup)."""
import math, random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUT = "/home/user/branding"
FONTS = "/home/user/autotube-system/assets/fonts"
ANTON = f"{FONTS}/Anton-Regular.ttf"
MBLACK = f"{FONTS}/Montserrat-Black.ttf"
MMED = "/usr/lib/R/library/grDevices/fonts/Montserrat/static/Montserrat-Medium.ttf"

YELLOW = (255, 215, 0)
INK = (13, 13, 16)
INK2 = (24, 24, 30)
WHITE = (245, 245, 245)
GREY = (150, 150, 160)


def font(path, size):
    return ImageFont.truetype(path, size)


# ── logo mark: stopwatch + check ────────────────────────────────────────────
def draw_mark(d, cx, cy, r, color=YELLOW, check=YELLOW, ticks=WHITE, sw=None):
    """Stopwatch ring of radius r centred on (cx, cy) with a bold check inside."""
    sw = sw or r * 0.22
    # crown (top button) + stem
    bw, bh = r * 0.42, r * 0.20
    stem_w, stem_h = r * 0.16, r * 0.16
    top = cy - r - sw / 2
    d.rounded_rectangle([cx - stem_w / 2, top - stem_h, cx + stem_w / 2, top + sw * 0.3], radius=stem_w * 0.2, fill=color)
    d.rounded_rectangle([cx - bw / 2, top - stem_h - bh, cx + bw / 2, top - stem_h + bh * 0.1], radius=bh * 0.45, fill=color)
    # side button at 45°
    ang = math.radians(-45)
    px, py = cx + (r + sw * 0.55) * math.cos(ang), cy + (r + sw * 0.55) * math.sin(ang)
    side = Image.new("RGBA", (int(r * 0.5), int(r * 0.5)), (0, 0, 0, 0))
    sd = ImageDraw.Draw(side)
    sd.rounded_rectangle([r * 0.25 - r * 0.08, r * 0.25 - r * 0.13, r * 0.25 + r * 0.08, r * 0.25 + r * 0.13],
                         radius=r * 0.04, fill=color)
    side = side.rotate(-45, resample=Image.BICUBIC)
    d._image.paste(side, (int(px - side.width / 2), int(py - side.height / 2)), side)
    # ring
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=int(sw))
    # minute ticks
    if ticks:
        for i in range(12):
            a = math.radians(i * 30 - 90)
            big = i % 3 == 0
            r1, r2 = r - sw * 0.9, r - sw * (1.55 if big else 1.25)
            d.line([cx + r1 * math.cos(a), cy + r1 * math.sin(a), cx + r2 * math.cos(a), cy + r2 * math.sin(a)],
                   fill=ticks, width=int(sw * (0.28 if big else 0.16)))
    # check mark
    cw = int(r * 0.23)
    pts = [(cx - r * 0.42, cy + r * 0.02), (cx - r * 0.12, cy + r * 0.32), (cx + r * 0.46, cy - r * 0.30)]
    d.line(pts, fill=check, width=cw, joint="curve")
    for x, y in (pts[0], pts[2]):
        d.ellipse([x - cw / 2, y - cw / 2, x + cw / 2, y + cw / 2], fill=check)


def radial_bg(w, h, center, inner=(34, 30, 10), outer=INK, radius=None):
    radius = radius or max(w, h) * 0.6
    g = Image.new("L", (w, h), 0)
    gd = ImageDraw.Draw(g)
    steps = 60
    for i in range(steps, 0, -1):
        rr = radius * i / steps
        gd.ellipse([center[0] - rr, center[1] - rr, center[0] + rr, center[1] + rr], fill=int(255 * (1 - i / steps) ** 1.6))
    g = g.filter(ImageFilter.GaussianBlur(radius / 25))
    return Image.composite(Image.new("RGB", (w, h), inner), Image.new("RGB", (w, h), outer), g)


# ── 1) profile picture 800x800 ──────────────────────────────────────────────
S = 3200
im = radial_bg(S, S, (S / 2, S * 0.52), inner=(40, 34, 6), radius=S * 0.62)
d = ImageDraw.Draw(im)
draw_mark(d, S / 2, S * 0.545, S * 0.285)
im.resize((800, 800), Image.LANCZOS).save(f"{OUT}/profile-picture-800.png", optimize=True)

# ── 2) watermark 150x150 transparent (Subscribe watermark on videos) ─────────
W = 1200
wm = Image.new("RGBA", (W, W), (0, 0, 0, 0))
wd = ImageDraw.Draw(wm)
wd.ellipse([W * 0.06, W * 0.06, W * 0.94, W * 0.94], fill=(13, 13, 16, 235))
draw_mark(wd, W / 2, W * 0.545, W * 0.27, ticks=None)
wm.resize((150, 150), Image.LANCZOS).save(f"{OUT}/video-watermark-150.png", optimize=True)

# ── 3) banner 2560x1440 (safe area 1546x423 centred) ─────────────────────────
BW, BH = 2560, 1440
SAFE = ((BW - 1546) // 2, (BH - 423) // 2, (BW + 1546) // 2, (BH + 423) // 2)
bn = radial_bg(BW, BH, (BW * 0.5, BH * 0.5), inner=(30, 27, 8), radius=BW * 0.55)
d = ImageDraw.Draw(bn, "RGBA")

# background texture: faint citation brackets / question marks / ticks (outside the text zone)
random.seed(7)
tex = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
td = ImageDraw.Draw(tex)
glyphs = ["[1]", "[2]", "[3]", "?", "[4]", "60s", "[5]", "?"]
placed = []
tries = 0
while len(placed) < 44 and tries < 5000:
    tries += 1
    x, y = random.randint(-20, BW - 60), random.randint(-20, BH - 60)
    if SAFE[0] - 380 < x < SAFE[2] + 40 and SAFE[1] - 140 < y < SAFE[3] + 30:
        continue          # keep the text zone and its surroundings clean
    g = random.choice(glyphs)
    f = font(ANTON, random.randint(46, 110))
    bx = td.textbbox((x, y), g, font=f)
    if any(not (bx[2] + 40 < p[0] or bx[0] > p[2] + 40 or bx[3] + 40 < p[1] or bx[1] > p[3] + 40) for p in placed):
        continue          # no overlapping glyphs
    placed.append(bx)
    a = random.randint(22, 48)
    td.text((x, y), g, font=f,
            fill=(255, 215, 0, a) if random.random() < 0.4 else (255, 255, 255, int(a * 0.8)))
bn = Image.alpha_composite(bn.convert("RGBA"), tex).convert("RGB")
d = ImageDraw.Draw(bn, "RGBA")

# diagonal accent stripes left & right of the desktop strip
for sx, flip in ((SAFE[0] - 330, 1), (SAFE[2] + 150, -1)):
    for k in range(3):
        x0 = sx + k * 70
        d.polygon([(x0, SAFE[1] - 40), (x0 + 38, SAFE[1] - 40), (x0 + 38 - 120 * flip, SAFE[3] + 40),
                   (x0 - 120 * flip, SAFE[3] + 40)], fill=(255, 215, 0, 60 - k * 18))

# logo mark inside safe area (left)
mark_r = 128
halo = 1.32
mcx, mcy = SAFE[0] + 16 + mark_r * halo, (SAFE[1] + SAFE[3]) // 2 + 14
d.ellipse([mcx - mark_r * halo, mcy - mark_r * halo, mcx + mark_r * halo, mcy + mark_r * halo], fill=(255, 215, 0, 20))
draw_mark(d, mcx, mcy, mark_r)

# wordmark, auto-fitted to the remaining safe width
tx = mcx + mark_r * halo + 44
avail = SAFE[2] - 12 - tx
def fit(path, text, maxsize):
    sz = maxsize
    while d.textlength(text, font=font(path, sz)) > avail:
        sz -= 2
    return font(path, sz)
f_big = fit(ANTON, "PROOF IN A MINUTE", 190)
TAG = "SURPRISING FACTS. REAL SOURCES. 60 SECONDS."
f_tag = fit(MBLACK, TAG, 46)
asc, desc = f_big.getmetrics()
block_h = asc + desc + 18 + f_tag.size * 1.25 + 26 + 60
y0 = (SAFE[1] + SAFE[3]) / 2 - block_h / 2 - 8
d.text((tx, y0), "PROOF", font=f_big, fill=YELLOW)
wproof = d.textlength("PROOF ", font=f_big)
d.text((tx + wproof, y0), "IN A MINUTE", font=f_big, fill=WHITE)
tag_y = y0 + asc + desc + 18
d.text((tx + 4, tag_y), TAG, font=f_tag, fill=WHITE)
f_pill = font(MBLACK, 30)
ptxt = "NEW SHORTS EVERY DAY"
pw = d.textlength(ptxt, font=f_pill)
py = tag_y + f_tag.size * 1.25 + 26
d.rounded_rectangle([tx + 4, py, tx + 4 + pw + 52, py + 58], radius=29, fill=YELLOW)
d.text((tx + 30, py + 11), ptxt, font=f_pill, fill=INK)
end_x = max(tx + d.textlength("PROOF IN A MINUTE", font=f_big), tx + 4 + d.textlength(TAG, font=f_tag))
print("sizes big/tag:", f_big.size, f_tag.size, "| right edge:", int(end_x), "<=", SAFE[2], "| top:", int(y0), ">=", SAFE[1], "| bottom:", int(py + 58), "<=", SAFE[3])
assert end_x <= SAFE[2] and py + 58 <= SAFE[3] and y0 >= SAFE[1] - 30, "text leaves the safe area"
bn.save(f"{OUT}/banner-2560x1440.png", optimize=True)

# ── 4) preview mockup: desktop strip + mobile + avatar at real sizes ─────────
pv = Image.new("RGB", (1800, 1250), (249, 249, 249))
pd = ImageDraw.Draw(pv)
fT, fS, fB = font(MBLACK, 30), font(MMED, 22), font(MBLACK, 40)
pd.text((40, 26), "Preview: how the channel art appears", font=fT, fill=(30, 30, 30))
# desktop: visible band = full width x 423 centre, shown scaled
pd.text((40, 86), "Desktop (visible band of the banner)", font=fS, fill=(90, 90, 90))
band = bn.crop((0, SAFE[1], BW, SAFE[3])).resize((1720, int(423 * 1720 / 2560)))
pv.paste(band, (40, 120))
av = Image.open(f"{OUT}/profile-picture-800.png").resize((160, 160), Image.LANCZOS)
mask = Image.new("L", (160, 160), 0); ImageDraw.Draw(mask).ellipse([0, 0, 159, 159], fill=255)
pv.paste(av, (40, 435), mask)
pd.text((225, 450), "Proof in a Minute", font=fB, fill=(15, 15, 15))
pd.text((226, 505), "@proofinaminute · Surprising facts. Real sources. 60 seconds.", font=fS, fill=(96, 96, 96))
pd.rounded_rectangle([226, 545, 386, 590], radius=22, fill=(15, 15, 15)); pd.text((252, 553), "Subscribe", font=font(MBLACK, 22), fill="white")
# mobile: safe area only
pd.text((40, 640), "Mobile (safe area only)", font=fS, fill=(90, 90, 90))
mob = bn.crop(SAFE).resize((900, int(423 * 900 / 1546)))
pv.paste(mob, (40, 674))
av2 = Image.open(f"{OUT}/profile-picture-800.png").resize((110, 110), Image.LANCZOS)
m2 = Image.new("L", (110, 110), 0); ImageDraw.Draw(m2).ellipse([0, 0, 109, 109], fill=255)
pv.paste(av2, (40, 950), m2)
pd.text((170, 965), "Proof in a Minute", font=font(MBLACK, 30), fill=(15, 15, 15))
pd.text((171, 1008), "@proofinaminute", font=fS, fill=(96, 96, 96))
# small sizes
pd.text((1010, 640), "Profile picture at real sizes", font=fS, fill=(90, 90, 90))
x = 1010
for s in (176, 88, 48, 32):
    a = Image.open(f"{OUT}/profile-picture-800.png").resize((s, s), Image.LANCZOS)
    m = Image.new("L", (s, s), 0); ImageDraw.Draw(m).ellipse([0, 0, s - 1, s - 1], fill=255)
    pv.paste(a, (x, 690), m); pd.text((x, 690 + s + 8), f"{s}px", font=fS, fill=(120, 120, 120)); x += s + 40
pd.text((1010, 930), "Video watermark (bottom-right of every video)", font=fS, fill=(90, 90, 90))
frame = Image.open("/home/user/first_upload/20260924-0008-01-container-deposit-legislation.jpg").resize((135, 240))
pv.paste(frame, (1010, 962))
wmk = Image.open(f"{OUT}/video-watermark-150.png").resize((34, 34), Image.LANCZOS)
pv.paste(wmk, (1010 + 135 - 42, 962 + 240 - 42), wmk)
pv.save(f"{OUT}/preview-mockup.png", optimize=True)
print("done")
