"""Video renderer: 1080x1920 Short with Ken Burns motion, crossfades, karaoke
captions, hook title card, progress bar, ducked procedural music, loudness norm.

Everything is ONE ffmpeg filter graph → fast, reproducible, zero paid tools.
"""
from __future__ import annotations

import logging
import random
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .common import FONTS_DIR, ffmpeg_bin, font_path

log = logging.getLogger("autotube.render")

XF = 0.30          # crossfade seconds
TAIL = 0.7         # silence after last word
SS = 1.5           # supersampling factor for smooth zoompan

# caption colour themes (ASS colours are &HAABBGGRR) — rotated per video for variety
THEMES = [
    {"hi": "&H0000F0FF", "box": "&H0000D7FF", "boxtxt": "&H00101010"},   # yellow
    {"hi": "&H0066FF33", "box": "&H0066FF33", "boxtxt": "&H00101010"},   # green
    {"hi": "&H00FFD000", "box": "&H00FFC400", "boxtxt": "&H00101010"},   # cyan
    {"hi": "&H003C8CFF", "box": "&H003C8CFF", "boxtxt": "&H00FFFFFF"},   # orange
    {"hi": "&H00FF66E0", "box": "&H00FF66E0", "boxtxt": "&H00FFFFFF"},   # pink
]


# ── image prep ────────────────────────────────────────────────────────────────
def compose_frame(src: Path, dst: Path, W: int, H: int) -> Path:
    """Portrait canvas: blurred, darkened cover-fill background + sharp fitted foreground."""
    cw, ch = int(W * SS), int(H * SS)
    with Image.open(src) as im:
        im = im.convert("RGB")
        # background: cover
        r = max(cw / im.width, ch / im.height)
        bg = im.resize((int(im.width * r) + 2, int(im.height * r) + 2), Image.LANCZOS)
        l, t = (bg.width - cw) // 2, (bg.height - ch) // 2
        bg = bg.crop((l, t, l + cw, t + ch)).filter(ImageFilter.GaussianBlur(40))
        bg = Image.eval(bg, lambda v: int(v * 0.45))
        # foreground: fit width (portrait images fill more)
        aspect = im.width / im.height
        if aspect < 0.75:        # already portrait-ish → cover the whole canvas
            fg, pos = None, None
            r = max(cw / im.width, ch / im.height)
            full = im.resize((int(im.width * r) + 2, int(im.height * r) + 2), Image.LANCZOS)
            l, t = (full.width - cw) // 2, (full.height - ch) // 2
            bg = full.crop((l, t, l + cw, t + ch))
        else:
            fw = cw
            fh = int(fw / aspect)
            if fh > ch * 0.62:
                fh = int(ch * 0.62)
                fw = int(fh * aspect)
            fg = im.resize((fw, fh), Image.LANCZOS)
            pos = ((cw - fw) // 2, int(ch * 0.40 - fh / 2))
            # soft shadow
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


def build_ass(segments: list[dict], hook_text: str, channel: str, W: int, H: int, theme: dict,
              font_name: str, out: Path, total: float) -> Path:
    fs = 112
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font_name},{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,5,60,60,0,1
Style: Hook,{font_name},96,{theme['boxtxt']},&H00FFFFFF,{theme['box']},{theme['box']},-1,0,0,0,100,100,1,0,3,18,0,8,90,90,210,1
Style: Brand,{font_name},46,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,2,0,1,3,0,8,40,40,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    cy = int(H * 0.72)
    # hook title card (first ~2.8 s) with pop-in
    if hook_text:
        lines.append(f"Dialogue: 2,{_ts(0)},{_ts(min(2.9, total))},Hook,,0,0,0,,"
                     f"{{\\fad(0,250)\\t(0,180,\\fscx112\\fscy112)\\t(180,320,\\fscx100\\fscy100)}}{_esc(hook_text.upper())}")
    lines.append(f"Dialogue: 1,{_ts(0)},{_ts(total)},Brand,,0,0,0,,{{\\alpha&H40&}}{_esc(channel)}")

    # karaoke word chunks
    for seg in segments:
        words = [w for w in seg["words"] if w["word"].strip()]
        chunks, cur = [], []
        for w in words:
            cur.append(w)
            if len(cur) >= 3 or w["word"][-1:] in ".,!?;:" or len(" ".join(x["word"] for x in cur)) > 15:
                chunks.append(cur)
                cur = []
        if cur:
            chunks.append(cur)
        for ci, ch in enumerate(chunks):
            c_end = chunks[ci + 1][0]["start"] if ci + 1 < len(chunks) else seg["end"] + 0.12
            for wi, w in enumerate(ch):
                st = w["start"]
                en = ch[wi + 1]["start"] if wi + 1 < len(ch) else c_end
                parts = []
                for wj, w2 in enumerate(ch):
                    txt = _esc(w2["word"].upper())
                    if wj == wi:
                        parts.append(f"{{\\c{theme['hi']}\\fscx108\\fscy108}}{txt}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}")
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


# ── main render ───────────────────────────────────────────────────────────────
def render(tts: dict, visuals: list[dict], music_wav: Path | None, hook_text: str, channel: str,
           cfg: dict, work: Path, out_mp4: Path, seed: int) -> dict:
    rng = random.Random(seed)
    W, H, fps = cfg["video"]["width"], cfg["video"]["height"], cfg["video"]["fps"]
    theme_idx = rng.randrange(len(THEMES))
    theme = THEMES[theme_idx]
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

    frames = []
    for i, v in enumerate(shots):
        frames.append(compose_frame(Path(v["path"]), work / f"frame{i:02d}.jpg", W, H))

    ass = build_ass(segs, hook_text, channel, W, H, theme, "Anton", work / "captions.ass", total)

    cmd = [ffmpeg_bin(), "-y", "-loglevel", "error", "-stats"]
    for f in frames:
        cmd += ["-i", str(f)]
    n = len(frames)
    cmd += ["-i", str(tts["audio"])]
    a_idx = n
    if music_wav:
        cmd += ["-i", str(music_wav)]
    m_idx = n + 1

    fg = []
    motions: list[str] = []
    for _ in range(n):             # never the same camera move twice in a row
        motions.append(rng.choice([m for m in ("in", "out", "left", "right", "up") if not motions or m != motions[-1]]))
    cw, ch = int(W * SS), int(H * SS)
    for i in range(n):
        d = durs[i] + (XF if i < n - 1 else 0)
        nf = max(2, int(round(d * fps)))
        mo = motions[i % len(motions)]
        z0, z1 = (1.0, 1.14) if mo != "out" else (1.14, 1.0)
        zexpr = f"{z0}+({z1 - z0})*on/{nf}"
        if mo == "left":
            x = f"(iw-iw/zoom)*(1-on/{nf})"
            y = "(ih-ih/zoom)/2"
        elif mo == "right":
            x = f"(iw-iw/zoom)*on/{nf}"
            y = "(ih-ih/zoom)/2"
        elif mo == "up":
            x = "(iw-iw/zoom)/2"
            y = f"(ih-ih/zoom)*(1-on/{nf})"
        else:
            x = "(iw-iw/zoom)/2"
            y = "(ih-ih/zoom)/2"
        fg.append(f"[{i}:v]scale={cw}:{ch},setsar=1,"
                  f"zoompan=z='{zexpr}':x='{x}':y='{y}':d={nf}:s={W}x{H}:fps={fps},"
                  f"format=yuv420p,trim=duration={d:.3f},setpts=PTS-STARTPTS,fps={fps},settb=1/{fps}[v{i}]")
    # crossfade chain
    last = "v0"
    offset = 0.0
    trans = ["fade", "smoothleft", "smoothup", "circleopen", "fadeblack", "slideleft", "zoomin"]
    for i in range(1, n):
        offset += durs[i - 1]
        t = rng.choice(trans)
        fg.append(f"[{last}][v{i}]xfade=transition={t}:duration={XF}:offset={offset - 0:.3f}[x{i}]")
        last = f"x{i}"
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
        fg.append("[nar][duck]amix=inputs=2:duration=first:normalize=0[mix]")
    else:
        fg.append("[sc]anullsink")
        fg.append("[nar]anull[mix]")
    lufs = cfg["video"].get("loudness_lufs", -14)
    fg.append(f"[mix]loudnorm=I={lufs}:TP=-1.5:LRA=11,atrim=duration={total:.3f}[aout]")

    cmd += ["-filter_complex", ";".join(fg), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-r", str(fps), "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-movflags", "+faststart",
            "-t", f"{total:.3f}", str(out_mp4)]
    log.info("rendering %d scenes, %.1fs …", n, total)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + res.stderr[-3000:])
    timeline, t = [], 0.0
    for v, d in zip(shots, durs):
        timeline.append({"seg": v["seg"], "path": str(v["path"]), "start": round(t, 3), "end": round(t + d, 3)})
        t += d
    return {"duration": total, "theme": theme_idx, "first_frame": frames[0], "timeline": timeline}
