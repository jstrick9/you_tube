"""Visual archetypes (AUDIT 4: "near-identical videos with minimal variation")."""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import render  # noqa: E402
from autotube.common import FONTS_DIR, ffmpeg_bin  # noqa: E402

ASS = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},140,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,7,3,5,150,150,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Cap,,0,0,0,,HELLO
"""


def test_archetypes_are_actually_distinct():
    """Three entries that differ only in name would satisfy nothing."""
    assert len(render.ARCHETYPES) >= 3
    for key in ("font", "caption_y", "hook_seconds"):
        values = [a[key] for a in render.ARCHETYPES]
        assert len(set(values)) > 1, f"every archetype has the same {key}"
    assert len({a["name"] for a in render.ARCHETYPES}) == len(render.ARCHETYPES)
    assert {a["stamp"] for a in render.ARCHETYPES} == {True, False}, "the end card must vary too"


def test_captions_stay_clear_of_the_shorts_ui():
    """The bottom ~18% is covered by YouTube's own chrome."""
    for a in render.ARCHETYPES:
        assert 0.45 <= a["caption_y"] <= 0.70, f"{a['name']} caption_y={a['caption_y']}"


def test_hook_card_never_outstays_the_opening():
    for a in render.ARCHETYPES:
        assert 0.8 <= a["hook_seconds"] <= 2.5, a["name"]


@pytest.mark.parametrize("arch", render.ARCHETYPES, ids=lambda a: a["name"])
def test_every_archetype_font_resolves_to_a_bundled_file(arch, tmp_path):
    """libass falls back to DejaVu silently when a family name is wrong.

    Nothing errors, nothing logs, and every video just keeps rendering in the same default
    typeface — so the variation this whole feature exists to create quietly does not happen.
    "Montserrat" does exactly that; the Black face only resolves as "Montserrat Black".
    """
    ass = tmp_path / "t.ass"
    ass.write_text(ASS.format(font=arch["font"]))
    r = subprocess.run(
        [ffmpeg_bin(), "-hide_banner", "-f", "lavfi", "-i", "color=black:s=1080x1920:d=1",
         "-vf", f"ass='{ass}':fontsdir='{FONTS_DIR}'", "-frames:v", "1", "-y", str(tmp_path / "o.png")],
        capture_output=True, text=True)
    picked = [l.split("->")[-1].strip() for l in r.stderr.splitlines() if "fontselect:" in l]
    assert picked, "ffmpeg did not report a font selection"
    assert "/usr/share/fonts" not in picked[0] and "DejaVu" not in picked[0], (
        f"{arch['font']!r} fell back to a system font: {picked[0]}")
