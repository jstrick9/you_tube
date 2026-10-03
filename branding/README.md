# Archive 13 — brand assets

Regenerate everything: `python branding/make_brand.py`

| File | Size | Where it goes |
|---|---|---|
| `profile-picture-800.png` | 800×800 | Channel avatar. YouTube masks it to a **circle** — the mark is sized to stay inside the inscribed circle. |
| `banner-2560x1440.png` | 2560×1440 | Channel banner. All text sits inside the central **1546×423** safe area, the only region guaranteed visible on every device. |
| `video-watermark-150.png` | 150×150 RGBA | Branding watermark overlaid on the player. Transparent background. |
| `preview-mockup.png` | 1280×720 | Not for upload — a proof sheet for eyeballing the set, including the avatar at 96/64/48/32/24px under the circular crop. |
| `src/emblem-source.png` | 1600×1600 | The source artwork. Everything else is derived from it. |

## Design notes

The mark is a dossier folder stamped **13**, with a redaction bar — unresolved records,
not verified facts. It replaces the old stopwatch-and-checkmark, which promised "a
minute" and "proven", both of which the channel abandoned.

The palette is **sampled from the emblem at build time**, not retyped as constants, so
the banner can never drift to a slightly different amber than the avatar.

Amber `#BF7F1E` · charcoal `#161616` · paper `#F2EEE4`
