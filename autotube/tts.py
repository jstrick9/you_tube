"""Narration: free neural TTS with word-level timings for karaoke captions.

Primary : edge-tts (Microsoft Edge neural voices, no key). Gives exact word boundaries.
Fallback: Piper (fully offline, open-source). Word timings are estimated per segment.

Each script segment is synthesized separately so segment boundaries (used for visual
cuts) are exact; segments are then concatenated with a short natural pause.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import re
import subprocess
import wave
from pathlib import Path

from .common import ffmpeg_bin

log = logging.getLogger("autotube.tts")

GAP = 0.18   # seconds of silence between segments
SR = 44100


def _probe_duration(path: Path) -> float:
    out = subprocess.run([ffmpeg_bin(), "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out)
    if not m:
        raise RuntimeError(f"cannot probe {path}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def _to_wav(src: Path, dst: Path) -> None:
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(src), "-ar", str(SR), "-ac", "1", str(dst)],
                   check=True)


def _clean_for_speech(text: str) -> str:
    text = re.sub(r"[#*_`~\[\]{}<>|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ── edge-tts ──────────────────────────────────────────────────────────────────
async def _edge_one(text: str, voice: str, rate: str, out_mp3: Path) -> list[dict]:
    import edge_tts
    words = []
    comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    with open(out_mp3, "wb") as f:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                s = chunk["offset"] / 1e7
                words.append({"word": chunk["text"], "start": s, "end": s + chunk["duration"] / 1e7})
    if out_mp3.stat().st_size < 1000:
        raise RuntimeError("edge-tts returned no audio")
    return words


async def _edge_all(texts, voice, rate, work: Path):
    sem = asyncio.Semaphore(3)

    async def run(i, t):
        async with sem:
            pauses = [3, 8, 20, 40]
            for attempt in range(len(pauses) + 1):
                try:
                    return await _edge_one(t, voice, rate, work / f"seg{i:02d}.mp3")
                except Exception as e:  # noqa: BLE001
                    if attempt == len(pauses):
                        raise
                    log.info("edge-tts retry %d for segment %d: %s", attempt + 1, i, e)
                    await asyncio.sleep(pauses[attempt])
    return await asyncio.gather(*[run(i, t) for i, t in enumerate(texts)])


# ── piper (offline) ───────────────────────────────────────────────────────────
PIPER_VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/{n}/{q}/en_US-{n}-{q}.onnx"


def _piper_model() -> Path:
    from .common import http
    name, q = os.environ.get("PIPER_VOICE", "ryan:high").split(":")
    d = Path(os.environ.get("PIPER_DIR", Path.home() / ".cache" / "piper"))
    d.mkdir(parents=True, exist_ok=True)
    onnx = d / f"en_US-{name}-{q}.onnx"
    if not onnx.exists():
        log.info("downloading Piper voice %s-%s (one-time)…", name, q)
        for suffix in ("", ".json"):
            r = http().get(PIPER_VOICE_URL.format(n=name, q=q) + suffix, timeout=300)
            r.raise_for_status()
            Path(str(onnx) + suffix).write_bytes(r.content)
    return onnx


def _piper_one(text: str, out_wav: Path) -> list[dict]:
    model = _piper_model()
    subprocess.run([sys.executable, "-m", "piper", "-m", str(model), "-f", str(out_wav)], input=text.encode(),
                   check=True, capture_output=True)
    dur = _probe_duration(out_wav)
    toks = text.split()
    weights = [len(re.sub(r"\W", "", t)) + 2 for t in toks]
    total = sum(weights) or 1
    t, words = 0.05, []
    usable = max(dur - 0.15, 0.1)
    for tok, w in zip(toks, weights):
        d = usable * w / total
        words.append({"word": re.sub(r"[^\w'%$.,-]", "", tok).strip(".,"), "start": t, "end": t + d * 0.92})
        t += d
    return words


# ── public API ────────────────────────────────────────────────────────────────
def synthesize(segments: list[str], voice: str, rate: str, work: Path, gaps: list[float] | None = None) -> dict:
    """Return {'audio': Path(wav), 'duration': float, 'segments': [{text,start,end,words:[...]}], 'engine': str}

    gaps[i] (optional) = seconds of silence BEFORE segment i (i >= 1); default GAP. A longer beat before the reveal
    line is where the renderer puts the riser / on-screen countdown ("comedic timing")."""
    work.mkdir(parents=True, exist_ok=True)
    texts = [_clean_for_speech(s) for s in segments]
    engine = "edge-tts"
    seg_files: list[Path] = []
    try:
        if os.environ.get("AUTOTUBE_TTS", "edge") != "edge":
            raise RuntimeError("edge disabled by env")
        all_words = asyncio.run(_edge_all(texts, voice, rate, work))
        for i in range(len(texts)):
            wav = work / f"seg{i:02d}.wav"
            _to_wav(work / f"seg{i:02d}.mp3", wav)
            seg_files.append(wav)
    except Exception as e:  # noqa: BLE001
        log.warning("edge-tts unavailable (%s) → falling back to offline Piper", e)
        engine = "piper"
        all_words, seg_files = [], []
        for i, t in enumerate(texts):
            raw = work / f"seg{i:02d}_raw.wav"
            all_words.append(_piper_one(t, raw))
            wav = work / f"seg{i:02d}.wav"
            _to_wav(raw, wav)
            seg_files.append(wav)

    # concatenate with gaps, offsetting word timings
    out = work / "narration.wav"
    silence = b"\x00\x00" * int(SR * GAP)
    t0, seg_meta = 0.0, []
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        for i, (wav, words) in enumerate(zip(seg_files, all_words)):
            with wave.open(str(wav), "rb") as r:
                frames = r.readframes(r.getnframes())
                dur = r.getnframes() / r.getframerate()
            w.writeframes(frames)
            seg_meta.append({"text": texts[i], "start": t0, "end": t0 + dur,
                             "words": [{**wd, "start": wd["start"] + t0, "end": min(wd["end"], dur) + t0}
                                       for wd in words]})
            t0 += dur
            if i < len(seg_files) - 1:
                g = GAP
                if gaps and i + 1 < len(gaps) and gaps[i + 1]:
                    g = max(GAP, min(1.6, float(gaps[i + 1])))
                w.writeframes(silence if g == GAP else b"\x00\x00" * int(SR * g))
                t0 += g
    return {"audio": out, "duration": t0, "segments": seg_meta, "engine": engine}
