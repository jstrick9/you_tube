"""Procedural sound effects (whoosh, ding, riser, hit, stamp, pop) — synthesized from math, so zero copyright /
Content ID risk, and slightly different every video (seeded), like the music bed.

`build_track()` renders every timed effect of a video into ONE stereo WAV that the renderer mixes under the
narration. Effects are what make a Short feel edited instead of slideshow-like: a whoosh on every cut, a ding when a
number lands, a riser that builds tension before the reveal, a hit on the reveal, and a stamp on the proof card.
"""
from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np

SR = 44100


def _t(dur: float) -> np.ndarray:
    return np.arange(int(dur * SR)) / SR


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _onepole(x: np.ndarray, cutoff: np.ndarray | float) -> np.ndarray:
    """Time-varying one-pole low-pass (cutoff in Hz, scalar or per-sample). Pure numpy, short signals only."""
    c = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    a = 1.0 - np.exp(-2 * np.pi * c / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a[i] * (x[i] - acc)
        y[i] = acc
    return y


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return x / m * peak


def whoosh(rng: np.random.Generator, dur: float = 0.38) -> np.ndarray:
    t = _t(dur)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    sweep = 300 + 5200 * np.sin(np.pi * t / dur) ** 1.5
    lp = _onepole(_noise(len(t), rng), sweep)
    hp = lp - _onepole(lp, 180.0)
    return _norm(hp * env, 0.8)


def ding(rng: np.random.Generator, dur: float = 0.6) -> np.ndarray:
    t = _t(dur)
    f = float(rng.choice([1318.5, 1396.9, 1568.0, 1760.0]))
    sig = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(2 * np.pi * 2.01 * f * t) * np.exp(-t * 9)
           + 0.2 * np.sin(2 * np.pi * 3.0 * f * t) * np.exp(-t * 14))
    env = np.exp(-t * 7) * np.clip(t / 0.004, 0, 1)
    return _norm(sig * env, 0.7)


def riser(rng: np.random.Generator, dur: float = 1.1) -> np.ndarray:
    t = _t(dur)
    p = t / dur
    f = 180 * (2 ** (p * 2.6))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.3 * np.sin(2 * np.pi * np.cumsum(f * 1.5) / SR)
    air = _onepole(_noise(len(t), rng), 400 + 7000 * p ** 2)
    env = p ** 2.2
    env[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    return _norm((0.55 * tone + 0.8 * air) * env, 0.75)


def hit(rng: np.random.Generator, dur: float = 0.7) -> np.ndarray:
    """Cinematic low boom for the reveal."""
    t = _t(dur)
    f = 95 * np.exp(-t * 5) + 42
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4.5)
    click = _onepole(_noise(len(t), rng), 2500.0) * np.exp(-t * 60)
    return _norm(np.tanh(1.8 * (body + 0.35 * click)), 0.9)


def stamp(rng: np.random.Generator, dur: float = 0.35) -> np.ndarray:
    """Rubber-stamp 'thunk' for the PROVEN card."""
    t = _t(dur)
    thud = np.sin(2 * np.pi * (140 * np.exp(-t * 18) + 60) * t) * np.exp(-t * 16)
    slap = _onepole(_noise(len(t), rng), 1800.0) * np.exp(-t * 45)
    return _norm(thud + 0.9 * slap, 0.9)


def pop(rng: np.random.Generator, dur: float = 0.12) -> np.ndarray:
    t = _t(dur)
    f = 900 * np.exp(-t * 30) + 300
    return _norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 35), 0.6)


KINDS = {"whoosh": whoosh, "ding": ding, "riser": riser, "hit": hit, "stamp": stamp, "pop": pop}
# relative loudness of each effect under the narration (dB); the renderer applies the overall sfx volume on top
GAIN_DB = {"whoosh": -15, "ding": -12, "riser": -5, "hit": -13, "stamp": -9, "pop": -12}   # tuned on a real render


def plan_events(timeline: list[dict], number_times: list[float], reveal_at: float | None, stamp_at: float | None,
                total: float, min_gap: float = 0.6) -> list[dict]:
    """Decide which effect plays when. Cuts get whooshes (not every one, never crowded), numbers get dings,
    the reveal gets riser→hit, the proof card gets a stamp."""
    ev: list[dict] = [{"kind": "pop", "t": 0.02}]
    if reveal_at is not None and reveal_at > 1.3:
        ev.append({"kind": "riser", "t": reveal_at - 1.1, "end": reveal_at})
        ev.append({"kind": "hit", "t": reveal_at})
    if stamp_at is not None:
        ev.append({"kind": "stamp", "t": stamp_at})
    for t in number_times[:4]:
        ev.append({"kind": "ding", "t": t})
    busy = sorted(e["t"] for e in ev)

    def free(t):
        return all(abs(t - b) >= min_gap for b in busy) and not (
            reveal_at is not None and reveal_at - 1.2 <= t <= reveal_at + 0.3)

    for seg in timeline[1:]:
        t = max(0.0, seg["start"] - 0.2)          # whoosh peaks right on the cut
        if t < total - 0.5 and free(t):
            ev.append({"kind": "whoosh", "t": t})
            busy.append(t)
    return sorted(ev, key=lambda e: e["t"])


def build_track(events: list[dict], total: float, out: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = int((total + 0.5) * SR)
    mix = np.zeros(n)
    for e in events:
        fn = KINDS.get(e["kind"])
        if not fn:
            continue
        snd = fn(rng) * 10 ** (GAIN_DB.get(e["kind"], -8) / 20)
        i = int(max(0.0, e["t"]) * SR)
        if i >= n:
            continue
        seg = snd[: n - i]
        mix[i:i + len(seg)] += seg
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    # slight stereo width: delay the right channel by a few samples, per video
    d = random.Random(seed).randint(8, 30)
    right = np.concatenate([np.zeros(d), mix[:-d]]) if n > d else mix
    st = np.stack([mix, right], axis=1)
    pcm = (np.clip(st, -1, 1) * 32767).astype("<i2")
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return out
