"""Procedural, copyright-free background music.

Every video gets a freshly synthesized ambient/lo-fi bed (random key, tempo,
progression, timbre, seeded per video). Because it's generated from math, there
is zero Content ID risk and no licensing to track — and no two videos share the
same soundtrack (helps against "repetitive content" signals).
"""
from __future__ import annotations

import random
import wave
from pathlib import Path

import numpy as np

SR = 44100
PROGRESSIONS = [
    [0, 5, 3, 4], [0, 3, 5, 4], [0, 4, 5, 3], [5, 3, 0, 4], [0, 5, 1, 4], [0, 2, 5, 4],
]
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]


def _note_freq(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _pad(freq: float, dur: float, bright: float, rng: random.Random) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    detune = [1.0, 1.0035, 0.9965]
    sig = np.zeros_like(t)
    for d in detune:
        ph = rng.random() * 6.28
        f = freq * d
        sig += np.sin(2 * np.pi * f * t + ph)
        sig += bright * 0.35 * np.sin(2 * np.pi * 2 * f * t + ph)
        sig += bright * 0.12 * np.sin(2 * np.pi * 3 * f * t + ph)
    att, rel = min(0.8, dur * 0.3), min(1.2, dur * 0.4)
    env = np.ones_like(t)
    a, r = int(att * SR), int(rel * SR)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    return sig * env / len(detune)


def _pluck(freq: float, dur: float) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    env = np.exp(-t * 6)
    return (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)) * env


def _lowpass(x: np.ndarray, alpha: float) -> np.ndarray:
    # one-pole IIR, vectorized via cumulative trick is unstable; do chunked loop in numpy
    y = np.empty_like(x)
    acc = 0.0
    for i in range(0, len(x), 4096):
        chunk = x[i:i + 4096]
        out = np.empty_like(chunk)
        for j, v in enumerate(chunk):
            acc += alpha * (v - acc)
            out[j] = acc
        y[i:i + 4096] = out
    return y


def generate(duration: float, out: Path, seed: int | None = None) -> Path:
    rng = random.Random(seed)
    root = rng.choice(range(45, 57))              # A2..G#3
    scale = rng.choice([MAJOR, MINOR, MINOR])
    bpm = rng.choice([72, 78, 84, 90, 96])
    beat = 60 / bpm
    bar = beat * 4
    prog = rng.choice(PROGRESSIONS)
    bright = rng.uniform(0.2, 0.7)
    arp_on = rng.random() < 0.7
    total = int((duration + 1.5) * SR)
    mix = np.zeros(total)

    t = 0.0
    i = 0
    while t < duration + 1.5:
        deg = prog[i % len(prog)]
        chord = [scale[(deg + k) % 7] + 12 * ((deg + k) // 7) for k in (0, 2, 4)]
        start = int(t * SR)
        # pad
        for n in chord:
            p = _pad(_note_freq(root + 12 + n), bar * 1.05, bright, rng) * 0.16
            end = min(total, start + len(p))
            mix[start:end] += p[:end - start]
        # bass
        b = _pad(_note_freq(root + chord[0]), bar, 0.1, rng) * 0.22
        end = min(total, start + len(b))
        mix[start:end] += b[:end - start]
        # arpeggio
        if arp_on:
            pattern = rng.choice([[0, 1, 2, 1], [0, 2, 1, 2], [2, 1, 0, 1], [0, 1, 2, 0]])
            steps = 8
            for s in range(steps):
                n = chord[pattern[s % 4]] + 24
                pl = _pluck(_note_freq(root + n), beat * 0.9) * 0.07
                st = start + int(s * beat / 2 * SR)
                end = min(total, st + len(pl))
                if st < total:
                    mix[st:end] += pl[:end - st]
        # soft kick-like pulse on beats 1 and 3
        for s in (0, 2):
            kt = np.arange(int(0.25 * SR)) / SR
            kick = np.sin(2 * np.pi * (55 + 60 * np.exp(-kt * 30)) * kt) * np.exp(-kt * 14) * 0.18
            st = start + int(s * beat * SR)
            end = min(total, st + len(kick))
            if st < total:
                mix[st:end] += kick[:end - st]
        t += bar
        i += 1

    # gentle vinyl-ish noise + lowpass warmth
    mix += np.random.default_rng(seed).normal(0, 0.004, total)
    # (warmth low-pass is applied by ffmpeg at mix time — much faster than Python)
    # fade in/out
    f = int(1.0 * SR)
    mix[:f] *= np.linspace(0, 1, f)
    mix[-f * 2:] *= np.linspace(1, 0, f * 2)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.8
    pcm = (mix * 32767).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).ravel()
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())
    return out
