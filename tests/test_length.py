"""Holding the Shorts length band (AUDIT 3: 37% of videos overshot 24-36s)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import render  # noqa: E402
from autotube.pipeline import MAX_SPEEDUP_PCT, fit_length, shown_duration, weakest_segment  # noqa: E402

CFG = {"video": {"target_seconds": [24, 36], "speech_rate": "+8%"},
       "content": {"reveal_pause": 0.55, "countdown_pause": 1.25}}
PLAN = {"voice": "v", "format": "sounds_fake"}

WORDS_PER_SEC = 3.0          # measured for en-US-AndrewNeural at +8%


def script_of(n_body, words_per_line=18):
    segs = [{"text": "hook " * 8}]
    for i in range(n_body):
        segs.append({"text": " ".join(f"w{i}_{j}" for j in range(words_per_line)),
                     "evidence": "quote here", "aside": "", "reveal": i == n_body - 1})
    segs.append({"text": "payoff line here"})
    return {"segments": segs}


def synth_factory(rate_effect=True):
    """A stand-in for edge-tts: duration follows word count and the requested rate."""
    def synth(lines, voice, rate, work, gaps=None):
        words = sum(len(l.split()) for l in lines)
        pct = int(str(rate).strip("+%"))
        secs = words / (WORDS_PER_SEC * ((1 + pct / 100) / 1.08 if rate_effect else 1))
        secs += sum(gaps or [])
        return {"duration": secs, "engine": "fake", "segments": [{"start": 0}] * len(lines)}
    return synth


# ── the measurement ───────────────────────────────────────────────────────────
def test_shown_duration_counts_the_renderer_tail():
    """Measuring narration alone let every video run TAIL seconds past its target."""
    assert shown_duration({"duration": 30.0}) == 30.0 + render.TAIL
    assert render.TAIL > 0


# ── the band ──────────────────────────────────────────────────────────────────
def test_a_short_script_is_left_alone():
    s = script_of(2)
    before = len(s["segments"])
    tts, _ = fit_length(s, PLAN, CFG, Path("/tmp"), synth_factory())
    assert shown_duration(tts) <= 36
    assert len(s["segments"]) == before, "nothing should be dropped from an in-band script"


def test_an_overlong_script_is_trimmed_into_the_band():
    s = script_of(7)                      # ~140 words -> well over 36s
    tts, _ = fit_length(s, PLAN, CFG, Path("/tmp"), synth_factory())
    assert shown_duration(tts) <= 36, shown_duration(tts)
    assert len(s["segments"]) < 9, "should have dropped at least one body line"


def test_the_hook_reveal_and_payoff_are_never_dropped():
    s = script_of(7)
    hook, payoff = s["segments"][0]["text"], s["segments"][-1]["text"]
    fit_length(s, PLAN, CFG, Path("/tmp"), synth_factory())
    assert s["segments"][0]["text"] == hook
    assert s["segments"][-1]["text"] == payoff
    assert any(x.get("reveal") for x in s["segments"]), "the reveal must survive trimming"


def test_speedup_rescues_a_script_with_nothing_safe_to_drop():
    """A 5-segment script protects three segments, so trimming runs out while still too long.
    The old code only intervened at the 59s cliff, which is how 44s videos shipped."""
    s = script_of(2, words_per_line=60)   # long lines, only 4 segments -> weakest_segment is None
    assert weakest_segment(s["segments"]) is None
    tts, _ = fit_length(s, PLAN, CFG, Path("/tmp"), synth_factory())
    assert tts is not None
    assert shown_duration(tts) < 44, "the speed-up fallback must have engaged"


def test_speedup_is_bounded_so_the_voice_never_turns_robotic():
    rates = []

    def synth(lines, voice, rate, work, gaps=None):
        rates.append(int(str(rate).strip("+%")))
        return {"duration": 300.0, "engine": "f", "segments": [{"start": 0}]}

    s = script_of(2, words_per_line=60)
    fit_length(s, PLAN, CFG, Path("/tmp"), synth)
    assert max(rates) <= 8 + MAX_SPEEDUP_PCT, rates


def test_a_hopeless_script_is_skipped_rather_than_shipped_over_60s():
    def synth(lines, voice, rate, work, gaps=None):
        return {"duration": 300.0, "engine": "f", "segments": [{"start": 0}]}

    tts, _ = fit_length(script_of(2, 60), PLAN, CFG, Path("/tmp"), synth)
    assert tts is None, "a Short over 59s loses its classification and must not be published"


def test_something_slightly_over_is_kept_not_thrown_away():
    def synth(lines, voice, rate, work, gaps=None):
        return {"duration": 40.0, "engine": "f", "segments": [{"start": 0}]}

    tts, _ = fit_length(script_of(2, 60), PLAN, CFG, Path("/tmp"), synth)
    assert tts is not None, "over target but well inside the Shorts limit is still publishable"


def test_reveal_pause_counts_toward_the_budget():
    s = script_of(3)
    _, gaps = fit_length(s, PLAN, CFG, Path("/tmp"), synth_factory())
    assert sum(gaps) > 0, "the reveal beat is real time and must be budgeted"
