"""Caption chunking and hook-card layout (AUDIT 1, secondary render issues)."""
import re
import pytest
from autotube.render import (_chunk_words, _hook_lines, fit_hook, DANGLERS,
                             CHUNK_MAX_WORDS, CHUNK_MAX_CHARS, HOOK_MAX_LINES)

SCRIPTS = [
    "Scientists just found something impossible under Antarctica.",
    "A lake sealed off from the sky for fifteen million years.",
    "It sits four kilometres below solid ice, and it is still liquid.",
    "The water never freezes because the planet's own heat warms it from below.",
    "Whatever lives down there evolved completely alone.",
    "He lost everything in a single afternoon.",
    "Nobody noticed for thirty years.",
    "It is still liquid.",
    "Wow.",
    "One.",
]


def mk(text, aside_from=None):
    out = []
    for i, w in enumerate(text.split()):
        out.append({"word": w, "start": i * 0.3, "end": i * 0.3 + 0.25,
                    "_aside": aside_from is not None and i >= aside_from})
    return out


def words_of(chunk):
    return " ".join(w["word"] for w in chunk)


@pytest.mark.parametrize("text", SCRIPTS)
def test_no_single_word_card_unless_the_whole_segment_is_one_word(text):
    """A lone word flashed for a fraction of a second is unreadable - and it lands on the payoff."""
    chunks = _chunk_words(mk(text))
    if len(text.split()) <= 1:
        return
    assert all(len(c) >= 2 for c in chunks), \
        f"orphan card in {text!r}: {[words_of(c) for c in chunks]}"


@pytest.mark.parametrize("text", SCRIPTS)
def test_chunks_stay_readable_and_lose_no_words(text):
    chunks = _chunk_words(mk(text))
    assert [w["word"] for c in chunks for w in c] == text.split(), "chunking dropped or reordered words"
    for c in chunks:
        assert len(c) <= CHUNK_MAX_WORDS
        assert len(words_of(c)) <= CHUNK_MAX_CHARS + 4


@pytest.mark.parametrize("text", SCRIPTS)
def test_chunks_are_in_ascending_time_order(text):
    chunks = _chunk_words(mk(text))
    starts = [c[0]["start"] for c in chunks]
    assert starts == sorted(starts)


def test_the_payoff_word_is_not_stranded_alone():
    chunks = _chunk_words(mk("Whatever lives down there evolved completely alone."))
    assert words_of(chunks[-1]) == "completely alone."


def test_a_card_does_not_end_on_a_dangling_function_word_when_it_can_be_moved():
    chunks = _chunk_words(mk("A lake sealed off from the sky for fifteen million years."))
    texts = [words_of(c) for c in chunks]
    assert "from the sky for" not in texts
    assert any(t.startswith("for ") for t in texts), texts


def test_asides_are_never_merged_into_a_spoken_card():
    """The aside is styled differently; mixing it into a normal card would mis-style both halves."""
    words = mk("Nobody noticed for thirty years allegedly", aside_from=5)
    for c in _chunk_words(words):
        assert len({w["_aside"] for w in c}) == 1, f"card mixes aside and narration: {words_of(c)}"


def test_chunker_handles_degenerate_input():
    assert _chunk_words([]) == []
    assert len(_chunk_words(mk("Wow."))) == 1


# ── hook card ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("hook", [
    "SEALED FOR 15 MILLION YEARS",
    "THIS LAKE HAS BEEN SEALED OFF FROM THE SKY FOR 15 MILLION YEARS",
    "A SHORT ONE",
    "WHY NOBODY NOTICED THIS FOR THIRTY YEARS AND WHAT IT MEANS FOR EVERYONE ALIVE TODAY",
])
def test_hook_card_never_grows_into_the_counter_zone(hook):
    """The card is top-anchored and grows down; Count sits at 0.30 H and Label at 0.25 H."""
    assert _hook_lines(fit_hook(hook)) <= HOOK_MAX_LINES


@pytest.mark.parametrize("hook", [
    "THIS LAKE HAS BEEN SEALED OFF FROM THE SKY FOR 15 MILLION YEARS",
    "WHY NOBODY NOTICED THIS FOR THIRTY YEARS AND WHAT IT MEANS FOR EVERYONE",
])
def test_a_trimmed_hook_does_not_end_mid_thought(hook):
    last = fit_hook(hook).split()[-1]
    assert re.sub(r"[^a-z]", "", last.lower()) not in DANGLERS, f"hook ends on {last!r}"


def test_hook_is_a_prefix_of_the_original_and_never_empty():
    for hook in ["SHORT", "THIS LAKE HAS BEEN SEALED OFF FROM THE SKY FOR 15 MILLION YEARS"]:
        f = fit_hook(hook)
        assert f and hook.startswith(f.rstrip(",;:-"))
    assert fit_hook("") == ""


def test_hook_card_ends_before_the_first_number_counter(tmp_path):
    """Counter and card are both anchored near the top, so they must not be on screen together."""
    from autotube import render

    segs, t = [], 0.0
    for tx in ["The vault held 2400 tonnes of gold.", "Nobody has opened it since."]:
        ws = []
        for w in tx.split():
            ws.append({"word": w, "start": round(t, 2), "end": round(t + 0.3, 2)})
            t += 0.3
        segs.append({"text": tx, "start": ws[0]["start"], "end": round(t, 2), "words": ws})

    out = render.build_ass(segs, "THE VAULT NOBODY OPENED", "AutoTube", 1080, 1920,
                           render.THEMES[0], "Anton", tmp_path / "c.ass", t,
                           fx={"counts": True}, arch=render.ARCHETYPES[0])
    body = out.read_text().split("[Events]")[1]

    def secs(ts):
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    hook_end = [secs(l.split(",")[2]) for l in body.splitlines() if ",Hook," in l]
    counts = [secs(l.split(",")[1]) for l in body.splitlines() if ",Count," in l]
    assert hook_end, "no hook card rendered"
    if counts:
        assert hook_end[0] <= min(counts), f"hook card still up at {hook_end[0]}s when counter fires at {min(counts)}s"
