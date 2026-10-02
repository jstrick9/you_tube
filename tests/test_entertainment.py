"""Entertainment layer: persona asides (jokes never smuggle in facts), formats that fit the topic, seasonal formats,
kinetic captions (count-ups, reveal flash, countdown, PROVEN stamp) and procedural sound effects."""
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, qa, render, sfx  # noqa: E402
from autotube.pipeline import build_description, effects_plan, pause_plan  # noqa: E402
from autotube.scriptwriter import ScriptWriter, spoken_text, tidy  # noqa: E402
from autotube.strategy import Strategy  # noqa: E402

CFG = common.load_config()
SOURCE = {"title": "Hunger stone", "url": "https://en.wikipedia.org/wiki/Hunger_stone",
          "text": ("A hunger stone is a type of hydrological landmark common in Central Europe. Hunger stones were "
                   "embedded into a river to commemorate droughts and to warn future generations. One stone in the "
                   "Elbe reads 'If you see me, then weep'. The oldest legible marks date from 1616. People carved "
                   "the stones during severe droughts when the water level was low.")}


def seg(text, aside="", emphasis=None, reveal=False):
    return {"text": text, "aside": aside, "emphasis": emphasis or [], "reveal": reveal,
            "visual": {"shows": "hunger stone in the Elbe", "queries": ["hunger stone river"]}, "evidence": ""}


def test_tidy_keeps_asides_as_jokes_only():
    s = {"segments": [seg("If you see me, then weep.", aside="Wow."),                 # hook: no aside
                      seg("People carved warnings into river stones.", aside="Cheerful bunch"),
                      seg("The oldest marks date from 1616.", aside="Over 400 years of bad news."),   # number → dropped
                      seg("They appear during droughts.", aside="Nature, please.", emphasis=["droughts", "unicorn"]),
                      seg("And that is why", aside="Also this one is way too long to be a real quick deadpan aside.")]}
    tidy(s, "sounds_fake", max_asides=2)
    asides = [x["aside"] for x in s["segments"]]
    assert asides == ["", "Cheerful bunch.", "", "Nature, please.", ""]
    assert s["segments"][3]["emphasis"] == ["droughts"]                  # only words that are really in the line
    assert [x["reveal"] for x in s["segments"]] == [False, False, False, True, False]   # default: last body line
    assert spoken_text(s["segments"][1]) == "People carved warnings into river stones. Cheerful bunch."


def test_tidy_single_reveal_never_on_hook():
    s = {"segments": [seg("a", reveal=True), seg("b", reveal=True), seg("c", reveal=True), seg("d")]}
    tidy(s, "backstory")
    assert [x["reveal"] for x in s["segments"]] == [False, True, False, False]


class ReviewLLM:
    lite = False

    def __init__(self):
        self.prompts = []

    def json(self, system, user, **kw):
        self.prompts.append(user)
        o = {"score": 9, "hook_strength": 9, "entertainment": 8, "factual_errors": [], "policy_concerns": [],
             "misleading_title": False, "advertiser_friendly": True}
        kw.get("validate") and kw["validate"](o)
        return o


class Strat:
    def category_weight(self, c):
        return 0.5

    def source_weight(self, s):
        return 0.5


def _script(aside):
    return {"title": "The river stones that say weep", "segments": [
        seg("If you see me, then weep."),
        dict(seg("Hunger stones were embedded into a river to warn future generations.", aside=aside),
             evidence="embedded into a river to commemorate droughts and to warn future generations"),
        dict(seg("People carved the stones during severe droughts."),
             evidence="People carved the stones during severe droughts when the water level was low"),
        dict(seg("The oldest legible marks date from 1616.", reveal=True),
             evidence="The oldest legible marks date from 1616"),
        seg("And that is why"),
    ]}


def test_reviewer_sees_asides_labelled_as_jokes():
    llm = ReviewLLM()
    ok, review = ScriptWriter(CFG, llm, Strat()).check(_script("Cheerful bunch."), SOURCE)
    assert ok, review
    p = llm.prompts[-1]
    assert "[ASIDE - joke/opinion, not a factual claim] Cheerful bunch." in p
    assert "ASIDES are humour, not claims" in p and "entertainment" in p


def test_numbers_in_asides_are_still_fact_checked():
    llm = ReviewLLM()
    ok, review = ScriptWriter(CFG, llm, Strat()).check(_script("All 9000 of them."), SOURCE)
    assert not ok and any("9000" in i for i in review["issues"])
    assert not llm.prompts                                               # rejected before the LLM is even asked


def test_blocked_words_in_asides_are_caught():
    """Asides go through the same screen as narration.

    Deliberate change: only genuinely non-negotiable terms are hard-blocked in speech now. A merely
    sensitive word ("killer vibes") is left to the LLM reviewer to judge in context, which is what
    YouTube's advertiser-friendly guidelines actually turn on.
    """
    w = ScriptWriter(CFG, ReviewLLM(), Strat())
    ok, review = w.check(_script("Pure torture."), SOURCE)
    assert not ok and any("never allowed" in i for i in review["issues"])


def _strategy():
    s = Strategy.__new__(Strategy)
    s.cfg, s.explore, s.state = CFG, 0.0, {"arms": {}, "updates": 0}
    for dim, opts in s.options().items():
        s.state["arms"][dim] = {o: {"a": 1.0, "b": 1.0, "n": 0} for o in opts}
    return s


def test_seasonal_formats(monkeypatch):
    s = _strategy()
    assert "creepy_true" in s.formats_in_season(10)
    assert "creepy_true" not in s.formats_in_season(3)
    assert "sounds_fake" in s.formats_in_season(3)


def test_fit_uses_a_format_the_topic_supports(monkeypatch):
    from autotube import strategy as st
    s = _strategy()
    monkeypatch.setattr(s, "formats_in_season", lambda month=None: [f for f in CFG["content"]["formats"]
                                                                     if f != "creepy_true"])
    monkeypatch.setattr(s, "seasonal_now", lambda: [])
    for _ in range(20):
        p = s.fit({"format": "dumbest_decision", "hook_style": "question", "voice": "v"},
                  {"formats": ["plot_twist", "myth_buster"]})
        assert p["format"] in ("plot_twist", "myth_buster")
    assert s.fit({"format": "what_if"}, {"formats": []})["format"] == "what_if"      # no info → keep the plan
    assert st  # module import ok


def test_fit_prefers_creepy_in_october_once(monkeypatch):
    s = _strategy()
    monkeypatch.setattr(s, "formats_in_season", lambda month=None: list(CFG["content"]["formats"]))
    monkeypatch.setattr(s, "seasonal_now", lambda: ["creepy_true"])
    topic = {"formats": ["creepy_true", "plot_twist"]}
    assert s.fit({"format": "plot_twist"}, topic, set())["format"] == "creepy_true"
    assert s.fit({"format": "plot_twist"}, topic, {"creepy_true"})["format"] == "plot_twist"


def test_selector_asks_for_fitting_formats():
    class LLM:
        lite = False
        prompt = ""

        def json(self, system, user, **kw):
            LLM.prompt = user
            return {"picks": [{"index": 0, "viral_score": 9, "category": "history", "wiki_query": "Hunger stone",
                               "formats": ["creepy_true", "plot_twist", "not_a_format"]}]}

    cands = [{"topic": "Hunger stone", "topic_key": "hunger stone", "sources": ["wikipedia"], "score": 0.9}]
    picks = ScriptWriter(CFG, LLM(), _strategy()).select_topics(cands, 1, set())
    assert "FORMATS we can make" in LLM.prompt and "ENTERTAINING" in LLM.prompt
    assert "not_a_format" not in picks[0]["formats"] and "plot_twist" in picks[0]["formats"]


def test_number_parsing_for_count_ups():
    assert render._number("1,500") == ("", 1500.0, 0, "")
    assert render._number("70%.") == ("", 70.0, 0, "%")
    assert render._number("$25.5") == ("$", 25.5, 1, "")
    assert render._number("1616") is None           # a year: never "count up" to a year
    assert render._number("7") is None              # too small to be worth animating
    assert render._number("hello") is None


def _tts():
    words = lambda text, t0: [{"word": w, "start": t0 + i * 0.3, "end": t0 + i * 0.3 + 0.28}  # noqa: E731
                              for i, w in enumerate(text.split())]
    texts = ["If you see me weep", "People carved 1,500 stones Cheerful bunch", "Guess when",
             "Only during droughts", "And that is why"]
    segs, t = [], 0.0
    for i, x in enumerate(texts):
        t += 1.25 if i == 3 else (0.18 if i else 0)
        ws = words(x, t)
        segs.append({"text": x, "start": t, "end": ws[-1]["end"], "words": ws})
        t = ws[-1]["end"]
    return {"segments": segs, "duration": t}


def _fx():
    script = {"segments": [seg("If you see me weep"), seg("People carved 1,500 stones", aside="Cheerful bunch.",
                                                          emphasis=["stones"]),
                           seg("Guess when"), seg("Only during droughts", reveal=True), seg("And that is why")]}
    return script, effects_plan(script, {"format": "guess_reveal"}, SOURCE, CFG)


def test_pause_plan_gives_the_reveal_a_beat():
    script, _ = _fx()
    assert pause_plan(script, {"format": "guess_reveal"}, CFG)[3] == CFG["content"]["countdown_pause"]
    assert pause_plan(script, {"format": "plot_twist"}, CFG)[3] == CFG["content"]["reveal_pause"]
    assert sum(pause_plan(script, {"format": "plot_twist"}, CFG)) == CFG["content"]["reveal_pause"]


def test_kinetic_captions(tmp_path):
    _, fx = _fx()
    tts = _tts()
    out = render.build_ass(tts["segments"], "IF YOU SEE ME, WEEP", "Proof in a Minute", 1080, 1920, render.THEMES[0],
                           "Anton", tmp_path / "c.ass", tts["duration"] + 0.7, fx=fx)
    a = out.read_text()
    assert ",Count,," in a and "1,500" in a                   # number count-up ends on the exact number
    assert a.count(",Countdown,,") == 3 and "LOCK IN YOUR GUESS" in a
    assert ",Fx,," in a                                        # reveal flash
    assert ",StampTxt,," in a and "PROVEN" in a and "Source: Wikipedia · Hunger stone" in a
    assert "\\i1}cheerful" in a                                # asides: italic, lower-case, own colour
    assert render.ASIDE_COL in a


def test_captions_without_fx_are_unchanged(tmp_path):
    tts = _tts()
    a = render.build_ass(tts["segments"], "HOOK", "Brand", 1080, 1920, render.THEMES[0], "Anton",
                         tmp_path / "c.ass", tts["duration"] + 0.7).read_text()
    assert ",Count,," not in a and "PROVEN" not in a


def test_sfx_events_and_track(tmp_path):
    timeline = [{"start": x} for x in (0.0, 2.0, 2.3, 5.0, 8.0, 11.0)]
    ev = sfx.plan_events(timeline, [3.4], reveal_at=8.0, stamp_at=12.0, total=14.0)
    kinds = [e["kind"] for e in ev]
    assert {"riser", "hit", "stamp", "ding", "whoosh", "pop"} <= set(kinds)
    wt = sorted(e["t"] for e in ev if e["kind"] == "whoosh")
    assert all(b - a >= 0.6 for a, b in zip(wt, wt[1:]))      # never crowded
    assert not any(6.8 <= t <= 8.3 for t in wt)                # the riser owns the lead-in to the reveal
    p = sfx.build_track(ev, 14.0, tmp_path / "s.wav", seed=3)
    with wave.open(str(p)) as w:
        assert w.getnchannels() == 2 and abs(w.getnframes() / w.getframerate() - 14.5) < 0.01


def test_narration_check_includes_asides():
    script = {"segments": [seg("People carved stones.", aside="Cheerful bunch.")]}
    tts = {"segments": [{"text": "People carved stones. Cheerful bunch.", "start": 0, "end": 2,
                         "words": [{"word": w} for w in "People carved stones Cheerful bunch".split()]}]}
    assert qa.narration_check(tts, script) == []


def test_description_has_the_brand_promise():
    d = build_description({"description": "About hunger stones.", "hashtags": ["#history"]}, SOURCE, [], CFG, "edge")
    assert "Sounds fake. It's proven." in d and "Source: Hunger stone" in d


def test_fit_avoids_repeating_a_format_in_one_run(monkeypatch):
    s = _strategy()
    monkeypatch.setattr(s, "formats_in_season", lambda month=None: [f for f in CFG["content"]["formats"]
                                                                     if f != "creepy_true"])
    monkeypatch.setattr(s, "seasonal_now", lambda: [])
    topic = {"formats": ["scale_shock", "backstory"]}
    for _ in range(10):
        assert s.fit({"format": "scale_shock"}, topic, {"scale_shock"})["format"] == "backstory"
    assert s.fit({"format": "scale_shock"}, {"formats": ["scale_shock"]}, {"scale_shock"})["format"] == "scale_shock"


def test_pinned_persona_voice_is_the_signature_narrator():
    """A recognisable channel has ONE narrator, and rotating voices also split the bandit's evidence
    five ways for no benefit. When persona.voice is set, every plan uses it."""
    s = _strategy()
    plans = s.plan(3)
    pinned = s.cfg.get("persona", {}).get("voice")
    assert pinned, "config should pin a signature narrator"
    assert {p["voice"] for p in plans} == {pinned}


def test_plan_varies_voices_when_none_is_pinned():
    s = _strategy()
    s.cfg = {**s.cfg, "persona": {**s.cfg.get("persona", {}), "voice": None}}
    plans = s.plan(3)
    assert len({p["voice"] for p in plans}) == 3


def test_flat_scripts_are_rewritten():
    class Flat(ReviewLLM):
        def json(self, system, user, **kw):
            o = super().json(system, user, **kw)
            o["entertainment"] = 5
            return o
    ok, review = ScriptWriter(CFG, Flat(), Strat()).check(_script("Cheerful bunch."), SOURCE)
    assert not ok and any("not entertaining enough" in i for i in review["issues"])


def test_small_numbers_count_up_only_with_a_scale_word():
    assert render._number("1.7") is None
    assert render._number("1.7", scaled=True) == ("", 1.7, 1, "")
