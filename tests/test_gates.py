"""Deterministic script gates — the rules the LLM reviewer kept missing."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autotube import gates  # noqa: E402
from autotube.common import load_config  # noqa: E402
from autotube.scriptwriter import spoken_text  # noqa: E402

CFG = load_config()
ROOT = Path(__file__).resolve().parent.parent


def _script(hook, *body, title="A Perfectly Reasonable Title", closer="and that is why it still happens today"):
    segs = [{"text": hook}] + [{"text": b} for b in body] + [{"text": closer}]
    return {"title": title, "segments": segs}


# ── the regression this module exists for ─────────────────────────────────────
def test_catches_the_sample_the_reviewer_scored_nine():
    """docs/samples/...whooping-crane.json opens 'Did you know...' and closes 'Follow Curious Minute
    for more wildlife wonders.' The LLM reviewer gave it score 9, hook_strength 9, and it shipped."""
    d = json.loads((ROOT / "docs/samples/20260923-1433-01-whooping-crane.json").read_text())
    assert d["review"]["score"] >= 9 and d["review"]["hook_strength"] >= 9, "fixture should be the lenient grade"
    issues = gates.run_all(d["script"], spoken_text, CFG)
    assert any("banned phrase" in i for i in issues)
    assert any("call-to-action" in i for i in issues)


def test_catches_the_second_sample_too():
    d = json.loads((ROOT / "docs/samples/20260923-1433-02-solar-panel.json").read_text())
    issues = gates.run_all(d["script"], spoken_text, CFG)
    assert any("banned phrase" in i for i in issues) and any("call-to-action" in i for i in issues)


# ── openers ───────────────────────────────────────────────────────────────────
def test_banned_openers():
    for hook in ["Did you know that octopuses have three hearts here",
                 "Have you ever wondered why the sky looks blue",
                 "Here are five facts about the deep ocean floor",
                 "In this video we look at the Roman concrete mystery",
                 "Hey guys welcome back to another amazing science short",
                 "Let's talk about why flamingos actually stand on one leg",
                 "You won't believe what divers found under the ice"]:
        assert gates.check_hook(hook), f"should be rejected: {hook}"


def test_good_hooks_pass():
    for hook in ["Antarctica has a waterfall that runs blood red",
                 "A teaspoon of neutron star weighs 6 billion tons",
                 "Your stomach lining replaces itself every four days",
                 "One California tree was already old when Egypt built pyramids"]:
        assert gates.check_hook(hook) == [], f"false positive on: {hook} -> {gates.check_hook(hook)}"


def test_hook_length_band():
    assert any("word" in i for i in gates.check_hook("Volcanoes"))
    assert any("12" in i for i in gates.check_hook(" ".join(["word"] * 20)))
    assert gates.check_hook("") == ["the hook is empty"]


def test_segment_count_is_consistent_with_the_writer_prompt_and_final_gate():
    body = {"text": "Each concrete new detail raises the stakes and advances the story"}
    payoff = {"text": "so the final fact completes the original question"}
    for count in (4, 5):
        segments = [{"text": "Antarctica has a waterfall that runs blood red"}]
        segments += [dict(body) for _ in range(count - 2)]
        segments.append(dict(payoff))
        assert not any("exactly 4 or 5" in issue for issue in gates.check_segment_lengths(segments, CFG))
    for count in (3, 6, 8):
        segments = [{"text": "Antarctica has a waterfall that runs blood red"}]
        segments += [dict(body) for _ in range(count - 2)]
        segments.append(dict(payoff))
        assert any("exactly 4 or 5" in issue for issue in gates.check_segment_lengths(segments, CFG))


def test_run_all_rejects_non_object_and_malformed_segment_shapes():
    assert any("JSON object" in issue for issue in gates.run_all(None, spoken_text, CFG))
    assert any("JSON object" in issue for issue in gates.run_all(
        {"title": "T", "segments": ["not an object"]}, spoken_text, CFG))
    assert any("non-empty text" in issue for issue in gates.run_all(
        {"title": "T", "segments": [{"text": ""}]}, spoken_text, CFG))


def test_segment_lengths_reject_dangling_turn_and_payoff():
    s = _script("Antarctica has a waterfall that runs blood red",
                "Which is actually the Manicouagan Reservoir's outer rim",
                "A second concrete fact reveals how the ancient crater formed underwater",
                closer="All 215 million years ago")
    issues = gates.run_all(s, spoken_text, CFG)
    assert any("segment 2" in i and "turn" in i for i in issues), issues
    assert any("complete, concrete answer" in i for i in issues), issues


# ── closers ───────────────────────────────────────────────────────────────────
def test_cta_closers_break_the_loop():
    for closer in ["Follow for more facts like this",
                   "Follow Curious Minute for more wildlife wonders.",
                   "Subscribe for daily science",
                   "Let me know in the comments what you think",
                   "Thanks for watching and see you next time",
                   "Follow us.",
                   "For more space facts, stick around"]:
        assert gates.check_closer(closer), f"should be rejected: {closer}"


def test_factual_closers_pass():
    for closer in ["and that is why the river still runs red today",
                   "which is exactly how it ended up on the moon",
                   "so the 21 birds became 830"]:
        assert gates.check_closer(closer) == [], f"false positive on: {closer}"


# ── titles ────────────────────────────────────────────────────────────────────
def test_clickbait_titles_rejected_real_ones_pass():
    assert gates.check_title("You Won't Believe What Lives Under Antarctica")
    assert gates.check_title("THIS CHANGES EVERYTHING ABOUT THE OCEAN")
    assert gates.check_title("The Tree That Was Old Before The Pyramids") == []


def test_no_false_positives_on_our_real_titles():
    hist = json.loads((ROOT / "state/history.json").read_text())
    flagged = [(h["title"], gates.check_title(h["title"])) for h in hist if gates.check_title(h.get("title", ""))]
    assert not flagged, f"gates must not reject titles we already ship: {flagged}"


# ── narration hygiene ─────────────────────────────────────────────────────────
def test_production_artifacts_in_speech():
    s = _script("Antarctica has a waterfall that runs blood red",
                "[cut to wide shot] Brine seeps through a crack in the ice",
                "It was first seen in 1911 🤯",
                "Read more at https://example.com #science",
                "The iron oxidises on contact with air",
                "The iron oxidises on contact with air")
    issues = gates.run_all(s, spoken_text, CFG)
    assert any("stage direction" in i for i in issues)
    assert any("emoji" in i for i in issues)
    assert any("URL" in i for i in issues)
    assert any("verbatim" in i for i in issues)


def test_a_clean_script_passes_every_gate():
    s = _script("Antarctica has a waterfall that runs blood red",
                "The colour comes from iron-rich brine trapped under the glacier",
                "It has been sealed away from sunlight for two million years",
                closer="so the ice keeps bleeding, unanswered")
    assert gates.run_all(s, spoken_text, CFG) == []


# ── independent reviewer routing ──────────────────────────────────────────────
def _llm(providers, review_providers, models):
    from autotube.llm import LLM
    return LLM({"llm": {"providers": providers, "review_providers": review_providers, **models}})


def test_review_runs_on_a_different_provider_than_the_writer(monkeypatch):
    """A writer grading its own draft agrees with itself, so review must route elsewhere."""
    import autotube.llm as L
    calls = []

    def fake(name):
        def f(model, system, user, temperature, images=None):
            calls.append(f"{name}:{model}")
            return '{"score": 8}'
        return f

    monkeypatch.setitem(L.PROVIDERS, "gemini", (fake("gemini"), "gemini_models"))
    monkeypatch.setitem(L.PROVIDERS, "groq", (fake("groq"), "groq_models"))
    llm = _llm(["gemini", "groq"], ["groq", "gemini"],
               {"gemini_models": ["g-1"], "groq_models": ["q-1"]})
    llm.json("sys", "write this")                       # writer
    writer = llm.last_used
    assert writer == "gemini:g-1"
    llm.review_json("sys", "review this", avoid=writer)  # reviewer
    assert llm.last_used == "groq:q-1", f"reviewer must differ from writer, got {llm.last_used}"
    assert calls == ["gemini:g-1", "groq:q-1"]


def test_avoid_skips_the_exact_writer_model(monkeypatch):
    """With only one provider keyed, review still avoids the specific model that wrote the draft."""
    import autotube.llm as L
    seen = []

    def fake(model, system, user, temperature, images=None):
        seen.append(model)
        return '{"score": 8}'

    monkeypatch.setitem(L.PROVIDERS, "groq", (fake, "groq_models"))
    llm = _llm(["groq"], ["groq"], {"groq_models": ["q-1", "q-2"], "groq_review_models": ["q-1", "q-2"]})
    llm.review_json("sys", "review", avoid="groq:q-1")
    assert seen == ["q-2"] and llm.last_used == "groq:q-2"


def test_review_falls_back_rather_than_failing_closed_on_routing(monkeypatch):
    """An imperfect review beats no review — but the caller can tell it wasn't independent."""
    import autotube.llm as L

    def only(model, system, user, temperature, images=None):
        return '{"score": 8}'

    monkeypatch.setitem(L.PROVIDERS, "groq", (only, "groq_models"))
    llm = _llm(["groq"], ["groq"], {"groq_models": ["q-1"], "groq_review_models": ["q-1"]})
    out = llm.review_json("sys", "review", avoid="groq:q-1")   # the only model IS the writer's
    assert out["score"] == 8
    assert llm.last_used == "groq:q-1", "fell back to the writer's model"


def test_scriptwriter_flags_a_non_independent_review():
    from autotube.scriptwriter import ScriptWriter
    cfg = load_config()

    class SameModel:
        lite = False
        last_used = "gemini:g-1"

        def review_json(self, system, user, avoid=None, **kw):
            return {"score": 9, "hook_strength": 9, "entertainment": 9, "coherence": 8,
                    "loops": True, "factual_errors": [], "misleading_title": False,
                    "advertiser_friendly": True, "policy_concerns": [],
                    "value_add": "A concise explanation of how the hive dries nectar.", "fixes": []}

        def json(self, *a, **k):
            return {"score": 9, "hook_strength": 9, "entertainment": 9, "coherence": 8,
                    "loops": True, "factual_errors": [], "misleading_title": False,
                    "advertiser_friendly": True, "policy_concerns": [],
                    "value_add": "A concise explanation of how the hive dries nectar.", "fixes": []}

    w = ScriptWriter.__new__(ScriptWriter)
    w.llm, w.src_chars = SameModel(), 4000
    w.cfg = {**cfg, "content": dict(cfg["content"], require_grounding=False)}
    src = {"title": "Bees", "text": "Bees make honey from nectar. " * 30}
    script = {"title": "Honey", "_writer_model": "gemini:g-1", "segments": [
        {"text": "Bees visit two million flowers to fill a single jar"},
        {"text": "A worker bee makes one twelfth of a teaspoon in life", "aside": "Busy little things."},
        {"text": "The hive fans its wings to dry the nectar down"},
        {"text": "so the jar in your cupboard will never spoil"}]}
    ok, review = w.check(script, src)
    assert ok and review["independent"] is False and review["reviewer"] == "gemini:g-1"
