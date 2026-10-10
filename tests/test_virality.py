"""Virality: topics must come from live trends, be scored for viral potential, and scripts must have strong hooks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, trends  # noqa: E402
from autotube.scriptwriter import ScriptWriter  # noqa: E402

CFG = common.load_config()


class Strat:
    def category_weight(self, c):
        return 0.5

    def source_weight(self, s):
        return {"youtube_outliers": 0.9}.get(s, 0.5)


class LLMPicks:
    lite = False

    def __init__(self, picks):
        self.picks, self.prompt = picks, ""

    def json(self, system, user, **kw):
        self.prompt = user
        o = {"picks": self.picks}
        kw.get("validate") and kw["validate"](o)
        return o


def cands():
    rows = [("Hunger stone", "wikipedia", 0.9, ["Wikipedia: 111,489 views/day, 1496.5x its normal"], 1496.5),
            ("Willie Mays", "wikipedia", 0.8, [], None),
            ("Octopus can taste with its arms", "youtube_outliers", 0.8,
             ["viral Short now: 2,000,000 views in 20h"], None),
            ("Dry policy news", "google_trends", 0.7, [], None)]
    return [{"topic": t, "topic_key": t.lower(), "source": s, "sources": [s], "score": sc,
             "context": ctx, "spike": spike}
            for t, s, sc, ctx, spike in rows]


def test_select_keeps_only_viral_topics_and_ranks_them():
    llm = LLMPicks([{"index": 0, "viral_score": 9, "category": "history", "wiki_query": "Hunger stone"},
                    {"index": 1, "viral_score": 9, "category": "nature", "wiki_query": "Octopus"},
                    {"index": 2, "viral_score": 8, "category": "culture", "risk": "high"}])
    w = ScriptWriter(CFG, llm, Strat())
    picks = w.select_topics(cands(), 2, set())
    assert [p["topic"] for p in picks] == ["Octopus can taste with its arms", "Hunger stone"]   # source weight breaks tie
    assert all(p["viral_score"] >= CFG["content"]["min_viral_score"] for p in picks)
    assert "1496.5x its normal" in llm.prompt and "2,000,000 views" in llm.prompt          # judged WITH evidence
    assert "Willie Mays" not in llm.prompt                                               # no Wikipedia-only fallback


def test_wikipedia_listing_without_spike_never_reaches_topic_selector():
    llm = LLMPicks([{"index": 0, "viral_score": 10, "category": "nature", "wiki_query": "Honey"}])
    w = ScriptWriter(CFG, llm, Strat())
    wiki_only = [{"topic": "Honey", "topic_key": "honey", "sources": ["wikipedia"], "score": 1.0}]
    assert w.select_topics(wiki_only, 1, set()) == []
    assert llm.prompt == ""


def test_selection_outage_returns_nothing_not_guesses():
    from autotube.llm import LLMError

    class Down(LLMPicks):
        def json(self, *a, **k):
            raise LLMError("down")

    assert ScriptWriter(CFG, Down([]), Strat()).select_topics(cands(), 2, set()) == []


def test_wikipedia_only_requires_a_verified_pageview_spike():
    wiki = {"sources": ["wikipedia"], "spike": 2.99}
    assert not trends.has_current_trend_evidence(wiki, 3.0)
    assert not trends.has_current_trend_evidence({"sources": ["wikipedia"]}, 3.0)
    assert trends.has_current_trend_evidence({"sources": ["wikipedia"], "spike": 3.0}, 3.0)
    assert not trends.has_current_trend_evidence({"sources": ["evergreen"]}, 3.0)
    assert not trends.has_current_trend_evidence({"sources": ["on_this_day"]}, 3.0)
    assert trends.has_current_trend_evidence({"sources": ["google_trends"]}, 3.0)
    assert trends.has_current_trend_evidence({"source": "reddit_science"}, 3.0)


def test_make_one_fails_closed_without_live_trend_evidence(monkeypatch):
    from autotube import pipeline

    def should_not_produce(*args, **kwargs):
        raise AssertionError("topic without current trend evidence must not be produced")

    monkeypatch.setattr(pipeline, "produce_assets", should_not_produce)
    topic = {"topic": "Honey", "sources": ["evergreen"], "score": 1.0}
    assert pipeline.make_one(CFG, object(), topic, {}, 1, "test-run") is None


def test_make_one_rejects_below_floor_viral_score_even_with_live_source(monkeypatch):
    from autotube import pipeline

    monkeypatch.setattr(pipeline, "produce_assets", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("weak viral score must be rejected before production")))
    topic = {"topic": "Current Google trend", "sources": ["google_trends"], "score": 0.9, "viral_score": 6}
    assert pipeline.make_one(CFG, object(), topic, {}, 1, "test-run") is None


def test_primary_source():
    assert trends.primary_source(["reddit_todayilearned", "wikipedia"]) == "wikipedia"
    assert trends.primary_source(["google_trends", "youtube_outliers"]) == "youtube_outliers"
    assert trends.primary_source(["reddit_science"]) == "reddit"
    assert trends.primary_source([]) == "evergreen"


def test_wikipedia_spikes_demote_perennial_pages(monkeypatch):
    series = {"Spiky": [100] * 27 + [120, 2500, 3000], "Perennial": [50000] * 30}
    monkeypatch.setattr(trends, "get_json", lambda url, **k: {"items": [
        {"views": v} for v in series["Spiky" if "Spiky" in url else "Perennial"]]})
    items = [{"topic": "Perennial", "wiki_title": "Perennial", "source": "wikipedia", "score": 1.0, "views": 50000},
             {"topic": "Spiky", "wiki_title": "Spiky", "source": "wikipedia", "score": 0.6, "views": 3000}]
    trends.wikipedia_spikes(items)
    by = {i["topic"]: i for i in items}
    assert by["Spiky"]["score"] > by["Perennial"]["score"]
    assert by["Perennial"]["score"] <= 0.5 and by["Spiky"]["spike"] >= 20


def test_weak_hook_is_rejected_and_rewritten():
    class Rev:
        lite = False

        def json(self, system, user, **kw):
            return {"score": 9, "hook_strength": 6, "coherence": 8, "factual_errors": [], "misleading_title": False,
                    "advertiser_friendly": True, "policy_concerns": []}

    w = ScriptWriter.__new__(ScriptWriter)
    w.cfg, w.llm, w.src_chars = CFG, Rev(), 4000
    cfg_c = dict(CFG["content"], require_grounding=False)
    w.cfg = {**CFG, "content": cfg_c}
    src = {"title": "X", "text": "Bees make honey. " * 50}
    # gate-clean script: the ONLY thing wrong with it is the reviewer's hook_strength of 6
    # One aside: content.min_commentary_ratio rejects narration with no narrator voice.
    segs = [{"text": "Bees visit two million flowers to make one jar of honey"},
            {"text": "A single worker bee makes a twelfth of a teaspoon in her life"},
            {"text": "The hive beats its wings to dry the nectar into honey", "aside": "Busy little things."},
            {"text": "Sealed in wax, it never spoils"},
            {"text": "so the jar in your cupboard could outlive you"}]
    ok, review = w.check({"title": "Honey", "segments": segs}, src)
    assert not ok and any("hook too weak" in i for i in review["issues"])


def test_titles_are_screened_for_shock_bait_not_vocabulary():
    class Rev:
        lite = False

        def json(self, system, user, **kw):
            return {"score": 9, "hook_strength": 9, "coherence": 8}

    w = ScriptWriter.__new__(ScriptWriter)
    w.llm, w.src_chars = Rev(), 4000
    w.cfg = {**CFG, "content": dict(CFG["content"], require_grounding=False)}
    src = {"title": "Hunger stone", "text": "stones during the Thirty Years War " * 40}
    seg = [{"text": "Carved during the Thirty Years War, this stone still warns us"},
           {"text": "It only appears when the river drops to a record low"},
           {"text": "The oldest marking on it is older than anyone alive",
            "aside": "Cheerful bunch."},   # min_commentary_ratio: narration needs a voice
           {"text": "Each line records a season nobody wanted to repeat"},
           {"text": "so when you can read it, the warning has already arrived"}]
    # This test reuses one body of segments to isolate the title as the only variable.
    # check() now registers each APPROVED script so later videos in the same run are
    # compared against it, which would make the second call collide with the first on
    # similarity rather than on its title. Clearing the in-flight history between calls
    # keeps the title the thing under test. Production never re-checks an approved
    # script: produce() returns as soon as check() says ok.
    def check(title):
        w._history = []
        return w.check({"title": title, "segments": seg}, src)[0]

    assert check("The stone that warns you")
    # Deliberate change: "war" in a title is no longer banned — "The Shortest War In History" is a
    # perfectly good educational title. Titles are screened for shock-bait instead.
    assert check("The war stone")
    assert not check("The murder stone")
