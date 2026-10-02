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
    rows = [("Hunger stone", "wikipedia", 0.9, ["Wikipedia: 111,489 views/day, 1496.5x its normal"]),
            ("Willie Mays", "wikipedia", 0.8, []),
            ("Octopus can taste with its arms", "youtube_outliers", 0.8, ["viral Short now: 2,000,000 views in 20h"]),
            ("Dry policy news", "google_trends", 0.7, [])]
    return [{"topic": t, "topic_key": t.lower(), "source": s, "sources": [s], "score": sc, "context": ctx}
            for t, s, sc, ctx in rows]


def test_select_keeps_only_viral_topics_and_ranks_them():
    llm = LLMPicks([{"index": 1, "viral_score": 5, "category": "sports", "wiki_query": "Willie Mays"},
                    {"index": 0, "viral_score": 9, "category": "history", "wiki_query": "Hunger stone"},
                    {"index": 2, "viral_score": 9, "category": "nature", "wiki_query": "Octopus"},
                    {"index": 3, "viral_score": 8, "category": "culture", "risk": "high"}])
    w = ScriptWriter(CFG, llm, Strat())
    picks = w.select_topics(cands(), 2, set())
    assert [p["topic"] for p in picks] == ["Octopus can taste with its arms", "Hunger stone"]   # source weight breaks tie
    assert all(p["viral_score"] >= CFG["content"]["min_viral_score"] for p in picks)
    assert "1496.5x its normal" in llm.prompt and "2,000,000 views" in llm.prompt          # judged WITH evidence


def test_selection_outage_returns_nothing_not_guesses():
    from autotube.llm import LLMError

    class Down(LLMPicks):
        def json(self, *a, **k):
            raise LLMError("down")

    assert ScriptWriter(CFG, Down([]), Strat()).select_topics(cands(), 2, set()) == []


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
            return {"score": 9, "hook_strength": 6, "factual_errors": [], "misleading_title": False,
                    "advertiser_friendly": True, "policy_concerns": []}

    w = ScriptWriter.__new__(ScriptWriter)
    w.cfg, w.llm, w.src_chars = CFG, Rev(), 4000
    cfg_c = dict(CFG["content"], require_grounding=False)
    w.cfg = {**CFG, "content": cfg_c}
    src = {"title": "X", "text": "Bees make honey. " * 50}
    # gate-clean script: the ONLY thing wrong with it is the reviewer's hook_strength of 6
    segs = [{"text": "Bees visit two million flowers to make one jar of honey"},
            {"text": "A single worker bee makes a twelfth of a teaspoon in her life"},
            {"text": "The hive beats its wings to dry the nectar into honey"},
            {"text": "Sealed in wax, it never spoils"},
            {"text": "so the jar in your cupboard could outlive you"}]
    ok, review = w.check({"title": "Honey", "segments": segs}, src)
    assert not ok and any("hook too weak" in i for i in review["issues"])


def test_war_mention_ok_in_narration_but_not_title():
    class Rev:
        lite = False

        def json(self, system, user, **kw):
            return {"score": 9, "hook_strength": 9}

    w = ScriptWriter.__new__(ScriptWriter)
    w.llm, w.src_chars = Rev(), 4000
    w.cfg = {**CFG, "content": dict(CFG["content"], require_grounding=False)}
    src = {"title": "Hunger stone", "text": "stones during the Thirty Years War " * 40}
    seg = [{"text": "Carved during the Thirty Years War, this stone still warns us"},
           {"text": "It only appears when the river drops to a record low"},
           {"text": "The oldest marking on it is older than anyone alive"},
           {"text": "Each line records a season nobody wanted to repeat"},
           {"text": "so when you can read it, the warning has already arrived"}]
    assert w.check({"title": "The stone that warns you", "segments": seg}, src)[0]
    assert not w.check({"title": "The war stone", "segments": seg}, src)[0]
