"""Virality: topics must come from live trends, be scored for viral potential, and scripts must have strong hooks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, trends  # noqa: E402
from autotube.scriptwriter import ScriptWriter  # noqa: E402

CFG = common.load_config()


def signal(source, url, evidence=None, captured_at=None):
    return {"source": source, "source_family": trends.source_family(source), "source_url": url,
            "captured_at": captured_at or common.now_utc().isoformat(timespec="seconds"),
            "evidence": evidence or {}}


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
    out = []
    for t, s, sc, ctx, spike in rows:
        candidate = {"topic": t, "topic_key": t.lower(), "source": s, "sources": [s], "score": sc,
                     "context": ctx, "spike": spike, "signals": []}
        if t == "Hunger stone":
            candidate["sources"].append("google_trends")
            candidate["signals"] = [
                signal("wikipedia", "https://en.wikipedia.org/wiki/Hunger_stone",
                       {"spike": 1496.5, "views": 111_489}),
                signal("google_trends", "https://trends.test/hunger-stone", {"traffic": 25_000}),
            ]
        elif t == "Willie Mays":
            candidate["signals"] = [signal("wikipedia", "https://en.wikipedia.org/wiki/Willie_Mays",
                                           {"views": 42_000})]
        elif t.startswith("Octopus"):
            candidate["signals"] = [signal(
                "youtube_outliers", "https://youtube.test/short",
                {"views": 2_000_000, "hours": 20, "vph": 100_000, "breakout": 4.2, "video_id": "abc123"})]
        else:
            candidate["signals"] = [signal("google_trends", "https://trends.test/dry-policy", {"traffic": 500})]
        out.append(candidate)
    return out


def test_select_keeps_only_viral_topics_and_ranks_them():
    llm = LLMPicks([{"index": 0, "viral_score": 9, "category": "history", "wiki_query": "Hunger stone",
                     "angle": "the stone appeared only when drought returned", "formats": ["creepy_true"]},
                    {"index": 1, "viral_score": 9, "category": "nature", "wiki_query": "Octopus",
                     "angle": "its arms taste prey before it reaches the mouth", "formats": ["sounds_fake"],
                     "why_trending": "An unverified claim from the selector."},
                    {"index": 2, "viral_score": 8, "category": "culture", "risk": "high"}])
    w = ScriptWriter(CFG, llm, Strat())
    picks = w.select_topics(cands(), 2, set())
    assert [p["topic"] for p in picks] == ["Octopus can taste with its arms", "Hunger stone"]   # source weight breaks tie
    assert all(p["viral_score"] >= CFG["content"]["min_viral_score"] for p in picks)
    assert "1496.5x its 30-day baseline" in llm.prompt and "2,000,000 views" in llm.prompt  # judged WITH evidence
    assert "100,000 views/hour" in llm.prompt
    assert "An unverified claim" not in picks[0]["why_trending"]
    assert "2,000,000 views in 20h" in picks[0]["why_trending"]
    assert "Willie Mays" not in llm.prompt                                               # no Wikipedia-only fallback


def test_selector_fails_closed_without_a_specific_angle_and_out_of_range_viral_score():
    llm = LLMPicks([
        {"index": 0, "viral_score": 9, "category": "history", "wiki_query": "Hunger stone",
         "angle": "Hunger stone", "formats": ["creepy_true"]},
        {"index": 1, "viral_score": 9, "category": "nature", "wiki_query": "Octopus",
         "angle": "octopus can taste with arms", "formats": ["sounds_fake"]},
        {"index": 2, "viral_score": 11, "category": "culture", "wiki_query": "Dry policy news",
         "angle": "a surprising policy fact changed everything", "formats": ["backstory"]},
    ])
    picks = ScriptWriter(CFG, llm, Strat()).select_topics(cands(), 2, set())
    assert picks == [], "subject-only angles and out-of-range viral scores must not enter production"


def test_selector_rejects_invalid_category_or_no_fitting_format():
    llm = LLMPicks([
        {"index": 0, "viral_score": 9, "category": "off_topic", "wiki_query": "Hunger stone",
         "angle": "the stone appeared only when drought returned", "formats": ["creepy_true"]},
        {"index": 1, "viral_score": 9, "category": "nature", "wiki_query": "Octopus",
         "angle": "its arms taste prey before it reaches the mouth", "formats": ["not_a_format"]},
    ])
    assert ScriptWriter(CFG, llm, Strat()).select_topics(cands(), 2, set()) == []


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


def test_trend_evidence_uses_measured_signals_and_distinct_families():
    cfg = CFG["trends"]
    wiki_low = {"signals": [signal("wikipedia", "https://wiki.test/page", {"spike": 2.99})]}
    wiki_spike = {"signals": [signal("wikipedia", "https://wiki.test/page", {"spike": 3.0})]}
    assert not trends.has_current_trend_evidence(wiki_low, 3.0, cfg)
    assert not trends.has_current_trend_evidence(wiki_spike, 3.0, cfg), "a large Wikipedia spike is corroboration, not proof"
    assert not trends.has_current_trend_evidence({"sources": ["google_trends"]}, 3.0, cfg)
    assert not trends.has_current_trend_evidence({"source": "reddit_science"}, 3.0, cfg)
    assert not trends.has_current_trend_evidence({"signals": [signal("evergreen", "https://x.test")]}, 3.0, cfg)
    assert not trends.has_current_trend_evidence({"signals": [signal("on_this_day", "https://x.test")]}, 3.0, cfg)

    # The same platform is only one family: repeated Reddit feeds and Reddit + Wikipedia
    # without a qualifying spike cannot pass.
    repeated_reddit = {"signals": [signal("reddit_science", "https://reddit.test/a"),
                                   signal("reddit_space", "https://reddit.test/b")]}
    assert not trends.has_current_trend_evidence(repeated_reddit, 3.0, cfg)
    chart_only = {"signals": [signal("youtube_chart", "https://youtube.test/chart", {"view_count": 50_000})]}
    assert not trends.has_current_trend_evidence(chart_only, 3.0, cfg)
    chart_reddit = {"signals": chart_only["signals"] + [signal("reddit_science", "https://reddit.test/a")]}
    assert trends.has_current_trend_evidence(chart_reddit, 3.0, cfg)
    hn_low = {"signals": [signal("hackernews", "https://news.ycombinator.com/item?id=1", {"points": 9}),
                           signal("reddit_science", "https://reddit.test/a")]}
    hn_confirmed = {"signals": [signal("hackernews", "https://news.ycombinator.com/item?id=1", {"points": 10}),
                                 signal("reddit_science", "https://reddit.test/a")]}
    assert not trends.has_current_trend_evidence(hn_low, 3.0, cfg)
    assert trends.has_current_trend_evidence(hn_confirmed, 3.0, cfg)
    reddit_wiki = {"signals": [signal("reddit_science", "https://reddit.test/a"),
                               signal("wikipedia", "https://wiki.test/page", {"spike": 2.99})]}
    assert not trends.has_current_trend_evidence(reddit_wiki, 3.0, cfg)
    reddit_wiki["signals"][1]["evidence"]["spike"] = 3.0
    assert trends.has_current_trend_evidence(reddit_wiki, 3.0, cfg)

    assert not trends.has_current_trend_evidence({"signals": [
        signal("google_trends", "https://trends.test/page", {"traffic": 999})]}, 3.0, cfg)
    assert trends.has_current_trend_evidence({"signals": [
        signal("google_trends", "https://trends.test/page", {"traffic": 1_000})]}, 3.0, cfg)
    assert not trends.has_current_trend_evidence({"signals": [
        signal("google_trends", "", {"traffic": 100_000})]}, 3.0, cfg), "unlinked numbers are not auditable"
    # Regression for the actual Oct 2026 upload: plausible view/hour numbers with no source URL
    # must not be treated as verified trend evidence.
    orphaned_outlier = {"signals": [signal("youtube_outliers", "", {
        "views": 67_504, "hours": 44.6, "vph": 1_512, "breakout": 67.5,
    })]}
    assert not trends.has_current_trend_evidence(orphaned_outlier, 3.0, cfg)

    weak_google_reddit = {"signals": [signal("google_trends", "https://trends.test/page", {"traffic": 200}),
                                        signal("reddit_science", "https://reddit.test/a")]}
    assert trends.has_current_trend_evidence(weak_google_reddit, 3.0, cfg)

    assert not trends.has_current_trend_evidence({"signals": [signal(
        "youtube_outliers", "https://youtube.test/short",
        {"views": 29_999, "hours": 20, "vph": 1_500, "video_id": "abc"})]}, 3.0, cfg)
    assert trends.has_current_trend_evidence({"signals": [signal(
        "youtube_outliers", "https://youtube.test/short",
        {"views": 30_000, "hours": 20, "vph": 1_500, "video_id": "abc"})]}, 3.0, cfg)
    assert not trends.has_current_trend_evidence({"signals": [signal(
        "youtube_outliers", "https://youtube.test/short",
        {"views": 3_000_000, "hours": 73, "vph": 40_000, "video_id": "abc"})]}, 3.0, cfg)
    no_capture = signal("google_trends", "https://trends.test/page", {"traffic": 5_000})
    no_capture["captured_at"] = "2026-01-01T00:00:00+00:00"
    assert not trends.has_current_trend_evidence({"signals": [no_capture]}, 3.0, cfg)
    missing_capture = signal("google_trends", "https://trends.test/page", {"traffic": 5_000})
    missing_capture.pop("captured_at")
    assert not trends.has_current_trend_evidence({"signals": [missing_capture]}, 3.0, cfg)
    verified_but_mislabeled = {"signals": [signal(
        "google_trends", "https://trends.test/page", {"traffic": 5_000})], "sources": ["evergreen"]}
    assert trends.has_current_trend_evidence(verified_but_mislabeled, 3.0, cfg)


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
    topic = {"topic": "Current Google trend", "sources": ["google_trends"], "score": 0.9, "viral_score": 6,
             "signals": [signal("google_trends", "https://trends.test/current", {"traffic": 20_000})]}
    assert pipeline.make_one(CFG, object(), topic, {}, 1, "test-run") is None


def test_google_trends_traffic_parser_preserves_k_and_magnitude():
    assert trends._parse_traffic("25K+") == 25_000
    assert trends._parse_traffic("1.2M+") == 1_200_000
    assert trends._parse_traffic("50,000+") == 50_000
    assert trends._parse_traffic(1234) == 1234
    assert trends._parse_traffic("unknown") == 0


def test_primary_source():
    assert trends.primary_source(["reddit_todayilearned", "wikipedia"]) == "reddit"
    assert trends.primary_source(["google_trends", "youtube_outliers"]) == "youtube_outliers"
    assert trends.primary_source(["reddit_science"]) == "reddit"
    assert trends.primary_source([]) == "evergreen"


def test_trend_merge_only_rewards_independent_sources_and_keeps_measured_evidence(monkeypatch):
    cfg = common.load_config()
    cfg["trends"]["wikipedia_spike_check"] = 0
    cfg["trends"]["on_this_day"] = False
    topic = "Octopus tastes with its arms"
    reddit_rows = {
        "science": [{"topic": topic, "source": "reddit_science", "score": 0.91,
                     "source_url": "https://reddit.test/science", "context": ["top post today"]}],
        "space": [{"topic": topic, "source": "reddit_space", "score": 0.88,
                   "source_url": "https://reddit.test/space", "context": ["top post today"]}],
    }
    monkeypatch.setattr(trends, "youtube_outliers", lambda cfg: [])
    monkeypatch.setattr(trends, "google_trends", lambda region: [])
    monkeypatch.setattr(trends, "wikipedia_top", lambda lang: [])
    monkeypatch.setattr(trends, "reddit_rss", lambda sub, limit=25: reddit_rows.get(sub, []))
    monkeypatch.setattr(trends, "hackernews", lambda: [])
    monkeypatch.setattr(trends, "youtube_chart", lambda region: [])
    monkeypatch.setattr(trends, "on_this_day", lambda lang: [])
    monkeypatch.setattr("autotube.safety.screen", lambda *a, **k: ("allow", ""))

    merged = trends.collect(cfg)
    assert merged == [], "multiple subreddit feeds are one family and cannot qualify alone"

    monkeypatch.setattr(trends, "google_trends", lambda region: [
        {"topic": topic, "source": "google_trends", "score": 0.80, "traffic": 100000,
         "source_url": "https://trends.test/octopus", "context": ["100,000 searches"]}])
    merged = trends.collect(cfg)
    candidate = next(c for c in merged if c["topic"] == topic)
    assert candidate["score"] == 1.0, "independent Google Trends confirmation should strengthen the signal"
    assert set(candidate["source_families"]) == {"reddit", "google_trends"}
    assert any(s["evidence"].get("traffic") == 100000 for s in candidate["signals"])


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
                    "advertiser_friendly": True, "policy_concerns": [], "fixes": []}

    w = ScriptWriter.__new__(ScriptWriter)
    w.cfg, w.llm, w.src_chars = CFG, Rev(), 4000
    cfg_c = dict(CFG["content"], require_grounding=False)
    w.cfg = {**CFG, "content": cfg_c}
    src = {"title": "X", "text": "Bees make honey. " * 50}
    # gate-clean script: the ONLY thing wrong with it is the reviewer's hook_strength of 6
    # One aside: content.min_commentary_ratio rejects narration with no narrator voice.
    segs = [{"text": "Bees visit two million flowers to make one jar of honey"},
            {"text": "A worker bee makes one twelfth of a teaspoon in life"},
            {"text": "The hive beats its wings to dry the nectar into honey", "aside": "Busy little things."},
            {"text": "A wax seal keeps the stored honey from absorbing moisture"},
            {"text": "so the jar in your cupboard could outlive you"}]
    ok, review = w.check({"title": "Honey", "segments": segs}, src)
    assert not ok and any("hook too weak" in i for i in review["issues"])


def test_titles_are_screened_for_shock_bait_not_vocabulary():
    class Rev:
        lite = False

        def json(self, system, user, **kw):
            return {"score": 9, "hook_strength": 9, "coherence": 8, "fixes": []}

    w = ScriptWriter.__new__(ScriptWriter)
    w.llm, w.src_chars = Rev(), 4000
    w.cfg = {**CFG, "content": dict(CFG["content"], require_grounding=False)}
    src = {"title": "Hunger stone", "text": "stones during the Thirty Years War " * 40}
    seg = [{"text": "Carved during the Thirty Years War, this stone still warns us"},
           {"text": "It only appears when the river drops to a record low"},
           {"text": "The oldest marking on it is older than anyone alive",
            "aside": "Cheerful bunch."},   # min_commentary_ratio: narration needs a voice
           {"text": "Each line records a season nobody wanted to repeat"},
           {"text": "so by reading it, the drought has already returned"}]
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
