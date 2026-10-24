"""Virality: topics must come from live trends, be scored for viral potential, and scripts must have strong hooks."""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, trends  # noqa: E402
from autotube.scriptwriter import ScriptWriter, names_subject_in_opening, retention_guidance  # noqa: E402

CFG = common.load_config()


def signal(source, url, evidence=None, captured_at=None):
    evidence = dict(evidence) if isinstance(evidence, dict) else {}
    hours = evidence.get("hours")
    if (source == "youtube_outliers" and "published_at" not in evidence
            and type(hours) in (int, float) and hours > 0):
        evidence["published_at"] = (common.now_utc() - timedelta(hours=hours)).isoformat(timespec="seconds")
    return {"source": source, "source_family": trends.source_family(source), "source_url": url,
            "captured_at": captured_at or common.now_utc().isoformat(timespec="seconds"),
            "evidence": evidence}


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
                "youtube_outliers", "https://www.youtube.com/shorts/abcdefghijk",
                {"views": 2_000_000, "hours": 20, "vph": 100_000, "breakout": 4.2, "video_id": "abcdefghijk"})]
        else:
            candidate["signals"] = [signal("google_trends", "https://trends.test/dry-policy", {"traffic": 500})]
        out.append(candidate)
    return out


def test_select_keeps_only_viral_topics_and_ranks_them():
    llm = LLMPicks([{"index": 0, "viral_score": 9, "category": "history", "risk": "none",
                     "script_subject": "Hunger stone", "wiki_query": "Hunger stone",
                     "angle": "the stone appeared only when drought returned",
                     "formats": ["creepy_true"]},
                    {"index": 1, "viral_score": 9, "category": "nature", "risk": "low",
                     "script_subject": "octopus", "wiki_query": "Octopus",
                     "angle": "its arms taste prey before it reaches the mouth", "formats": ["sounds_fake"],
                     "why_trending": "An unverified claim from the selector."},
                    {"index": 2, "viral_score": 8, "category": "culture", "risk": "high"}])
    w = ScriptWriter(CFG, llm, Strat())
    candidates = cands()
    candidates[2]["lane"] = "science_nature"
    picks = w.select_topics(candidates, 2, set())
    assert [p["topic"] for p in picks] == ["Octopus can taste with its arms", "Hunger stone"]   # source weight breaks tie
    assert all(p["viral_score"] >= CFG["content"]["min_viral_score"] for p in picks)
    assert "1496.5x its 30-day baseline" in llm.prompt and "2,000,000 views" in llm.prompt  # judged WITH evidence
    assert "100,000 views/hour" in llm.prompt
    assert "trend context (discovery only): viral Short now: 2,000,000 views in 20h" in llm.prompt
    assert "source lane=science_nature" in llm.prompt
    assert '"script_subject"' in llm.prompt and "never copy their headline" in llm.prompt
    assert "non-generic literal" in llm.prompt and "not a translation" in llm.prompt
    assert "Do not stop at four" in llm.prompt and "inflate scores" in llm.prompt and "pad below the floor" in llm.prompt
    assert "deep-sea exploration" in llm.prompt and "photographable" in llm.prompt
    assert "not a shocking" in llm.prompt and "name the creature or object, not its behavior" in llm.prompt
    assert "Avoid causation, comparisons, exact timing/location" in llm.prompt
    assert "explicit `risk`=`none|low|high`" in llm.prompt
    assert "never turn M38A1 into M38A1C/D" in llm.prompt
    assert len(llm.prompt) < 5_000, "selector context must stay compact for free-tier models"
    assert picks[0]["script_subject"] == "octopus"
    assert "An unverified claim" not in picks[0]["why_trending"]
    assert "2,000,000 views in 20h" in picks[0]["why_trending"]
    assert "Willie Mays" not in llm.prompt                                               # no Wikipedia-only fallback


def test_selector_normalizes_topic_keys_before_applying_retry_exclusions():
    candidate = cands()[2]
    candidate.update({
        "topic": "How Astronauts Live, Sleep, Shower & Cry in Space!",
        # Punctuation removal in raw YouTube titles can leave doubled spaces.
        "topic_key": "how astronauts live sleep shower  cry in space",
    })
    llm = LLMPicks([])

    picks = ScriptWriter(CFG, llm, Strat()).select_topics(
        [candidate], 1, set(), exclude={"how astronauts live sleep shower cry in space"})

    assert picks == []
    assert llm.prompt == "", "a recently rejected key must be removed before model selection"


def test_selector_rejects_generic_compilations_and_mismatched_source_subjects():
    candidate = cands()[2]
    candidate.update({
        "topic": "Rare Human History Photos (Part 49)",
        "topic_key": common.normalize_topic_key("Rare Human History Photos (Part 49)"),
    })
    generic_pick = {
        "index": 0, "viral_score": 9, "category": "history", "risk": "low",
        "script_subject": "Rare Human History Photos", "wiki_query": "Area 51",
        "angle": "A disputed historical photograph supposedly shows a hidden desert town.",
        "formats": ["creepy_true"],
    }
    assert ScriptWriter(CFG, LLMPicks([generic_pick]), Strat()).select_topics([candidate], 1, set()) == []

    candidate.update({"topic": "History of Pickles", "topic_key": "history of pickles"})
    unsupported_page = {
        "index": 0, "viral_score": 9, "category": "history", "risk": "low",
        "script_subject": "Pickles", "wiki_query": "Military of ancient Rome",
        "angle": "Ancient Roman soldiers carried pickled cucumbers in their rations.",
        "formats": ["backstory"],
    }
    assert ScriptWriter(CFG, LLMPicks([unsupported_page]), Strat()).select_topics(
        [candidate], 1, set()) == []


def test_selector_rejects_compound_subject_from_only_a_generic_trend_adjective():
    candidate = cands()[2]
    candidate.update({
        "topic": "We Thought We Knew the Ocean, Then This Giant Appeared!",
        "topic_key": common.normalize_topic_key("We Thought We Knew the Ocean, Then This Giant Appeared!"),
        "lane": "science_nature",
    })
    unsupported = {
        "index": 0, "viral_score": 9, "category": "nature", "risk": "low",
        "script_subject": "Amazon Cachalot", "wiki_query": "Cachalot",
        "angle": "Amazon cachalots use clicks to find prey in dark water.",
        "formats": ["sounds_fake"],
    }
    assert ScriptWriter(CFG, LLMPicks([unsupported]), Strat()).select_topics(
        [candidate], 1, set()) == []

    candidate.update({
        "topic": "TIPU SULTAN'S ROCKETS THAT DEFEATED THE BRITISHERS!!",
        "topic_key": common.normalize_topic_key("TIPU SULTAN'S ROCKETS THAT DEFEATED THE BRITISHERS!!"),
        "lane": "history_mystery",
    })
    canonical_subject = {
        "index": 0, "viral_score": 9, "category": "history", "risk": "low",
        "script_subject": "Mysorean rockets", "wiki_query": "Mysorean rockets",
        "angle": "Iron-cased Mysorean rockets forced the British to develop their own rocket artillery.",
        "formats": ["backstory"],
    }
    picks = ScriptWriter(CFG, LLMPicks([canonical_subject]), Strat()).select_topics(
        [candidate], 1, set())
    assert len(picks) == 1
    assert picks[0]["script_subject"] == "Mysorean rockets"


def test_dog_lapping_outlier_uses_narrow_primary_research_without_relaxing_live_gates():
    topic = "Dog Paani Kaise Peeta Hai? Tongue Ka Asli Secret!"
    video_id = "F9IluEaMjJ4"
    candidate = {
        "topic": topic, "topic_key": topic.lower(), "source": "youtube_outliers",
        "sources": ["youtube_outliers"], "score": 0.95,
        "context": ["viral Short: 388,032 views in 35.4h"], "lane": "science_nature",
        "signals": [signal("youtube_outliers", f"https://www.youtube.com/shorts/{video_id}", {
            "views": 388_032, "hours": 35.4, "vph": 10_961, "breakout": 84.7, "video_id": video_id,
        })],
    }
    llm = LLMPicks([{
        "index": 0, "viral_score": 8, "category": "animals", "risk": "low",
        "script_subject": "Dog", "wiki_query": "Hair of the dog",
        "angle": "Dogs pull a column of water into their mouths by curling their tongues.",
        "formats": ["sounds_fake"],
    }])

    picks = ScriptWriter(CFG, llm, Strat()).select_topics([candidate], 1, set())

    assert len(picks) == 1
    assert picks[0]["script_subject"] == "Dog"
    assert picks[0]["wiki_query"] == "Dog"                         # not the ambiguous idiom article
    assert picks[0]["angle"] == "Dogs curl their tongues to pull a larger liquid column into their mouths."
    assert picks[0]["research_query"] == "Dogs lap using acceleration-driven open pumping"
    assert picks[0]["viral_score"] >= CFG["content"]["min_viral_score"]
    assert picks[0]["lane"] == "science_nature"


def test_selector_prompt_caps_candidates_to_a_bounded_live_shortlist():
    import copy

    template = cands()[2]
    rows = []
    for i in range(40):
        row = copy.deepcopy(template)
        row["topic"] = f"Octopus can taste prey with its arms {i}"
        row["topic_key"] = f"octopus taste arms {i}"
        rows.append(row)
    llm = LLMPicks([])
    assert ScriptWriter(CFG, llm, Strat()).select_topics(rows, 1, set()) == []
    assert sum(f"Octopus can taste prey with its arms {i}" in llm.prompt for i in range(40)) == 24


def test_selector_requires_a_concrete_subject_connected_to_the_live_trend():
    candidate = cands()[2]
    base = {"index": 0, "viral_score": 9, "category": "nature", "risk": "low",
            "script_subject": "octopus", "wiki_query": "Octopus",
            "angle": "sensory cells on its arms identify chemical traces",
            "formats": ["sounds_fake"]}

    picks = ScriptWriter(CFG, LLMPicks([base]), Strat()).select_topics([candidate], 1, set())
    assert len(picks) == 1 and picks[0]["script_subject"] == "octopus"

    missing = dict(base)
    missing.pop("script_subject")
    assert ScriptWriter(CFG, LLMPicks([missing]), Strat()).select_topics([candidate], 1, set()) == []

    missing_risk = dict(base)
    missing_risk.pop("risk")
    assert ScriptWriter(CFG, LLMPicks([missing_risk]), Strat()).select_topics([candidate], 1, set()) == []

    unrelated = dict(base, script_subject="Stereopsis")
    assert ScriptWriter(CFG, LLMPicks([unrelated]), Strat()).select_topics([candidate], 1, set()) == []

    # A malformed backup must not invalidate a valid sibling pick; each candidate is
    # still individually gated, so only the concrete, trend-connected subject survives.
    resilient = ScriptWriter(CFG, LLMPicks([unrelated, base]), Strat()).select_topics([candidate], 1, set())
    assert len(resilient) == 1 and resilient[0]["script_subject"] == "octopus"
    malformed_backup = ScriptWriter(CFG, LLMPicks(["not a pick object", base]), Strat()) \
        .select_topics([candidate], 1, set())
    assert len(malformed_backup) == 1 and malformed_backup[0]["script_subject"] == "octopus"

    too_long = dict(base, script_subject="octopus senses food with chemical receptor cells")
    assert ScriptWriter(CFG, LLMPicks([too_long]), Strat()).select_topics([candidate], 1, set()) == []


def test_selector_rejects_unverified_model_variant_designation():
    candidate = cands()[2]
    candidate.update({
        "topic": "This Jeep Could Launch A Nuke",
        "topic_key": "this jeep could launch a nuke",
        "score": 0.95,
        "context": ["viral Short now: 190,100 views in 35.3h"],
        "lane": "history_mystery",
        "sources": ["youtube_outliers"],
        "signals": [signal("youtube_outliers", "https://www.youtube.com/shorts/jeep1234567",
                            {"views": 190_100, "hours": 35.3, "vph": 5_380, "video_id": "jeep1234567"})],
    })
    base = {"index": 0, "viral_score": 9, "category": "history", "risk": "low",
            "wiki_query": "Willys M38A1", "angle": "the military jeep carried a nuclear launcher",
            "formats": ["creepy_true"]}
    invented = dict(base, script_subject="M38A1C Nuclear Jeep")
    assert ScriptWriter(CFG, LLMPicks([invented]), Strat()).select_topics([candidate], 1, set()) == []

    verified = dict(base, script_subject="M38A1 Jeep")
    picks = ScriptWriter(CFG, LLMPicks([verified]), Strat()).select_topics([candidate], 1, set())
    assert len(picks) == 1 and picks[0]["script_subject"] == "M38A1 Jeep"


def test_selector_rejects_known_off_lane_youtube_chart_topics_before_provider_call():
    sports = {
        "topic": "NFL overtime rules", "topic_key": "nfl overtime rules",
        "source": "google_trends", "sources": ["google_trends", "youtube_chart"],
        "score": 0.95, "signals": [
            signal("google_trends", "https://trends.test/nfl", {"traffic": 50_000}),
            signal("youtube_chart", "https://www.youtube.com/watch?v=abcdefghijk",
                   {"view_count": 80_000, "video_id": "abcdefghijk", "yt_category": 17}),
        ],
    }
    llm = LLMPicks([])
    assert ScriptWriter(CFG, llm, Strat()).select_topics([sports], 1, set()) == []
    assert llm.prompt == "", "explicit sports chart metadata should be filtered before selection"


def test_selector_rejects_sports_trend_context_even_if_the_model_could_relabel_it():
    candidates = [
        {
            "topic": "Dortmund", "topic_key": "dortmund", "source": "google_trends",
            "sources": ["google_trends"], "score": 0.95,
            "context": ["Borussia Dortmund advance in the UEFA Champions League"],
            "signals": [signal("google_trends", "https://trends.test/dortmund", {"traffic": 25_000})],
        },
        {
            "topic": "canadiens vs capitals", "topic_key": "canadiens vs capitals",
            "source": "google_trends", "sources": ["google_trends"], "score": 0.95,
            "context": ["FAQ: How to watch Wednesday's Canadiens-Capitals game"],
            "signals": [signal("google_trends", "https://trends.test/canadiens", {"traffic": 25_000})],
        },
        {
            "topic": "new york city vs inter miami", "topic_key": "new york city vs inter miami",
            "source": "google_trends", "sources": ["google_trends"], "score": 0.95,
            "context": ["LIVE MATCH: Inter Miami vs. NYCFC: Match Preview & Start Time for 2026"],
            "signals": [signal("google_trends", "https://trends.test/inter-miami", {"traffic": 25_000})],
        },
        {
            "topic": "Funny Moments in Cricket History!", "topic_key": "funny moments in cricket history",
            "source": "google_trends", "sources": ["google_trends"], "score": 0.95,
            "context": ["A viral cricket Short from this week's sports roundup"],
            "signals": [signal("google_trends", "https://trends.test/cricket", {"traffic": 25_000})],
        },
    ]
    for candidate in candidates:
        llm = LLMPicks([])
        assert ScriptWriter(CFG, llm, Strat()).select_topics([candidate], 1, set()) == []
        assert llm.prompt == "", "out-of-scope sports context must be filtered before model selection"


def test_selector_rejects_category_that_conflicts_with_explicit_trend_lane():
    candidate = cands()[2]
    candidate["lane"] = "science_nature"
    llm = LLMPicks([{"index": 0, "viral_score": 9, "category": "history", "risk": "none",
                     "script_subject": "octopus", "wiki_query": "Octopus",
                     "angle": "its arms taste prey before it reaches the mouth",
                     "formats": ["sounds_fake"]}])
    writer = ScriptWriter(CFG, llm, Strat())
    assert writer.select_topics([candidate], 1, set()) == []
    assert "HARD EDITORIAL SCOPE" in llm.prompt
    first_prompt = llm.prompt
    assert writer.select_topics([candidate], 1, set()) == []
    assert llm.prompt == first_prompt, "a malformed pick must not burn another selection round"


def test_selector_rejects_configuration_that_weakens_viral_floor():
    import copy

    cfg = copy.deepcopy(CFG)
    cfg["content"]["min_viral_score"] = 6
    llm = LLMPicks([])
    assert ScriptWriter(cfg, llm, Strat()).select_topics(cands(), 1, set()) == []
    assert llm.prompt == "", "invalid low floors must fail before invoking the provider"


def test_selector_rejects_malformed_live_trend_thresholds_before_provider_call():
    import copy

    for key, value in (("min_google_trend_traffic", True),
                       ("min_independent_trend_families", 1),
                       ("min_hackernews_points", "10"),
                       ("outlier_window_hours", float("nan"))):
        cfg = copy.deepcopy(CFG)
        cfg["trends"][key] = value
        llm = LLMPicks([])
        assert ScriptWriter(cfg, llm, Strat()).select_topics(cands(), 1, set()) == [], key
        assert llm.prompt == "", key


def test_selector_deduplicates_repeated_provider_picks():
    pick = {"index": 0, "viral_score": 9, "category": "history", "risk": "none",
            "script_subject": "Hunger stone", "wiki_query": "Hunger stone",
            "angle": "the stone appeared only when drought returned",
            "formats": ["creepy_true"]}
    picks = ScriptWriter(CFG, LLMPicks([pick, dict(pick)]), Strat()).select_topics(cands(), 2, set())
    assert len(picks) == 1
    assert picks[0]["topic"] == "Hunger stone"


def test_selector_fails_closed_without_a_specific_angle_and_out_of_range_viral_score():
    llm = LLMPicks([
        {"index": 0, "viral_score": 9, "category": "history", "risk": "none",
         "script_subject": "Hunger stone", "wiki_query": "Hunger stone",
         "angle": "Hunger stone", "formats": ["creepy_true"]},
        {"index": 1, "viral_score": 9, "category": "nature", "risk": "low",
         "script_subject": "octopus", "wiki_query": "Octopus",
         "angle": "octopus can taste with arms", "formats": ["sounds_fake"]},
        {"index": 2, "viral_score": 11, "category": "culture", "risk": "none",
         "script_subject": "dry policy news", "wiki_query": "Dry policy news",
         "angle": "a surprising policy fact changed everything", "formats": ["backstory"]},
    ])
    picks = ScriptWriter(CFG, llm, Strat()).select_topics(cands(), 2, set())
    assert picks == [], "subject-only angles and out-of-range viral scores must not enter production"


def test_selector_rejects_invalid_category_or_no_fitting_format():
    llm = LLMPicks([
        {"index": 0, "viral_score": 9, "category": "off_topic", "risk": "none",
         "script_subject": "Hunger stone", "wiki_query": "Hunger stone",
         "angle": "the stone appeared only when drought returned", "formats": ["creepy_true"]},
        {"index": 1, "viral_score": 9, "category": "nature", "risk": "none",
         "script_subject": "octopus", "wiki_query": "Octopus",
         "angle": "its arms taste prey before it reaches the mouth", "formats": ["not_a_format"]},
    ])
    assert ScriptWriter(CFG, llm, Strat()).select_topics(cands(), 2, set()) == []


def test_selector_fails_closed_on_malformed_scores_indices_risks_and_angles():
    bad_picks = [
        {"index": 0, "viral_score": True, "category": "history", "risk": "none",
         "script_subject": "Hunger stone", "wiki_query": "Hunger stone",
         "angle": "the stone appeared when drought returned", "formats": ["creepy_true"]},
        {"index": True, "viral_score": 9, "category": "history", "risk": "none",
         "script_subject": "Hunger stone", "wiki_query": "Hunger stone",
         "angle": "the stone appeared when drought returned", "formats": ["creepy_true"]},
        {"index": 0, "viral_score": 9, "category": "history",
         "script_subject": "Hunger stone", "wiki_query": "Hunger stone",
         "angle": "the stone appeared when drought returned", "formats": ["creepy_true"]},
        {"index": 0, "viral_score": 9, "category": "history", "risk": "none",
         "script_subject": "Hunger stone", "wiki_query": "Hunger stone",
         "angle": ["the stone appeared", "when drought returned"], "formats": ["creepy_true"]},
    ]
    assert ScriptWriter(CFG, LLMPicks(bad_picks), Strat()).select_topics(cands(), 2, set()) == []


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


def test_fresh_retention_data_guides_opening_without_becoming_a_platform_claim():
    now = common.now_utc()
    summary = {
        "updated": now.isoformat(),
        "average_view_target": {
            "n": 33, "median_avg_view_pct": 59.4, "gate_pct": 70.0, "provisional_n": 5,
            "learning_qualified": {"n": 28, "passing": 8, "minimum_sample_views": 50,
                                    "median_avg_view_pct": 58.3},
        },
        "retention": {"videos": 27, "where": "hook",
                      "worst_drop": {"from_pct": 10, "to_pct": 20}},
    }
    guidance = retention_guidance(CFG, summary, now)
    assert "10-20% of runtime" in guidance
    assert "6-9 words" in guidance and "subject by beat two" in guidance
    assert "not a YouTube ranking rule" in guidance and "not viewer-survival percentages" in guidance
    assert "n=28" in guidance and "sample-qualified at ≥50 engaged views" in guidance
    assert "5 low- or unknown-sample values excluded" in guidance

    legacy = {**summary, "average_view_target": {"n": 28, "median_avg_view_pct": 58.3, "gate_pct": 70}}
    legacy_guidance = retention_guidance(CFG, legacy, now)
    assert "n=28; sample-count qualification not available in this older summary" in legacy_guidance

    stale = dict(summary, updated=(now - timedelta(days=15)).isoformat())
    assert not retention_guidance(CFG, stale, now), "stale retention should not guide current scripts"
    low_sample = {**summary, "average_view_target": {
        "n": 33, "median_avg_view_pct": 59.4, "gate_pct": 70,
        "learning_qualified": {"n": 4, "minimum_sample_views": 50, "median_avg_view_pct": 50},
    }}
    assert not retention_guidance(CFG, low_sample, now), "small sample-qualified cohorts should not guide scripts"


def test_subject_is_named_by_the_hook_or_early_turn():
    early = {"segments": [
        {"text": "Some fish spend three quarters of life on land."},
        {"text": "Mudskippers crawl on their fins across the mud."},
        {"text": "They can stay out of water for hours."},
    ]}
    delayed = {"segments": [
        {"text": "Some fish spend three quarters of life on land."},
        {"text": "They crawl and leap across the muddy shore."},
        {"text": "The mudskipper can breathe through its skin."},
    ]}
    assert names_subject_in_opening(early, "Mudskipper")
    assert not names_subject_in_opening(delayed, "Mudskipper")
    assert names_subject_in_opening({"segments": [{"text": "Astronomy in 2026 is remarkable."}]},
                                    "2026 in astronomy")


def test_signal_record_preserves_source_lane_and_youtube_category():
    row = {"topic": "NFL rules", "source": "youtube_chart", "lane": "science_nature",
           "yt_category": "17"}
    saved = trends._signal_record(row, common.now_utc().isoformat())
    assert saved["evidence"]["yt_category"] == "17"
    assert saved["evidence"]["lane"] == "science_nature"


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
    chart_only = {"signals": [signal("youtube_chart", "https://www.youtube.com/watch?v=abcdefghijk",
                                      {"view_count": 50_000, "video_id": "abcdefghijk"})]}
    assert not trends.has_current_trend_evidence(chart_only, 3.0, cfg)
    chart_reddit = {"signals": chart_only["signals"] + [signal("reddit_science", "https://reddit.test/a")]}
    assert trends.has_current_trend_evidence(chart_reddit, 3.0, cfg)
    mismatched_chart = {"signals": [dict(chart_only["signals"][0],
                                         source_url="https://www.youtube.com/watch?v=zyxwvutsrqp"),
                                    signal("reddit_science", "https://reddit.test/a")]}
    assert not trends.has_current_trend_evidence(mismatched_chart, 3.0, cfg)
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
        "youtube_outliers", "https://www.youtube.com/shorts/abcdefghijk",
        {"views": 29_999, "hours": 20, "vph": 1_500, "video_id": "abcdefghijk"})]}, 3.0, cfg)
    assert trends.has_current_trend_evidence({"signals": [signal(
        "youtube_outliers", "https://www.youtube.com/shorts/abcdefghijk",
        {"views": 30_000, "hours": 20, "vph": 1_500, "video_id": "abcdefghijk"})]}, 3.0, cfg)
    assert not trends.has_current_trend_evidence({"signals": [signal(
        "youtube_outliers", "https://www.youtube.com/shorts/abcdefghijk",
        {"views": 3_000_000, "hours": 73, "vph": 40_000, "video_id": "abcdefghijk"})]}, 3.0, cfg)

    def outlier(url, video_id="abcdefghijk", **extra):
        evidence = {"views": 50_000, "hours": 20, "vph": 2_500, "video_id": video_id, **extra}
        return {"signals": [signal("youtube_outliers", url, evidence)]}

    assert trends.has_current_trend_evidence(
        outlier("https://www.youtube.com/watch?v=abcdefghijk"), 3.0, cfg)
    assert trends.has_current_trend_evidence(
        outlier("https://youtu.be/abcdefghijk"), 3.0, cfg)
    assert not trends.has_current_trend_evidence(
        outlier("https://unrelated.example/shorts/abcdefghijk"), 3.0, cfg)
    assert not trends.has_current_trend_evidence(
        outlier("https://www.youtube.com/shorts/zyxwvutsrqp"), 3.0, cfg)
    assert not trends.has_current_trend_evidence(
        outlier("http://www.youtube.com/shorts/abcdefghijk"), 3.0, cfg)
    assert not trends.has_current_trend_evidence(
        outlier("https://www.youtube.com/watch?v=abcdefghijk&v=zyxwvutsrqp"), 3.0, cfg)
    assert not trends.has_current_trend_evidence(
        outlier("https://www.youtube.com/shorts/abcdefghijk", video_id="short-id"), 3.0, cfg)
    conflicting_id = outlier("https://www.youtube.com/shorts/abcdefghijk")
    conflicting_id["signals"][0]["video_id"] = "zyxwvutsrqp"
    assert not trends.has_current_trend_evidence(conflicting_id, 3.0, cfg)
    malformed_url = outlier("https://")
    assert not trends.has_current_trend_evidence(malformed_url, 3.0, cfg)
    assert not trends.has_current_trend_evidence(
        outlier("https://www.youtube.com/shorts/abcdefghijk", vph=1), 3.0, cfg)
    assert not trends.has_current_trend_evidence(
        outlier("https://www.youtube.com/shorts/abcdefghijk", vph=True), 3.0, cfg)
    assert not trends.has_current_trend_evidence(
        outlier("https://www.youtube.com/shorts/abcdefghijk", published_at=(
            common.now_utc() + timedelta(hours=1)).isoformat()), 3.0, cfg)
    stale = outlier("https://www.youtube.com/shorts/abcdefghijk",
                    published_at=(common.now_utc() - timedelta(days=10)).isoformat())
    assert not trends.has_current_trend_evidence(stale, 3.0, cfg)
    missing_publication = outlier("https://www.youtube.com/shorts/abcdefghijk")
    missing_publication["signals"][0]["evidence"].pop("published_at")
    assert not trends.has_current_trend_evidence(missing_publication, 3.0, cfg)

    no_capture = signal("google_trends", "https://trends.test/page", {"traffic": 5_000})
    no_capture["captured_at"] = "2026-01-01T00:00:00+00:00"
    assert not trends.has_current_trend_evidence({"signals": [no_capture]}, 3.0, cfg)
    missing_capture = signal("google_trends", "https://trends.test/page", {"traffic": 5_000})
    missing_capture.pop("captured_at")
    assert not trends.has_current_trend_evidence({"signals": [missing_capture]}, 3.0, cfg)
    verified_but_mislabeled = {"signals": [signal(
        "google_trends", "https://trends.test/page", {"traffic": 5_000})], "sources": ["evergreen"]}
    assert trends.has_current_trend_evidence(verified_but_mislabeled, 3.0, cfg)


def test_youtube_outlier_age_is_compared_to_capture_time_not_later_attempt_time(monkeypatch):
    captured_at = common.now_utc()
    published_at = (captured_at - timedelta(hours=48)).isoformat()
    later_attempt = captured_at + timedelta(minutes=10)
    evidence = {"views": 250_000, "hours": 48, "vph": round(250_000 / 48),
                "video_id": "abcdefghijk", "published_at": published_at}
    candidate = {"signals": [signal(
        "youtube_outliers", "https://www.youtube.com/watch?v=abcdefghijk", evidence,
        captured_at=captured_at.isoformat())]}
    monkeypatch.setattr(trends, "now_utc", lambda: later_attempt)

    assert trends.has_current_trend_evidence(candidate, 3.0, CFG["trends"]), (
        "a valid fallback candidate must not age out merely because an earlier script attempt took ten minutes")


def test_trend_gate_rejects_boolean_evidence_and_malformed_thresholds():
    cfg = dict(CFG["trends"])
    current = {"signals": [signal("google_trends", "https://trends.test/page", {"traffic": 100_000})]}
    assert not trends.has_current_trend_evidence(current, 3.0, {**cfg, "min_google_trend_traffic": True})
    assert not trends.has_current_trend_evidence(current, True, cfg)
    assert not trends.has_current_trend_evidence(current, 3.0, {**cfg, "min_independent_trend_families": 1.5})
    assert not trends.has_current_trend_evidence(current, 3.0, {**cfg, "min_google_trend_traffic": "1000"})
    boolean_evidence = {"signals": [signal("google_trends", "https://trends.test/page", {"traffic": True})]}
    assert not trends.has_current_trend_evidence(boolean_evidence, 3.0, cfg)
    string_evidence = {"signals": [signal("google_trends", "https://trends.test/page", {"traffic": "100000"})]}
    assert not trends.has_current_trend_evidence(string_evidence, 3.0, cfg)


def test_make_one_fails_closed_without_live_trend_evidence(monkeypatch):
    from autotube import pipeline

    def should_not_produce(*args, **kwargs):
        raise AssertionError("topic without current trend evidence must not be produced")

    monkeypatch.setattr(pipeline, "produce_assets", should_not_produce)
    topic = {"topic": "Honey", "sources": ["evergreen"], "score": 1.0}
    assert pipeline.make_one(CFG, object(), topic, {}, 1, "test-run") is None


def test_make_one_rejects_configuration_that_weakens_viral_floor(monkeypatch):
    import copy
    from autotube import pipeline

    cfg = copy.deepcopy(CFG)
    cfg["content"]["min_viral_score"] = 6
    monkeypatch.setattr(pipeline, "produce_assets", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("invalid low viral floor must be rejected before production")))
    topic = {"topic": "Current Google trend", "angle": "a rare object surged in searches",
             "sources": ["google_trends"], "score": 0.9, "viral_score": 9,
             "signals": [signal("google_trends", "https://trends.test/current", {"traffic": 20_000})]}
    assert pipeline.make_one(cfg, object(), topic, {}, 1, "test-run") is None


def test_make_one_rejects_below_floor_viral_score_even_with_live_source(monkeypatch):
    from autotube import pipeline

    monkeypatch.setattr(pipeline, "produce_assets", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("weak viral score must be rejected before production")))
    topic = {"topic": "Current Google trend", "sources": ["google_trends"], "score": 0.9, "viral_score": 6,
             "signals": [signal("google_trends", "https://trends.test/current", {"traffic": 20_000})]}
    assert pipeline.make_one(CFG, object(), topic, {}, 1, "test-run") is None


def test_make_one_rejects_non_finite_or_coerced_viral_scores(monkeypatch):
    from autotube import pipeline

    def should_not_produce(*args, **kwargs):
        raise AssertionError("invalid viral scores must not reach production")

    monkeypatch.setattr(pipeline, "produce_assets", should_not_produce)
    for score in (float("nan"), float("inf"), True, "9"):
        topic = {"topic": "Current Google trend", "angle": "a rare object suddenly surged in searches",
                 "sources": ["google_trends"], "score": 0.9, "viral_score": score,
                 "signals": [signal("google_trends", "https://trends.test/current", {"traffic": 20_000})]}
        assert pipeline.make_one(CFG, object(), topic, {}, 1, "test-run") is None


def test_youtube_chart_boolean_view_count_is_not_a_trending_signal(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    monkeypatch.setattr(trends, "get_json", lambda *a, **k: {"items": [{
        "id": "abcdefghijk", "snippet": {"title": "Trending topic", "tags": [],
                                         "categoryId": "22", "publishedAt": "2026-10-10T12:00:00Z"},
        "statistics": {"viewCount": True},
    }]})
    row = trends.youtube_chart("US")[0]
    assert row["view_count"] == 0
    signals = [trends._signal_record(row, common.now_utc().isoformat(timespec="seconds")),
               signal("reddit_science", "https://reddit.test/post")]
    assert not trends.has_current_trend_evidence({"signals": signals}, 3.0, CFG["trends"])


def test_google_trends_traffic_parser_preserves_k_and_magnitude():
    assert trends._parse_traffic(True) == 0
    assert trends._parse_traffic(1.5) == 0
    assert trends._parse_traffic(",1,000") == 0
    assert trends._parse_traffic("1,,000") == 0
    assert trends._parse_traffic(float("nan")) == 0
    assert trends._provider_count(True) is None
    assert trends._provider_count("1000") == 1000
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


def test_trend_collection_uses_configured_reddit_sources_once(monkeypatch):
    cfg = common.load_config()
    cfg["trends"]["wikipedia_spike_check"] = 0
    cfg["trends"]["on_this_day"] = False
    cfg["trends"]["reddit_subreddits"] = ["science", "archaeology", "SCIENCE", "bad/name", 7]
    calls = []
    monkeypatch.setattr(trends, "youtube_outliers", lambda cfg: [])
    monkeypatch.setattr(trends, "google_trends", lambda region: [])
    monkeypatch.setattr(trends, "wikipedia_top", lambda lang: [])
    monkeypatch.setattr(trends, "reddit_rss", lambda sub, limit=25: calls.append((sub, limit)) or [])
    monkeypatch.setattr(trends, "hackernews", lambda: [])
    monkeypatch.setattr(trends, "youtube_chart", lambda region: [])
    monkeypatch.setattr(trends, "on_this_day", lambda lang: [])
    monkeypatch.setattr("autotube.safety.screen", lambda *a, **k: ("allow", ""))

    assert trends.collect(cfg) == []
    assert calls == [("science", 15), ("archaeology", 10)]


def test_trend_merge_only_rewards_independent_sources_and_keeps_measured_evidence(monkeypatch, caplog):
    caplog.set_level("INFO", logger="autotube.trends")
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
    assert "reddit_science:1, reddit_space:1" in caplog.text

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
            return {"score": 9, "hook_strength": 6, "entertainment": 8, "coherence": 8, "naturalness": 9, "loops": True,
                    "factual_errors": [], "misleading_title": False, "advertiser_friendly": True,
                    "policy_concerns": [], "value_add": "A concise explanation of the supported fact.", "fixes": []}

    w = ScriptWriter.__new__(ScriptWriter)
    w.cfg, w.llm, w.src_chars = CFG, Rev(), 4000
    cfg_c = dict(CFG["content"], require_grounding=False)
    w.cfg = {**CFG, "content": cfg_c}
    src = {"title": "X", "text": "Bees make honey and visit two million flowers for one honey jar. " * 50}
    # Gate-clean script: the ONLY thing wrong with it is the reviewer's hook_strength of 6.
    segs = [{"text": "Bees need two million flowers for one honey jar"},
            {"text": "A worker bee makes one twelfth of a teaspoon in life"},
            {"text": "The hive beats its wings to dry the nectar into honey"},
            {"text": "A wax seal keeps the stored honey from absorbing moisture"},
            {"text": "so the jar in your cupboard could outlive you"}]
    ok, review = w.check({"title": "Honey", "segments": segs}, src)
    assert not ok and any("hook too weak" in i for i in review["issues"])


def test_titles_are_screened_for_shock_bait_not_vocabulary():
    class Rev:
        lite = False

        def json(self, system, user, **kw):
            return {"score": 9, "hook_strength": 9, "entertainment": 8, "coherence": 8, "naturalness": 9,
                    "loops": True, "factual_errors": [], "misleading_title": False,
                    "advertiser_friendly": True, "policy_concerns": [],
                    "value_add": "A concise explanation of the supported fact.", "fixes": []}

    w = ScriptWriter.__new__(ScriptWriter)
    w.llm, w.src_chars = Rev(), 4000
    w.cfg = {**CFG, "content": dict(CFG["content"], require_grounding=False)}
    src = {"title": "Hunger stone", "text": "stones during the Thirty Years War " * 40}
    seg = [{"text": "The Thirty Years War stone still warns us"},
           {"text": "It only appears when the river drops to a record low"},
           {"text": "The oldest marking on it is older than anyone alive"},
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
