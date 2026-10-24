from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, scriptwriter  # noqa: E402
from autotube.curated_drafts import NOKIA_ANGLE, NOKIA_PROFILE_ID, nokia_draft, nokia_metadata  # noqa: E402


CFG = common.load_config()
SOURCE_TEXT = """Nokia's mobile phone market share reached its all-time high figure of 40.4% in 2007.
In 2011, Nokia announced a strategic partnership with Microsoft, under which it would adopt Windows Phone 7 as its primary operating system on smartphones.
After the announcement, demand for Symbian dropping when consumers realized Nokia's focus and attention would be elsewhere.
Elop announced cost-cutting measures by shedding 10,000 employees by the end of the year and the closure of the Salo manufacturing plant.
In September 2013, Nokia announced the sale of its mobile and devices division to Microsoft.
Experts agree that Nokia would have been in a cash crisis had it not sold the division to Microsoft.
"""


class OfflineButReviewableLLM:
    lite = False
    last_used = ""

    def __init__(self):
        self.writer_calls = []
        self.review_calls = []

    def json(self, system, user, **kwargs):
        self.writer_calls.append((system, user))
        raise RuntimeError("all text-draft and metadata providers unavailable")

    def review_json(self, system, user, **kwargs):
        self.review_calls.append((system, user, kwargs.get("avoid")))
        review = {
            "factual_errors": [], "misleading_title": False, "advertiser_friendly": True,
            "policy_concerns": [], "value_add": "The 2011 Windows Phone switch preceded a Symbian-demand drop and a sale that averted a cash crisis.",
            "hook_strength": 8, "entertainment": 7, "score": 8, "loops": False,
            "coherence": 8, "naturalness": 9, "trend_alignment": 9,
            "trend_alignment_reason": "It directly follows the Nokia Windows Phone shift through Symbian demand and the Microsoft sale.",
            "turn_strength": 8, "turn_strength_reason": "The platform switch directly advances the story.", "fixes": [],
        }
        kwargs["validate"](review)
        self.last_used = "fake:independent-review"
        return review


def _topic():
    captured = datetime.now(timezone.utc)
    published = captured - timedelta(hours=2)
    return {
        "topic": "Kebangkitan dan Kebangkrutan Nokia",
        "topic_key": "kebangkitan dan kebangkrutan nokia",
        "sources": ["youtube_outliers"],
        "lane": "history_mystery",
        "category": "history",
        "script_subject": "Nokia",
        "wiki_query": "Nokia",
        "angle": NOKIA_ANGLE,
        "research_query": "Nokia 2011 Windows Phone partnership and mobile division sale",
        "curated_profile": NOKIA_PROFILE_ID,
        "signals": [{
            "source": "youtube_outliers",
            "source_url": "https://www.youtube.com/watch?v=h1i-SkdL8Tc",
            "captured_at": captured.isoformat(),
            "evidence": {
                "video_id": "h1i-SkdL8Tc", "views": 100_000, "hours": 2.0,
                "vph": 50_000, "breakout": 10.0,
                "published_at": published.isoformat(),
            },
        }],
        "why_trending": "Current viral YouTube Short with independently measured velocity.",
        "viral_score": 10,
    }


def _source(text=SOURCE_TEXT):
    return {"title": "Nokia", "summary": "Nokia mobile-phone history.",
            "url": "https://en.wikipedia.org/wiki/Nokia", "text": text}


def test_nokia_profile_is_limited_to_a_live_youtube_decline_story():
    profile = scriptwriter._curated_research_profile(
        "Kebangkitan dan Kebangkrutan Nokia", ["youtube_outliers"], "history_mystery")

    assert profile is not None
    assert profile["curated_profile"] == NOKIA_PROFILE_ID
    assert profile["script_subject"] == "Nokia"
    assert profile["wiki_query"] == "Nokia"
    assert profile["angle"] == NOKIA_ANGLE
    assert scriptwriter._curated_research_profile(
        "Kebangkitan dan Kebangkrutan Nokia", ["reddit_history"], "history_mystery") is None
    assert scriptwriter._curated_research_profile(
        "Nokia launches a new phone", ["youtube_outliers"], "history_mystery") is None
    assert scriptwriter._curated_research_profile(
        "Nokia stock falls sharply", ["youtube_outliers"], "history_mystery") is None
    assert scriptwriter._curated_research_profile(
        "Kebangkitan dan Kebangkrutan Nokia", ["youtube_outliers"], "science_nature") is None


def test_curated_nokia_route_rejects_stale_trend_evidence(monkeypatch):
    topic = _topic()
    stale_capture = datetime.now(timezone.utc) - timedelta(hours=37)
    topic["signals"][0]["captured_at"] = stale_capture.isoformat()
    topic["signals"][0]["evidence"]["published_at"] = (stale_capture - timedelta(hours=2)).isoformat()
    llm = OfflineButReviewableLLM()
    writer = scriptwriter.ScriptWriter(CFG, llm, object())
    monkeypatch.setattr(scriptwriter, "read_json", lambda *_args, **_kwargs: [])

    with pytest.raises(AssertionError, match="fresh, currently qualifying trend evidence"):
        writer.write(topic, {
            "format": "backstory", "hook_style": "bold_claim", "voice": "test", "comment_device": "none",
        }, _source())
    assert llm.writer_calls == []


def test_nokia_draft_is_source_locked_and_passes_the_same_writer_and_review_gates(monkeypatch):
    topic, source = _topic(), _source()
    assert all(quote in source["text"] for quote in (
        "would adopt Windows Phone 7 as its primary operating system on smartphones",
        "demand for Symbian dropping when consumers realized Nokia's focus and attention would be elsewhere",
        "shedding 10,000 employees by the end of the year and the closure of the Salo",
        "would have been in a cash crisis had it not sold the division to Microsoft",
        "In September 2013, Nokia announced the sale of its mobile and devices division to Microsoft.",
    ))
    draft = nokia_draft(topic, source)
    assert draft is not None
    assert scriptwriter.source_supports_angle(NOKIA_ANGLE, "Nokia", source["text"])
    assert scriptwriter.delivers_angle(draft, NOKIA_ANGLE, "Nokia")
    assert nokia_draft(topic, _source(source["text"].replace("Windows Phone 7", "Android"))) is None
    assert nokia_draft({**topic, "sources": ["reddit_history"]}, source) is None

    llm = OfflineButReviewableLLM()
    writer = scriptwriter.ScriptWriter(CFG, llm, object())
    monkeypatch.setattr(scriptwriter, "read_json", lambda *_args, **_kwargs: [])
    script = writer.write(topic, {
        "format": "backstory", "hook_style": "bold_claim", "voice": "test", "comment_device": "none",
    }, source)

    assert script["_writer_model"] == f"curated:{NOKIA_PROFILE_ID}"
    assert len(script["segments"]) == 5
    assert len(llm.writer_calls) == 1, "the curated draft must not require an external writer call"
    assert llm.writer_calls[0][0] == scriptwriter.WRITER_SYSTEM
    assert "Write YouTube metadata" in llm.writer_calls[0][1]
    assert scriptwriter.metadata_schema_issues(script) == []
    assert all(6 <= len(segment["text"].split()) <= 11 for segment in script["segments"])
    assert scriptwriter._final_shape_check(script) is None

    ok, review = writer.check(script, source, topic)
    assert ok, review
    assert review["independent"] is True
    assert llm.review_calls and llm.review_calls[0][2] == f"curated:{NOKIA_PROFILE_ID}"


def test_final_shape_check_rejects_a_post_metadata_shape_regression(monkeypatch):
    topic, source = _topic(), _source()
    llm = OfflineButReviewableLLM()
    writer = scriptwriter.ScriptWriter(CFG, llm, object())
    broken_segments = nokia_draft(topic, source)["segments"]
    broken_segments[0]["emphasis"] = []
    valid_metadata = nokia_metadata({"_writer_model": f"curated:{NOKIA_PROFILE_ID}"})
    monkeypatch.setattr(writer, "metadata", lambda *_args, **_kwargs: {
        **valid_metadata, "segments": broken_segments,
    })

    with pytest.raises(AssertionError, match="FINAL SHAPE CHECK failed"):
        writer.write(topic, {
            "format": "backstory", "hook_style": "bold_claim", "voice": "test", "comment_device": "none",
        }, source)


def test_final_shape_check_rejects_post_metadata_upload_shape_regression(monkeypatch):
    topic, source = _topic(), _source()
    writer = scriptwriter.ScriptWriter(CFG, OfflineButReviewableLLM(), object())
    broken_metadata = nokia_metadata({"_writer_model": f"curated:{NOKIA_PROFILE_ID}"})
    broken_metadata["hashtags"] = ["#Nokia"]
    monkeypatch.setattr(writer, "metadata", lambda *_args, **_kwargs: broken_metadata)

    with pytest.raises(AssertionError, match="FINAL SHAPE CHECK failed.*hashtags"):
        writer.write(topic, {
            "format": "backstory", "hook_style": "bold_claim", "voice": "test", "comment_device": "none",
        }, source)


def test_generic_metadata_outage_fallback_satisfies_upload_schema():
    writer = scriptwriter.ScriptWriter(CFG, OfflineButReviewableLLM(), object())
    script = {
        "title": "How a Stone Records Floods",
        "segments": [{"text": "A hunger stone was carved when river water dropped."}],
    }

    metadata = writer.metadata(script, {"title": "Hunger stone"})

    assert scriptwriter.metadata_schema_issues(metadata) == []
    assert 8 <= len(metadata["tags"]) <= 12
    assert len(metadata["hashtags"]) == 3


def test_nokia_metadata_fallback_is_valid_upload_metadata():
    meta = nokia_metadata({"_writer_model": f"curated:{NOKIA_PROFILE_ID}"})
    assert meta is not None
    assert scriptwriter.metadata_schema_issues(meta) == []
