"""The appeal packet.

Enforcement against faceless channels is automated, channel-wide and retroactive, and
honest creators get swept up. Successful appeals turn on documentary evidence of
originality, which has to exist before it is needed - after a strike the work directory
is gone and the model that wrote the script has been deprecated.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from autotube import provenance  # noqa: E402

CFG = {"persona": {"voice": "en-US-AndrewNeural"}}


@pytest.fixture(autouse=True)
def tmp_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "DIR", tmp_path / "provenance")
    return tmp_path / "provenance"


def _res(**over):
    res = {
        "script": {"title": "The lighthouse that emptied itself", "segments": [
            {"text": "Three keepers vanished from a locked lighthouse", "evidence": "the door was bolted"},
            {"text": "Two coats were gone, one still hung by the door", "evidence": "a coat remained",
             "aside": "Tidy of them."}]},
        "source": {"title": "Flannan Isles", "url": "https://en.wikipedia.org/wiki/Flannan_Isles",
                   "also": [{"title": "BBC", "url": "https://bbc.co.uk/x"}]},
        "topic": {"topic": "Flannan Isles", "category": "mysteries", "sources": ["wikipedia"],
                  "score": 0.91, "spike": 3.5, "why_trending": "Wikipedia pageviews reached 3.5x baseline",
                  "angle": "the island lighthouse went silent with no explanation",
                  "viral_score": 8, "context": ["3.5x its 30-day pageview baseline"],
                  "signals": [{"source": "wikipedia", "evidence": {"spike": 3.5}}]},
        "plan": {"format": "creepy_true", "hook_style": "bold_claim"},
        "review": {"score": 8, "coherence": 8, "trend_alignment": 9,
                   "trend_alignment_reason": "Covers the recent spike angle",
                   "reviewer": "anthropic:claude", "independent": True, "issues": []},
    }
    res.update(over)
    return res


def _entry(**over):
    e = {"title": "CASE #007 — the lighthouse", "video_id": "abc123", "series": "unsolved",
         "episode": 7, "duration": 18.2, "file": "x.mp4", "publish_at": "2026-10-04T12:00:00",
         "created_at": "2026-10-03T09:00:00", "llm": "openai:gpt", "tts_engine": "edge",
         "run_id": "r1", "sketch": ["aa", "bb"], "max_similarity": 0.07, "commentary_ratio": 0.08}
    e.update(over)
    return e


def _written(d):
    return json.loads(next(Path(d).glob("*.json")).read_text())


# ── recording ───────────────────────────────────────────────────────────────
def test_record_writes_a_file_and_returns_its_path(tmp_archive):
    p = provenance.record(_res(), _entry(), CFG)
    assert p and Path(p).exists()


def test_the_record_answers_the_accusation_claim_by_claim(tmp_archive):
    """'No commentary, scraped content' is rebutted by showing where each line came
    from - this is the section an appeal actually leans on."""
    provenance.record(_res(), _entry(), CFG)
    g = _written(tmp_archive)["grounding"]
    assert g["primary"]["url"].startswith("https://")
    assert g["corroboration"][0]["title"] == "BBC"
    assert len(g["claims"]) == 2
    assert all(c["says"] and c["evidence"] for c in g["claims"])


def test_the_record_preserves_the_narrator_commentary_separately(tmp_archive):
    provenance.record(_res(), _entry(), CFG)
    assert _written(tmp_archive)["script"]["asides"] == ["Tidy of them."]


def test_the_record_shows_why_this_topic_was_chosen(tmp_archive):
    """Editorial selection is the difference between a show and bulk enumeration."""
    provenance.record(_res(), _entry(), CFG)
    sel = _written(tmp_archive)["selection"]
    assert sel["why_trending"] == "Wikipedia pageviews reached 3.5x baseline"
    assert sel["trend_sources"] == ["wikipedia"] and sel["trend_score"] == 0.91
    assert sel["wikipedia_spike"] == 3.5
    assert sel["viral_score"] == 8 and sel["format"] == "creepy_true"
    assert sel["angle"] == "the island lighthouse went silent with no explanation"
    assert sel["signals"] == [{"source": "wikipedia", "evidence": {"spike": 3.5}}]


def test_the_record_pins_the_models_used(tmp_archive):
    """A record written today must stay interpretable after these are deprecated."""
    provenance.record(_res(), _entry(), CFG)
    pipe = _written(tmp_archive)["pipeline"]
    assert pipe["llm"] == "openai:gpt" and pipe["voice"] == "en-US-AndrewNeural"


def test_the_record_captures_independence_of_review(tmp_archive):
    provenance.record(_res(), _entry(), CFG)
    assert _written(tmp_archive)["review"]["independent"] is True
    review = _written(tmp_archive)["review"]
    assert review["trend_alignment"] == 9 and review["coherence"] == 8


def test_the_record_does_not_copy_the_source_text(tmp_archive):
    """Archiving scraped page text is the one thing that would make the evidence
    look like the offence. The URL is enough."""
    res = _res()
    res["source"]["text"] = "SOME VERY LONG SCRAPED ARTICLE BODY"
    provenance.record(res, _entry(), CFG)
    assert "SOME VERY LONG SCRAPED" not in next(Path(tmp_archive).glob("*.json")).read_text()


def test_records_do_not_overwrite_each_other(tmp_archive):
    provenance.record(_res(), _entry(title="CASE #007 — one"), CFG)
    provenance.record(_res(), _entry(title="CASE #008 — two"), CFG)
    assert len(list(Path(tmp_archive).glob("*.json"))) == 2


def test_filenames_are_safe_for_any_title(tmp_archive):
    provenance.record(_res(), _entry(title="CASE #9: what?! / 50% \\ *done*"), CFG)
    assert len(list(Path(tmp_archive).glob("*.json"))) == 1


def test_recording_never_raises_and_never_takes_down_a_run(tmp_archive):
    """The video is already rendered and uploaded by this point. Insurance that can
    destroy the thing it insures is worse than no insurance."""
    assert provenance.record(None, _entry(), CFG) == ""
    assert provenance.record(_res(), {}, CFG) != ""


def test_a_record_with_missing_optional_fields_still_writes(tmp_archive):
    assert provenance.record({"script": {}}, {"title": "bare"}, {}) != ""


# ── dossier ─────────────────────────────────────────────────────────────────
def test_dossier_is_safe_before_anything_has_been_published(tmp_archive):
    d = provenance.dossier()
    assert d["episodes"] == 0 and d["highest_similarity_between_any_episode_and_its_predecessors"] is None


def test_dossier_aggregates_the_channel_level_claim(tmp_archive):
    for i in range(3):
        provenance.record(_res(), _entry(title=f"CASE #{i}", max_similarity=0.05 * i), CFG)
    d = provenance.dossier()
    assert d["episodes"] == 3
    assert d["with_cited_primary_source"] == 3
    assert d["independently_reviewed"] == 3
    assert d["with_narrator_commentary"] == 3
    assert d["highest_similarity_between_any_episode_and_its_predecessors"] == 0.1
    assert d["series"] == ["unsolved"]


def test_dossier_reports_source_diversity(tmp_archive):
    for i in range(3):
        res = _res()
        res["source"]["url"] = f"https://en.wikipedia.org/wiki/page{i}"
        provenance.record(res, _entry(title=f"CASE #{i}"), CFG)
    assert provenance.dossier()["distinct_sources"] == 3


def test_dossier_survives_a_corrupt_record(tmp_archive):
    provenance.record(_res(), _entry(), CFG)
    (Path(tmp_archive) / "broken.json").write_text("{not json")
    assert provenance.dossier()["episodes"] == 1


def test_dossier_can_be_limited_to_recent_episodes(tmp_archive):
    for i in range(5):
        provenance.record(_res(), _entry(title=f"CASE #{i}"), CFG)
    assert provenance.dossier(limit=2)["episodes"] == 2
