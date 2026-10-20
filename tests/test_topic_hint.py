from autotube import pipeline


def test_topic_hint_can_focus_a_fresh_candidate_by_exact_youtube_video_id():
    video_id = "F9IluEaMjJ4"
    candidate = {
        "topic": "Dog Paani Kaise Peeta Hai? Tongue Ka Asli Secret!",
        "topic_key": "dog paani kaise peeta hai tongue ka asli secret",
        "signals": [{
            "source": "youtube_outliers",
            "source_url": f"https://www.youtube.com/shorts/{video_id}",
            "evidence": {"video_id": video_id},
        }],
    }

    assert pipeline.filter_topic_candidates_by_hint([candidate], video_id) == [candidate]
    assert pipeline.filter_topic_candidates_by_hint([candidate], "dog tongue") == [candidate]
    assert pipeline.filter_topic_candidates_by_hint([candidate], "unrelated subject") == []


def test_empty_topic_hint_leaves_candidates_unchanged_and_does_not_create_one():
    candidates = [{"topic": "current candidate", "topic_key": "current candidate"}]

    assert pipeline.filter_topic_candidates_by_hint(candidates, "") is candidates
    assert pipeline.filter_topic_candidates_by_hint([], "F9IluEaMjJ4") == []
