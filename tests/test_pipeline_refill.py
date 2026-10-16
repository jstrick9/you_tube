from autotube.common import load_config


def test_pipeline_refills_fallback_queue_after_two_rejections_without_duplicates(monkeypatch, tmp_path):
    from autotube import pipeline

    cfg = load_config()
    cfg["media"]["require_llm_verdict"] = False
    cfg["schedule"]["topic_reserve_minutes"] = 0
    events = []
    first_batch = [
        {"topic": "first", "topic_key": "first", "category": "history_mystery"},
        {"topic": "second", "topic_key": "second", "category": "history_mystery"},
        {"topic": "third", "topic_key": "third", "category": "history_mystery"},
    ]
    backup = {"topic": "fourth", "topic_key": "fourth", "category": "history_mystery"}

    class FakeLLM:
        def __init__(self, _cfg):
            pass

        @staticmethod
        def has_available_provider():
            return True

    class FakeStrategy:
        def __init__(self, _cfg):
            pass

        @staticmethod
        def plan(n):
            return [{} for _ in range(n)]

        @staticmethod
        def fit(plan, _topic, _used_formats):
            return dict(plan)

    class FakeWriter:
        def __init__(self, _cfg, _llm, _strategy):
            self.calls = 0

        def select_topics(self, _candidates, _n, _recent, exclude=None):
            self.calls += 1
            excluded = set(exclude or ())
            events.append(("select", self.calls, excluded))
            if self.calls == 1:
                return [dict(row) for row in first_batch]
            if self.calls == 2:
                return [dict(backup)]
            return []

    monkeypatch.setattr(pipeline, "LLM", FakeLLM)
    monkeypatch.setattr(pipeline, "Strategy", FakeStrategy)
    monkeypatch.setattr(pipeline, "ScriptWriter", FakeWriter)
    monkeypatch.setattr(pipeline.trends, "recent_topics", lambda *_args: set())
    monkeypatch.setattr(pipeline.trends, "collect", lambda _cfg: [{"topic": "live"}])
    monkeypatch.setattr(pipeline.run_report, "current_receipt", lambda: None)
    monkeypatch.setattr(pipeline, "read_json", lambda *_args: [])
    monkeypatch.setattr(pipeline, "WORK_DIR", tmp_path / "work")

    def reject_topic(_cfg, _writer, topic, _plan, _idx, _run_id):
        events.append(("attempt", topic["topic"]))
        return None

    monkeypatch.setattr(pipeline, "make_one", reject_topic)

    assert pipeline.run(cfg, count=1, upload=False) == []

    second_selection = next(i for i, event in enumerate(events)
                            if event[0] == "select" and event[1] == 2)
    third_attempt = events.index(("attempt", "third"))
    assert second_selection < third_attempt, "refill should happen after two misses, before queued fallbacks"
    assert events[second_selection][2] == {"first", "second", "third"}, \
        "attempted and still-queued candidates must be excluded from the fresh selection"
    assert [event[1] for event in events if event[0] == "attempt"] == [
        "first", "second", "third", "fourth",
    ]
