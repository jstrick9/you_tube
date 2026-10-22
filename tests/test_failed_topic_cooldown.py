from datetime import datetime, timedelta, timezone

from autotube.pipeline import recently_failed_topic_keys


def test_recent_rejected_topic_is_cooled_down_but_acceptances_and_old_failures_are_not():
    now = datetime(2026, 10, 22, 13, 30, tzinfo=timezone.utc)
    reports = [
        {
            "started_at": (now - timedelta(hours=2)).isoformat(),
            "topic_attempts": [
                {"topic_key": "recent bad angle", "outcome": "rejected", "stage": "script_research"},
                {"topic_key": "recent good topic", "outcome": "accepted", "stage": "upload"},
                {"topic_key": "upload failure", "outcome": "rejected", "stage": "upload"},
            ],
        },
        {
            "started_at": (now - timedelta(hours=7)).isoformat(),
            "topic_attempts": [
                {"topic_key": "old failed topic", "outcome": "rejected", "stage": "script_draft"},
            ],
        },
    ]

    assert recently_failed_topic_keys(reports, now=now, cooldown_hours=6) == {"recent bad angle"}


def test_failed_topic_cooldown_fails_open_for_malformed_receipts_and_invalid_configuration():
    now = datetime(2026, 10, 22, 13, 30, tzinfo=timezone.utc)
    reports = [
        None,
        {"started_at": "not-a-date", "topic_attempts": [
            {"topic_key": "bad date", "outcome": "rejected", "stage": "script_draft"},
        ]},
    ]

    assert recently_failed_topic_keys(reports, now=now) == set()
    assert recently_failed_topic_keys([], now=now, cooldown_hours=0) == set()
    assert recently_failed_topic_keys([], now=now, cooldown_hours=float("nan")) == set()
