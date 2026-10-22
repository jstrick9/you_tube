from datetime import datetime, timedelta, timezone

from autotube.common import normalize_topic_key
from autotube.pipeline import recently_failed_topic_keys


def test_failed_trends_stay_cooled_down_across_same_day_top_ups():
    now = datetime(2026, 10, 22, 13, 30, tzinfo=timezone.utc)
    reports = [
        {
            "started_at": (now - timedelta(hours=2)).isoformat(),
            "topic_attempts": [
                {"topic_key": "recent bad angle", "outcome": "rejected", "stage": "script_research"},
                # Older receipts collapse title whitespace; live `&` removal used to
                # leave a double space, so both sides must share the same canonicalizer.
                {"topic_key": "how astronauts live sleep shower  cry in space",
                 "outcome": "rejected", "stage": "script_research"},
                {"topic_key": "recent good topic", "outcome": "accepted", "stage": "upload"},
                {"topic_key": "upload failure", "outcome": "rejected", "stage": "upload"},
            ],
        },
        {
            # This reproduces the 9-hour gap between the earlier failed attempt and
            # the later top-up that selected the same unsupported angles again.
            "started_at": (now - timedelta(hours=9)).isoformat(),
            "topic_attempts": [
                {"topic_key": "same day failed angle", "outcome": "rejected", "stage": "script_draft"},
            ],
        },
        {
            "started_at": (now - timedelta(hours=25)).isoformat(),
            "topic_attempts": [
                {"topic_key": "expired failed topic", "outcome": "rejected", "stage": "script_draft"},
            ],
        },
    ]

    cooled = recently_failed_topic_keys(reports, now=now)
    assert cooled == {
        "recent bad angle", "same day failed angle",
        "how astronauts live sleep shower cry in space",
    }
    assert normalize_topic_key("How Astronauts Live, Sleep, Shower & Cry in Space!") in cooled
    assert recently_failed_topic_keys(reports, now=now, cooldown_hours=6) == {
        "recent bad angle", "how astronauts live sleep shower cry in space"}


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
