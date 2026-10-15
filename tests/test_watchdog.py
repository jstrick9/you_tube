"""The schedule watchdog retries missing triggers without replaying late cron events."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from autotube import watchdog
from autotube.watchdog import scheduled_slot_utc, should_dispatch_top_up, should_skip_delayed_schedule

ROOT = Path(__file__).resolve().parents[1]

UTC = timezone.utc
NOW = datetime(2026, 10, 13, 17, 39, tzinfo=UTC)


def run(run_id, started, status="completed", number=1):
    return {"id": run_id, "run_number": number, "status": status,
            "run_started_at": started.isoformat(), "created_at": started.isoformat()}


def test_watchdog_schedule_covers_each_daily_and_top_up_slot():
    daily = yaml.safe_load((ROOT / ".github/workflows/daily.yml").read_text())
    watchdog = yaml.safe_load((ROOT / ".github/workflows/watchdog.yml").read_text())
    daily_crons = [item["cron"] for item in daily[True]["schedule"]]
    watchdog_crons = [item["cron"] for item in watchdog[True]["schedule"]]
    assert "17 10 * * *" in daily_crons
    assert "47 11-23/4 * * *" in daily_crons
    assert watchdog_crons == ["35 10 * * *", "5 0,12,16,20 * * *"]
    steps = daily["jobs"]["produce"]["steps"]
    first_step = steps[0]
    assert first_step["uses"] == "actions/checkout@v4"
    assert first_step["with"]["ref"] == "main", "queued top-ups must read the latest committed target"
    preflight = next(step for step in steps if step.get("id") == "need")
    assert 'echo "proceed=true"' in preflight["run"]
    for name in ("Install", "Cache CLIP model weights", "Cache depth model (living-photo motion)",
                 "Cache offline TTS voice (fallback only)", "Learn from previous videos",
                 "Produce & schedule today's videos"):
        step = next(item for item in steps if item.get("name") == name)
        assert "steps.need.outputs.proceed == 'true'" in str(step.get("if", ""))
    fallback = next(step for step in steps if step.get("name") == "Ensure every Actions run has a production receipt")
    assert "AUTOTUBE_SKIP_REASON" in fallback.get("env", {})


def test_scheduled_slots_resolve_the_daily_and_top_up_crons_in_utc():
    assert scheduled_slot_utc("17 10 * * *", NOW) == datetime(2026, 10, 13, 10, 17, tzinfo=UTC)
    assert scheduled_slot_utc("47 11-23/4 * * *", NOW) == datetime(2026, 10, 13, 15, 47, tzinfo=UTC)
    assert scheduled_slot_utc("5 0,12,16,20 * * *", NOW) == datetime(2026, 10, 13, 16, 5, tzinfo=UTC)
    assert scheduled_slot_utc("not a cron", NOW) is None


def test_delayed_cron_is_skipped_when_recovery_already_started_after_its_slot():
    recovery = run("recovery", datetime(2026, 10, 13, 16, 5, tzinfo=UTC), number=105)
    skip, reason = should_skip_delayed_schedule(
        "47 11-23/4 * * *", [recovery], NOW, current_run_id="delayed-cron")
    assert skip and "already started after this scheduled slot" in reason


def test_schedule_guard_does_not_compare_a_run_against_itself():
    current = run("current", datetime(2026, 10, 13, 16, 5, tzinfo=UTC))
    skip, _ = should_skip_delayed_schedule(
        "47 11-23/4 * * *", [current], NOW, current_run_id="current")
    assert not skip


def test_schedule_guard_allows_the_regular_next_slot_after_an_earlier_run():
    morning = run("morning", datetime(2026, 10, 13, 10, 17, tzinfo=UTC))
    skip, _ = should_skip_delayed_schedule(
        "47 11-23/4 * * *", [morning], datetime(2026, 10, 13, 15, 47, tzinfo=UTC))
    assert not skip


def test_schedule_guard_defers_if_another_production_run_is_active():
    active = run("active", datetime(2026, 10, 13, 10, 17, tzinfo=UTC), status="in_progress")
    skip, reason = should_skip_delayed_schedule(
        "47 11-23/4 * * *", [active], NOW, current_run_id="current")
    assert skip and "already active" in reason


def test_dispatch_uses_main_with_only_the_safe_top_up_input(monkeypatch):
    class Response:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        assert timeout == 20
        assert request.full_url == "https://api.github.com/repos/owner/repo/actions/workflows/daily.yml/dispatches"
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Bearer test-token"
        assert json.loads(request.data) == {
            "ref": "main", "inputs": {"top_up": "true"},
        }
        return Response()

    monkeypatch.setattr(watchdog.urllib.request, "urlopen", fake_urlopen)
    watchdog.dispatch_top_up("owner/repo", "test-token")


def test_watchdog_only_dispatches_when_work_remains_and_no_recent_attempt_exists():
    old = run("old", NOW - timedelta(hours=3))
    recover, _ = should_dispatch_top_up(1, [old], NOW, cooldown_minutes=90)
    assert recover

    recent = run("recent", NOW - timedelta(minutes=30))
    recover, reason = should_dispatch_top_up(1, [recent], NOW, cooldown_minutes=90)
    assert not recover and "within the last 90 minutes" in reason

    active = run("active", NOW - timedelta(hours=3), status="in_progress")
    recover, reason = should_dispatch_top_up(1, [active], NOW, cooldown_minutes=90)
    assert not recover and "already active" in reason

    recover, reason = should_dispatch_top_up(0, [], NOW)
    assert not recover and "already satisfied" in reason
