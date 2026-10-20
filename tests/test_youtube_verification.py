from datetime import datetime, timezone

import pytest

from autotube import youtube


PUBLISH_AT = datetime(2026, 10, 20, 23, 30, tzinfo=timezone.utc)
VIDEO_ID = "verified12345"
CFG = {"upload": {"default_privacy": "public"}}


def scheduled_status(**changes):
    value = {
        "video_id": VIDEO_ID,
        "title": "A verified Short",
        "privacy_status": "private",
        "upload_status": "uploaded",
        "publish_at": "2026-10-20T23:30:00.000Z",
        "rejection_reason": None,
    }
    value.update(changes)
    return value


def test_verify_upload_confirms_exact_scheduled_id_privacy_and_publish_time(monkeypatch):
    monkeypatch.setattr(youtube, "video_status", lambda video_id: scheduled_status())

    result = youtube.verify_upload(VIDEO_ID, PUBLISH_AT, CFG)

    assert result["video_id"] == VIDEO_ID
    assert result["privacy_status"] == "private"
    assert result["upload_status"] == "uploaded"
    assert result["publish_at"] == "2026-10-20T23:30:00.000Z"


@pytest.mark.parametrize("changes", [
    {"video_id": "different123"},
    {"privacy_status": "public"},
    {"upload_status": "failed"},
    {"publish_at": "2026-10-20T23:31:00.000Z"},
])
def test_verify_upload_rejects_any_mismatched_publication_state(monkeypatch, changes):
    monkeypatch.setattr(youtube, "video_status", lambda _video_id: scheduled_status(**changes))
    monkeypatch.setattr(youtube.time, "sleep", lambda *_args: None)

    with pytest.raises(RuntimeError, match="did not confirm"):
        youtube.verify_upload(VIDEO_ID, PUBLISH_AT, CFG)


def test_verify_upload_retries_one_indexing_delay_then_returns_verified_state(monkeypatch):
    responses = [None, scheduled_status()]
    calls = []
    monkeypatch.setattr(youtube, "video_status", lambda _video_id: calls.append(_video_id) or responses.pop(0))
    monkeypatch.setattr(youtube.time, "sleep", lambda *_args: None)

    assert youtube.verify_upload(VIDEO_ID, PUBLISH_AT, CFG)["publish_at"] == "2026-10-20T23:30:00.000Z"
    assert calls == [VIDEO_ID, VIDEO_ID]


def test_verify_immediate_upload_uses_configured_privacy_and_rejects_unexpected_schedule(monkeypatch):
    monkeypatch.setattr(youtube, "video_status", lambda _video_id: scheduled_status(
        privacy_status="public", publish_at=None))

    result = youtube.verify_upload(VIDEO_ID, None, CFG)

    assert result["privacy_status"] == "public"
    assert result["publish_at"] is None
