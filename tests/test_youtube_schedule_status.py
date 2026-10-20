from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import youtube_schedule_status as schedule_status  # noqa: E402


def test_scheduled_uploads_uses_youtube_status_and_excludes_public_or_unscheduled(monkeypatch):
    class Request:
        def __init__(self, result):
            self.result = result

        def execute(self):
            return self.result

    class Channels:
        def list(self, **kwargs):
            assert kwargs == {"part": "contentDetails", "mine": True, "maxResults": 1}
            return Request({"items": [{"contentDetails": {
                "relatedPlaylists": {"uploads": "uploads-playlist"},
            }}]})

    class PlaylistItems:
        def list(self, **kwargs):
            assert kwargs["playlistId"] == "uploads-playlist"
            return Request({"items": [
                {"contentDetails": {"videoId": "scheduled123"}},
                {"contentDetails": {"videoId": "publicvideo1"}},
                {"contentDetails": {"videoId": "unscheduled1"}},
            ]})

    class Videos:
        def list(self, **kwargs):
            assert kwargs["part"] == "snippet,status"
            return Request({"items": [
                {"id": "scheduled123", "snippet": {"title": "Next Short"}, "status": {
                    "privacyStatus": "private", "uploadStatus": "processed",
                    "publishAt": "2026-10-21T16:00:00Z",
                }},
                {"id": "publicvideo1", "snippet": {"title": "Already public"}, "status": {
                    "privacyStatus": "public", "uploadStatus": "processed",
                }},
                {"id": "unscheduled1", "snippet": {"title": "Private draft"}, "status": {
                    "privacyStatus": "private", "uploadStatus": "uploaded",
                }},
            ]})

    class Service:
        def channels(self):
            return Channels()

        def playlistItems(self):
            return PlaylistItems()

        def videos(self):
            return Videos()

    monkeypatch.setattr(schedule_status.youtube, "service", lambda: Service())
    result = schedule_status.scheduled_uploads()
    assert result == [{
        "video_id": "scheduled123",
        "title": "Next Short",
        "privacy_status": "private",
        "upload_status": "processed",
        "publish_at": "2026-10-21T16:00:00Z",
    }]


def test_timestamp_parsing_normalizes_utc_and_handles_invalid_values():
    from datetime import datetime, timezone

    assert schedule_status._timestamp("2026-10-21T16:00:00Z") == datetime(2026, 10, 21, 16, tzinfo=timezone.utc)
    assert schedule_status._timestamp("not-a-date") is None
