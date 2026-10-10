"""Regression tests for rolling Analytics-window freshness and qualified Shorts views."""
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import analytics  # noqa: E402
from autotube.common import load_config  # noqa: E402


class _Strategy:
    def __init__(self, cfg):
        self.state = {"reward_schema_version": analytics.REWARD_SCHEMA_VERSION, "updates": 0}

    def age(self, now):
        pass

    def save(self):
        pass

    def report(self):
        return {}


def _run(monkeypatch, retention_row):
    now = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    entry = {
        "video_id": "video-1",
        "title": "Trend-backed Short",
        "created_at": "2026-06-01T12:00:00+00:00",
        "publish_at": "2026-10-10T02:00:00+00:00",
        "status": "scheduled",
        "choice": {},
        "metrics": {
            "views": 5000, "likes": 10, "comments": 1, "privacy": "public",
            "hours_live": 100, "vph": 7, "vph_basis": "engaged_views",
            "a_views": 6000, "engaged_views_90d": 900, "avg_view_pct": 91.0,
            "avg_view_pct_raw": 3474.0, "avg_view_dur": 18, "subs_gained": 4, "subs_lost": 1, "shares": 2,
            "engaged_views_window": {"start": "2026-07-12", "end": "2026-10-09"},
            "checked_at": "2026-10-09T12:00:00+00:00",
        },
    }
    history = [entry]
    written = {}
    query = {}

    monkeypatch.setattr(analytics, "now_utc", lambda: now)
    monkeypatch.setattr(analytics, "read_json", lambda name, default: history)
    monkeypatch.setattr(analytics, "write_json", lambda name, data: written.setdefault(name, data))
    monkeypatch.setattr(analytics, "Strategy", _Strategy)
    monkeypatch.setattr(analytics.originality, "channel_audit", lambda *a, **k: {})
    monkeypatch.setattr(analytics.youtube, "video_stats", lambda ids: {
        "video-1": {"views": 5000, "likes": 10, "comments": 1, "privacy": "public",
                    "upload_status": "processed", "rejection": None}})

    def retention(ids, start, end):
        query.update(start=start, end=end)
        return {"video-1": dict(retention_row)} if retention_row is not None else {}

    monkeypatch.setattr(analytics.youtube, "retention", retention)
    monkeypatch.setattr(analytics.youtube, "retention_curves", lambda *a, **k: {})
    result = analytics.run(load_config())
    return entry["metrics"], result, query, written


def test_missing_analytics_row_clears_stale_window_values_and_marks_fallback(monkeypatch):
    metrics, result, query, written = _run(monkeypatch, None)

    assert query == {"start": "2026-07-13", "end": "2026-10-10"}  # 90 inclusive calendar dates
    for key in ("a_views", "engaged_views_90d", "avg_view_pct", "avg_view_pct_raw", "avg_view_dur",
                "subs_gained", "subs_lost", "shares"):
        assert key not in metrics
    assert metrics["engaged_views_window"] == {"start": "2026-07-13", "end": "2026-10-10"}
    assert metrics["vph_basis"] == "public_views_fallback"
    assert result["shorts_monetization"]["engaged_views_90d"] == 0
    assert result["shorts_monetization"]["coverage"] == 0.0
    assert "analytics_summary.json" in written


def test_current_analytics_row_replaces_stale_values_and_uses_engaged_velocity(monkeypatch):
    metrics, result, query, written = _run(monkeypatch, {
        "a_views": 1200, "engaged_views_90d": 240, "avg_view_pct": 76.5,
        "avg_view_dur": 14.2, "subs_gained": 3, "subs_lost": 1, "shares": 8,
    })

    assert query == {"start": "2026-07-13", "end": "2026-10-10"}
    assert metrics["a_views"] == 1200 and metrics["engaged_views_90d"] == 240
    assert metrics["avg_view_pct"] == 76.5 and metrics["vph_basis"] == "engaged_views"
    assert metrics["vph"] == 24.0  # 240 engaged views over the 10 live hours
    assert metrics["engaged_views_window"] == {"start": "2026-07-13", "end": "2026-10-10"}
    assert result["shorts_monetization"]["engaged_views_90d"] == 240
    assert result["shorts_monetization"]["coverage"] == 1.0
    assert "history.json" in written and "analytics_summary.json" in written
