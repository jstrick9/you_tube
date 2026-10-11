"""Coverage for the internal average-view target watchlist and its dashboard rendering."""
from pathlib import Path

from autotube.analytics import average_view_watchlist
from scripts import dashboard


def _video(video_id, pct, *, views=100, engaged=40, title=None):
    return {
        "video_id": video_id,
        "title": title or f"Video {video_id}",
        "topic": f"Topic for {video_id}",
        "created_at": "2026-10-10T12:00:00+00:00",
        "duration": 17.5,
        "choice": {"category": "history", "format": "plot_twist", "hook_style": "question"},
        "metrics": {
            "avg_view_pct": pct,
            "views": views,
            "engaged_views_90d": engaged,
        },
    }


def test_watchlist_lists_only_valid_under_target_videos_lowest_first():
    videos = [
        _video("near", 69.9, views=900, engaged=350),
        _video("lowest", 31.25, views=400, engaged=180),
        _video("meets", 70.0),
        _video("missing", None),
        _video("zero-is-invalid", 0),
        _video("implausible", 401),
    ]

    result = average_view_watchlist(videos, {"average_view_gate_pct": 70.0})

    assert result["gate_pct"] == 70.0
    assert result["evaluated"] == 3
    assert result["below_target"] == 2
    assert result["shown"] == 2
    assert [video["id"] for video in result["videos"]] == ["lowest", "near"]
    assert result["videos"][0]["avg_view_pct"] == 31.25
    assert result["videos"][0]["duration_seconds"] == 17.5
    assert result["videos"][0]["engaged_views_90d"] == 180


def test_watchlist_counts_all_under_target_rows_but_caps_displayed_rows():
    videos = [_video(f"video-{i:02}", float(i + 1)) for i in range(12)]

    result = average_view_watchlist(videos, {"average_view_gate_pct": 70.0}, limit=5)

    assert result["evaluated"] == 12
    assert result["below_target"] == 12
    assert result["shown"] == 5
    assert len(result["videos"]) == 5
    assert result["videos"][0]["id"] == "video-00"


def _render_dashboard(monkeypatch, tmp_path, history, summary):
    cfg = {
        "channel": {"name": "Archive 13"},
        "analytics": {"average_view_gate_pct": 70.0, "evaluate_after_hours": 48},
    }

    class StubStrategy:
        def __init__(self, _cfg):
            self.state = {"updates": 0}

        def report(self):
            return {}

    monkeypatch.setattr(dashboard, "load_config", lambda: cfg)
    monkeypatch.setattr(dashboard, "read_json", lambda name, default: (
        history if name == "history.json" else summary))
    monkeypatch.setattr(dashboard, "Strategy", StubStrategy)
    monkeypatch.setattr(dashboard, "ROOT", Path(tmp_path))
    dashboard.main()
    return (Path(tmp_path) / "dashboard.html").read_text(encoding="utf-8")


def test_dashboard_shows_under_target_details_and_escapes_video_text(monkeypatch, tmp_path):
    watchlist = average_view_watchlist([_video(
        "abc_123", 42.3, views=1200, engaged=700, title="<script>alert('x')</script>")],
        {"average_view_gate_pct": 70.0})
    summary = {
        "updated": "2026-10-11T16:48:01+00:00",
        "average_view_target": {
            "n": 1, "passing": 0, "gate_pct": 70.0, "median_avg_view_pct": 42.3,
        },
        "average_view_watchlist": watchlist,
    }

    page = _render_dashboard(monkeypatch, tmp_path, [], summary)

    assert "Videos below the 70% average-view target" in page
    assert "1 of 1 measured videos are below 70%" in page
    assert "Engaged views (90d)" in page
    assert "https://www.youtube.com/watch?v=abc_123" in page
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in page
    assert "<script>alert('x')</script>" not in page


def test_dashboard_falls_back_to_existing_history_snapshot(monkeypatch, tmp_path):
    video = _video("old-snapshot", 55.0, title="Older under-target video")
    video["metrics"]["hours_live"] = 72
    summary = {
        "average_view_target": {
            "n": 1, "passing": 0, "gate_pct": 70.0, "median_avg_view_pct": 55.0,
        },
    }

    page = _render_dashboard(monkeypatch, tmp_path, [video], summary)

    assert "Older under-target video" in page
    assert "1 of 1 measured videos are below 70%" in page
