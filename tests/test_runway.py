"""Runway Dev adapter tests: all API calls are mocked; no paid generation is made."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube import common, runway  # noqa: E402


class JsonResponse:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status
        self.headers = {}

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        pass


class StreamResponse(JsonResponse):
    def __init__(self, data=b"mock video bytes", status=200):
        super().__init__({}, status)
        self.data = data

    def iter_content(self, size):
        yield self.data


class FakeRunwaySession:
    def __init__(self):
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url, headers, json, timeout))
        return JsonResponse({"id": "task-123", "estimatedCost": {"credits": 60}})

    def get(self, url, headers=None, timeout=None, stream=False):
        self.calls.append(("GET", url, headers, timeout, stream))
        if "/v1/tasks/task-123" in url:
            return JsonResponse({"id": "task-123", "status": "SUCCEEDED",
                                 "output": ["https://outputs.runway.test/task-123.mp4"],
                                 "cost": {"credits": 60}})
        if url == "https://outputs.runway.test/task-123.mp4":
            return StreamResponse()
        raise AssertionError(f"unexpected URL: {url}")


def _terms_ready_cfg():
    cfg = common.load_config()
    settings = cfg["media"]["runway_dev"]
    settings["enabled"] = True
    settings["account_terms_confirmed"] = True
    settings["account_terms_reviewed_on"] = "2026-10-13"
    settings["api_schema_confirmed"] = True
    settings["api_schema_reviewed_on"] = "2026-10-13"
    settings["pricing_confirmed"] = True
    settings["pricing_reviewed_on"] = "2026-10-13"
    settings["hosted_veo_terms_confirmed"] = False
    settings["hosted_veo_terms_reviewed_on"] = ""
    return cfg


def test_default_config_keeps_all_runway_confirmations_and_global_synthetic_flag_false():
    cfg = common.load_config()
    settings = cfg["media"]["runway_dev"]
    assert settings["enabled"] is True
    for key in ("account_terms_confirmed", "api_schema_confirmed", "pricing_confirmed",
                "hosted_veo_terms_confirmed"):
        assert settings[key] is False
    assert cfg["compliance"]["contains_synthetic_media"] is False


def test_model_payloads_pin_distinct_runway_routes_portrait_size_and_no_audio():
    cfg = _terms_ready_cfg()
    client = runway.RunwayDev(cfg)
    gen = {"id": "gen4.5", "duration": 5, "ratio": "720:1280", "spec": runway.MODEL_SPECS["gen4.5"]}
    veo = {"id": "veo3.1_fast", "duration": 8, "ratio": "1080:1920", "spec": runway.MODEL_SPECS["veo3.1_fast"]}

    assert client._payload(gen, "stylized fossil illustration") == {
        "model": "gen4.5", "promptText": "stylized fossil illustration", "ratio": "720:1280", "duration": 5,
    }
    assert client._payload(veo, "stylized fossil illustration") == {
        "model": "veo3.1_fast", "promptText": "stylized fossil illustration", "ratio": "1080:1920",
        "duration": 8, "audio": False,
    }
    assert set(runway.MODEL_SPECS) == {"gen4.5", "veo3.1_fast"}
    assert runway.API_BASE == "https://api.dev.runwayml.com"
    assert runway.MODEL_SPECS["gen4.5"]["credits_per_second"] == 12
    assert runway.MODEL_SPECS["veo3.1_fast"]["credits_per_second"] == 10


def test_provider_is_dormant_until_api_key_and_account_rights_are_confirmed(tmp_path, monkeypatch):
    cfg = common.load_config()
    monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)

    class NoNetwork:
        def post(self, *args, **kwargs):
            raise AssertionError("dormant adapter must not make a network call")

        def get(self, *args, **kwargs):
            raise AssertionError("dormant adapter must not make a network call")

    client = runway.RunwayDev(cfg, session=NoNetwork(), ledger_path=tmp_path / "ledger.json")
    ready, reason = client.readiness()
    assert ready is False and "RUNWAYML_API_SECRET" in reason
    assert client.generate_candidates("illustration", tmp_path) == []
    assert not (tmp_path / "ledger.json").exists()

    monkeypatch.setenv("RUNWAYML_API_SECRET", "fake-key")
    client = runway.RunwayDev(cfg, session=NoNetwork(), ledger_path=tmp_path / "ledger.json")
    ready, reason = client.readiness()
    assert ready is False and "terms" in reason.lower()
    assert client.generate_candidates("illustration", tmp_path) == []
    assert not (tmp_path / "ledger.json").exists()


def test_api_schema_and_pricing_reviews_are_separate_paid_call_gates(tmp_path, monkeypatch):
    cfg = _terms_ready_cfg()
    monkeypatch.setenv("RUNWAYML_API_SECRET", "unit-test-key")

    class NoNetwork:
        def post(self, *args, **kwargs):
            raise AssertionError("an unconfirmed provider must not make a network call")

        def get(self, *args, **kwargs):
            raise AssertionError("an unconfirmed provider must not make a network call")

    client = runway.RunwayDev(cfg, session=NoNetwork(), ledger_path=tmp_path / "ledger.json")
    settings = cfg["media"]["runway_dev"]
    settings["api_schema_confirmed"] = False
    ready, reason = client.readiness()
    assert ready is False and "schema" in reason.lower()
    assert client.generate_candidates("illustration", tmp_path) == []
    assert not (tmp_path / "ledger.json").exists()

    settings["api_schema_confirmed"] = True
    settings["api_schema_reviewed_on"] = "not-a-date"
    ready, reason = client.readiness()
    assert ready is False and "schema" in reason.lower()
    settings["api_schema_reviewed_on"] = "2026-10-13"
    settings["pricing_confirmed"] = False
    ready, reason = client.readiness()
    assert ready is False and "pricing" in reason.lower()
    settings["pricing_confirmed"] = True
    settings["pricing_reviewed_on"] = "2099-01-01"
    ready, reason = client.readiness()
    assert ready is False and "pricing" in reason.lower()


def test_runway_hosted_veo_needs_its_own_terms_confirmation(tmp_path, monkeypatch):
    cfg = _terms_ready_cfg()
    monkeypatch.setenv("RUNWAYML_API_SECRET", "fake-key")
    client = runway.RunwayDev(cfg, ledger_path=tmp_path / "ledger.json")

    active = client._active_models()
    assert [item["id"] for item in active] == ["gen4.5"]

    cfg["media"]["runway_dev"]["hosted_veo_terms_confirmed"] = True
    cfg["media"]["runway_dev"]["hosted_veo_terms_reviewed_on"] = "2026-10-13"
    client = runway.RunwayDev(cfg, ledger_path=tmp_path / "ledger.json")
    assert [item["id"] for item in client._active_models()] == ["gen4.5", "veo3.1_fast"]


def test_daily_budget_is_reserved_before_submission_and_never_overruns(tmp_path):
    cfg = _terms_ready_cfg()
    settings = cfg["media"]["runway_dev"]
    settings["daily_credit_budget"] = 60
    client = runway.RunwayDev(cfg, ledger_path=tmp_path / "ledger.json",
                              clock=lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))

    ticket = client._reserve("gen4.5", 60, "safe illustration")
    assert ticket == ("2026-10-13", 0)
    assert client._reserve("gen4.5", 1, "second request") is None
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    day = ledger["days"]["2026-10-13"]
    assert day["reserved_credits"] == 60
    assert day["tasks"][0]["status"] == "reserved"
    assert len(day["tasks"]) == 1


def test_configured_credit_cap_is_the_amount_reserved_before_task_submission(tmp_path, monkeypatch):
    cfg = _terms_ready_cfg()
    cfg["media"]["runway_dev"]["models"] = [{
        "id": "gen4.5", "enabled": True, "duration": 5,
        "ratio": "720:1280", "max_credits": 75,
    }]
    monkeypatch.setenv("RUNWAYML_API_SECRET", "unit-test-key")
    client = runway.RunwayDev(cfg, ledger_path=tmp_path / "ledger.json",
                              clock=lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))
    client._post_task = lambda payload: (_ for _ in ()).throw(RuntimeError("mocked submit failure"))

    assert client._generate_one(client._active_models()[0], "stylized illustration") is None
    row = json.loads((tmp_path / "ledger.json").read_text())["days"]["2026-10-13"]["tasks"][0]
    assert row["credits_reserved"] == 75
    assert row["status"] == "failed"


def test_corrupt_credit_ledger_fails_closed_instead_of_resetting_the_daily_cap(tmp_path):
    cfg = _terms_ready_cfg()
    path = tmp_path / "ledger.json"
    path.write_text("not valid json", encoding="utf-8")
    client = runway.RunwayDev(cfg, ledger_path=path,
                              clock=lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))

    assert client._reserve("gen4.5", 60, "illustration") is None
    assert path.read_text(encoding="utf-8") == "not valid json"


def test_actual_cost_overrun_is_accounted_once_and_blocks_further_reservations(tmp_path):
    cfg = _terms_ready_cfg()
    cfg["media"]["runway_dev"]["daily_credit_budget"] = 60
    client = runway.RunwayDev(cfg, ledger_path=tmp_path / "ledger.json",
                              clock=lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))

    ticket = client._reserve("gen4.5", 60, "safe illustration")
    client._update_reservation(ticket, status="succeeded", actual_credits=70)
    client._update_reservation(ticket, actual_credits=70)  # final provenance update must not double-count

    ledger = json.loads((tmp_path / "ledger.json").read_text())
    day = ledger["days"]["2026-10-13"]
    assert day["reserved_credits"] == 70
    assert day["tasks"][0]["budget_overrun_credits"] == 10
    assert client._reserve("gen4.5", 1, "next request") is None


def test_mocked_gen4_5_generation_records_cost_hash_rights_and_never_persists_signed_url(
        tmp_path, monkeypatch):
    cfg = _terms_ready_cfg()
    cfg["media"]["runway_dev"]["models"] = [{
        "id": "gen4.5", "enabled": True, "duration": 5,
        "ratio": "720:1280", "max_credits": 60,
    }]
    monkeypatch.setenv("RUNWAYML_API_SECRET", "unit-test-key")
    session = FakeRunwaySession()
    client = runway.RunwayDev(cfg, session=session, ledger_path=tmp_path / "runway_usage.json",
                              clock=lambda: datetime(2026, 10, 13, tzinfo=timezone.utc), sleep=lambda _: None)
    monkeypatch.setattr(runway, "probe_video", lambda path: (720, 1280, 5.0))
    monkeypatch.setattr(runway, "ffmpeg_bin", lambda: "ffmpeg-mock")

    def fake_poster(cmd, **kwargs):
        Image.new("RGB", (720, 1280), (20, 80, 120)).save(cmd[-1], "JPEG")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(runway.subprocess, "run", fake_poster)
    candidates = client.generate_candidates("A clearly stylized fossil illustration", tmp_path / "work")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["synthetic"] is True and candidate["model"] == "gen4.5"
    assert candidate["task_id"] == "task-123" and len(candidate["file_sha256"]) == 64
    assert candidate["ratio"] == "720:1280" and candidate["actual_duration_seconds"] == 5.0
    assert candidate["actual_cost_credits"] == 60 and candidate["credits_reserved"] == 60
    assert candidate["rights_confirmation"]["runway_account_terms_confirmed"] is True

    method, url, headers, body, timeout = session.calls[0]
    assert method == "POST" and url == f"{runway.API_BASE}/v1/text_to_video"
    assert headers["Authorization"] == "Bearer unit-test-key"
    assert headers["X-Runway-Version"] == "2024-11-06"
    assert body["model"] == "gen4.5" and "audio" not in body
    assert body["ratio"] == "720:1280" and body["duration"] == 5
    output_downloads = [call for call in session.calls if call[0] == "GET" and "outputs.runway.test" in call[1]]
    assert output_downloads and output_downloads[0][2] is None, "signed output URLs must not receive the API key"

    ledger = json.loads((tmp_path / "runway_usage.json").read_text())
    row = ledger["days"]["2026-10-13"]["tasks"][0]
    assert row["status"] == "output_validated" and row["actual_credits"] == 60
    assert row["file_sha256"] == candidate["file_sha256"]
    assert "outputs.runway.test" not in json.dumps(ledger), "signed output links must not be persisted"


def test_invalid_provider_cost_estimate_cancels_created_task_and_keeps_reservation(tmp_path, monkeypatch):
    cfg = _terms_ready_cfg()
    cfg["media"]["runway_dev"]["models"] = [{
        "id": "gen4.5", "enabled": True, "duration": 5,
        "ratio": "720:1280", "max_credits": 60,
    }]
    monkeypatch.setenv("RUNWAYML_API_SECRET", "unit-test-key")

    class InvalidEstimateSession:
        def __init__(self):
            self.deleted = []

        def post(self, *args, **kwargs):
            return JsonResponse({"id": "untrusted-estimate-task", "estimatedCost": {"credits": "unknown"}})

        def delete(self, url, **kwargs):
            self.deleted.append(url)
            return JsonResponse({})

    session = InvalidEstimateSession()
    client = runway.RunwayDev(cfg, session=session, ledger_path=tmp_path / "ledger.json",
                              clock=lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))
    assert client.generate_candidates("stylized illustration", tmp_path / "work") == []
    assert session.deleted == [f"{runway.API_BASE}/v1/tasks/untrusted-estimate-task"]
    row = json.loads((tmp_path / "ledger.json").read_text())["days"]["2026-10-13"]["tasks"][0]
    assert row["status"] == "cancelled_invalid_cost_estimate"
    assert row["credits_reserved"] == 60


def test_runway_local_output_is_copied_without_fetching_a_signed_url(tmp_path, monkeypatch):
    from autotube import media

    source = tmp_path / "validated-runway.mp4"
    source.write_bytes(b"already downloaded and probed")
    def no_network():
        raise AssertionError("unexpected network fetch")

    monkeypatch.setattr(media, "http", no_network)
    destination = media._download_video({"_local_path": source, "url": "runway://task/123",
                                         "title": "AI-generated illustration"},
                                       tmp_path / "media", max_mb=1)
    assert destination is not None and destination.read_bytes() == source.read_bytes()


def test_generated_video_and_poster_badges_render_as_visible_pixels(tmp_path):
    import subprocess
    from autotube import render

    ordinary = render.video_chain(0, 9 / 16, 540, 960, 1)
    generated = render.video_chain(0, 9 / 16, 540, 960, 1, synthetic=True)
    assert "AI ILLUSTRATION" not in ordinary and "overlay=0:0" not in ordinary
    assert "overlay=0:0" in generated

    badge = render.ai_illustration_overlay(tmp_path / "badge.png", 540, 960)
    video_badge = tmp_path / "video-badge.png"
    command = [common.ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
               "color=c=blue:s=540x960:r=1:d=1", "-loop", "1", "-i", str(badge),
               "-filter_complex", generated + ",format=rgb24[v]", "-map", "[v]", "-frames:v", "1",
               str(video_badge)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    with Image.open(video_badge) as frame:
        assert sum(frame.convert("RGB").getpixel((40, 70))) < 100
        assert frame.crop((27, 34, 250, 74)).convert("L").point(
            lambda value: 255 if value > 210 else 0).getbbox()

    poster = tmp_path / "poster.jpg"
    Image.new("RGB", (540, 960), (20, 80, 180)).save(poster, "JPEG")
    render.label_ai_illustration(poster)
    with Image.open(poster) as labelled:
        assert sum(labelled.convert("RGB").getpixel((20, 35))) < 150
        assert labelled.crop((27, 34, 250, 74)).convert("L").point(
            lambda value: 255 if value > 210 else 0).getbbox()


def test_youtube_upload_uses_per_video_synthetic_disclosure_flag(tmp_path, monkeypatch):
    import types
    import autotube.youtube as youtube

    errors = types.ModuleType("googleapiclient.errors")
    errors.HttpError = RuntimeError
    http_module = types.ModuleType("googleapiclient.http")
    http_module.MediaFileUpload = lambda *args, **kwargs: (args, kwargs)
    api_module = types.ModuleType("googleapiclient")
    api_module.__path__ = []
    monkeypatch.setitem(sys.modules, "googleapiclient", api_module)
    monkeypatch.setitem(sys.modules, "googleapiclient.errors", errors)
    monkeypatch.setitem(sys.modules, "googleapiclient.http", http_module)

    captured = []

    class Request:
        def next_chunk(self):
            return None, {"id": "test-video"}

    class Videos:
        def insert(self, **kwargs):
            captured.append(kwargs["body"])
            return Request()

    class Service:
        def videos(self):
            return Videos()

    monkeypatch.setattr(youtube, "service", lambda: Service())
    video = tmp_path / "upload.mp4"
    video.write_bytes(b"video")
    cfg = {"compliance": {"made_for_kids": False, "contains_synthetic_media": False},
           "upload": {"category_id": "27", "default_privacy": "public"},
           "channel": {"language": "en"}, "schedule": {"notify_subscribers": False}}
    meta = {"title": "A sourced Short", "description": "description", "tags": [],
            "contains_synthetic_media": True}

    youtube.upload(video, meta, cfg, None)

    assert captured[0]["status"]["containsSyntheticMedia"] is True
    assert captured[0]["status"]["privacyStatus"] == "public"
    meta["contains_synthetic_media"] = False
    youtube.upload(video, meta, cfg, None)
    assert captured[1]["status"]["containsSyntheticMedia"] is False
