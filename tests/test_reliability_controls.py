import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_budgets_enforce_hard_caps_and_safe_minima():
    from autotube.recovery import RecoveryBudgets

    capped = RecoveryBudgets.from_config({
        "content": {"max_drafts": 1000, "selection_rounds": 500,
                    "max_topic_attempts_per_video": 99},
        "qa": {"max_repairs": 10_000},
    })
    assert capped.report() == {
        "script_drafts": 4, "visual_repairs": 2, "selection_rounds": 3,
        "topics_per_video": 6,
    }
    minimums = RecoveryBudgets.from_config({
        "content": {"max_drafts": 0, "selection_rounds": -1,
                    "max_topic_attempts_per_video": 0},
        "qa": {"max_repairs": -1},
    })
    assert minimums.report() == {
        "script_drafts": 1, "visual_repairs": 0, "selection_rounds": 1,
        "topics_per_video": 1,
    }


def test_provider_cooldowns_persist_expire_and_bound_repeated_5xx(monkeypatch):
    from autotube import common
    from autotube.provider_health import ProviderCircuitBreakers

    now = [datetime(2026, 10, 13, 12, tzinfo=timezone.utc)]
    persisted = {}
    monkeypatch.setattr(common, "read_json", lambda name, default: persisted.get(name, default))
    monkeypatch.setattr(common, "write_json", lambda name, data: persisted.update({name: data}))
    circuits = ProviderCircuitBreakers(clock=lambda: now[0])

    until = circuits.open("groq", 429, "rate_limited", 5, 60)
    assert circuits.is_open("groq")
    assert datetime.fromisoformat(until) - now[0] == timedelta(seconds=60)
    assert persisted["provider_cooldowns.json"]["providers"]["groq"]["status_code"] == 429

    assert circuits.note_5xx("cerebras", 503) is False
    assert circuits.note_5xx("cerebras", 503) is False
    restarted = ProviderCircuitBreakers(clock=lambda: now[0])
    assert restarted.note_5xx("cerebras", 503) is True
    assert restarted.is_open("cerebras")
    circuits = restarted

    now[0] += timedelta(seconds=61)
    assert circuits.is_open("groq") is False
    assert "groq" not in persisted["provider_cooldowns.json"]["providers"]


def test_provider_circuit_persistence_failure_does_not_break_routing(monkeypatch):
    from autotube import common
    from autotube.provider_health import ProviderCircuitBreakers

    monkeypatch.setattr(common, "read_json", lambda name, default: {})
    monkeypatch.setattr(common, "write_json", lambda *_: (_ for _ in ()).throw(OSError("read only")))
    circuits = ProviderCircuitBreakers(clock=lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))
    circuits.open("gemini", 402, "billing_or_entitlement", 3600)
    assert circuits.is_open("gemini")


def test_retry_after_http_date_and_transient_waits_are_honored(monkeypatch, memory_provider_circuits):
    from email.utils import format_datetime
    from autotube import llm

    now = datetime.now(timezone.utc)
    date_header = format_datetime(now + timedelta(seconds=20), usegmt=True)
    delay = llm._retry_after_header(date_header)
    assert delay is not None and 0 <= delay <= 21
    assert llm._retry_after_header("not a date") is None
    assert llm._retry_after_header("1m12.5s") == 72.5
    assert llm._retry_after_header("250ms") == 0.25

    calls, slept = [], []

    def transient(model, system, user, temp):
        calls.append(model)
        raise llm.LLMError("HTTP 503: unavailable", status_code=503, retry_after_seconds=9)

    monkeypatch.setitem(llm.PROVIDERS, "fake", (transient, "fake_models"))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: slept.append(seconds))
    client = llm.LLM.__new__(llm.LLM)
    client.cfg = {"fake_models": ["one"]}
    client.order = ["fake"]
    client.temperature = 0.7
    client._dead = set()
    client._blocked_providers = set()
    client.circuits = memory_provider_circuits()
    client.last_used = None

    with pytest.raises(llm.LLMError):
        client.json("system", "user", attempts_per_model=2)
    assert calls == ["one", "one"]
    assert slept == [9]


def test_long_retry_after_opens_provider_circuit_instead_of_retrying_early(
        monkeypatch, memory_provider_circuits):
    from autotube import llm

    calls, slept = [], []

    def limited(model, system, user, temp):
        calls.append(model)
        raise llm.LLMError("HTTP 429: limited", status_code=429, retry_after_seconds=70)

    monkeypatch.setitem(llm.PROVIDERS, "fake", (limited, "fake_models"))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: slept.append(seconds))
    client = llm.LLM.__new__(llm.LLM)
    client.cfg = {"fake_models": ["one", "two"]}
    client.order = ["fake"]
    client.temperature = 0.7
    client._dead = set()
    client._blocked_providers = set()
    client.circuits = memory_provider_circuits()

    with pytest.raises(llm.LLMError):
        client.json("system", "user", attempts_per_model=2)
    assert calls == ["one"]
    assert slept == []
    assert client.circuits.is_open("fake")


def test_failure_classification_preserves_diagnostics_without_raw_errors():
    from autotube.run_report import classify_failure

    assert classify_failure("script_draft", "all models failed: HTTP 402 payment needed") == "billing_or_entitlement"
    assert classify_failure("script_draft", "HTTP 429: quota exceeded") == "provider_rate_limited"
    assert classify_failure("script_draft", "HTTP 503: unavailable") == "provider_transient_5xx"
    assert classify_failure("script_draft", "HTTP 403: forbidden") == "provider_auth_or_permission"
    assert classify_failure("script_draft", "ReadTimeout") == "provider_transient_network"


def test_text_router_stops_when_all_configured_provider_circuits_are_open(
        memory_provider_circuits):
    from autotube.llm import LLM

    router = LLM({"llm": {"providers": ["pollinations"],
                           "pollinations_models": ["openai-fast"]}})
    assert router.has_available_provider() is True
    router.circuits.open("pollinations", 429, "rate_limited", 60, 60)
    assert router.has_available_provider() is False


def test_run_receipt_records_zero_output_and_never_persists_raw_exception(tmp_path, monkeypatch):
    from autotube import common, run_report

    monkeypatch.setattr(common, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(common, "STATE_DIR", tmp_path / "state")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    receipt = run_report.RunReceipt("test-123", 2, upload=True)
    token = run_report.activate(receipt)
    try:
        receipt.set_stage("script_review")
        receipt.note_failure("script_review", "script_quality_rejected", exception_type="LLMError", attempt=4)
        receipt.note_topic("rejected", category="script_quality_rejected")
        receipt.finish(0, 2, 0, 0)
        artifact = receipt.persist()
    finally:
        run_report.deactivate(token)

    row = json.loads(artifact.read_text(encoding="utf-8"))
    assert row["target_count"] == 2
    assert row["produced_count"] == 0
    assert row["outcome"] == "no_output"
    assert row["stage"] == "script_review"
    assert json.loads((tmp_path / "state" / "run_reports.json").read_text())[0] == row

    failed = run_report.RunReceipt("test-456", 1, upload=True)
    failed.set_stage("script_draft")
    failed.fail(RuntimeError("provider response included api_key=SHOULD_NOT_LEAK"))
    failed_artifact = failed.persist()
    text = failed_artifact.read_text(encoding="utf-8")
    assert "SHOULD_NOT_LEAK" not in text
    assert "failed" in text


def test_cli_persists_receipt_when_pipeline_returns_no_video(tmp_path, monkeypatch):
    from autotube import common, pipeline, run_report
    from autotube import __main__ as cli

    monkeypatch.setattr(common, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(common, "STATE_DIR", tmp_path / "state")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda _path=None: {
        "upload": {"enabled": True}, "schedule": {"videos_per_day": 1},
    })
    monkeypatch.setattr(cli, "setup_logging", lambda *_: None)
    monkeypatch.setattr(cli, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_write_github_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "now_utc", lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))
    observed = []

    def no_video(*args, **kwargs):
        observed.append(run_report.current_receipt())
        return []

    monkeypatch.setattr(pipeline, "run", no_video)
    import autotube.llm as llm
    monkeypatch.setattr(llm, "rate_limited", lambda: set())

    assert cli.main(["run", "-n", "1", "--dry-run"]) == 1
    assert observed and observed[0] is not None
    reports = list((tmp_path / "output").glob("run-report-*.json"))
    assert len(reports) == 1
    row = json.loads(reports[0].read_text())
    assert row["target_count"] == 1 and row["produced_count"] == 0
    assert row["outcome"] == "no_output"


def test_cli_persists_failure_receipt_without_raw_exception(tmp_path, monkeypatch):
    from autotube import common, pipeline
    from autotube import __main__ as cli

    monkeypatch.setattr(common, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(common, "STATE_DIR", tmp_path / "state")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda _path=None: {
        "upload": {"enabled": True}, "schedule": {"videos_per_day": 1},
    })
    monkeypatch.setattr(cli, "setup_logging", lambda *_: None)
    monkeypatch.setattr(cli, "notify", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "now_utc", lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))

    def fail(*args, **kwargs):
        raise RuntimeError("provider body carried token=DO_NOT_STORE")

    monkeypatch.setattr(pipeline, "run", fail)
    assert cli.main(["run", "-n", "1"]) == 1
    report = next((tmp_path / "output").glob("run-report-*.json"))
    text = report.read_text(encoding="utf-8")
    row = json.loads(text)
    assert row["outcome"] == "failed"
    assert row["failures"][-1]["category"] == "pipeline_error"
    assert "DO_NOT_STORE" not in text


def test_top_up_noop_still_gets_an_already_satisfied_receipt(tmp_path, monkeypatch):
    from autotube import common, pipeline
    from autotube import __main__ as cli

    monkeypatch.setattr(common, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(common, "STATE_DIR", tmp_path / "state")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda _path=None: {
        "upload": {"enabled": True}, "schedule": {"videos_per_day": 1},
    })
    monkeypatch.setattr(cli, "setup_logging", lambda *_: None)
    monkeypatch.setattr(pipeline, "remaining_today", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(pipeline, "now_utc", lambda: datetime(2026, 10, 13, tzinfo=timezone.utc))

    assert cli.main(["run", "--top-up"]) == 0
    report = next((tmp_path / "output").glob("run-report-*.json"))
    row = json.loads(report.read_text())
    assert row["outcome"] == "already_satisfied"
    assert row["target_count"] == 0 and row["produced_count"] == 0


def test_actions_fallback_creates_one_receipt_for_preflight_noop(tmp_path, monkeypatch):
    path = ROOT / "scripts" / "fallback_run_report.py"
    spec = importlib.util.spec_from_file_location("fallback_run_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(module, "STATE", tmp_path / "state")
    monkeypatch.setenv("GITHUB_RUN_ID", "98765")
    monkeypatch.setenv("AUTOTUBE_NEEDED", "0")
    monkeypatch.setenv("AUTOTUBE_NEED_OUTCOME", "success")
    monkeypatch.setenv("AUTOTUBE_INSTALL_OUTCOME", "skipped")
    monkeypatch.setenv("AUTOTUBE_PRODUCTION_OUTCOME", "skipped")

    assert module.main() == 0
    assert module.main() == 0
    report = tmp_path / "output" / "run-report-98765.json"
    row = json.loads(report.read_text())
    assert row["outcome"] == "already_satisfied"
    assert row["target_count"] == 0 and row["target_count_known"] is True
    rows = json.loads((tmp_path / "state" / "run_reports.json").read_text())
    assert len(rows) == 1


def test_workflow_artifact_keeps_receipts_even_if_video_production_is_skipped():
    workflow = (ROOT / ".github" / "workflows" / "daily.yml").read_text()
    assert "Ensure every Actions run has a production receipt" in workflow
    assert "if: always()" in workflow
    assert "fallback_run_report.py" in workflow
    assert "actions/upload-artifact@v4" in workflow
