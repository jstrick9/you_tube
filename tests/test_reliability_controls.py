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
    row = client.circuits.snapshot()["fake"]
    assert row["reason"] == "rate_limited"
    assert row["cooldown_seconds"] == llm.DEFAULT_429_COOLDOWN_SECONDS


def test_repeated_generic_429_uses_two_hour_circuit(monkeypatch, memory_provider_circuits):
    from autotube import llm

    calls, slept = [], []

    def limited(model, system, user, temp):
        calls.append(model)
        raise llm.LLMError("HTTP 429: rate limit reached", status_code=429,
                           retry_after_seconds=27)

    monkeypatch.setitem(llm.PROVIDERS, "fake", (limited, "fake_models"))
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: slept.append(seconds))
    client = llm.LLM.__new__(llm.LLM)
    client.cfg = {"fake_models": ["one"]}
    client.order = ["fake"]
    client.temperature = 0.7
    client._dead = set()
    client._blocked_providers = set()
    client.circuits = memory_provider_circuits()

    with pytest.raises(llm.LLMError):
        client.json("system", "user", attempts_per_model=2)
    assert calls == ["one", "one"]
    assert slept == [27]
    row = client.circuits.snapshot()["fake"]
    assert row["reason"] == "rate_limited"
    assert row["cooldown_seconds"] == 2 * 60 * 60


def test_explicit_quota_exhaustion_uses_day_long_circuit_without_retry(
        monkeypatch, memory_provider_circuits):
    from autotube import llm

    calls = []

    def exhausted(model, system, user, temp):
        calls.append(model)
        raise llm.LLMError(
            'HTTP 429: {"message":"You exceeded your current quota, please check your plan and billing details."}',
            status_code=429, retry_after_seconds=27)

    monkeypatch.setitem(llm.PROVIDERS, "fake", (exhausted, "fake_models"))
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
    row = client.circuits.snapshot()["fake"]
    assert row["reason"] == "quota_exhausted"
    assert row["cooldown_seconds"] == llm.DEFAULT_QUOTA_COOLDOWN_SECONDS == 24 * 60 * 60


def test_model_request_too_large_tries_next_model_without_provider_cooldown(
        monkeypatch, memory_provider_circuits):
    from autotube import llm

    calls = []

    def models(model, system, user, temp):
        calls.append(model)
        if model == "small-context":
            raise llm.LLMError("HTTP 429: Request too large for model; submit a smaller request",
                               status_code=429, retry_after_seconds=0.001)
        return '{"ok": true}'

    monkeypatch.setitem(llm.PROVIDERS, "fake", (models, "fake_models"))
    client = llm.LLM.__new__(llm.LLM)
    client.cfg = {"fake_models": ["small-context", "fallback"]}
    client.order = ["fake"]
    client.temperature = 0.7
    client._dead = set()
    client._blocked_providers = set()
    client.circuits = memory_provider_circuits()
    client.last_used = None
    before_limited = llm.rate_limited()

    assert client.json("system", "user") == {"ok": True}
    assert calls == ["small-context", "fallback"]
    assert client.last_used == "fake:fallback"
    assert not client.circuits.is_open("fake")
    assert llm.rate_limited() == before_limited


def test_failure_classification_preserves_diagnostics_without_raw_errors():
    from autotube.run_report import classify_failure

    assert classify_failure("script_draft", "all models failed: HTTP 402 payment needed") == "billing_or_entitlement"
    assert classify_failure("script_draft", "HTTP 429: quota exceeded") == "provider_rate_limited"
    assert classify_failure("script_draft", "HTTP 503: unavailable") == "provider_transient_5xx"
    assert classify_failure("script_draft", "HTTP 403: forbidden") == "provider_auth_or_permission"
    assert classify_failure("script_draft", "ReadTimeout") == "provider_transient_network"


def test_provider_read_timeout_is_skipped_for_rest_of_run(monkeypatch, memory_provider_circuits):
    """A dead endpoint must not consume two long request timeouts for every topic."""
    import requests
    import autotube.llm as L

    calls = []

    def slow(_model, _system, _user, _temperature):
        calls.append("slow")
        raise requests.exceptions.ReadTimeout("provider did not respond")

    def healthy(_model, _system, _user, _temperature):
        calls.append("healthy")
        return '{"ok": true}'

    monkeypatch.setitem(L.PROVIDERS, "slow", (slow, "slow_models"))
    monkeypatch.setitem(L.PROVIDERS, "healthy", (healthy, "healthy_models"))
    router = L.LLM({"llm": {"providers": ["slow", "healthy"],
                              "slow_models": ["model-a", "model-b"],
                              "healthy_models": ["model-c"]}})

    assert router.json("system", "user") == {"ok": True}
    assert calls == ["slow", "healthy"]
    assert "slow" in router._blocked_text_providers
    assert "slow" not in router._blocked_providers
    assert not router.circuits.is_open("slow"), "one-run network failures must not persist cooldowns"


def test_text_network_timeout_does_not_disable_configured_vision_route(monkeypatch, memory_provider_circuits):
    import requests
    import autotube.llm as L

    calls = []

    def dual_route(model, _system, _user, _temperature, images=None):
        calls.append((model, images))
        if images is None:
            raise requests.exceptions.ReadTimeout("text endpoint did not respond")
        return '{"color":"red"}'

    monkeypatch.setitem(L.PROVIDERS, "fake", (dual_route, "fake_models"))
    monkeypatch.setattr(L, "_provider_key", lambda _provider: "test-key")
    router = L.LLM.__new__(L.LLM)
    router.cfg = {"fake_models": ["text-model"], "vision_providers": ["fake"],
                  "fake_vision_models": ["vision-model"]}
    router.order = ["fake"]
    router.temperature = 0.7
    router._dead = set()
    router._blocked_providers = set()
    router._blocked_text_providers = set()
    router._blocked_vision_providers = set()
    router.circuits = memory_provider_circuits()
    router.last_used = ""

    with pytest.raises(L.LLMError, match="All LLM providers failed"):
        router.json("system", "text request")
    assert "fake" in router._blocked_text_providers
    assert "fake" not in router._blocked_providers
    assert router.vision_available(), "a text timeout alone must not retire the image route"

    def require_red(value):
        assert value.get("color") == "red"

    assert router.vision_json("vision system", "inspect image", [b"jpeg"], validate=require_red) == {"color": "red"}
    assert calls == [("text-model", None), ("vision-model", [b"jpeg"])]


def test_repeated_text_5xx_circuit_does_not_disable_vision_route(monkeypatch, memory_provider_circuits):
    import autotube.llm as L

    def split_routes(_model, _system, _user, _temperature, images=None):
        if images is None:
            raise L.LLMError("HTTP 503: text endpoint unavailable", status_code=503)
        return '{"color":"red"}'

    monkeypatch.setitem(L.PROVIDERS, "fake", (split_routes, "fake_models"))
    monkeypatch.setattr(L, "_provider_key", lambda _provider: "test-key")
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)
    router = L.LLM.__new__(L.LLM)
    router.cfg = {"fake_models": ["text-model"], "vision_providers": ["fake"],
                  "fake_vision_models": ["vision-model"]}
    router.order = ["fake"]
    router.temperature = 0.7
    router._dead = set()
    router._blocked_providers = set()
    router._blocked_text_providers = set()
    router._blocked_vision_providers = set()
    router.circuits = memory_provider_circuits()
    router.last_used = ""

    for _ in range(3):
        with pytest.raises(L.LLMError, match="All LLM providers failed"):
            router.json("system", "text request", attempts_per_model=1)
    assert router.circuits.is_open("fake")
    assert "fake" in router._blocked_text_providers
    assert router.vision_available(), "a repeated text 5xx circuit must not suppress image QA"
    assert router.vision_json("vision system", "inspect image", [b"jpeg"]) == {"color": "red"}

    router.circuits.open("fake", 429, "rate_limited", 60)
    assert not router.vision_available(), "provider-wide quota circuits still apply to vision"


def test_vision_schema_error_gets_one_same_model_correction(monkeypatch, memory_provider_circuits):
    import autotube.llm as L

    calls = []
    valid = {"images": [{"n": 1, "shows": "a specific object", "score": 8, "legible": True}]}

    def provider(_model, _system, user, temperature, images=None):
        calls.append((user, temperature, images))
        return '{"images":[]}' if len(calls) == 1 else json.dumps(valid)

    def validate(value):
        if not isinstance(value.get("images"), list) or len(value["images"]) != 1:
            raise ValueError("vision response needs exactly 1 image verdicts")

    monkeypatch.setitem(L.PROVIDERS, "fake", (provider, "fake_models"))
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)
    router = L.LLM.__new__(L.LLM)
    router.cfg = {"vision_providers": ["fake"], "fake_vision_models": ["vision-model"]}
    router.order = ["fake"]
    router.temperature = 0.7
    router._dead = set()
    router._blocked_providers = set()
    router._blocked_text_providers = set()
    router._blocked_vision_providers = set()
    router.circuits = memory_provider_circuits()
    router.last_used = ""

    assert router.vision_json("vision system", "return one verdict", [b"jpeg"], validate=validate) == valid
    assert len(calls) == 2
    assert "VALIDATION FAILURE: vision response needs exactly 1 image verdicts" in calls[1][0]


def test_validation_retry_repeats_schema_and_lowers_temperature(monkeypatch, memory_provider_circuits):
    """A correction attempt should make structural requirements easier to follow."""
    import autotube.llm as L

    calls = []

    def draft(_model, _system, user, temperature):
        calls.append((user, temperature))
        if len(calls) == 1:
            return '{"title":"Draft","segments":[]}'
        return '{"title":"Draft","segments":[' + ','.join(
            '{"text":"Fact","evidence":"source quote"}' for _ in range(4)) + ']}'

    def validate(obj):
        segments = obj.get("segments")
        assert isinstance(segments, list) and 4 <= len(segments) <= 5, "need exactly 4 or 5 segments"
        assert all(isinstance(item.get("evidence"), str) for item in segments), \
            "every segment must include textual evidence"

    monkeypatch.setitem(L.PROVIDERS, "schema", (draft, "schema_models"))
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)
    router = L.LLM.__new__(L.LLM)
    router.cfg = {"schema_models": ["model"]}
    router.order = ["schema"]
    router.temperature = 0.8
    router._dead = set()
    router._blocked_providers = set()
    router.circuits = memory_provider_circuits()
    router.last_used = ""

    assert router.json("system", "original request", validate=validate) == {
        "title": "Draft", "segments": [{"text": "Fact", "evidence": "source quote"}] * 4,
    }
    assert len(calls) == 2, "schema correction stays within the bounded two-call model budget"
    retry_prompt, retry_temperature = calls[1]
    assert "VALIDATION FAILURE: need exactly 4 or 5 segments" in retry_prompt
    assert "preserve every required key" in retry_prompt
    assert "string evidence field" in retry_prompt
    assert "Prefer exactly five segment objects in this order: hook, turn, escalation, escalation, payoff" in retry_prompt
    assert "complete, non-null visual object" in retry_prompt
    assert calls[0][1] == 0.8 and retry_temperature == 0.3


def test_word_count_validation_retry_gives_a_targeted_repair_instruction(monkeypatch, memory_provider_circuits):
    import autotube.llm as L

    calls = []
    responses = ['{"text":"The turn arrives too late."}',
                 '{"text":"A specific source-backed consequence changes the whole story for everyone."}']

    def draft(_model, _system, user, _temperature):
        calls.append(user)
        return responses[len(calls) - 1]

    def validate(value):
        count = len(value["text"].split())
        assert 9 <= count <= 11, (
            f"word-count repair required: segment 2 turn is {count} words; it must be 9-11 factual words")

    monkeypatch.setitem(L.PROVIDERS, "wordcheck", (draft, "wordcheck_models"))
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)
    router = L.LLM.__new__(L.LLM)
    router.cfg = {"wordcheck_models": ["model"]}
    router.order = ["wordcheck"]
    router.temperature = 0.8
    router._dead = set()
    router._blocked_providers = set()
    router.circuits = memory_provider_circuits()
    router.last_used = ""

    result = router.json("system", "write a turn", validate=validate)
    assert result["text"] == "A specific source-backed consequence changes the whole story for everyone."
    assert "VALIDATION FAILURE: word-count repair required" in calls[1]
    assert "count whitespace-separated words in each spoken text" in calls[1]
    assert "preserve already-valid beats" in calls[1]


def test_visual_validation_retry_repeats_the_concrete_subject_requirement(monkeypatch, memory_provider_circuits):
    import autotube.llm as L

    calls = []

    def draft(_model, _system, user, temperature):
        calls.append((user, temperature))
        if len(calls) == 1:
            return '{"shows":"a dark cave","queries":["cave wall","shadows"]}'
        return '{"shows":"a stone sphere close-up","queries":["stone sphere photo","stone sphere detail"]}'

    def validate(obj):
        blob = " ".join([obj.get("shows", "")] + obj.get("queries", []))
        assert "stone" in blob.lower() and "sphere" in blob.lower(), \
            "no shot shows 'stone sphere' itself — every visual is ambient scenery"

    monkeypatch.setitem(L.PROVIDERS, "visual", (draft, "visual_models"))
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)
    router = L.LLM.__new__(L.LLM)
    router.cfg = {"visual_models": ["model"]}
    router.order = ["visual"]
    router.temperature = 0.8
    router._dead = set()
    router._blocked_providers = set()
    router.circuits = memory_provider_circuits()
    router.last_used = ""

    assert router.json("system", "draw the stone sphere", validate=validate)["shows"] == \
        "a stone sphere close-up"
    retry_prompt, retry_temperature = calls[1]
    assert "VALIDATION FAILURE: no shot shows 'stone sphere' itself" in retry_prompt
    assert "Every segment must include visual={shows: non-empty string, queries: [2-3 non-empty strings]}" in retry_prompt
    assert "visual.shows must explicitly name the concrete subject itself" in retry_prompt
    assert "must include its distinctive noun(s)" in retry_prompt
    assert calls[0][1] == 0.8 and retry_temperature == 0.3


def test_trend_angle_validation_retry_keeps_story_specific_and_source_bound(monkeypatch, memory_provider_circuits):
    import json
    import autotube.llm as L

    calls = []

    def draft(_model, _system, user, temperature):
        calls.append((user, temperature))
        text = "The Nintendo DS sold millions of units." if len(calls) == 1 else \
            "The Nintendo DS could not render 3D, so developers faked it with flat images."
        return json.dumps({"text": text})

    def validate(obj):
        assert "faked" in obj["text"], \
            "this is not the story that trended. The angle was: the DS faked 3D"

    monkeypatch.setitem(L.PROVIDERS, "angle", (draft, "angle_models"))
    monkeypatch.setattr(L.time, "sleep", lambda *_: None)
    router = L.LLM.__new__(L.LLM)
    router.cfg = {"angle_models": ["model"]}
    router.order = ["angle"]
    router.temperature = 0.8
    router._dead = set()
    router._blocked_providers = set()
    router.circuits = memory_provider_circuits()
    router.last_used = ""

    result = router.json("system", "SOURCE TEXT: the effect used flat images", validate=validate)
    assert "faked" in result["text"]
    retry_prompt, retry_temperature = calls[1]
    assert "The selected current-trend angle is the required story" in retry_prompt
    assert "make its specific claim and distinctive details the through-line" in retry_prompt
    assert "Use only facts supported by SOURCE TEXT" in retry_prompt
    assert "never invent facts to match an unsupported angle" in retry_prompt
    assert calls[0][1] == 0.8 and retry_temperature == 0.3


def test_text_router_stops_when_all_configured_provider_circuits_are_open(
        memory_provider_circuits):
    from autotube.llm import LLM

    router = LLM({"llm": {"providers": ["pollinations"],
                           "pollinations_models": ["openai-fast"]}})
    assert router.has_available_provider() is True
    router.circuits.open("pollinations", 429, "rate_limited", 60, 60)
    assert router.has_available_provider() is False


def test_live_vision_model_is_kept_as_a_cloudflare_text_fallback():
    from autotube.common import load_config

    models = load_config()["llm"]["cloudflare_models"]
    assert models[:2] == ["@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                          "@cf/meta/llama-4-scout-17b-16e-instruct"]


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
        receipt.note_grounding_repair("Kola Borehole", "Kola Superdeep Borehole",
                                      "180 degrees, nearly twice model prediction",
                                      "180 °C, higher than expected",
                                      "higher-than-expected temperatures of 180 °C")
        receipt.note_grounding_repair("example", "source", "original", "corrected",
                                      "API_KEY=SHOULD_NOT_LEAK github_pat_1234567890123456")
        receipt.finish(0, 2, 0, 0)
        artifact = receipt.persist()
    finally:
        run_report.deactivate(token)

    row = json.loads(artifact.read_text(encoding="utf-8"))
    assert row["target_count"] == 2
    assert row["produced_count"] == 0
    assert row["outcome"] == "no_output"
    assert row["stage"] == "script_review"
    assert row["grounding_repairs"][0]["source_quote"] == "higher-than-expected temperatures of 180 °C"
    assert "SHOULD_NOT_LEAK" not in artifact.read_text(encoding="utf-8")
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
    monkeypatch.setenv("AUTOTUBE_PROCEED", "false")
    monkeypatch.setenv("AUTOTUBE_SKIP_REASON", "")
    monkeypatch.setenv("AUTOTUBE_NEED_OUTCOME", "success")
    monkeypatch.setenv("AUTOTUBE_INSTALL_OUTCOME", "skipped")
    monkeypatch.setenv("AUTOTUBE_PRODUCTION_OUTCOME", "skipped")

    assert module.main() == 0
    assert module.main() == 0
    report = tmp_path / "output" / "run-report-98765.json"
    row = json.loads(report.read_text())
    assert row["outcome"] == "already_satisfied"
    assert row["target_count"] == 0 and row["target_count_known"] is True
    assert "preflight_skip_reason" not in row
    rows = json.loads((tmp_path / "state" / "run_reports.json").read_text())
    assert len(rows) == 1


def test_delayed_schedule_skip_preserves_remaining_count_and_is_not_already_satisfied(tmp_path, monkeypatch):
    path = ROOT / "scripts" / "fallback_run_report.py"
    spec = importlib.util.spec_from_file_location("fallback_run_report_superseded", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(module, "STATE", tmp_path / "state")
    monkeypatch.setenv("GITHUB_RUN_ID", "late-cron")
    monkeypatch.setenv("AUTOTUBE_NEEDED", "2")
    monkeypatch.setenv("AUTOTUBE_PROCEED", "false")
    monkeypatch.setenv("AUTOTUBE_SKIP_REASON", "delayed_schedule_superseded")
    monkeypatch.setenv("AUTOTUBE_NEED_OUTCOME", "success")
    monkeypatch.setenv("AUTOTUBE_INSTALL_OUTCOME", "skipped")
    monkeypatch.setenv("AUTOTUBE_PRODUCTION_OUTCOME", "skipped")

    assert module.main() == 0
    row = json.loads((tmp_path / "output" / "run-report-late-cron.json").read_text())
    assert row["outcome"] == "superseded"
    assert row["target_count"] == 2 and row["target_count_known"] is True
    assert row["preflight_skip_reason"] == "delayed_schedule_superseded"
    assert row["produced_count"] == 0 and row["topic_attempts"] == []


def test_workflow_artifact_keeps_receipts_even_if_video_production_is_skipped():

    workflow = (ROOT / ".github" / "workflows" / "daily.yml").read_text()
    assert "Ensure every Actions run has a production receipt" in workflow
    assert "if: always()" in workflow
    assert "fallback_run_report.py" in workflow
    assert "actions/upload-artifact@v4" in workflow
