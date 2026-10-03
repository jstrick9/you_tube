"""Credentials plumbing and reviewer independence.

Two failure modes here are invisible until a live run and expensive when they land:
a secret that exists but is never passed into the job, and a "reviewer" that turns out
to be the same model that wrote the script.
"""
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
WF = ROOT / ".github" / "workflows"

# Read from the environment but never supplied by CI secrets:
# local TTS paths, config overrides, and the keyless provider's optional token.
LOCAL_ONLY = {"AUTOTUBE_CONFIG", "AUTOTUBE_TTS", "AUTOTUBE_DEPTH_MODEL",
              "PIPER_DIR", "PIPER_VOICE", "POLLINATIONS_TOKEN"}


def env_vars_read_by_code() -> set[str]:
    src = "\n".join(p.read_text() for p in (ROOT / "autotube").glob("*.py"))
    found = set(re.findall(r'environ(?:\.get)?\(\s*["\']([A-Z_]+)["\']', src))
    return found - LOCAL_ONLY


def env_block(workflow: str) -> set[str]:
    text = (WF / workflow).read_text()
    return set(re.findall(r'^\s+([A-Z_]+):\s*\$\{\{\s*secrets\.', text, re.M))


def test_the_daily_job_passes_every_secret_the_code_reads():
    """A secret that exists in the repo but is not mapped into the job is invisible:
    the code sees an empty string and silently degrades."""
    missing = env_vars_read_by_code() - env_block("daily.yml")
    assert not missing, f"daily.yml never passes these: {sorted(missing)}"


def test_the_analytics_job_passes_what_analytics_needs():
    """Analytics talks to the YouTube Analytics API over OAuth and uses no LLM."""
    need = {"YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"}
    assert need <= env_block("analytics.yml")
    src = (ROOT / "autotube" / "analytics.py").read_text()
    assert "LLM(" not in src, "analytics must not need an LLM; analytics.yml passes no LLM keys"


def test_no_workflow_references_a_secret_the_code_never_reads():
    """Dead secret wiring is a slow trap: it looks configured and does nothing."""
    referenced = env_block("daily.yml") | env_block("analytics.yml")
    known = env_vars_read_by_code() | {
        # notification sinks, read via common.notify
        "NTFY_TOPIC", "DISCORD_WEBHOOK_URL",
        # OAuth pair consumed by the google client, not via os.environ directly
        "YT_CLIENT_ID", "YT_CLIENT_SECRET",
    }
    assert not (referenced - known), f"workflow passes unused secrets: {sorted(referenced - known)}"


# ── reviewer independence ───────────────────────────────────────────────────
def test_the_reviewer_is_never_the_model_that_wrote_the_draft():
    """A writer grading its own draft agrees with itself.

    This used to require the reviewer to open on a *different provider*. That is no
    longer the right guarantee. Gemini is the only vision-capable provider and image
    verification is a hard gate, so spending its free-tier quota on reviews is what
    429'd the vision call and discarded a finished script. Reviews now prefer Groq,
    which is also where the writer starts.

    Independence is enforced per model, not per provider, and Groq's models are not one
    family - gpt-oss is OpenAI's and qwen is Alibaba's, different lineages with
    different failure modes. A live run confirmed the pairing works in practice
    (gpt-oss-120b reviewing a qwen draft and vice versa). What must never happen is a
    model reviewing itself, so that is what this pins.
    """
    llm = CFG["llm"]
    reviewers = llm["review_providers"]
    assert reviewers, "there must be a review order"
    # Whichever provider writes first must offer at least two models, so the reviewer
    # always has somewhere else to go when it lands on the same provider.
    first = llm["providers"][0]
    if reviewers[0] == first:
        pool = llm.get(f"{first}_review_models") or llm.get(f"{first}_models", [])
        assert len(pool) >= 2, (
            f"review opens on {first}, the same provider that writes, so {first} needs "
            f"at least two models for the reviewer to differ from the writer")
    # And a second provider must remain reachable as a fallback.
    assert len(reviewers) >= 2, "review needs a fallback provider"
    # Both keyed providers must stay in the review order so the fallback is real.
    assert {"gemini", "groq"} <= set(reviewers), \
        "both keyed providers should remain reachable for review"


def test_every_review_provider_can_actually_be_reached():
    """A review_providers entry with no models configured silently does nothing."""
    for p in CFG["llm"]["review_providers"]:
        models = CFG["llm"].get(f"{p}_review_models") or CFG["llm"].get(f"{p}_models")
        assert models, f"review provider {p} has no models"


def test_the_writer_model_is_recorded_and_excluded_from_review():
    """Independence is only real if the writer's exact model is passed as `avoid`."""
    src = (ROOT / "autotube" / "scriptwriter.py").read_text()
    assert 'script["_writer_model"] = self.llm.last_used' in src
    assert 'avoid=script.get("_writer_model")' in src
    assert 'review["independent"] = bool(review["reviewer"]) and review["reviewer"] != script.get("_writer_model")' in src


def test_independence_is_recorded_rather_than_assumed():
    """The provenance dossier reports it per episode, so a run where the fallback
    collapsed onto one provider is visible instead of quietly counted as reviewed."""
    src = (ROOT / "autotube" / "provenance.py").read_text()
    assert '"independent": review.get("independent")' in src
    assert "independently_reviewed" in src


def test_groq_retries_without_json_mode_when_the_model_refuses_it():
    """Reasoning models 400 on response_format=json_object; we must ask again without it.

    This single line is why the review stage had no model but the writer's own: every
    Groq call 400'd, Groq was retired on first use, and the skip path logged nothing.
    """
    import autotube.llm as L

    calls = []

    class R:
        def __init__(self, code, text, payload=None):
            self.status_code, self.text, self._p = code, text, payload

        def json(self):
            return self._p

    class FakeHTTP:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.append(dict(json))
            if "response_format" in json:
                return R(400, '{"error":{"code":"json_validate_failed"}}')
            return R(200, "", {"choices": [{"message": {"content": '{"ok": true}'}}]})

    orig = L.http
    L.http = lambda: FakeHTTP()
    try:
        out = L._openai_compatible("https://api.groq.com/openai/v1/chat/completions", "k",
                             "openai/gpt-oss-120b", "sys", "usr", 0.0, json_mode=True)
    finally:
        L.http = orig

    assert len(calls) == 2, "should retry once without the json_object constraint"
    assert "response_format" in calls[0] and "response_format" not in calls[1]
    assert out


def test_every_configured_model_id_is_syntactically_a_real_candidate():
    """Guard the class of bug that cost three production runs: invented model ids.

    Cannot hit the network here, so this pins the ids to the provider catalogues that
    `doctor` verified live, and will fail loudly if someone edits one by hand again.
    """
    cfg = CFG["llm"]
    groq_real = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b",
                 "qwen/qwen3.8-27b", "allam-2-7b"}
    for key in ("groq_models", "groq_review_models", "groq_vision_models"):
        for m in cfg.get(key, []) or []:
            assert m in groq_real, (
                f"{key}: {m!r} is not in Groq's catalogue. Run `autotube doctor` — it "
                f"lists the live model ids. Do not guess at model names.")


def test_length_instruction_rejects_both_ends_not_just_short():
    """The budget instruction used to say only "Too short = rejected".

    That is a one-sided push, and the model duly overshot: 58, 68 and 76 words against a
    52-word ceiling in a single live run, which became the largest single cause of
    abandoned scripts once the quality gates stopped being the bottleneck.
    """
    src = (ROOT / "autotube" / "scriptwriter.py").read_text()
    assert "Too short = rejected." not in src, "one-sided length instruction is back"
    assert "words per segment" in src, "per-segment budget is easier to hit than a total"
    assert "too short AND too long" in src.lower() or "Both ends are rejected" in src


def test_rate_limited_runs_are_not_reported_as_broken_builds():
    """Free-tier exhaustion is an operating condition, not a defect.

    Both ended as "produced nothing, exit 1", so five rate-limited runs in one day sat
    next to real failures and trained the red X to mean nothing. The alert must still
    fire; only the exit code changes.
    """
    import autotube.llm as L

    src = (ROOT / "autotube" / "__main__.py").read_text()
    assert "rate_limited()" in src, "the run must distinguish rate limiting from failure"
    assert "notify(" in src, "a rate-limited run must still alert; it just must not go red"

    before = L.rate_limited()
    L.note_rate_limited("groq:openai/gpt-oss-20b")
    assert "groq:openai/gpt-oss-20b" in L.rate_limited()
    assert L.rate_limited() >= before


def test_provider_errors_keep_enough_body_to_act_on():
    """200 chars cut 429s off before they said per-minute or per-day."""
    src = (ROOT / "autotube" / "llm.py").read_text()
    assert "r.text[:200]" not in src, "429 bodies truncated before the useful part"
    assert src.count("retry after") >= 2, "both error paths should surface retry-after"
