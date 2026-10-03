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
def test_the_reviewer_tries_a_different_provider_first():
    """A writer grading its own draft agrees with itself. With GEMINI and GROQ both
    configured the writer resolves to gemini and the reviewer to groq."""
    writers = CFG["llm"]["providers"]
    reviewers = CFG["llm"]["review_providers"]
    assert writers[0] != reviewers[0], "review order must not open with the writer's provider"
    available = [p for p in writers if p in ("gemini", "groq")]
    avail_rev = [p for p in reviewers if p in ("gemini", "groq")]
    assert available[0] != avail_rev[0], \
        "with only gemini+groq keys, writer and reviewer must still differ"


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
