"""Create a dependency-free run receipt if Actions never reached the AutoTube CLI."""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
STATE = ROOT / "state"
KEEP = 30


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "-", run_id)[:100]
    receipt_path = OUTPUT / f"run-report-{safe_id}.json"
    if receipt_path.exists():
        return 0

    needed = os.environ.get("AUTOTUBE_NEEDED", "")
    need_outcome = os.environ.get("AUTOTUBE_NEED_OUTCOME", "")
    install_outcome = os.environ.get("AUTOTUBE_INSTALL_OUTCOME", "")
    production_outcome = os.environ.get("AUTOTUBE_PRODUCTION_OUTCOME", "")
    if needed == "0":
        stage, outcome, category = "preflight", "already_satisfied", None
    elif need_outcome == "failure":
        stage, outcome, category = "preflight", "failed", "preflight_failed"
    elif install_outcome == "failure":
        stage, outcome, category = "install", "failed", "workflow_install_failed"
    elif production_outcome == "failure":
        stage, outcome, category = "production_cli", "failed", "workflow_producer_step_failed"
    else:
        stage, outcome, category = "workflow", "producer_not_started", "producer_not_started"

    now = utc_now()
    target_value = needed
    if not target_value.isdigit():
        target_value = os.environ.get("AUTOTUBE_REQUESTED_COUNT", "").strip()
    target_known = target_value.isdigit()
    target = max(0, int(target_value)) if target_known else None
    row = {
        "schema_version": 1,
        "run_id": safe_id,
        "workflow_run_id": run_id,
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "started_at": now,
        "finished_at": now,
        "duration_seconds": 0,
        "mode": "workflow_fallback",
        "top_up": os.environ.get("AUTOTUBE_TOP_UP", "false").lower() == "true",
        "target_count": target,
        "target_count_known": target_known,
        "produced_count": 0,
        "scheduled_count": 0,
        "upload_failure_count": 0,
        "outcome": outcome,
        "stage": stage,
        "stage_entries": {stage: 1},
        "retry_budgets": {},
        "topic_attempts": [],
        "failures": ([{"at": now, "stage": stage, "category": category}] if category else []),
        "retry_events": [],
        "provider_events": [],
    }
    try:
        atomic_json(receipt_path, row)
        state_path = STATE / "run_reports.json"
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else []
            previous = previous if isinstance(previous, list) else []
        except (OSError, json.JSONDecodeError):
            previous = []
        atomic_json(state_path, (previous + [row])[-KEEP:])
    except OSError as exc:
        print(f"fallback receipt write failed: {type(exc).__name__}")
        return 1
    print(f"wrote fallback run receipt: outcome={outcome}, stage={stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
