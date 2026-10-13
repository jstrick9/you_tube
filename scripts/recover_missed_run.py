"""Dispatch one safe top-up when a scheduled production attempt is missing or late."""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autotube.watchdog import dispatch_top_up, list_daily_runs, should_dispatch_top_up  # noqa: E402


def main() -> int:
    try:
        needed = int(os.environ.get("NEEDED", "0"))
    except ValueError:
        print("watchdog: invalid remaining-video count; refusing dispatch")
        return 1
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    try:
        runs = list_daily_runs(repository, token, api_url)
    except (RuntimeError, ValueError) as exc:
        print(f"watchdog: {exc}; refusing dispatch")
        return 1
    recover, reason = should_dispatch_top_up(needed, runs, datetime.now(timezone.utc), cooldown_minutes=90)
    print(f"watchdog: {reason}")
    if recover:
        try:
            dispatch_top_up(repository, token, api_url)
        except (RuntimeError, ValueError) as exc:
            print(f"watchdog: {exc}")
            return 1
        message = (f"AutoTube watchdog dispatched a safe top-up for {needed} remaining video(s). "
                   "The daily workflow rechecks its cap before producing.")
        print(message)
    else:
        message = f"AutoTube watchdog did not dispatch: {reason}."
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"{message}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
