"""How many videos today — the single source of truth for the daily target.

Deliberately self-contained: stdlib only, no imports from the rest of the package.
`scripts/needed_today.py` runs as a pre-flight gate before the heavy install, with only
PyYAML available, so anything it shares with the pipeline has to be importable without
requests, PIL or numpy. That constraint is why this is its own module rather than
living in common.py.

It exists because the logic was duplicated. The pipeline understood a [lo, hi] range in
videos_per_day; the pre-flight copy only ever handled a plain int, and called int() on
the list. Every scheduled top-up run therefore died in about eighteen seconds, which is
the whole mechanism that exists to cover GitHub silently dropping a cron. Two copies of
one rule is one copy too many.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def now_utc() -> datetime:
    """Indirected so tests can simulate other days.

    common.now_utc is the same thing, but importing common here would pull in requests,
    which is not installed when the pre-flight gate runs.
    """
    return datetime.now(timezone.utc)


def daily_target(cfg: dict) -> int:
    """How many videos today. Accepts a fixed int or a [lo, hi] range in videos_per_day.

    A fixed count forever is the same fingerprint problem as a fixed clock time. Varying it is
    free, but it must be stable for the whole day or the top-up runs would disagree with each
    other about whether the day is finished, so it is derived from the date.
    """
    v = cfg["schedule"]["videos_per_day"]
    if isinstance(v, (list, tuple)):
        lo, hi = int(v[0]), int(v[-1])
        if hi <= lo:
            return max(0, lo)
        tz = ZoneInfo(cfg["channel"].get("timezone", "UTC"))
        day = now_utc().astimezone(tz).date()
        h = hashlib.md5(f"count:{day.isoformat()}".encode()).hexdigest()
        return lo + int(h[:8], 16) % (hi - lo + 1)
    return max(0, int(v))
