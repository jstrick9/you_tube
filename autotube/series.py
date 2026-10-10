"""Recurring, numbered Archive 13 series.

CASE and FIELD provide a recognizable editorial structure and make the two approved
content lanes legible to viewers. Numbering is a branding convention; it does not
promise subscriber growth or affect a YouTube monetization decision. The YPP subscriber
requirement is tracked separately in ``docs/MONETIZATION.md``.

Format may repeat; substance may not. Each episode still needs a distinct trend-backed
topic, grounded claims, original explanation, and its own visual treatment. Series
membership must never block a video, and episode numbers are derived from history to
avoid accidental reuse or gaps.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"#(\d{1,4})\b")

# Numbers handed out during this process but not yet in history.json. A run publishes
# up to three videos and history is only written at the end, so without this all three
# read the same file and all three claim the same episode number - two duplicate #047s
# on the channel, which is worse than no numbering at all. Reset at the start of a run.
_ISSUED: set[tuple[str, int]] = set()


def reset() -> None:
    _ISSUED.clear()


def definitions(cfg: dict) -> dict:
    return (cfg.get("content") or {}).get("series") or {}


def assign(category: str | None, cfg: dict) -> str | None:
    """Which series does a topic in this category belong to?

    Returns None when series are switched off or the category is unmapped - an
    unmapped topic still publishes, it just publishes unbranded. Series membership
    must never be able to block a video.
    """
    for key, spec in definitions(cfg).items():
        if category and category in (spec.get("categories") or []):
            return key
    return None


def episode_number(key: str, history: list[dict]) -> int:
    """Next episode number for a series.

    Counted from history rather than stored in a counter file on purpose: a counter
    is state that can drift out of sync with what was actually published, and a
    series that skips from #12 to #15 because three runs failed after incrementing
    advertises that nobody is home. Derived numbering cannot drift.

    We take max-seen + 1 rather than len() + 1 so that deleting an old entry or
    pruning history never reissues a number that is already live on the channel.
    """
    seen = [0]
    for h in history:
        if h.get("series") != key:
            continue
        n = h.get("episode")
        if isinstance(n, int) and n > 0:
            seen.append(n)
        else:                                    # pre-series entries, or hand-edited history
            m = _NUM.search(h.get("title") or "")
            if m:
                seen.append(int(m.group(1)))
    seen += [n for k, n in _ISSUED if k == key]
    nxt = max(seen) + 1
    _ISSUED.add((key, nxt))
    return nxt


def decorate_title(title: str, key: str | None, number: int, cfg: dict, limit: int = 100) -> str:
    """Prefix the episode label, but never at the cost of the title itself.

    The hook is what earns the view; the series label only earns the follow. If the
    combined string would overflow YouTube's limit the label is dropped rather than
    the title truncated - a clipped hook costs a view today, a missing label costs a
    fraction of a subscriber.
    """
    spec = definitions(cfg).get(key or "", {})
    tag = (spec.get("tag") or "").strip()
    if not tag or number < 1:
        return title
    prefix = f"{tag} #{number:03d} — "
    return prefix + title if len(prefix) + len(title) <= limit else title
