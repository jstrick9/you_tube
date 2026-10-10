"""Per-lane editorial profiles for a more coherent Archive 13 voice.

Lanes already existed for topic discovery. These profiles can also help a history mystery and a
science/nature story use fitting pacing and emphasis without changing the shared pipeline. Editorial
variety supports original treatment, but is not a monetization guarantee or a response to any unpublished
similarity threshold.

Deliberately prompt-level. These profiles steer the writer; they do not fork the pipeline, so there is one
code path to maintain and no new failure mode.
"""
from __future__ import annotations

import logging

log = logging.getLogger("autotube.lanes")

# Used when a topic belongs to no lane we recognise. Matches the previous single voice,
# so an unmapped topic behaves exactly as it did before this module existed.
DEFAULT_PROFILE: dict = {
    "tone": "",
    "hook_style": "",
    "pacing": "",
}


def lane_for(topic: dict, cfg: dict) -> str | None:
    """Which lane a topic belongs to.

    Outlier-sourced topics carry `lane` directly. Wikipedia spikes and anniversaries do
    not, so fall back to the category-to-lane mapping the lanes already declare.
    """
    lanes = (cfg.get("trends") or {}).get("lanes") or {}
    explicit = topic.get("lane")
    if explicit and explicit in lanes:
        return explicit
    cat = (topic.get("category") or "").strip().lower()
    if cat:
        for name, spec in lanes.items():
            if cat in [c.lower() for c in (spec.get("categories") or [])]:
                return name
    return None


def profile_for(topic: dict, cfg: dict) -> dict:
    """The editorial profile for this topic's lane, falling back to the shared voice."""
    lanes = (cfg.get("trends") or {}).get("lanes") or {}
    name = lane_for(topic, cfg)
    if not name:
        return dict(DEFAULT_PROFILE)
    prof = (lanes.get(name) or {}).get("profile") or {}
    return {**DEFAULT_PROFILE, **prof}


def writer_rule(topic: dict, cfg: dict) -> str:
    """The profile as a prompt fragment, or "" when the lane has no profile.

    Returning an empty string for an unprofiled lane matters: it keeps the prompt byte
    for byte what it was, so adding this module cannot quietly change existing output.
    """
    p = profile_for(topic, cfg)
    bits = []
    if p.get("tone"):
        bits.append(f"VOICE: {p['tone']}")
    # Deliberately not emitted here: the hook is stated once, by the writer, as the
    # single HOOK line. Emitting it from both places is what produced a prompt saying
    # "HOOK STYLE: question" and "never open with a question" in consecutive lines.
    if p.get("pacing"):
        bits.append(f"PACING: {p['pacing']}")
    if not bits:
        return ""
    name = lane_for(topic, cfg)
    log.debug("lane profile applied: %s", name)
    return "\n".join(bits) + "\n"
