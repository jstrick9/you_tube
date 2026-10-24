"""Narrow source-locked drafts for known, live trend profiles.

These drafts do not approve or publish themselves: the normal local script gates,
independent review, originality/safety checks, licensed-visual retrieval, visual QA,
final-render QA, and YouTube upload checks still run unchanged. A profile is ignored
unless its exact live-trend selector tag, lane, article title, angle, and source quotes
all match.
"""
from __future__ import annotations

from copy import deepcopy

NOKIA_PROFILE_ID = "nokia_phone_decline_v1"
NOKIA_ANGLE = (
    "After Nokia adopted Windows Phone in 2011, Symbian demand fell; its 2013 Microsoft deal "
    "helped avert a cash crisis."
)

# The line-level evidence fields are exact source excerpts, not model-generated claims.
# The hook's evidence may be empty under the existing script schema; all four factual
# body/payoff beats carry their own distinct, verbatim quote.
_NOKIA_EVIDENCE = (
    "would adopt Windows Phone 7 as its primary operating system on smartphones",
    "demand for Symbian dropping when consumers realized Nokia's focus and attention would be elsewhere",
    "shedding 10,000 employees by the end of the year and the closure of the Salo",
    "would have been in a cash crisis had it not sold the division to Microsoft",
    "In September 2013, Nokia announced the sale of its mobile and devices division to Microsoft.",
)

_NOKIA_DRAFT = {
    "title": "Nokia's 40.4% Share—and the Phone Sale to Microsoft",
    "segments": [
        {
            "text": "Nokia's handset market share hit 40.4% in Q4 2007.",
            "aside": "",
            "emphasis": ["40.4%"],
            "reveal": False,
            "visual": {
                "shows": "Nokia mobile phones from 2007, with Nokia branding visible.",
                "queries": [
                    "Nokia mobile phones 2007 product photo",
                    "Nokia 2007 handset collection archival image",
                ],
            },
            "evidence": "",
        },
        {
            "text": "Nokia bet its 2011 smartphone future on Microsoft's Windows Phone 7.",
            "aside": "",
            "emphasis": ["Windows Phone 7"],
            "reveal": False,
            "visual": {
                "shows": "A Nokia smartphone running Microsoft's Windows Phone 7.",
                "queries": [
                    "Nokia Windows Phone 7 smartphone product photo",
                    "Nokia Lumia Windows Phone 2011 handset image",
                ],
            },
            "evidence": _NOKIA_EVIDENCE[0],
        },
        {
            "text": "Symbian demand fell when Nokia shifted its smartphone focus.",
            "aside": "",
            "emphasis": ["Symbian"],
            "reveal": False,
            "visual": {
                "shows": "A Nokia handset displaying its Symbian operating system.",
                "queries": [
                    "Nokia Symbian smartphone Symbian OS screen photo",
                    "Nokia Symbian phone product image",
                ],
            },
            "evidence": _NOKIA_EVIDENCE[1],
        },
        {
            "text": "Nokia cut 10,000 jobs and shut its Salo plant.",
            "aside": "",
            "emphasis": ["10,000", "Salo"],
            "reveal": False,
            "visual": {
                "shows": "Nokia's Salo manufacturing plant in Finland.",
                "queries": [
                    "Nokia Salo manufacturing plant Finland exterior",
                    "Nokia Salo factory site archival photo",
                ],
            },
            "evidence": _NOKIA_EVIDENCE[2],
        },
        {
            "text": "Nokia's 2013 Microsoft deal helped avert a cash crisis.",
            "aside": "",
            "emphasis": ["Microsoft", "cash crisis"],
            "reveal": True,
            "visual": {
                "shows": "Nokia and Microsoft Lumia devices displayed together.",
                "queries": [
                    "Nokia Microsoft Lumia devices together photo",
                    "Nokia Microsoft mobile division sale image",
                ],
            },
            "evidence": f"{_NOKIA_EVIDENCE[4]} {_NOKIA_EVIDENCE[3]}",
        },
    ],
}


def nokia_draft(topic: dict, source: dict) -> dict | None:
    """Return the constrained Nokia draft only when its live/source identity still matches."""
    if not isinstance(topic, dict) or not isinstance(source, dict):
        return None
    if (topic.get("curated_profile") != NOKIA_PROFILE_ID
            or topic.get("lane") != "history_mystery"
            or "youtube_outliers" not in (topic.get("sources") or [])
            or topic.get("script_subject") != "Nokia"
            or topic.get("wiki_query") != "Nokia"
            or topic.get("angle") != NOKIA_ANGLE
            or str(source.get("title") or "").strip().casefold() != "nokia"):
        return None
    text = source.get("text")
    if not isinstance(text, str) or not all(quote in text for quote in _NOKIA_EVIDENCE):
        return None
    draft = deepcopy(_NOKIA_DRAFT)
    draft["_curated_profile"] = NOKIA_PROFILE_ID
    return draft


def nokia_metadata(script: dict) -> dict | None:
    """Source-grounded metadata fallback for the Nokia draft if every writer API is unavailable."""
    if not isinstance(script, dict) or script.get("_writer_model") != f"curated:{NOKIA_PROFILE_ID}":
        return None
    return {
        "description": (
            "Nokia's handset market share reached 40.4% in Q4 2007. This Short traces the 2011 Windows Phone move, "
            "falling Symbian demand, and the later mobile-division sale to Microsoft that helped avert a cash crisis."
        ),
        "tags": [
            "Nokia", "Nokia history", "Nokia mobile phones", "Windows Phone 7",
            "Symbian", "Nokia Salo plant", "Microsoft mobile division", "Nokia market share",
        ],
        "hashtags": ["#Nokia", "#History", "#Shorts"],
        "thumbnail_text": "40.4% TO MICROSOFT",
    }
