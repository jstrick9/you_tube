# Long-form pipeline — deferred design sketch

> **Not active scope.** AutoTube remains Shorts-only. Do not resume or implement this proposal unless the user explicitly changes scope. This is a historical design note, not current monetization advice. Earlier RPM, revenue, and view projections were not verified; consult [`MONETIZATION.md`](MONETIZATION.md) for current official YPP requirements.

## Why this was considered

An earlier draft explored whether long-form watch hours could be an alternate YPP route. Shorts also have their own engaged-view eligibility route, so Shorts do not represent “nothing that leads to monetisation.” No long-form build is authorized by this note.

The 4,000 valid public watch-hours route is one standard YPP option; the required subscriber count and other channel policies still apply. Do not infer earnings from generic RPM estimates.

## What was proposed (unimplemented)

This section records ideas only. It does not change the current Shorts-only roadmap.

| Module | Reuse | Change proposed |
|---|---|---|
| `trends.py` | full | none — same lanes, same corroboration |
| `research` / grounding | full | fetch more per topic; the two-outlet rule is unchanged |
| `safety.py`, `gates.py` | full | `check_turn`/`check_hook` apply per chapter |
| `media.py` | full | more shots per video |
| `qa.py` | mostly | frame sampling would need to scale with duration |
| `strategy.py` | full | add a `length` dimension |
| `analytics.py` | partial | watch-hours would matter for long-form; engaged views remain the Shorts metric |
| `render.py` | no | 9:16 full-bleed is wrong for 16:9 long-form |
| `scriptwriter.py` | no | a long-form script is not a scaled-up Short |

## Draft design notes

A long-form script would need distinct chapters, their own evidence, and a 16:9 renderer. This is an archive of possible work only; no code changes, channel-format changes, or Shorts derivatives are currently planned.
