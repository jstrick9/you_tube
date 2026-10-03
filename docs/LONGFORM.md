# Long-form pipeline — design

**Decision:** Shorts become top-of-funnel. Long-form becomes the product and the monetisation path.

## Why

```
939 median views × 2.03 videos/day × 90 days  =    171,540
YPP Shorts requirement                        = 10,000,000
                                        gap   =         58×
```

Even on reaching it, Shorts RPM of $0.05–0.15 pays roughly $500–1,500 per quarter. The long-form
requirement is **4,000 watch hours + 1,000 subscribers**, at an RPM of $4–15 — between 30× and 100×
the revenue per view. Shorts views do not count toward watch hours, so the current channel is
accumulating nothing that leads to monetisation.

4,000 hours is reachable: 8-minute videos at 40% average view duration earn ~3.2 minutes each, so
**75,000 views total** clears it. That is 80 videos at 940 views — roughly the audience the channel
already reaches, redirected into a format that counts.

## What already carries over

This is the part that makes the pivot cheap. The existing stack is format-agnostic:

| Module | Reuse | Change needed |
|---|---|---|
| `trends.py` | full | none — same lanes, same corroboration |
| `research` / grounding | full | fetch more per topic; the two-outlet rule is unchanged |
| `safety.py`, `gates.py` | full | `check_turn`/`check_hook` apply per chapter |
| `media.py` | full | ~10× more shots per video |
| `qa.py` | mostly | frame sampling must scale with duration, not be a fixed count |
| `strategy.py` | full | add `length` as a dimension so the bandit learns the trade |
| `analytics.py` | partial | watch-hours become the reach metric; `vph` is a Shorts idiom |
| `render.py` | **no** | 9:16 full-bleed is wrong for 16:9 long-form |
| `scriptwriter.py` | **no** | a 1,200-word script is not a scaled-up 78-word one |

## The two real pieces of work

### 1. Chaptered scriptwriting

A long-form script is not a long Short. It is 6–10 chapters, each with its own hook and its own
turn, chained so every chapter opens a loop the next one closes. The existing retention machinery
applies *per chapter* rather than once per video — which is exactly what `check_turn()` already
does, so it generalises.

Budget: ~150 words/chapter, 8 chapters ≈ 1,200 words ≈ 8 minutes at the current speech rate.

Grounding cost scales linearly and is the main new expense: 8 chapters each needing verifiable
evidence means roughly 8× the research calls per video. At 1 long-form video/day this is cheaper
than the current 3 Shorts/day.

### 2. 16:9 rendering

Not a variant of the Shorts renderer — a sibling. Landscape, captions as an accessory rather than
the main visual element, slower shot cadence (4–6s against the current 1.5–2.5s), chapter title
cards, and an actual intro/outro. The `ARCHETYPES` mechanism carries over unchanged.

## Sequencing

1. `length` as a bandit dimension (`short` | `long`), defaulting to a fixed split, so cadence is
   controlled by config rather than a code branch.
2. Chaptered scriptwriter behind a format flag — reuses every existing gate.
3. 16:9 renderer.
4. Analytics: watch-hours as the reach metric for long-form; keep `vph` for Shorts. The reward
   function needs a per-format target, because 40% retention on 8 minutes and 40% on 26 seconds are
   not comparable achievements.
5. Shorts become derivative: cut the strongest 30 seconds out of each long-form video. One research
   pass, one grounding pass, two assets. This is where the funnel actually closes, and it drops
   Shorts production cost to near zero.

Step 5 is the one that makes the hybrid worth doing rather than just running two channels.

## Honest risks

- **4,000 hours is a much slower clock than it looks.** At 80 videos and one per day, that is three
  months of production before the threshold is even in reach, assuming current per-video audience
  holds at 8 minutes — it may not, because long-form is a harder sell to a cold audience.
- **Long-form is far more exposed to the Generic or Repetitive Content policy.** Eight minutes of
  TTS over stock footage is much more obviously mass-produced than 26 seconds of it. The voice
  problem stops being cosmetic and becomes the central quality risk.
- **QA cost scales with duration** and the vision checks are the most expensive part of a run.
- This does not fix the retention cliff; it changes what the cliff costs. The script work in
  `6cfe1be` matters more here, not less.
