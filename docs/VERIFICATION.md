# Verification Audit — 2026-10-03

Post-implementation audit of everything shipped in the Shorts-quality, monetization and
demonetization-risk work. Scope: is it tested, is it wired, and does it do what the
commit messages claim.

Method was deliberately not "read the code and agree with it". Every claim below was
either executed or is backed by a test that fails when the claim stops being true.

---

## Summary

| | |
|---|---|
| Tests | **397 passed, 1 skipped** (skip is environmental: `onnxruntime` absent) |
| Modules importing clean | 23 / 23 |
| Git | working tree clean, `HEAD == origin/main` |
| Coverage, decision logic | `provenance` 100%, `series` 100%, `originality` 98%, `safety` 97%, `sfx` 97%, `qa` 95%, `gates` 95%, `strategy` 95% |
| Coverage, overall | 57% — the remainder is network/IO and the render path (see Residual risk) |
| **Defects found** | **3, all fixed in `ab6ec59`** |

---

## Defects found and fixed

All three were **wiring** defects. Every component involved was well covered in
isolation (98–100%), and `pipeline.run()` and `analytics.run()` were at **0%** — which
is precisely where the components are connected to each other. Unit tests cannot see
this class of bug.

### 1. The similarity gate was blind inside a single run — *serious*

One `ScriptWriter` is constructed per run, it reads `history.json` once, and history is
only written at the **end** of the run. A run makes up to three videos, so videos two
and three were never compared against video one.

This is the worst possible place for the gate to be blind: same-run videos are the most
likely to collide, because they are drawn from the same day's trends, the same lane and
the same format pool.

Reproduced before fixing — two near-identical scripts, both approved:

```
video 1 approved: True
video 2 approved: True   <-- identical script, SAME RUN
```

After: `video 2 approved: False — too close to a published episode (92% phrase overlap)`.

Approved drafts now register into the in-memory history. **Only approved drafts** — a
rejected draft is about to be rewritten on the same topic, and holding it against its
own replacement would guarantee the retry fails and the topic is abandoned.

### 2. Appeal evidence was written for videos that never published

`provenance.record()` ran whenever `do_upload` was set, including when the upload raised
and left `status=upload_failed` with no `video_id`. The dossier would then claim episodes
that are not on the channel.

This matters more than it sounds. The dossier exists to be handed to a policy reviewer.
A reviewer who checks one claimed episode and cannot find it discredits the entire
archive — **inflated evidence is worse than missing evidence**. Now gated on `video_id`.

### 3. `content.series.*.remit` was config nothing read

Found by sweeping every config key against the source. Added in the series commit and
never wired, so the remit — the thing that defines what an episode of `CASE` or `FIELD`
actually *is* — had no effect. The numbering was a prefix stapled to whatever the trend
feed produced, which is the opposite of what a series is for. It now reaches the writer
prompt, and a test asserts no key under `series` is ever unread again.

---

## Gaps closed (found, not created, by the audit)

- **`_subs_progress` had zero tests** despite being the headline subscriber metric.
  Now covered, including that it returns `None` rather than inventing an arrival date
  for a channel with no data or one that is losing subscribers.
- **The CLI had zero tests.** `dossier` and `report` now covered.
  `doctor` is deliberately excluded: it probes live endpoints, cost 3.4s and a network
  dependency. A test that fails on a train is worse than no test.

---

## Verified working (executed, not assumed)

- Similarity gate fires inside `ScriptWriter.check` — rejects a duplicate at 100% overlap.
- Series assignment and title decoration: `mysteries → CASE #007 — …`,
  `space → FIELD #007 — …`, unmapped category publishes unbranded rather than failing.
- `subscribersGained` is genuinely in the Analytics request string and unpacked.
- `channel_audit` and `_subs_progress` both run against the **real** `state/history.json`
  and **degrade honestly**: no fingerprints yet → *"not enough published episodes with a
  recorded script fingerprint yet"*; no subscriber data → `null` projections, not zeros
  dressed up as facts.
- Config is self-consistent: reward weights sum to exactly 1.0; `channel.categories`,
  the union of lane categories, and the union of series categories are all the same set.
- All three CLI commands exit 0 offline.

---

## Residual risk — not fixed, and why

**1. The render + upload path is unverified (`pipeline.py` 44%, `make_one` largely
uncovered).** It needs ffmpeg assets, API credentials and network. No API keys exist in
this environment, so **no live end-to-end dry run was possible**. This is the single
largest remaining unknown. The first real run should be `--dry-run --keep-work` with the
output inspected by eye.

**2. With no API keys the system silently runs on one keyless provider.** `doctor`
reports `LLM: {'ok': True}` with every key unset, because `pollinations:openai` needs
none. It works — but writer and reviewer are then the **same** provider, so
`review["independent"]` is always false. `require_independent_review` is `False`, so the
run proceeds and merely logs it. Consequence: the provenance dossier would record
**zero independently reviewed episodes**, directly weakening the appeal evidence it
exists to provide.
*Recommendation: set at least one API key.* I did not flip
`require_independent_review` to `True` — that fails closed and would halt all production
if a second provider is unreachable. That is a decision with an availability cost and
belongs to you.

**3. The existing 30 history entries have no script fingerprint.** The originality
defences start from the next upload. `channel_audit` deliberately excludes them rather
than counting them as evidence of variety, which would make it most reassuring exactly
when it knows least.

**4. Low coverage on I/O wrappers** — `music` 14%, `tts` 18%, `research` 23%,
`youtube` 25%, `motion` 28%, `trends` 32%, `sources` 35%. These are mostly network and
subprocess calls where a test would assert that a mock was called. Noted, not chased.

**5. Two pre-existing dead config keys**: `compliance.script_mention_ok` and
`upload.playlist_per_category`. Harmless, not mine, left alone.
