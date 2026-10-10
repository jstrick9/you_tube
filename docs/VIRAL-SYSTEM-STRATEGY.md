# Building the best automated viral system — GitHub survey and strategy

Researched 3 Oct 2026. Repos below were read from the GitHub API and their READMEs, not
from round-up articles. Star counts, licences and last-push dates are as returned by the
API that day.

---

## 1. The remake strategy, answered directly

You described taking viral Shorts and remaking them "similar to the original but
different enough to look original". That tooling exists and is popular — **`hypit-ai/hypit`,
19,022★**, whose tagline is literally *"Clone any viral video with AI agents — 1 command,
100 variants, 100M views."*

I am not going to recommend it, and the reason is your stated goal: build original,
monetizable Archive 13 content. Recreating another creator's Short or generating many
surface-level variants conflicts with that goal and creates reused/inauthentic-content
risk. YouTube's policy describes these content categories, but it does **not** publish
fixed script-similarity percentages, daily upload counts, or other numerical enforcement
triggers. See `MONETIZATION.md` for the verified policy and threshold references.

| Practice | AutoTube rule |
|---|---|
| Repackage another creator's Short | Do not do it; use current trend evidence only to choose a topic and make a genuinely original treatment |
| Reuse facts or a template | Add original explanation/value and make every episode's substance distinct |
| Use realistic altered or synthetic content | Disclose when YouTube's current rules require it |

**Editorial principle for this project:**

> **Format can repeat. Substance cannot.**

This is an internal creative rule aligned with the goal of original work, not a quote or a guarantee of a YouTube review outcome.

The original, trend-informed version of the idea is already partly built here:

- **Use common formats, not someone else's execution.** A hook style or narrative format can be a creative starting point, but scripts, visuals and claims must be independently developed and valuable.
- **Mine outliers for *why* they worked**, then apply that to your own subject matter.
  `trends.py` already has `youtube_outliers()`. That is the compliant version of
  "find a viral video and remake it".
- **A repeatable format can coexist with original substance.** CASE/FIELD branding is a channel choice, not a policy guarantee; keep each episode's facts, explanation and visual treatment distinct.

Recreating another creator's Short can create reused-content and originality concerns, so AutoTube uses viral examples only as evidence of topical interest. It does not copy their footage, script, sequence, or distinctive presentation. This reduces an avoidable risk but cannot predict a review outcome.

---

## 2. The landscape, by category

### A. Viral-clone / variant factories — **reject**
| Repo | Stars | Licence | Verdict |
|---|---|---|---|
| `hypit-ai/hypit` | 19,022 | NOASSERTION | Offers cloning and bulk-variant tooling; not aligned with this project's originality constraints. Also needs a paid agent/model service. |

### B. Whole-pipeline replacements — **mine for ideas, do not adopt**
| Repo | Stars | Licence | Fit |
|---|---|---|---|
| `harry0703/MoneyPrinterTurbo` | 128,218 | MIT | Mature, but does **less** than this repo: no provenance, no originality gate, no compliance layer |
| `calesthio/OpenMontage` | 62,636 | AGPL-3.0 | Agentic, 12 pipelines. **See §2b — my first assessment of this was wrong** |
| `ATH-MaaS/Pixelle-Video` | 28,607 | Apache-2.0 | Strong, but leans on ComfyUI / 48GB GPU hosts — not free on our runner |
| `rushindrasinha/youtube-shorts-pipeline` (Verticals v3) | 2,306 | MIT | **Closest analogue to us.** Worth reading properly — see §3 |

Switching wholesale would change the project's architecture and remove its existing
provenance, originality and QA layers. Those safeguards are useful for this workflow,
but do not determine or guarantee a monetization decision.

### 2b. Correction: OpenMontage does have a genuine free path

My first pass said it "needs paid providers". **That was wrong**, and worth correcting
because the mistake was dismissing a 62k★ project on a skim.

It ships a documented **"What You Get With Zero API Keys"** path: Piper TTS for
narration, **Archive.org + NASA + Wikimedia Commons** for real footage, Remotion and
HyperFrames for composition, FFmpeg for post, and built-in word-level captions. GPU is
explicitly optional — it unlocks local video models, it is not required.

Two things follow from reading it properly:

1. **It independently validates the plan in `FREE-STACK-RESEARCH.md`.** Its free
   real-footage route is a documentary-montage pipeline building a **CLIP-searchable
   corpus from Archive.org, NASA and Wikimedia** — arrived at separately, the same three
   sources, for the same reason. That is strong corroboration that archival montage is
   the right free architecture.
2. **The blocker is operational, not financial.** OpenMontage is driven by an AI coding
   assistant, not by a scheduler. Its top level is `.agents`, `.claude`, `.codex`,
   `.cursor`, `.windsurfrules`, `AGENTS.md`, `CLAUDE.md`, `CODEX.md`, `COPILOT.md`,
   `CURSOR.md`; the documented workflow is *"open the project in your AI coding
   assistant and tell it what you want"*. There is no headless generate command —
   `python -m backlot` is a project viewer.

So running it daily and unattended would require adapting its interactive workflow,
accepting non-deterministic output, and rebuilding the provenance, originality and QA
checks this repo already has. Those checks support this project's process but do not
determine a monetization outcome. Minor issues on top: it narrates with **Piper, which is
archived**, and composes with Remotion.

**Conclusion: borrow, don't adopt.** The valuable, portable idea is the CLIP-searchable
Archive.org/NASA/Wikimedia montage corpus. That drops into our cron-native pipeline
behind the existing vision gate, and is already item 3 on the roadmap.

### C. Long-form → Shorts clippers — **only on your own footage**
`Anil-matcha/AI-Youtube-Shorts-Generator` (5,223★ MIT), `zhouxiaoka/autoclip` (9,132★
MIT), `jipraks/yt-short-clipper` (1,022★ MIT). All solid. All produce *reused content*
if pointed at someone else's video. Safe only against footage you own or public domain.

### D. Local video generation — **unusable here**
`Wan2.2` (17,707★), `LTX-Video` (11,011★), `CogVideo` (13,055★), `Sana` (9,191★).
8–80GB VRAM. We have no GPU. Covered in `FREE-STACK-RESEARCH.md`.

### E. Component wins — **adopt**
`Zulko/moviepy` (14,945★ MIT), `modelscope/FunClip` (6,361★ MIT),
`nadermx/backgroundremover` (8,084★ MIT), `SYSTRAN/faster-whisper` (25,683★ MIT).

---

## 3. The one idea worth stealing: niche intelligence

From **Verticals v3** (`rushindrasinha/youtube-shorts-pipeline`, MIT). Its central
concept is that **every pipeline stage reads from a niche profile** which shapes script
tone, visual style, caption styling and pacing — rather than those being global
settings. It ships 15 niches.

We have two lanes (`history_mystery`, `science_nature`) but they only really affect topic
selection. Script voice, caption styling, music and pacing are shared. Giving each lane
a full profile would:

1. Make the two lanes **visibly distinct**, which directly attacks the "generic and
   repetitive" rejection category.
2. Raise per-lane performance, since a history hook and a science hook want different
   pacing.
3. Add variety to creative treatment without relying on any unpublished numerical similarity threshold.

This is a config-and-prompt change, not a rewrite, and it is the highest value-per-hour
item I found.

---

## 4. Monetization thresholds and analytics (verified 9 Oct 2026)

Use the official YPP eligibility panel in YouTube Studio as the source of truth. The
expanded tier currently requires 500 subscribers, 3 valid public uploads in 90 days, and
either 3M qualified Shorts views in 90 days or 3,000 qualified watch hours in 12 months.
Standard ad-revenue eligibility currently requires 1,000 subscribers and either 10M
qualified public Shorts views in 90 days or 4,000 qualified watch hours in 12 months.
YouTube has announced new standard-tier thresholds of 20M qualified Shorts views or
8,000 qualified watch hours for new applicants from 1 February 2027; the expanded-tier
thresholds remain unchanged.

AutoTube's tracker uses YouTube Analytics `engagedViews` for its Shorts-view progress
proxy because raw public Shorts `views` now include starts/replays. It tracks
`averageViewPercentage` as average percentage watched and keeps `audienceWatchRatio` as a
relative curve only. The 70% average-view goal and 15–20 second duration are internal
experiments, **not** official recommendation thresholds. This system cannot guarantee
that an upload will hit either the YPP thresholds or the internal target.

Original, valuable and advertiser-friendly content is essential. YouTube does not publish
the numeric policy triggers previously quoted in this memo; do not infer a fixed
script-variation %, daily-upload count, RPM or channel-enforcement deadline from them.
See [`MONETIZATION.md`](MONETIZATION.md) for the official sources and current detail.

---

## 5. Recommended system — in priority order

Ordered by effect on your actual goal, not by novelty.

1. **Unblock production.** The pipeline shipped zero videos on 3 Oct because both LLM
   providers were rate-limited. Add 2–3 no-credit-card providers (Cerebras, Cloudflare,
   Mistral, Cohere). Nothing else matters while output is zero.
2. **Niche profiles per lane** (§3). Best quality-and-compliance gain per hour.
3. **Internet Archive b-roll.** 10,461 public-domain films, keyless, verified. Real
   footage, on-brand, and it is the visual-supply fix that also unlocks long-form.
4. **Outlier-driven format mining.** Extend `youtube_outliers()` to extract *structure*
   from winners — hook type, beat count, caption density — and feed it to the writer as
   format guidance, without copying a creator's execution. This keeps the source video
   as trend evidence rather than a template to clone.
5. **Raise the 150-min cap.** Actions minutes are free and unlimited on a public repo.
6. **Cloudflare FLUX, non-depictive only.** Backgrounds, texture, thumbnails. Never as
   evidence behind a factual claim.

What I would **not** build: a clone pipeline, a face-swap stage, or anything that raises
output volume without adding substantive originality or viewer value. Those do not serve
the project's stated content goals.
