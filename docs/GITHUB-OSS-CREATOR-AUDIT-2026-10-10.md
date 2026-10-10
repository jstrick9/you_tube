# GitHub open-source audit for Archive 13

**Audit date:** 10 October 2026  
**Scope:** GitHub projects that could help AutoTube find timely Shorts topics, write original scripts, source/verify visuals, narrate/caption, render, and measure results. The goal is a Shorts-only Archive 13 workflow that earns genuine subscriber growth without undermining originality, asset rights, or YouTube Partner Program (YPP) eligibility.

> **Bottom line:** No GitHub project, star count, model, or automation recipe can guarantee a viral Short, subscriber growth, a 70% average view percentage, or YPP approval. Stars measure developer attention—not video performance. The best fit is to keep AutoTube’s evidence-led, strict-QA pipeline and borrow only targeted ideas/components. Do not replace the workflow with a generic video factory or a viral-Short cloning tool.

## Executive findings

1. **AutoTube already has the right high-level architecture for this channel.** `autotube/trends.py` gathers Google Trends, YouTube outliers/chart, Reddit, Hacker News, and Wikipedia pageview/spike signals; Wikipedia/evergreen alone is not enough. `autotube/media.py` already queries Wikimedia, NASA, Internet Archive, Openverse, and configured stock sources. `autotube/vision.py` uses OpenCLIP for cheap candidate ranking and a vision model for approval; `autotube/qa.py` checks the finished rendered frames against narration and the intended shot. `autotube/render.py` already renders vertical Shorts with FFmpeg. An all-in-one replacement would discard working, channel-specific controls without evidence it would improve retention.
2. **The meaningful gap is creative performance, not GitHub popularity.** The latest analytics snapshot (`state/analytics_summary.json`, updated 2026-10-10 18:54 UTC) records **6 of 29 videos (20.7%) at or above the 70% average-view target; median 56.22%**. The retention curve’s largest relative drop is 22.5% from the 10% to 20% runtime points. That is a relative watch-ratio change—not “22.5% of viewers left.” The snapshot estimates 1.03 subscribers per 1,000 public views over 90 days, but engaged-view coverage is incomplete. These are small/noisy samples; they support controlled hook and pacing tests, not a promise that any code change will hit 70%.
3. **One actionable model-terms issue deserves priority.** AutoTube’s current CLIP config selects `ViT-B-32` with `laion2b_s34b_b79k`. OpenCLIP’s *code* is MIT-licensed, but that specific checkpoint’s model card says it is research-oriented and lists deployed use cases as out of scope. This is a scope/terms warning, not a conclusion that the MIT code license is invalid. Before making the model a commercial production dependency, document a rights review or benchmark a checkpoint whose model card clearly permits the intended use (for example, the Apache-2.0 `timm/ViT-B-16-SigLIP2` model card). Keep the existing vision gate whichever checkpoint is used.
4. **Best selective integrations/references:**
   - Test `faster-whisper` as an optional post-TTS transcription check, especially for the estimated-timing Piper fallback and proper-noun pronunciation. Do not use it to replace edge-TTS’s direct word boundaries without a measured benefit.
   - Borrow Verticals v3’s *per-niche profile* idea for distinct `history_mystery` and `science_nature` writing/pacing—not its generic generation pipeline.
   - Study OpenMontage’s archival footage indexing and quality-control concepts; its documentary-source idea substantially overlaps AutoTube’s existing sources, so do not adopt its whole agent-driven stack.
   - Keep MoneyPrinterTurbo as a workflow/UX reference, not a publish button.
5. **Explicitly reject the clone/repurpose shortcut.** Hypit is popular precisely for cloning a reference Short into many variants; that conflicts with the stated “format can repeat; substance cannot” rule. Long-video clip generators can be useful only on footage Archive 13 owns or has permission to use. No creator’s YouTube video should be downloaded, cut up, or used as a visual source for AutoTube.

## How the audit was scored

The table below is ranked by **fit to Archive 13**, not stars. GitHub API counts and last-push dates were retrieved on 10 October 2026; they are point-in-time snapshots and will change. A license shown for a repository applies to its code unless stated otherwise. Model weights, voices, source media, fonts, and APIs may have separate terms. No project was installed, benchmarked on channel assets, or integrated as part of this audit.

“Success” here means visible project adoption/maintenance plus usefulness to this workflow. It does **not** mean the project has demonstrated channel growth, viral performance, watch retention, or YPP acceptance. This refresh complements the earlier [`VIRAL-SYSTEM-STRATEGY.md`](VIRAL-SYSTEM-STRATEGY.md) and [`FREE-STACK-RESEARCH.md`](FREE-STACK-RESEARCH.md) rather than replacing their implementation notes.

## Ranked project scorecard

| Rank / project | GitHub snapshot (stars; license; last push) | Fit for Archive 13 | Decision |
|---|---:|---|---|
| **1. faster-whisper** | 25,805; MIT; Oct 10, 2026 | Local transcription, VAD, and word timestamps. Most useful as an independent check of the *finished narration audio* versus the script, not as another script generator. | **Small optional POC.** Benchmark CPU time and proper-noun error rate on existing narration. Keep exact edge-TTS word boundaries unless the test demonstrates better sync. |
| **2. Verticals v3** (`youtube-shorts-pipeline`) | 2,318; MIT; Jun 9, 2026 | Closest architectural analogue: topic discovery, resumable stages, and 15 niche profiles. The useful idea is that niche identity can shape tone, visual style, captions, and pacing. | **Reference-only.** Adapt the profile concept to the two approved lanes. Last push is older than the other active candidates; do not take the dependency. |
| **3. Openverse** | 381; MIT; Oct 10, 2026 | Openly licensed/public-domain image and audio discovery. AutoTube already uses its API. | **Keep current use, strengthen per-item verification.** Openverse itself warns that its license metadata may be inaccurate; visit the asset’s source and retain the exact license, author, and attribution. |
| **4. OpenMontage** | 66,132; AGPL-3.0; Oct 3, 2026 | Large agentic production project; its public README documents zero-key documentary montage paths using Archive.org, NASA, and Wikimedia, plus Remotion/HyperFrames and FFmpeg. | **Reference-only.** Study asset-corpus indexing, documentary montage, and contract-test patterns. Its AI-coding-assistant workflow is not a drop-in scheduled AutoTube job, and AGPL terms make code-level integration a deliberate licensing decision. |
| **5. MoneyPrinterTurbo** | 129,477; MIT; Oct 10, 2026 | Highly adopted end-to-end short-video generator. Its README describes automated scripts, media, subtitles, background music, and rendering. | **Reference-only.** Compare ergonomics and provider fallbacks. A one-click pipeline is not evidence that a topic is currently trending, a visual depicts the narration, the script is substantively original, or the output grows a channel. Keep AutoTube’s own evidence and publish gates. |
| **6. OpenCLIP** | 14,189; code MIT; Sep 30, 2026 | Image/text similarity is already in `autotube/vision.py` and pre-ranks candidate shots. | **Keep the component; resolve the checkpoint warning.** Code license and model-card use scope are distinct. Benchmark any replacement against labeled Archive 13 shot matches before changing config. |
| **7. Kokoro** | 9,244; Apache-2.0; Aug 6, 2025 | Small, CPU-friendly TTS candidate; can be auditioned as a different narration voice. | **Voice test only.** It has not been actively pushed for over a year in this snapshot, and current edge-TTS already returns word-boundary events. Do a blind quality test before considering a switch; verify the exact weights and voice terms. |
| **8. WhisperX** | 24,455; BSD-2-Clause; Sep 26, 2026 | Adds forced word alignment and optional diarization to ASR. | **Not a current dependency.** AutoTube has one narrator and edge-TTS timing. WhisperX is heavier than needed for that case; check the separately downloaded aligner/diarization model terms before any commercial use. |
| **9. Chatterbox** | 26,834; MIT code; Jul 21, 2026 | Expressive TTS with model variants, including a newer 110M Nano variant. Its official model card reports CPU operation at 3× real-time on 8 cores; AutoTube’s runner is smaller. It supports voice prompting/cloning and adds a Perth watermark. | **Do not replace TTS now.** Eight-core speed is not a four-vCPU GitHub-runner benchmark. Voice cloning adds consent/identity risk without solving a measured channel problem. If tested later, use a voice Archive 13 is authorized to use. |
| **10. OpenShorts** | 6,390; MIT core (with a separate commercial license for cloud code); Oct 9, 2026 | Transcribes long footage, finds highlight clips, reframes, and captions them. | **Only for Archive 13-owned/licensed source footage.** It is a repurposing tool, not an original Shorts-topic-to-script solution; the user’s current scope is Shorts-only and no long-form build. Do not use its URL-ingest path on another creator’s YouTube content. |
| **11. AutoClip** | 9,305; MIT; Oct 10, 2026 | Similar transcript-led clipping/packaging for long interviews, podcasts, and courses. | **Reference-only for owned footage.** It cannot solve the present need for distinct, trend-grounded Archive 13 stories. Its ability to accept a link does not grant rights to the linked video. |
| **12. Remotion** | 63,061; custom/source-available terms; Oct 10, 2026 | Strong React framework for highly designed motion graphics and templated compositions. | **Do not migrate the renderer.** AutoTube already renders 1080×1920 with FFmpeg, motion, crossfades, captions, title card, music ducking, and loudness normalization. A single infographic experiment might be justified; check the current license first (individuals and eligible organizations of up to three people may use it commercially under Remotion’s free terms). |
| **13. ComfyUI** | 136,795; GPL-3.0; Oct 10, 2026 | Extremely popular node-based image/video generation workflow. | **No factual-visual integration.** Generated scenery is not historical/scientific evidence and could contradict the “Every file is real” promise. If any photorealistic synthetic scene is ever used, follow YouTube’s current altered/synthetic-content disclosure rules. GPL terms also deserve review before embedding or distributing its code. |
| **14. GroundingDINO** | 10,660; Apache-2.0; Aug 12, 2024 | Open-vocabulary object detection could find literal objects in a frame. | **Defer.** It cannot prove event identity, date, context, or a scientific claim, and its repository has been quiet for over two years in this snapshot. Consider only after a measured false-match benchmark. |
| **15. PySceneDetect** | 5,227; BSD-3-Clause; Oct 10, 2026 | Detects scene cuts; useful for media-editing diagnostics. | **Optional QA utility, not a growth lever.** It can measure shot changes or pacing, but it does not know whether the visual is true, sourced, or compelling. |
| **16. OpenAI Whisper** | 110,294; MIT code and weights; Aug 31, 2026 | Broad ASR baseline used by a large ecosystem. | **Prefer faster-whisper for a local throughput test.** Whisper is a parent-model reference, not another required runtime. Validate model size and latency on the actual runner. |
| **17. yt-dlp** | 196,672; Unlicense; Oct 10, 2026 | Popular site extractor/downloader; some clipper projects use it for URL ingestion. | **Do not use to source competitor footage.** The software license does not convey rights to the downloaded audiovisual content or override YouTube’s terms/API policies. Use compliant public metadata for trend research and licensed/owned media for the edit. |
| **18. MoviePy** | 14,966; MIT; Aug 26, 2026 | Convenient Python video-editing wrapper. | **No migration.** AutoTube already has a direct FFmpeg renderer, so adding MoviePy would duplicate an existing capability without demonstrated output improvement. |
| **19. Hypit** | 20,803; modified Apache-2.0 with extra conditions; Oct 9, 2026 | Its pitch is to clone a viral reference and produce many variants. | **Reject for this channel.** It directly conflicts with the no-cloning rule and introduces a custom license, model-service cost, and originality/reused-content risk. Stars are not a defense against those issues. |

## What YouTube monetization requires of this workflow

- YouTube’s monetization policy applies to Shorts as well as long-form. It distinguishes **inauthentic** mass-produced/repetitive material from authentic work, and reused material needs meaningful original commentary, transformation, or educational/entertainment value. A repeatable format or series is not itself a reason to copy another creator’s execution. The practical Archive 13 rule remains: **format can repeat; substance cannot.** No software can guarantee a channel-level YPP review outcome.
- AI tools are not automatically disqualifying. YouTube requires disclosure when a creator uses AI to meaningfully alter or generate **realistic** content that could be mistaken for real people, events, or places. For Archive 13’s factual history/science claims, real archival evidence is the safer default; use generated visuals only when their illustrative role is clear and policy disclosures are applied when required.
- The current standard Shorts ad-revenue entry threshold is 1,000 subscribers plus 10 million qualified public Shorts views in 90 days. YouTube’s current notice says that, from **February 1, 2027**, new applicants will need 1,000 subscribers plus 20 million qualified Shorts views in 90 days (or the qualified watch-hours alternative). The expanded YPP tier is a different, earlier feature tier—not full Shorts ad revenue. Check the YPP panel in Studio before relying on thresholds because policy can change.
- YouTube Analytics defines `engagedViews` as Shorts views that passed the first frame or the viewer chose to continue; `averageViewPercentage` is the average percent of a video watched. These are the right analytics targets for the stated work. They are not guarantees of distribution or monetization.

## Current AutoTube fit and gaps

### What is already stronger than a generic generator

- **Trend evidence:** Google Trends, YouTube outlier Shorts, YouTube’s popular chart, current Reddit feeds, Hacker News, and Wikipedia spikes. The pipeline does not accept a plain Wikipedia page or evergreen seed as sufficient trend evidence. The outlier search also captures views, age, views/hour, and views relative to the source channel; AutoTube uses the **topic/interest signal**, not the Short’s footage or script.
- **Script alignment:** the writer has a current-trend angle rubric and a reviewer gate, rather than treating the broad subject as enough. A high LLM “viral potential” score is still an estimate; observed audience behavior is the stronger signal.
- **Real visuals and provenance:** Wikimedia, NASA, Internet Archive, Openverse, and configured stock sources can supply real assets. Each selected shot is checked against its narration line and carries provenance. The same is true of QA-researched replacements, which must meet a stronger match threshold.
- **Final-output gate:** final QA checks the rendered MP4—not just a pre-render storyboard—against narration, intended shots, selected asset descriptions, and hook/title. Failed renders are rejected rather than published. Keep this gate strict; do not accept an unverified replacement after the final QA attempt.
- **Renderer:** a custom FFmpeg path already provides the core vertical-video features a Remotion/MoviePy migration would be expected to add. New render technology is not the current highest-value lever.

### Gaps that external stars do not solve

1. **Trend-proof strength is uneven.** `has_current_trend_evidence()` allows one current Reddit top-day feed item to qualify. This is live evidence, but a single post is not the same as demonstrated broad virality. `viral_score` is a model judgment, not an observed view count. For high-confidence “trending now” topics, prioritize measured YouTube outlier velocity or Google Trends and keep weaker one-feed items as discovery leads until corroborated by another independent source family. YouTube’s `mostPopular` chart can corroborate, but its catalog spans broad categories and is not a niche-specific history/science trend feed by itself.
2. **The first seconds are the measured retention bottleneck.** Prioritize original, specific opening claims and earlier payoff; test hook and pacing variants inside the two approved lanes. Do not copy competitor wording, sequence, or visuals. Evaluate in YouTube Analytics, then stratify by lane, format, hook, duration, and trend source.
3. **Synthetic-image policy/brand fit needs a deliberate rule.** Root `config.yaml` currently includes `pixazo` among media sources. The final-QA repair search excludes AI-generated imagery, but the initial candidate pool includes it. For factual evidence shots, prefer real footage/photos with provenance. If Pixazo remains enabled, restrict it to clearly illustrative, non-photorealistic graphics/backgrounds and apply YouTube’s disclosure requirements to realistic synthetic depictions.
4. **Checkpoint/model terms need an inventory.** The code license, pretrained weights, TTS voice, and image/video source are separate questions. Resolve the OpenCLIP model-card warning; confirm the selected TTS voice/model terms; and keep asset-specific rights/attribution in the provenance record.
5. **Retention and subscriber conversion are analytics outcomes.** The current snapshot records 9,344 engaged Shorts views over 90 days against the 3M expanded-YPP Shorts-view threshold (coverage is partial). That gap should be treated as a growth/format problem to improve with experiments—not as something a more popular renderer can solve.

## Prioritized action plan

### P0 — finish safeguards before shipping

- **Keep the final QA-attempt guard.** A replacement is not valid until it has been re-rendered and passed final-frame QA. If the last permitted attempt fails, stop; preserve the rejection evidence and do not publish.
- **Resolve the active OpenCLIP checkpoint/model-card issue.** Document a rights review or run a labeled match benchmark for an alternative with clearer commercial scope. For example, `timm/ViT-B-16-SigLIP2` is available through OpenCLIP and identifies an Apache-2.0 model license; still test CPU speed, retrieval quality, model terms, and the existing thresholds before changing config.
- **Keep per-item rights provenance.** For Openverse, independently verify the creator/source page and license. For NASA, credit NASA and check for third-party marks/credits, recognizable people, or endorsement implications; not every asset on a NASA site is automatically unrestricted for every commercial context.

### P1 — spend effort where channel analytics point

- Run controlled, originality-preserving hook/pacing experiments in `history_mystery` and `science_nature`. Hold other factors as constant as practical; measure first-seconds retention, `averageViewPercentage`, engaged views, and subscribers per 1,000 engaged views after the analytics window. Do not declare a win from one Short.
- Preserve concrete trend evidence in each topic record: source URL, capture time, measured volume/velocity (where available), and source family. Consider requiring a second independent source family for weak single-feed signals; do not let a Wikipedia fact alone make the episode “trending.”
- Build distinct per-lane profiles inspired by Verticals v3: hook voice, sentence length, pacing, caption density, music, and visual motifs. Keep every episode’s research, claims, script, and actual visuals distinct.
- If narration errors or synchronization remain after current edge-TTS timing, run a **small optional faster-whisper QA trial** on the completed narration audio. Compare transcript-to-script edits and word timing, paying special attention to Piper fallback and historical/scientific proper nouns. Do not add a large ASR dependency to every run unless measured benefits exceed runtime and error cost.

### P2 — defer unless evidence justifies it

- Blind-test Kokoro against the current voice using identical scripts, without changing the renderer and trend system at the same time.
- Use WhisperX only if word-level timing errors persist and single-speaker alignment proves inadequate.
- Try Remotion only for a specific motion-graphic experiment that the current FFmpeg path cannot render well.
- Try GroundingDINO only for a labeled object-verification benchmark; it cannot replace the current factual/frame QA.
- Use OpenShorts/AutoClip only to clip Archive 13-owned or properly licensed source footage. Do not restart the deferred long-form build.

## Compliance / rights guardrails

- Use YouTube APIs only for allowed public metadata and channel analytics. YouTube API Services Developer Policies prohibit API clients from downloading or caching audiovisual YouTube content without prior written approval and prohibit scraping YouTube/Google applications. A downloader’s open-source license does not license the downloaded video.
- Verify each media item’s license and attribution. Openverse explicitly says its license information may be inaccurate. NASA media is generally not copyrighted in the United States, but NASA’s guidelines flag third-party material, logos/identifiers, identifiable people, and implied endorsement as exceptions or restrictions to check.
- Distinguish licenses for repository code, model checkpoints/weights, voice assets, and media. Do not assume “MIT project” means every model or source asset used by that project is MIT.
- Avoid lookalike/clone variants, stock-template mass production, and fabricated photorealistic historical/scientific scenes. No fixed numeric “script difference” or upload-frequency threshold should be invented; follow YouTube’s published policy and make the original value obvious to a viewer and reviewer.

## Sources

### Official YouTube / platform sources

- [YouTube channel monetization policies](https://support.google.com/youtube/answer/1311392?hl=en)
- [YouTube Shorts monetization policies](https://support.google.com/youtube/answer/12504220?hl=en)
- [Current YPP entry requirements and upcoming February 2027 changes](https://support.google.com/youtube/answer/12843009?hl=en)
- [YouTube Analytics API metric definitions](https://developers.google.com/youtube/analytics/metrics)
- [YouTube Data API `videos.list`](https://developers.google.com/youtube/v3/docs/videos/list)
- [YouTube API Services Developer Policies](https://developers.google.com/youtube/terms/developer-policies)
- [YouTube altered/synthetic content disclosure guidance](https://support.google.com/youtube/answer/14328491?hl=en)

### Official GitHub project pages and license/model cards

- [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo)
- [OpenMontage](https://github.com/calesthio/OpenMontage)
- [Verticals v3 / youtube-shorts-pipeline](https://github.com/rushindrasinha/youtube-shorts-pipeline)
- [Hypit](https://github.com/hypit-ai/hypit) and its [custom license](https://github.com/hypit-ai/hypit/blob/main/LICENSE)
- [OpenShorts](https://github.com/mutonby/openshorts)
- [AutoClip](https://github.com/zhouxiaoka/autoclip)
- [Openverse](https://github.com/WordPress/openverse) and its [license-verification warning](https://docs.openverse.org/api/reference/made_with_ov.html)
- [OpenCLIP code](https://github.com/mlfoundations/open_clip), [pretrained-checkpoint license notes](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md), current [`laion2b_s34b_b79k` model card](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K), and candidate [SigLIP2 OpenCLIP model card](https://huggingface.co/timm/ViT-B-16-SigLIP2)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [WhisperX](https://github.com/m-bain/whisperX), and [OpenAI Whisper](https://github.com/openai/whisper)
- [Kokoro](https://github.com/hexgrad/kokoro), [Chatterbox](https://github.com/resemble-ai/chatterbox) and [Chatterbox Nano model card](https://huggingface.co/ResembleAI/chatterbox-nano)
- [Remotion source and license FAQ](https://www.remotion.dev/docs/license/faq)
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect), [MoviePy](https://github.com/Zulko/moviepy), and [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [NASA Images and Media Usage Guidelines](https://www.nasa.gov/nasa-brand-center/images-and-media/)

---

This is a workflow and licensing assessment, not legal advice. Recheck the exact license, model card, source-media terms, and current YouTube rules before commercial deployment; do not rely on star counts or third-party summaries as clearance.
