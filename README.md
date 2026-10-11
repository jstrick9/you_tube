# AutoTube: automated, trend-adaptive, $0 faceless YouTube Shorts

AutoTube researches currently trending or viral topics, turns selected topics into **fact-checked Shorts** (licensed visuals, neural narration, karaoke captions, original music), **schedules them on your channel**, then **learns from channel analytics** and adjusts what it makes next. The configured 15–20 second duration and 70% average-view target are internal experiments, not YouTube distribution rules.

After a one-time setup of about 45 minutes it runs on GitHub Actions' free tier. It needs no server, no paid APIs and no human in the loop.

```
 ┌──────────── daily (GitHub Actions cron) ────────────┐        ┌─── nightly ───┐
 │                                                     │        │               │
 Trends ─► Safety filter ─► LLM topic pick ─► Wikipedia grounding ─► Script (format/hook/voice
 (Google Trends,  (blocklist,   (weighted by     (source text,       chosen by bandit)
  Wikipedia,       tragedies,    learned          living-person        │
  Reddit TIL,      living ppl,   category         & sensitive-topic    ▼
  HN, YT chart)    dedupe)       preference)      checks)          Fact-check: numbers ⊂ source,
                                                                    evidence quotes verified,
                                                                    LLM policy review ≥ 7/10
                                                                    (auto-rewrite ×3)
                                                                       │
   YouTube Analytics ◄── publishAt ◄── Upload ◄── Render ◄── Visuals + TTS + Music
   (engagedViews, avg view %) scheduling   (API)    (ffmpeg)   (CC/PD only, credited;
        │                                                     edge-tts / Piper;
        └──► reward ─► Thompson-sampling bandit ─► tomorrow's plan    procedural, Content-ID-free)
```

---

## 1. What it does

| Stage | How | Cost |
|---|---|---|
| **Trend analysis** | Collects current signals from fact/explainer Shorts that broke out in the last 72 h, Google Trends, Reddit's daily top posts, Hacker News, and YouTube's current popular chart; Wikipedia is used for discovery, but a Wikipedia-only topic qualifies only at ≥3× its own 30-day pageview baseline. Multi-source agreement strengthens the score. Same-day anniversaries are disabled and cannot qualify alone. | Free |
| **Virality gate** | The LLM scores each eligible candidate's viral potential 0–10 **with its trend evidence** (e.g. "2M views in 20 h", "1,496× its normal"): scroll-stopping surprise, broad appeal, proven demand, emotional pull, and whether real photos can show it. Dry news, scores, gossip and tragedies are capped low. Only picks ≥ 7 survive. They're ranked by viral score, trend strength, and which categories and **trend sources** our own analytics show get views. There is no evergreen fallback; if picks fail the accuracy checks, it re-selects only from remaining eligible live trends (up to 3 rounds). | Free tier |
| **Retention structure** | Shorts use a focused story structure and an internal 15–20 s duration experiment. `averageViewPercentage` is tracked against an internal 70% average-view target; this is not a viewer-completion rate or official distribution gate. The hook opens with a specific fact, each body line adds substance, the ending may loop as a creative choice (no analytics gain is assumed), and a new verified shot appears about every 3.5 s. A randomized, logged, balanced-block beat-two study compares a source-backed consequence-first segment-two opening with a next-distinct-surprise-first segment-two opening across distinct currently trending topics; it tests which supported story beat leads without reusing topics/scripts or relaxing factual and originality standards. A reviewer turn-strength score/reason is optional telemetry only, never a publish gate. | – |
| **Topic selection** | An LLM picks educational angles on the trends, weighted by the categories that have performed best on your channel. Gossip, tragedies, politics and living-person biographies are rejected. | Free tier |
| **Grounding** | Each factual line is checked against a specific source record. A relevant Wikipedia article is the usual backbone; thin pages may be corroborated with cited sources, and breaking-news topics require independent coverage. Source material supports facts; separate live signals establish trend eligibility. | Free |
| **Fact-checking** | Each segment carries a source quote; numeric claims are checked against the retrieved material; an automated reviewer checks accuracy, title, and advertiser-friendliness and must score ≥ 7/10. Independent review is used when a separate provider is available, but is not required by default. Failed drafts are rewritten with that feedback, up to 4 attempts. | Free |
| **Entertainment & identity** | One recognisable narrator and one promise: **"Every file is real. That's the problem."** Two numbered running series, **CASE** (history/mystery) and **FIELD** (science/nature), so a viewer has a reason to subscribe rather than just watch. The persona in `config.yaml` is a dry-witted, incredulous narrator who reacts to real facts. Up to 2 short **asides** per script ("Cheerful bunch.") are spoken after a line. They are labelled as jokes for the fact-checker, so they aren't rejected for being opinion. They are still checked for numbers and blocked words, and are dropped if they carry a number, sit on the hook, or run too long. Every video ends with a green **PROVEN ✓ stamp** plus its source card, and the description carries the promise. | Free |
| **Originality & appeal evidence** | Internal originality checks compare a draft with recent scripts and encourage a distinctive narrator voice; their numerical settings are not YouTube enforcement thresholds. Every published episode also writes `state/provenance/<date>-<title>.json` — the trend signal, claims and evidence, sources, review score and independence, and models used. `python -m autotube dossier` aggregates the records. | Free |
| **Formats** | 11 entertainment-first formats: sounds-fake-but-true, ranked weird→insane, guess-before-the-reveal (on-screen 3-2-1), plot twist, myths named by the source, could-you-handle-it, the most absurd decision, scale shock, backstory, what-if, and **creepy-but-true (October only)**. The topic selector restricts formats to those the facts can honestly support; performance learning uses measured average-view and engaged-view analytics. | – |
| **Editing feel** | **Procedural sound effects** (`sfx.py`, synthesized, so there's no Content ID risk): a whoosh on cuts, a ding when a number lands, a riser, a beat of silence and a hit before the reveal, and a stamp on the proof card. **Kinetic captions:** keywords pop in a second colour, numbers **count up** on screen, the reveal gets a flash, and asides appear in italic lower case. | Free |
| **Variety** | 11 formats × 5 hook styles × 5 voices × 5 caption themes × 7 transition styles × a unique soundtrack and SFX mix per video. | – |
| **Voice** | edge-tts neural voices with exact word timings. If that fails, it falls back automatically to **Piper**, which is fully offline and open source. | Free |
| **Visuals (moving footage first)** | **Real video clips** from Wikimedia Commons video and the NASA Image & Video Library (keyless), plus Pexels videos if you add a key. Clips get a bonus over stills when ranking, but only when the vision judge has already approved them. The judge sees the exact frame (Commons seek-thumbnail / NASA preview), and the clip is cut to a cut-free window **around that same frame**, found by frame correlation, so what plays is what was verified. If the approved frame can't be found, the clip is dropped. If no clip matches, verified photos from Commons/Openverse/Pexels are used, and they aren't left static: a local depth model (Depth Anything V2 Small, Apache-2.0, 27 MB, CPU) turns each one into a **2.5D "living photo"** camera move (dolly-in, drift, rise with real parallax). If that fails, it falls back to Ken Burns. **Only PD/CC0/CC-BY/CC-BY-SA licenses are accepted, never NC/ND.** Every author and license goes into the description. No AI-generated video is used. If realistic altered or synthetic scenes are ever added, follow YouTube's current disclosure rules; disclosure alone does not establish originality or ad suitability. | Free |
| **Visual match check (fail-closed)** | The script gives every line a concrete shot description and specific searches. Candidate images are **looked at, not matched by file name**: CLIP (local, CPU) pre-ranks them, then Gemini's free vision model scores a numbered contact sheet against the exact narration line. Same-name buildings/streets/logos, text, maps, generic stand-ins and bystanders are rejected. **Every line** must get an image scoring ≥ 7/10. A line with no match gets one re-search with new queries; if it still fails, the whole topic is skipped. There's no "close enough" fallback. | Free |
| **Final QA gate** | After rendering, one frame per shot is pulled **from the finished MP4** and judged against the narration, the intended-shot request and selected-asset provenance. A blind adversarial second look challenges positive visual matches. Failed shots can be swapped for a verified spare and re-rendered (up to 2 repairs); anything still failing is saved as `*-REJECTED` and **never uploaded**. Verdicts and context go into the run's JSON, plus a `*.shots.jpg` contact sheet. | Free |
| **Fails closed** | If the vision judge or the script fact-checker can't be reached (for example, a Gemini outage), nothing is published and the run stops. The 3 daily runs (main, backup, catch-up) fill the missing slots later. A run that produces nothing fails the workflow, and GitHub emails you. | – |
| **Music** | Synthesized per video (random key, tempo and progression), so there's zero Content ID risk. | Free |
| **Render** | 1080×1920 at 30 fps, Ken Burns motion, crossfades, word-by-word karaoke captions, a hook title card, a progress bar, sidechain-ducked music, and loudness normalized to −14 LUFS. | Free (ffmpeg) |
| **Publishing** | YouTube Data API resumable upload with `publishAt` scheduling (3 daily slots), `containsSyntheticMedia` disclosure, the made-for-kids flag, and full metadata. | Free quota |
| **Learning** | After the configured maturity window, each video is scored using `averageViewPercentage` and, when Analytics provides it, Shorts `engagedViews` per hour. Raw public `views` is retained separately and used only as a labeled fallback when engaged-view data is unavailable. The retention report also stores `startedWatching`, `stoppedWatching`, and `totalSegmentImpressions` by 5% of video runtime, with coverage and definitions; these are in-video playback events, not feed impressions, unique-viewer shares, or swipe outcomes. YouTube Analytics API does not expose Studio's **Shown in feed** or **Viewed vs swiped away** metrics here, and AutoTube does not estimate them. Rewards update the decaying Thompson-sampling bandit; none of these metrics is treated as a hard recommendation or publication gate. | Free |
| **Withdrawn videos** | If you set a scheduled video to Private in Studio, the nightly analytics run notices and marks it `withdrawn`. It then gets no reward or learning, and its slot is freed. | Free |
| **Ops** | State is committed to the repo, which also keeps the cron alive. Failure alerts go to ntfy/Discord, rendered videos are kept as artifacts for 7 days, and there's a dashboard. | Free |

---

## 2. Setup (about 45 minutes, once)

### Step 0: prerequisites
* A Google account with a YouTube channel (a **Brand Account** channel is recommended).
* A GitHub account.

### Step 1: put the code on GitHub
Push this folder to a GitHub repository. No secrets are stored in the code; they live in *Settings → Secrets and variables → Actions*.
* **Private repo** (e.g. `jstrick9/you_tube`): free accounts get **2,000 Actions minutes/month**. A day's run takes roughly 15–25 min with a Gemini key (~45 min keyless), plus ~1 min for the nightly analytics, so it fits within the free allowance. After enabling, confirm the first *scheduled* run shows up under the Actions tab. If schedules don't fire on your plan, either make the repo public or trigger it from a free external cron (see §5).
* **Public repo**: unlimited free minutes. Schedules are disabled after 60 days without commits, but the daily state commit prevents that.

### Step 2: get an LLM key (5 min, free)
* **Gemini** (recommended): https://aistudio.google.com/apikey → create a key → add it as the repo secret `GEMINI_API_KEY`.
* Optional extra failover: **Groq** (`GROQ_API_KEY`) and/or **OpenRouter** (`OPENROUTER_API_KEY`).
* With no key at all, the system falls back to Pollinations' keyless endpoint. That works (it's how this build was tested), but it's slower and less reliable.

### Step 3: YouTube API credentials (20 min, **no credit card**)
> 💳 **You do not need billing or a credit card for anything in this project.** The YouTube Data API and YouTube Analytics API are free and work in a project with no billing account. If Google shows a "Start free trial" / "Activate" banner or asks for a card, that's the optional $300 Cloud trial. Close it or click *Dismiss*, and **never click "Enable billing"**. Always go straight to `https://console.cloud.google.com` rather than the `cloud.google.com/free` marketing page, which leads into the card-required trial signup.

1. Go to https://console.cloud.google.com, create a project, then **APIs & Services → Library**. Enable **YouTube Data API v3** and **YouTube Analytics API**.
2. Open **Google Auth Platform → Branding**: add an app name, support email and developer email.
3. Under **Audience**, choose External, add yourself as a test user, then click **Publish app → In production**. ⚠️ If you skip this, refresh tokens expire every 7 days and automation silently stops.
4. Under **Clients → Create client**, choose **Web application**. Under **Authorized redirect URIs**, add exactly `https://developers.google.com/oauthplayground` and click **Create**. Copy the **Client ID** and **Client secret** (no need to download the JSON).
5. **Get the refresh token in your browser** (nothing to install):
   1. Open https://developers.google.com/oauthplayground and click the ⚙️ gear (top right).
   2. Tick **Use your own OAuth credentials**, then paste your Client ID and Client secret. Leave *Access type* on **Offline**. Close the panel.
   3. In the left box **"Input your own scopes"**, paste this line (space-separated) and click **Authorize APIs**:
      `https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/yt-analytics.readonly`
   4. Choose the Google account (or Brand Account) that **owns the channel**. At "Google hasn't verified this app", click *Advanced → Go to … (unsafe)*, then allow all permissions.
   5. Back in the Playground, under Step 2, click **Exchange authorization code for tokens** and copy the **Refresh token** (starts with `1//`).
   6. Optional: click ⚙️ and untick "Use your own OAuth credentials" so the Playground forgets them.
   *(Alternative for people with Python locally: create a **Desktop app** client instead and run `python scripts/get_refresh_token.py client_secret.json`.)*
6. Add repo secrets: `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`. **Never commit** the client secret JSON or these values to the repo.
7. **Verify:** Actions → **AutoTube check connection** → *Run workflow*. The log shows your channel name and ✓ for each permission.

### Step 4: ⚠️ request the API compliance audit (required for public videos)
YouTube locks **every video uploaded by an unaudited API project created after 28 July 2020 as private**. This is documented Google policy, and there's no way around it within the Terms of Service.
* Submit the free **YouTube API Services Audit & Quota Extension form**: https://support.google.com/youtube/contact/yt_api_form
* Describe it honestly: *"Internal tool that uploads original, fact-checked educational Shorts produced by our own channel to our own channel. Single user, uploads only to the authorizing channel; at most 3 uploads/day; reads our own analytics."* A short demo of the OAuth flow and pipeline can help. Review timing varies and Google may ask follow-up questions.
* **Before audit approval**, the YouTube API project may keep uploads private. AutoTube does not add a manual per-video approval or publication step. Complete the required API audit so publication can proceed under YouTube's project rules; until then, set `upload.enabled: false` if you want render-only runs and workflow artifacts.

### Step 5: optional extras
| Secret | Purpose |
|---|---|
| `PEXELS_API_KEY` | More stock visuals (https://www.pexels.com/api/) |
| `YOUTUBE_API_KEY` | Adds the YouTube trending chart as a trend source (costs 1 quota unit per run) |
| `NTFY_TOPIC` | Push notifications to your phone via the free ntfy app, no account needed |
| `DISCORD_WEBHOOK_URL` | Notifications in Discord |
| `GROQ_API_KEY` | Backup LLM (https://console.groq.com, free, no card). Fewer outage days because the script writer, fact-checker and image judge (Llama 4 Scout vision) can all fail over when Gemini is busy. |

### Step 6: first run
Optionally trigger **Actions → AutoTube daily → Run workflow** with `dry_run` enabled to confirm the render-only path. Then enable normal uploads; the scheduled workflow runs autonomously. Automated source checks, fact review, and final rendered-frame QA block failures—there is no required human approval step.

### Customize
Everything is in `config.yaml`: niche, categories, videos per day, publish times, voices, formats, blocklist, quality threshold, music volume and so on.

---

## 3. Compliance

These are the rules that decide whether a channel like this survives. The system is designed around them.

| Policy | Risk for automated channels | How AutoTube handles it |
|---|---|---|
| **YPP inauthentic / reused content** | YouTube assesses the channel under its current monetization policies; AutoTube does not infer fixed similarity, upload-count, or commentary triggers. | The pipeline selects a distinct, trend-backed subject for each upload, grounds claims in sources, adds Archive 13 narration/value-add, and runs local originality checks. Formats may repeat; substance must not. These safeguards do not guarantee monetization eligibility. |
| **Altered/synthetic media disclosure** | Required for *realistic* content showing things that didn't happen | The pipeline uses real licensed footage and photos (parallax camera moves on a real photo are an effect, not a fabricated scene), so the default is `containsSyntheticMedia: false`, which is correct because no realistic fabricated scenes are produced. If you add AI-generated realistic imagery, set it to `true`. The description also transparently notes the AI-assisted script and synthetic voice. |
| **Misinformation** | Unsupported or inaccurate claims | Source-grounded writing, numeric checks, evidence quotes, and an automated reviewer (independent when a separate provider is available). Unverifiable scripts are discarded. |
| **Copyright / Content ID** | Stock music and unlicensed images | Procedurally generated music. Images are PD/CC0/CC-BY/CC-BY-SA only, with attribution. NC/ND licenses are rejected in code. |
| **Advertiser-friendly guidelines** | Sensitive subjects can require context, especially in titles or graphic framing | Context-aware topic screening, title restrictions, family-friendly script checks, and an automated `advertiser_friendly` review. This is a conservative internal safeguard, not a guarantee of ad suitability. |
| **Spam / deceptive practices** | Clickbait titles, tag stuffing | The title must be supported by the content (checked by the reviewer), there are ≤ 25 relevant tags, and no misleading thumbnails. |
| **Made for kids (COPPA)** | Mislabeling | Explicit `selfDeclaredMadeForKids` from config. |
| **API Terms of Service** | Quota gaming, multiple projects | One project only. It uses ~3 of the 100 daily upload calls and fewer than 200 of the 10,000 daily units, and goes through the official audit. |

**Monetization note:** The system is designed to produce original, sourced, advertiser-friendly Shorts and runs without a human approval gate. YouTube evaluates YPP eligibility at the channel level under its current policies; AutoTube's quality checks and analytics targets are not an approval guarantee. Platform review is external to this automated pipeline.

---

## 4. Commands

```bash
pip install -r requirements.txt          # plus ffmpeg (or imageio-ffmpeg is used automatically)
python -m autotube doctor                # check keys, LLM, YouTube auth
python -m autotube trends                # today's ranked, filtered trends
python -m autotube run --dry-run -n 1    # render one video locally, no upload
python -m autotube run                   # full daily run
python -m autotube analytics             # pull stats → update learning model
python -m autotube report                # what the model has learned
python scripts/dashboard.py              # HTML dashboard → dashboard.html
python tests/test_core.py                # safety-logic unit tests
```

---

## 5. Limits & caveats (researched September 2026)

* **YouTube quota:** uploads now use their own bucket (100 `videos.insert` calls/day). Other calls share the 10,000-unit pool. There is no paid option, and creating extra projects to get more quota violates the Terms.
* **Private lock until audit:** see Step 4. This is the one step that needs Google's approval.
* **GitHub Actions:** keep the repo **public**. Actions minutes are free and unlimited for public repos, while private repos get 2,000 minutes a month, which the strict visual checks can use up (this happened in Sept 2026). All keys live in GitHub Secrets, which are never exposed in a public repo or its logs. Scheduled backup and catch-up runs check first whether anything is still needed and exit in seconds if not. GitHub's terms say Actions should relate to the repository's software project; running the project's own pipeline on a schedule is a common, low-burden use, but it's a grey area. The alternative is `deploy/crontab.txt` on any always-on machine. Scheduled workflows are disabled after 60 days without activity; the daily state commit prevents this.
* **Free LLM tiers change often.** Gemini's free quotas were cut in late 2025 and 2026. That's why the router fails over across Gemini, Groq, OpenRouter and Pollinations, and why each video needs only about 4–8 calls.
* **edge-tts** uses Microsoft's public Edge read-aloud endpoint, which is unofficial. If it breaks, the offline Piper fallback takes over automatically.
* **Alignment is enforced, but by an AI judge.** Every shot is checked twice (on selection and again on the final frames), and anything unverified is withheld rather than published. The trade-off: on days when the free vision model is overloaded, fewer (or zero) videos go out.
* **"Viral" can't be guaranteed.** The system maximizes the controllable factors (trend timing, hooks, retention-friendly pacing, captions) and learns from results, but reach is up to the algorithm and the audience.

---

## 6. Project layout

```
autotube/
  trends.py        multi-source trend collection, merge, safety filter, dedupe
  scriptwriter.py  topic selection, grounded writing, programmatic + LLM review loop
  research.py      Wikipedia grounding, living-person/sensitive checks, number fact-check
  llm.py           free-tier LLM router with failover + truncated-JSON repair
  strategy.py      Thompson-sampling bandit (category/format/hook/voice) with decay
  tts.py           edge-tts (word timings) → Piper offline fallback
  media.py         licensed video/image sourcing, clip cutting, per-line shot selection, attribution
  motion.py        2.5D depth-parallax "living photo" motion for stills
  vision.py        pixel-level relevance check (CLIP + Gemini vision contact sheets)
  qa.py            final gate: frames of the rendered MP4 re-judged vs. the spoken words
  music.py         procedural copyright-free soundtrack
  sfx.py           procedural sound effects (whoosh, ding, riser, hit, stamp)
  render.py        ffmpeg graph: Ken Burns, xfade, kinetic ASS captions, count-ups, PROVEN stamp, SFX, ducking, loudnorm
  youtube.py       OAuth refresh, resumable upload, publishAt, stats, analytics
  analytics.py     reward computation → bandit update → summary
  pipeline.py      orchestrator
.github/workflows/ daily.yml (produce + schedule), analytics.yml (nightly learning)
scripts/           get_refresh_token.py, dashboard.py
deploy/            crontab.txt, env.example (self-hosted alternative)
state/             history.json, strategy.json, analytics_summary.json (auto-committed)
```
