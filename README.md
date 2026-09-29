# AutoTube: automated, trend-adaptive, $0 faceless YouTube Shorts

AutoTube researches what's trending each day, turns the best topics into **fact-checked 30–55 s Shorts** (licensed visuals, neural narration, karaoke captions, original music), **schedules them on your channel**, then **learns from each video's performance** and adjusts what it makes next.

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
   (views/h, avg view %)  scheduling    (API)     (ffmpeg)   (CC/PD only, credited;
        │                                                     edge-tts / Piper;
        └──► reward ─► Thompson-sampling bandit ─► tomorrow's plan    procedural, Content-ID-free)
```

---

## 1. What it does

| Stage | How | Cost |
|---|---|---|
| **Trend analysis** | Merges 5 signals: Google Trends daily RSS, Wikipedia top pageviews, Reddit r/todayilearned + r/science + r/space, Hacker News, and (optionally) the YouTube mostPopular chart. Topics that show up in several sources get a higher score. | Free, no keys |
| **Topic selection** | An LLM picks educational angles on the trends, weighted by the categories that have performed best on your channel. Gossip, tragedies, politics and living-person biographies are rejected. | Free tier |
| **Grounding** | Every video is written **only** from a fetched Wikipedia article. | Free |
| **Fact-checking** | (1) Every number over 12 in the script must appear in the source. (2) Each segment carries a verbatim evidence quote that's matched against the source. (3) A separate LLM reviewer checks accuracy, misleading titles and advertiser-friendliness, and must score it ≥ 7/10. Failed drafts are rewritten with that feedback, up to 3 times. | Free |
| **Variety** | 6 formats (facts, backstory, myth-vs-fact, timeline, by-the-numbers, what-if) × 4 hook styles × 5 voices × 5 caption themes × 7 transition styles × a unique soundtrack per video. | – |
| **Voice** | edge-tts neural voices with exact word timings. If that fails, it falls back automatically to **Piper**, which is fully offline and open source. | Free |
| **Visuals** | Wikimedia Commons (images from the source article first), Openverse, and Pexels (optional key). **Only PD/CC0/CC-BY/CC-BY-SA licenses are accepted, never NC/ND.** Each asset's author and license go into the description. | Free |
| **Visual match check (fail-closed)** | The script gives every line a concrete shot description and specific searches. Candidate images are **looked at, not matched by file name**: CLIP (local, CPU) pre-ranks them, then Gemini's free vision model scores a numbered contact sheet against the exact narration line. Same-name buildings/streets/logos, text, maps, generic stand-ins and bystanders are rejected. **Every line** must get an image scoring ≥ 7/10. A line with no match gets one re-search with new queries; if it still fails, the whole topic is skipped. There's no "close enough" fallback. | Free |
| **Final QA gate** | After rendering, one frame per shot is pulled **from the finished MP4** and judged again against the words being spoken at that moment, together with checks of the title, hook text and narration. A failed shot is swapped for a spare that was verified for that line and the video is re-rendered (up to 2 repairs). Anything still failing is saved as `*-REJECTED` and **never uploaded**. The QA verdicts go into the run's JSON, plus a `*.shots.jpg` contact sheet. | Free |
| **Fails closed** | If the vision judge or the script fact-checker can't be reached (for example, a Gemini outage), nothing is published and the run stops. The 3 daily runs (main, backup, catch-up) fill the missing slots later. A run that produces nothing fails the workflow, and GitHub emails you. | – |
| **Music** | Synthesized per video (random key, tempo and progression), so there's zero Content ID risk. | Free |
| **Render** | 1080×1920 at 30 fps, Ken Burns motion, crossfades, word-by-word karaoke captions, a hook title card, a progress bar, sidechain-ducked music, and loudness normalized to −14 LUFS. | Free (ffmpeg) |
| **Publishing** | YouTube Data API resumable upload with `publishAt` scheduling (3 daily slots), `containsSyntheticMedia` disclosure, the made-for-kids flag, and full metadata. | Free quota |
| **Learning** | Each video is scored at 48 h: 0.6 × views/hour percentile + 0.4 × average-view-% percentile, relative to your own channel. That score updates a decaying Thompson-sampling bandit over category, format, hook and voice. | Free |
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
* Describe it honestly: *"Internal tool that uploads original, fact-checked educational Shorts produced by our own channel to our own channel. Single user, uploads only to the authorizing channel; ~3 uploads/day; reads our own analytics."* A short demo video of the OAuth flow and pipeline helps. Reviews typically take 1–4 weeks and may involve follow-up questions.
* **Until approval**, the system still runs daily, but uploads arrive as *private*. You can publish them with one click in Studio, or set `upload.enabled: false` and use the rendered files from the Actions artifacts.

### Step 5: optional extras
| Secret | Purpose |
|---|---|
| `PEXELS_API_KEY` | More stock visuals (https://www.pexels.com/api/) |
| `YOUTUBE_API_KEY` | Adds the YouTube trending chart as a trend source (costs 1 quota unit per run) |
| `NTFY_TOPIC` | Push notifications to your phone via the free ntfy app, no account needed |
| `DISCORD_WEBHOOK_URL` | Notifications in Discord |
| `GROQ_API_KEY` | Backup LLM (https://console.groq.com, free, no card). Fewer outage days because the script writer, fact-checker and image judge (Llama 4 Scout vision) can all fail over when Gemini is busy. |

### Step 6: first run
**Actions → AutoTube daily → Run workflow**, with `dry_run` ticked. Download the videos from the run's artifacts, check them, then run again without dry run. From then on it runs every day on its own.

### Customize
Everything is in `config.yaml`: niche, categories, videos per day, publish times, voices, formats, blocklist, quality threshold, music volume and so on.

---

## 3. Compliance

These are the rules that decide whether a channel like this survives. The system is designed around them.

| Policy | Risk for automated channels | How AutoTube handles it |
|---|---|---|
| **YPP "inauthentic content"** (renamed from "repetitious content" on 15 July 2025) | YouTube explicitly targets *"narrative stories with only superficial differences"* and *"slideshows that all have the same narration."* | A different real subject every video, 6 structurally different formats, grounded facts that add educational value, 5 rotating voices, a unique soundtrack, varied motion, captions and transitions, 45-day topic dedupe, and a modest volume (3/day by default). **Human creative input still helps a lot with YPP review** (see below). |
| **Altered/synthetic media disclosure** | Required for *realistic* content showing things that didn't happen | The pipeline uses real licensed photos, so the default is `containsSyntheticMedia: false`, which is correct because no realistic fabricated scenes are produced. If you add AI-generated realistic imagery, set it to `true`. The description also transparently notes the AI-assisted script and synthetic voice. |
| **Misinformation** | AI hallucinations | Source-only writing, numeric verification, evidence quotes and an independent reviewer. Unverifiable scripts are discarded, never published. |
| **Copyright / Content ID** | Stock music and unlicensed images | Procedurally generated music. Images are PD/CC0/CC-BY/CC-BY-SA only, with attribution. NC/ND licenses are rejected in code. |
| **Advertiser-friendly guidelines** | Tragedy-baiting trends | A 60+ term blocklist is applied to trends, source articles and final scripts, plus the reviewer's `advertiser_friendly` check. Living-person biographies are skipped by default. |
| **Spam / deceptive practices** | Clickbait titles, tag stuffing | The title must be supported by the content (checked by the reviewer), there are ≤ 25 relevant tags, and no misleading thumbnails. |
| **Made for kids (COPPA)** | Mislabeling | Explicit `selfDeclaredMadeForKids` from config. |
| **API Terms of Service** | Quota gaming, multiple projects | One project only. It uses ~3 of the 100 daily upload calls and fewer than 200 of the 10,000 daily units, and goes through the official audit. |

**Honest note on monetization:** AutoTube produces content that is fact-checked, varied and genuinely informative, which is what YouTube says it rewards. But YPP approval involves human reviewers judging the channel as a whole. The channels that do best add some human element: picking the niche, occasionally reviewing output, writing a channel trailer, replying to comments. The system runs without you, but a few minutes a week of your attention improves both the monetization odds and the results.

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
  media.py         licensed image sourcing, per-line shot selection, attribution
  vision.py        pixel-level relevance check (CLIP + Gemini vision contact sheets)
  qa.py            final gate: frames of the rendered MP4 re-judged vs. the spoken words
  music.py         procedural copyright-free soundtrack
  render.py        ffmpeg graph: Ken Burns, xfade, ASS karaoke captions, ducking, loudnorm
  youtube.py       OAuth refresh, resumable upload, publishAt, stats, analytics
  analytics.py     reward computation → bandit update → summary
  pipeline.py      orchestrator
.github/workflows/ daily.yml (produce + schedule), analytics.yml (nightly learning)
scripts/           get_refresh_token.py, dashboard.py
deploy/            crontab.txt, env.example (self-hosted alternative)
state/             history.json, strategy.json, analytics_summary.json (auto-committed)
```
