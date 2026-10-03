# Research: free open-source routes to longer, better videos

Researched 3 Oct 2026. Every claim marked **[verified]** was tested live from this
workspace during the research; everything else is sourced reading and should be treated
as weaker. The point of the distinction is that three separate defects in this repo have
come from trusting a plausible-looking string that nobody had actually called.

---

## 1. The constraint that eliminates most of the field

Almost every "best open-source AI video model 2026" list recommends Wan 2.2,
HunyuanVideo, LTX, Mochi, CogVideoX or MAGI-2. **None of them are usable here.** They
need 8–80 GB of VRAM and this pipeline runs on GitHub-hosted runners, which have no GPU
at all. A list that does not mention VRAM is not answering our question.

What we actually have is better than it first looks:

| | Value | Source |
|---|---|---|
| Repo visibility | **public** | [verified] GitHub API |
| Actions minutes | **free and unlimited** | public repos, standard runners |
| Runner shape | **4 vCPU / 16 GB RAM** | public repos get double the private shape |
| Job ceiling | 6 h (we currently cap at 150 min) | `daily.yml` |

So compute is not scarce — **wall-clock is**. We can spend an hour of CPU per video and
it costs nothing. That reframes the question from "what can we afford" to "what finishes
on 4 cores". It also means the 150-minute timeout is a self-imposed limit, not a
platform one.

---

## 2. What is already good and should not be replaced

Worth stating plainly, because the temptation with a research task is to recommend
replacing things that are working.

- **`render.py` is not the bottleneck.** It already does 1080×1920, Ken Burns motion,
  crossfades, karaoke captions, a hook title card, a progress bar, ducked procedural
  music and loudness normalisation — as a single ffmpeg filter graph. Swapping it for
  Remotion or Revideo would be a large rewrite for no quality gain, and Remotion's
  licence is `NOASSERTION` [verified] because it carries a paid company licence. Avoid.
- **edge-tts is decent.** It is a Microsoft neural voice and free. Kokoro is a quality
  upgrade on paper but not a dramatic one (see §4).
- **The vision gate is correct.** "Nothing unverified is published" is what makes the
  provenance dossier worth anything. Nothing below should weaken it.

---

## 3. The finding that matters most: real archival *motion* footage

The pipeline currently animates **still images**. The single largest quality and length
unlock is real public-domain **film**, and it is free and keyless.

**Internet Archive** [verified live]:
- `10,461` public-domain films in the Prelinger collection alone
- keyless JSON search and metadata APIs
- sample item `FarWeste1955` returns licence `creativecommons.org/licenses/publicdomain/`
- playable `.mp4` confirmed, and HTTP **206** range requests work, so we can fetch a
  clip without pulling a 63 MB file

**Why this beats every AI video model for this channel specifically:**
1. It is **free and runs on CPU** — we are downloading, not generating.
2. It is **real**, which is the entire brand: *"Every file is real. That's the problem."*
3. It is **public domain**, so it is safe for a monetised channel.
4. It is the evidence a monetisation appeal wants — sourced archival material, not
   synthetic footage.
5. It supplies the **visual variety that actually caps video length**. A 3–10 minute
   video needs 60–200 distinct shots; that, not the renderer, is why long-form was hard.

**The honest caveat:** the footage is `640×480` and often 4:3 [verified]. Upscaling that
to a 1080×1920 vertical frame is real work. Mitigations, in order of preference: use it
as *insert* b-roll between stills rather than full-frame; blurred-pillarbox composite
(the renderer already does background blur); Real-ESRGAN upscaling on CPU (free but
slow — viable precisely because minutes are unlimited).

**Other verified keyless archives:**
- **NASA Images API** — keyless, works [verified]. Public domain. Strong for
  `science_nature`.
- **Library of Congress** — `fo=json` responds [verified] but my test query returned 0
  results; the query format needs work before trusting it.
- Already wired up: Wikimedia Commons, Openverse.

---

## 4. Ranked recommendations

| # | Change | Effort | Why |
|---|---|---|---|
| 1 | **Internet Archive film as b-roll** | M | Biggest quality jump; on-brand; unlocks long-form |
| 2 | **Raise the 150-min job cap** | XS | Minutes are free; we are leaving budget unused |
| 3 | **faster-whisper for forced alignment** | S | 25,683★, MIT, active [verified]. True word-level caption timing |
| 4 | **NASA + LoC as extra visual sources** | S | More real supply = fewer "no image shows this line" failures |
| 5 | **Kokoro TTS** | M | 9,136★, Apache-2.0, CPU, MOS 4.2. A real but incremental gain |

On **#3**: the renderer already draws karaoke captions, but timing is only as good as
the timestamps feeding it. Note Groq's catalogue includes `whisper-large-v3` and
`whisper-large-v3-turbo` [verified earlier today] — we already hold that key, so
alignment could be an API call rather than a CPU job.

On **#5**: Kokoro is Apache-2.0 and CPU-friendly, but its last push was 2025-08-06
[verified] — stable, not actively developed. Chatterbox (26,658★, MIT) is better quality
and actively maintained but wants a GPU, so it is out on this runner.

---

## 5. Rejected, with reasons

Recording these so they are not re-proposed later.

- **Gemini image generation** — the Feb 2026 write-ups claiming "500 free images/day"
  are **out of date**. As of Sept 2026 the Gemini API has **no free tier for image
  models**; every image is billed. Would have been a costly recommendation.
- **Pollinations image generation** — keyless and fast (4.2 s) [verified], but the
  returned image carried a **`pollinations.ai` watermark despite `nologo=true`**, came
  back `576×1024` rather than the requested 1080×1920, and prompt adherence was poor (a
  request for a 1940s gas cylinder produced a bottle). Not usable for a monetised
  channel.
- **AI-generated imagery in general** — contradicts *"Every file is real"*, weakens the
  originality evidence behind a monetisation appeal, and edges toward the disclosure
  rules for realistic synthetic media. This channel's advantage is that its material is
  genuine; generating it away is a bad trade.
- **Piper TTS** — **archived** [verified]. Do not adopt.
- **Remotion / Revideo** — licence is not free for commercial use; `revideo` under the
  name I checked does not exist [verified].
- **Local video diffusion (Wan, Hunyuan, LTX, Mochi, CogVideoX, MAGI-2)** — no GPU.
- **MoneyPrinterTurbo / ShortGPT** — 128k★ and 8k★ respectively, MIT. Worth *reading*
  for ideas, but they are whole-pipeline replacements that do less than this repo
  already does (no provenance, no originality gate, no compliance layer). ShortGPT is
  also stale (last push 2025-02-10) [verified].

---

## 6. Suggested order of work

1. **Raise the job cap** and add Internet Archive as a *source*, behind the existing
   vision gate so footage is still verified to show what the line claims.
2. **Insert-cut b-roll**: keep stills as the spine, cut 1.5–3 s of archival film over
   the strongest beat. Lowest-risk way to get the quality gain without a render rewrite.
3. **Only then** consider long-form. The visual-supply problem is what made 3–10 minutes
   infeasible before; it should be solved and proven on Shorts first.

A note on scope: this reverses the earlier "Shorts only" decision. Everything above
improves Shorts on its own, and long-form becomes a config change rather than a new
pipeline — which is roughly what `docs/LONGFORM.md` already assumed.
