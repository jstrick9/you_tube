"""Credentials plumbing and reviewer independence.

Two failure modes here are invisible until a live run and expensive when they land:
a secret that exists but is never passed into the job, and a "reviewer" that turns out
to be the same model that wrote the script.
"""
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
WF = ROOT / ".github" / "workflows"

# Read from the environment but never supplied by CI secrets:
# local TTS paths, config overrides, and the keyless provider's optional token.
LOCAL_ONLY = {"AUTOTUBE_CONFIG", "AUTOTUBE_TTS", "AUTOTUBE_DEPTH_MODEL",
              "PIPER_DIR", "PIPER_VOICE", "POLLINATIONS_TOKEN"}


def env_vars_read_by_code() -> set[str]:
    src = "\n".join(p.read_text() for p in (ROOT / "autotube").glob("*.py"))
    found = set(re.findall(r'environ(?:\.get)?\(\s*["\']([A-Z_]+)["\']', src))
    return found - LOCAL_ONLY


def env_block(workflow: str) -> set[str]:
    text = (WF / workflow).read_text()
    return set(re.findall(r'^\s+([A-Z_]+):\s*\$\{\{\s*secrets\.', text, re.M))


def test_the_daily_job_passes_every_secret_the_code_reads():
    """A secret that exists in the repo but is not mapped into the job is invisible:
    the code sees an empty string and silently degrades."""
    missing = env_vars_read_by_code() - env_block("daily.yml")
    assert not missing, f"daily.yml never passes these: {sorted(missing)}"


def test_the_analytics_job_passes_what_analytics_needs():
    """Analytics talks to the YouTube Analytics API over OAuth and uses no LLM."""
    need = {"YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"}
    assert need <= env_block("analytics.yml")
    src = (ROOT / "autotube" / "analytics.py").read_text()
    assert "LLM(" not in src, "analytics must not need an LLM; analytics.yml passes no LLM keys"


def test_no_workflow_references_a_secret_the_code_never_reads():
    """Dead secret wiring is a slow trap: it looks configured and does nothing."""
    referenced = env_block("daily.yml") | env_block("analytics.yml")
    known = env_vars_read_by_code() | {
        # notification sinks, read via common.notify
        "NTFY_TOPIC", "DISCORD_WEBHOOK_URL",
        # OAuth pair consumed by the google client, not via os.environ directly
        "YT_CLIENT_ID", "YT_CLIENT_SECRET",
    }
    assert not (referenced - known), f"workflow passes unused secrets: {sorted(referenced - known)}"


# ── reviewer independence ───────────────────────────────────────────────────
def test_the_reviewer_is_never_the_model_that_wrote_the_draft():
    """A writer grading its own draft agrees with itself.

    This used to require the reviewer to open on a *different provider*. That is no
    longer the right guarantee. Gemini is the only vision-capable provider and image
    verification is a hard gate, so spending its free-tier quota on reviews is what
    429'd the vision call and discarded a finished script. Reviews now prefer Groq,
    which is also where the writer starts.

    Independence is enforced per model, not per provider, and Groq's models are not one
    family - gpt-oss is OpenAI's and qwen is Alibaba's, different lineages with
    different failure modes. A live run confirmed the pairing works in practice
    (gpt-oss-120b reviewing a qwen draft and vice versa). What must never happen is a
    model reviewing itself, so that is what this pins.
    """
    llm = CFG["llm"]
    reviewers = llm["review_providers"]
    assert reviewers, "there must be a review order"
    # Whichever provider writes first must offer at least two models, so the reviewer
    # always has somewhere else to go when it lands on the same provider.
    first = llm["providers"][0]
    if reviewers[0] == first:
        pool = llm.get(f"{first}_review_models") or llm.get(f"{first}_models", [])
        assert len(pool) >= 2, (
            f"review opens on {first}, the same provider that writes, so {first} needs "
            f"at least two models for the reviewer to differ from the writer")
    # And a second provider must remain reachable as a fallback.
    assert len(reviewers) >= 2, "review needs a fallback provider"
    # Both keyed providers must stay in the review order so the fallback is real.
    assert {"gemini", "groq"} <= set(reviewers), \
        "both keyed providers should remain reachable for review"


def test_every_review_provider_can_actually_be_reached():
    """A review_providers entry with no models configured silently does nothing."""
    for p in CFG["llm"]["review_providers"]:
        models = CFG["llm"].get(f"{p}_review_models") or CFG["llm"].get(f"{p}_models")
        assert models, f"review provider {p} has no models"


def test_the_writer_model_is_recorded_and_excluded_from_review():
    """Independence is only real if the writer's exact model is passed as `avoid`."""
    src = (ROOT / "autotube" / "scriptwriter.py").read_text()
    assert 'script["_writer_model"] = self.llm.last_used' in src
    assert 'avoid=script.get("_writer_model")' in src
    assert 'review["independent"] = bool(review["reviewer"]) and review["reviewer"] != script.get("_writer_model")' in src


def test_independence_is_recorded_rather_than_assumed():
    """The provenance dossier reports it per episode, so a run where the fallback
    collapsed onto one provider is visible instead of quietly counted as reviewed."""
    src = (ROOT / "autotube" / "provenance.py").read_text()
    assert '"independent": review.get("independent")' in src
    assert "independently_reviewed" in src


def test_groq_retries_without_json_mode_when_the_model_refuses_it():
    """Reasoning models 400 on response_format=json_object; we must ask again without it.

    This single line is why the review stage had no model but the writer's own: every
    Groq call 400'd, Groq was retired on first use, and the skip path logged nothing.
    """
    import autotube.llm as L

    calls = []

    class R:
        def __init__(self, code, text, payload=None):
            self.status_code, self.text, self._p = code, text, payload

        def json(self):
            return self._p

    class FakeHTTP:
        def post(self, url, headers=None, json=None, timeout=None):
            calls.append(dict(json))
            if "response_format" in json:
                return R(400, '{"error":{"code":"json_validate_failed"}}')
            return R(200, "", {"choices": [{"message": {"content": '{"ok": true}'}}]})

    orig = L.http
    L.http = lambda: FakeHTTP()
    try:
        out = L._openai_compatible("https://api.groq.com/openai/v1/chat/completions", "k",
                             "openai/gpt-oss-120b", "sys", "usr", 0.0, json_mode=True)
    finally:
        L.http = orig

    assert len(calls) == 2, "should retry once without the json_object constraint"
    assert "response_format" in calls[0] and "response_format" not in calls[1]
    assert out


def test_every_configured_model_id_is_syntactically_a_real_candidate():
    """Guard the class of bug that cost three production runs: invented model ids.

    Cannot hit the network here, so this pins the ids to the provider catalogues that
    `doctor` verified live, and will fail loudly if someone edits one by hand again.
    """
    cfg = CFG["llm"]
    groq_real = {"openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b",
                 "qwen/qwen3.8-27b", "allam-2-7b"}
    for key in ("groq_models", "groq_review_models", "groq_vision_models"):
        for m in cfg.get(key, []) or []:
            assert m in groq_real, (
                f"{key}: {m!r} is not in Groq's catalogue. Run `autotube doctor` — it "
                f"lists the live model ids. Do not guess at model names.")


def test_length_instruction_rejects_both_ends_not_just_short():
    """The budget instruction used to say only "Too short = rejected".

    That is a one-sided push, and the model duly overshot: 58, 68 and 76 words against a
    52-word ceiling in a single live run, which became the largest single cause of
    abandoned scripts once the quality gates stopped being the bottleneck.
    """
    src = (ROOT / "autotube" / "scriptwriter.py").read_text()
    assert "Too short = rejected." not in src, "one-sided length instruction is back"
    assert "words per segment" in src, "per-segment budget is easier to hit than a total"
    assert "too short AND too long" in src.lower() or "Both ends are rejected" in src


def test_rate_limited_runs_are_not_reported_as_broken_builds():
    """Free-tier exhaustion is an operating condition, not a defect.

    Both ended as "produced nothing, exit 1", so five rate-limited runs in one day sat
    next to real failures and trained the red X to mean nothing. The alert must still
    fire; only the exit code changes.
    """
    import autotube.llm as L

    src = (ROOT / "autotube" / "__main__.py").read_text()
    assert "rate_limited()" in src, "the run must distinguish rate limiting from failure"
    assert "notify(" in src, "a rate-limited run must still alert; it just must not go red"

    before = L.rate_limited()
    L.note_rate_limited("groq:openai/gpt-oss-20b")
    assert "groq:openai/gpt-oss-20b" in L.rate_limited()
    assert L.rate_limited() >= before


def test_provider_errors_keep_enough_body_to_act_on():
    """200 chars cut 429s off before they said per-minute or per-day."""
    src = (ROOT / "autotube" / "llm.py").read_text()
    assert "r.text[:200]" not in src, "429 bodies truncated before the useful part"
    assert src.count("retry after") >= 2, "both error paths should surface retry-after"


def test_a_long_quota_window_retires_the_model_instead_of_sleeping():
    """Groq answered "retry after 1100" and the run kept sleeping 45s and asking again.

    Five models each backing off the cap, across every topic, took one run to 51 minutes
    with nothing produced. A quota window measured in minutes cannot be waited out
    inside a run, so the provider's own advice should end the attempt, not pace it.
    """
    from autotube.llm import advertised_retry

    assert advertised_retry("HTTP 429 (retry after 1100): {}") == 1100.0
    assert advertised_retry("HTTP 429 (retry after 30): {}") == 30.0
    assert advertised_retry("HTTP 429 (retry after 1ms): {}") == 0.001, "ms must not read as seconds"
    assert advertised_retry("HTTP 429: no header") == 0.0

    import re as _re
    src = (ROOT / "autotube" / "llm.py").read_text()
    m = _re.search(r"advertised_retry\(msg\)\s*>\s*(\d+)", src)
    assert m, "long quota windows must retire the model rather than pace retries"
    # Sleeps are capped at 45s across a few attempts, so a window of a minute or two is
    # genuinely waitable and should not retire a model; anything past a few minutes is not.
    threshold = int(m.group(1))
    assert 90 <= threshold <= 300, (
        f"retirement threshold {threshold}s: below ~90s it discards models that would "
        f"have come back within the backoff ladder, above ~300s it stalls the run")


def test_top_up_cadence_stays_within_free_tier_budget():
    """Top-ups guard against GitHub dropping crons, but they must not starve each other.

    A satisfied top-up is free: it reads history.json and exits. The cost appears on a
    bad day, when every remaining slot attempts a full production of 50+ LLM calls.
    At the old two-hour cadence that was seven attempts competing for the same ~20
    minute Groq quota window, so one failed morning could lock out the whole day.
    """
    wf = yaml.safe_load((WF / "daily.yml").read_text())
    crons = [c["cron"] for c in wf[True]["schedule"]]
    topups = [c for c in crons if "/" in c.split()[1]]
    assert topups, "the drop-resilience top-up slot should still exist"
    step = int(topups[0].split()[1].split("/")[1])
    assert step >= 4, (
        f"top-ups every {step}h will stack full production attempts on a failed day; "
        f"keep them at 4h or sparser")
    # Still enough slots to survive GitHub silently dropping a scheduled run.
    lo, hi = (int(x) for x in topups[0].split()[1].split("/")[0].split("-"))
    assert len(range(lo, hi + 1, step)) >= 3, "need several independent chances per day"


# ── scheduled top-up pre-flight ─────────────────────────────────────────────
def test_preflight_gate_handles_a_videos_per_day_range():
    """Every scheduled top-up run died in 18 seconds before this.

    videos_per_day became a [lo, hi] range, but the pre-flight copy of the rule still
    called int() on it. Top-ups are the mechanism that covers GitHub silently dropping
    a cron, so the safety net had been down since the two changes met.
    """
    import subprocess

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "needed_today.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"pre-flight gate crashed: {r.stderr[-400:]}"
    assert "still needed" in r.stdout


def test_the_daily_target_rule_exists_only_once():
    """The crash was duplication, not arithmetic — so pin that there is one copy."""
    from autotube.target import daily_target

    pipeline_src = (ROOT / "autotube" / "pipeline.py").read_text()
    assert "def daily_target" not in pipeline_src, "pipeline must import the shared rule"
    script_src = (ROOT / "scripts" / "needed_today.py").read_text()
    assert "videos_per_day" not in script_src, "the pre-flight gate must not re-derive it"

    # Both forms, and a range must be stable within a day or top-ups disagree with
    # each other about whether the day is already finished.
    cfg = {"schedule": {"videos_per_day": [1, 3]}, "channel": {}}
    assert daily_target(cfg) == daily_target(cfg)
    assert 1 <= daily_target(cfg) <= 3
    assert daily_target({"schedule": {"videos_per_day": 2}, "channel": {}}) == 2
    assert daily_target({"schedule": {"videos_per_day": [2, 2]}, "channel": {}}) == 2


def test_preflight_gate_imports_without_the_heavy_dependencies():
    """It runs before the full install, with only PyYAML present."""
    import subprocess

    probe = (
        "import sys;\n"
        "sys.modules.update({m: None for m in ('requests','PIL','numpy','feedparser')});\n"
        f"sys.path.insert(0, {str(ROOT)!r});\n"
        "from autotube.target import daily_target;\n"
        "print(daily_target({'schedule': {'videos_per_day': [1, 3]}, 'channel': {}}))\n"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert r.returncode == 0, f"shared rule pulled in a heavy import: {r.stderr[-400:]}"


def test_new_free_providers_are_wired_end_to_end():
    """A provider is only real if the key reaches it AND doctor can check its models.

    The pipeline produced nothing on 3 Oct because both text providers were rate
    limited at once. These three are permanent free tiers needing no credit card; they
    exist to make that single point of failure a three-deep one.
    """
    import autotube.llm as L

    for p in ("cerebras", "mistral", "cloudflare"):
        assert p in L.PROVIDERS, f"{p} missing from the provider registry"
        assert p in L.KEYED_PROVIDERS, f"{p} missing from the keyed map (would be invisible)"
        assert p in CFG["llm"]["providers"], f"{p} configured but never used for writing"
        assert CFG["llm"].get(f"{p}_models"), f"{p} has no models configured"

    # Guessed model ids have cost this repo three production runs, so doctor must be
    # able to check these against the live catalogue rather than us trusting the string.
    src = (ROOT / "autotube" / "llm.py").read_text()
    for p in ("cerebras", "mistral", "cloudflare"):
        assert f'"{p}"' in src.split("def available_models")[1].split("def audit_models")[0], \
            f"available_models() cannot verify {p} ids"


# ── per-lane editorial profiles ─────────────────────────────────────────────
def test_the_two_lanes_do_not_sound_like_one_channel():
    """A single house voice across every upload is measurable template similarity.

    That is the "generic or repetitive content" rejection category, and bulk
    demonetisation starts at five videos sharing a template with under 20% script
    variation. So the lanes differing is a compliance property, not a style preference.
    """
    from autotube import lanes

    hist = lanes.writer_rule({"category": "history"}, CFG)
    sci = lanes.writer_rule({"category": "space"}, CFG)
    assert hist and sci, "both lanes should carry a profile"
    assert hist != sci, "the two lanes must not produce the same writer guidance"
    for rule in (hist, sci):
        assert "VOICE:" in rule and "PACING:" in rule


def test_lane_resolves_from_category_when_not_stated():
    """Only outlier topics carry `lane`; Wikipedia spikes and anniversaries do not."""
    from autotube import lanes

    assert lanes.lane_for({"lane": "history_mystery"}, CFG) == "history_mystery"
    assert lanes.lane_for({"category": "history"}, CFG) == "history_mystery"
    assert lanes.lane_for({"category": "Space"}, CFG) == "science_nature", "case-insensitive"
    assert lanes.lane_for({"category": "nonsense"}, CFG) is None


def test_an_unknown_lane_leaves_the_prompt_exactly_as_it_was():
    """Adding this module must not quietly change output for unmapped topics."""
    from autotube import lanes

    assert lanes.writer_rule({"category": "nonsense"}, CFG) == ""
    assert lanes.writer_rule({}, CFG) == ""
    assert lanes.profile_for({}, CFG) == lanes.DEFAULT_PROFILE


def test_the_writer_actually_uses_the_lane_rule():
    """A profile nothing reads is just config."""
    src = (ROOT / "autotube" / "scriptwriter.py").read_text()
    assert "lanes.writer_rule(topic, self.cfg)" in src
    assert "{lane_rule}" in src, "the rule must be interpolated into the prompt"


def test_vision_is_not_a_single_point_of_failure():
    """Vision is a hard gate: no vision model, no video, however good the script.

    Gemini was the only provider that could serve it, and a 429 there ended two of the
    last three full runs *after* the script had been written and verified. A second
    keyed provider is the difference between a slow run and a lost one.
    """
    vis = CFG["llm"]["vision_providers"]
    assert len(vis) >= 2, f"vision has only {vis} — one rate limit discards finished work"
    # Derived, not literal. A hardcoded set here is the same bug this suite exists to
    # catch: it silently falls behind the router every time a provider is added.
    import autotube.llm as L
    keyed = set(L.KEYED_PROVIDERS)
    for p in vis:
        assert p in keyed, f"{p} is not a keyed provider; vision never uses the keyless tier"
        assert CFG["llm"].get(f"{p}_vision_models") or p == "gemini", \
            f"{p} is in vision_providers but has no vision models configured"


def test_doctor_reports_every_provider_key_it_depends_on():
    """A hardcoded key list falls behind the router and hides working providers.

    Mistral's key was set and its catalogue fetched fine, while doctor's own key report
    never mentioned it — making a working provider look identical to a missing one.
    """
    src = (ROOT / "autotube" / "__main__.py").read_text()
    assert "KEYED_PROVIDERS" in src, "doctor must derive its key list from the router"
    assert "CLOUDFLARE_ACCOUNT_ID" in src, "Cloudflare needs an account id as well as a token"
    import autotube.llm as L
    for env in L.KEYED_PROVIDERS.values():
        assert env in src or "KEYED_PROVIDERS" in src


# ── Internet Archive public-domain footage ──────────────────────────────────
def _fake_archive(monkeypatch, docs, meta):
    import autotube.media as M

    def fake_get_json(url, params=None, headers=None, **kw):
        if "advancedsearch" in url:
            return {"response": {"docs": docs}}
        return meta
    monkeypatch.setattr(M, "get_json", fake_get_json)
    return M


def test_archive_refuses_items_that_are_not_public_domain(monkeypatch):
    """archive.org is a host, not a rights clearinghouse.

    An unfiltered search there is a licensing problem, not a free-footage win, so a
    declared licence that is not PD/CC must be rejected even inside a PD collection.
    """
    M = _fake_archive(monkeypatch,
                      [{"identifier": "x", "title": "nuclear test", "licenseurl": "http://example.com/all-rights"}],
                      {"metadata": {"title": "nuclear test"},
                       "files": [{"name": "a.mp4", "size": "999", "width": "640", "height": "480"}]})
    assert M.archive_video_search("nuclear test", 3) == []


def test_archive_rejects_noncommercial_licences(monkeypatch):
    """A monetised channel cannot use NC material."""
    M = _fake_archive(monkeypatch,
                      [{"identifier": "x", "title": "nuclear test",
                        "licenseurl": "http://creativecommons.org/licenses/by-nc/4.0/"}],
                      {"metadata": {"title": "nuclear test"},
                       "files": [{"name": "a.mp4", "size": "999"}]})
    assert M.archive_video_search("nuclear test", 3) == []


def test_archive_drops_irrelevant_hits_before_they_cost_a_vision_call(monkeypatch):
    """Vision is the scarcest quota and the hard gate; noise there is expensive.

    archive.org's keyword ranking is weak enough to answer "deep sea ocean" with
    "COLORADO PLATEAU", so an item sharing no query word with its own title or
    description is dropped for free rather than screened for a vision call.
    """
    M = _fake_archive(monkeypatch,
                      [{"identifier": "x", "title": "Colorado Plateau",
                        "licenseurl": "http://creativecommons.org/licenses/publicdomain/"}],
                      {"metadata": {"title": "Colorado Plateau", "description": "a film about canyons"},
                       "files": [{"name": "a.mp4", "size": "999"}]})
    assert M.archive_video_search("submarine", 3) == [], "irrelevant item should be dropped"
    assert len(M.archive_video_search("canyons", 3)) == 1, "a real term match should pass"


def test_archive_asset_matches_the_shape_other_video_sources_use(monkeypatch):
    M = _fake_archive(monkeypatch,
                      [{"identifier": "dugout", "title": "Project Dugout",
                        "licenseurl": "http://creativecommons.org/licenses/publicdomain/"}],
                      {"metadata": {"title": "Project Dugout", "creator": "AEC"},
                       "files": [{"name": "big.mp4", "size": "900", "width": "640", "height": "480"},
                                 {"name": "small.mp4", "size": "100"}]})
    a = M.archive_video_search("dugout", 3)[0]
    for k in ("kind", "url", "thumb", "page", "title", "license", "author", "source", "w", "h", "variants"):
        assert k in a, f"asset missing {k} — would break the common video path"
    assert a["kind"] == "video"
    assert a["url"].endswith("big.mp4"), "should prefer the largest derivative"
    assert "publicdomain" in a["license"].lower() or "public domain" in a["license"].lower()


def test_archive_is_actually_wired_into_the_source_list():
    import autotube.media as M
    assert "archive_video" in CFG["media"]["sources"]
    src = (ROOT / "autotube" / "media.py").read_text()
    assert "archive_video_search" in src.split("def _candidates_for")[1], "registered but never dispatched"


def test_cloudflare_accepts_the_common_misspelling(monkeypatch):
    """The secrets were created as CLOUDFARE_*; the code read CLOUDFLARE_*.

    The provider then looked configured, reported nothing, and silently never ran —
    the failure mode that has cost this repo more time than any other. Accepting both
    spellings is one line.
    """
    import autotube.llm as L

    for v in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID",
              "CLOUDFARE_API_TOKEN", "CLOUDFARE_ACCOUNT_ID"):
        monkeypatch.delenv(v, raising=False)
    assert L.cloudflare_creds() == (None, None)

    monkeypatch.setenv("CLOUDFARE_API_TOKEN", "t")
    monkeypatch.setenv("CLOUDFARE_ACCOUNT_ID", "a")
    assert L.cloudflare_creds() == ("t", "a"), "misspelled secrets must still work"
    assert L._provider_key("cloudflare") == "t"

    # A token without an account id is not usable, and must not count as configured.
    monkeypatch.delenv("CLOUDFARE_ACCOUNT_ID")
    assert L._provider_key("cloudflare") is None


def test_both_spellings_reach_the_workflows():
    for wf in ("daily.yml", "check.yml"):
        s = (WF / wf).read_text()
        for v in ("CLOUDFLARE_API_TOKEN", "CLOUDFARE_API_TOKEN",
                  "CLOUDFLARE_ACCOUNT_ID", "CLOUDFARE_ACCOUNT_ID"):
            assert v in s, f"{wf} does not pass {v}"


def test_generated_imagery_is_labelled_and_last():
    """"Every file is real" is the channel's claim and the appeal's evidence.

    Generated frames are allowed as a last resort, but they must never be silently
    mixed into a provenance record whose whole value is being truthful about what the
    footage is.
    """
    import autotube.media as M
    src = (ROOT / "autotube" / "media.py").read_text()

    body = src.split("def pixazo_image")[1].split("\ndef ")[0]
    assert '"synthetic": True' in body, "generated assets must flag themselves"
    assert "AI-generated" in body, "licence/author must say so plainly"

    dispatch = src.split("def _candidates_for")[1].split("\ndef ")[0]
    i_px = dispatch.index("pixazo_image")
    for real in ("commons_search", "archive_video_search", "openverse_search"):
        assert dispatch.index(real) < i_px, f"{real} must be tried before generated imagery"

    assert M.pixazo_image("x") == [] or os.environ.get("PIXAZO_API_KEY"), \
        "must no-op without a key"


def test_pixazo_is_configured_last_in_the_source_order():
    srcs = CFG["media"]["sources"]
    assert srcs[-1] == "pixazo", f"generated imagery should be the final fallback, got {srcs}"


# ── series numbering ────────────────────────────────────────────────────────
def test_package_returns_series_and_episode(monkeypatch, tmp_path):
    """Exercise package() rather than grep it.

    The previous version of this test searched pipeline.py for the fix and passed
    while the behaviour was still broken, because package() builds two dicts: a
    `record` written beside the video, and a separate return value that run() reads to
    build the history entry. The fix had gone into the first one. A test that reads
    source can only confirm that code exists, not that it runs.
    """
    import types
    import autotube.pipeline as P

    monkeypatch.setattr(P, "render", types.SimpleNamespace(thumbnail=lambda *a, **k: "t.jpg"))
    monkeypatch.setattr(P, "_shots_sheet", lambda *a, **k: None)
    monkeypatch.setattr(P, "build_description", lambda *a, **k: "desc")

    out = tmp_path / "v.mp4"
    out.write_bytes(b"x")
    a = {"script": {"segments": [{"text": "a b c"}], "tags": ["t"]},
         "source": {"title": "T", "url": "u"}, "visuals": {"shots": []}, "out": out,
         "review": {"score": 8}, "plan": {"format": "f"}, "tts": {"engine": "e"},
         "hook_card": "h", "title": "FIELD #003 — X", "topic": {"topic": "x"},
         "series": "field", "episode": 3}
    r = {"first_frame": "f", "theme": "th", "duration": 18.5, "timeline": [], "archetype": "arch"}
    rep = {"passed": True, "attempt": 1, "issues": [], "frames": 5, "meta": {}}

    res = P.package(a, r, rep, {"video": {}})
    assert res is not None
    assert res.get("series") == "field", "run() reads this dict to write history"
    assert res.get("episode") == 3


def test_episode_numbers_advance_once_history_records_them():
    from autotube.series import episode_number

    assert episode_number("unsolved", []) == 1
    hist = [{"series": "unsolved", "episode": 1}, {"series": "unsolved", "episode": 2}]
    assert episode_number("unsolved", hist) == 3, "must continue, not restart"
    # max+1, not len+1: pruning history must never reissue a live number
    assert episode_number("unsolved", [{"series": "unsolved", "episode": 9}]) == 10
    assert episode_number("field", hist) == 1, "series are numbered independently"


# ── prompt assembly, exercised rather than grepped ──────────────────────────
def _capture_prompt(category="history"):
    """Build a real writer prompt and return it, by stubbing the LLM mid-call."""
    import yaml as _y
    from autotube.scriptwriter import ScriptWriter

    cfg = _y.safe_load((ROOT / "config.yaml").read_text())
    seen = {}

    class FakeLLM:
        lite = False
        last_used = "fake:model"

        def json(self, system, user, **kw):
            seen["user"] = user
            raise RuntimeError("captured")

    class FakeStrat:
        def fit(self, *a, **k):
            return {}

    w = ScriptWriter(cfg, FakeLLM(), FakeStrat())
    try:
        w.write({"topic": "A strange disappearance", "category": category},
                {"format": "sounds_fake", "hook_style": "bold_claim", "voice": "v"},
                {"title": "T", "url": "u", "text": "Sourced text.", "summary": "s"})
    except Exception:
        pass
    return seen.get("user", "")


def test_lane_voice_actually_reaches_the_prompt():
    """Grepping for "{lane_rule}" proves the placeholder exists, not that it is filled.

    The series bug was exactly this shape - a value computed, carried into one dict and
    never into the one that mattered - and its test passed throughout by reading source.
    """
    hist = _capture_prompt("history")
    sci = _capture_prompt("space")
    assert "cool, documentary" in hist, "history lane voice missing from the real prompt"
    assert "bright, fast, delighted" in sci, "science lane voice missing from the real prompt"
    assert hist != sci, "the two lanes must produce different prompts"
    # The hook is stated once, as a single authoritative HOOK line: emitting it from
    # both the lane and the bandit produced "HOOK STYLE: question" directly above
    # "never open with a question", and the model obeyed the wrong one.
    for probe in ("VOICE:", "PACING:", "HOOK:"):
        assert probe in hist
    assert "HOOK STYLE:" not in hist, "the generic hook line must not coexist with the lane's"
    assert "Never open with a question" in hist


def test_length_budget_actually_reaches_the_prompt():
    p = _capture_prompt("history")
    assert "words per segment" in p, "per-segment budget missing"
    assert "Both ends are rejected" in p, "one-sided length instruction is back"


# ── closing line must not be the hook again ─────────────────────────────────
def test_rejects_the_echoed_endings_actually_published():
    """Corpus is three real scripts this channel published, not invented examples.

    Each spent its final line restating its first — the moment a viewer decides to
    replay or leave, used to deliver nothing. The prompt had offered "...and that is
    why" as a worked example, so the model was obeying it.
    """
    from autotube.scriptwriter import echoes_hook

    published_failures = [
        ("In 2008, a nightmare washed up on a New York beach.",
         "And that is why in 2008, a nightmare washed up on a New York beach."),
        ("A plane punched a hole in the sky.",
         "A plane just punched a hole in the sky."),
        ("193 years old. This tortoise is still walking.",
         "So yes, he is 193 years old."),
    ]
    for hook, last in published_failures:
        assert echoes_hook(hook, last) >= 0.6, f"should be caught: {last!r}"

    # A real loop shares meaning, not words, and must still pass.
    good = [
        ("A waterfall that runs blood red.", "So the ice down there is still bleeding."),
        ("193 years old. This tortoise is still walking.", "He has outlived thirty-one governors."),
    ]
    for hook, last in good:
        assert echoes_hook(hook, last) < 0.6, f"false positive on a good loop: {last!r}"


def test_the_writer_rejects_an_echoed_ending_end_to_end():
    """The check must fire inside validate(), where it costs the model a retry."""
    src = (ROOT / "autotube" / "scriptwriter.py").read_text()
    # Several functions define a local validate(); pick the script one by the word-count
    # assertion that only it carries.
    bodies = [b for b in src.split("def validate(")[1:] if "narration is" in b]
    assert bodies, "could not locate the script validator"
    assert "echoes_hook" in bodies[0], "the gate must run during validation, not as advice"

    # and the prompt must no longer teach the failure
    assert "'...and that is why' before a hook" not in src, "the bad worked example is back"
    assert "STRONGEST REMAINING FACT" in src


# ── the video must show its own subject ─────────────────────────────────────
def test_rejects_the_all_ambient_shotlist_actually_published():
    """Corpus is the real Montauk Monster episode's visual requests.

    It asked for "beach in Montauk New York with waves", "shoreline with driftwood and
    seaweed" and "waves washing onto sandy beach". The vision gate scored those 8, 10
    and 10 — correctly, because they were delivered exactly. The viewer was promised a
    nightmare and shown six stock beaches and two living raccoons.

    Easy requests score highest, so nothing penalised asking for scenery. This is the
    thing that penalises it.
    """
    from autotube.scriptwriter import names_subject

    published = {"segments": [
        {"visual": {"shows": "beach in Montauk New York with waves", "queries": ["montauk beach waves"]}},
        {"visual": {"shows": "raccoon sitting outdoors on grass", "queries": ["raccoon grass"]}},
        {"visual": {"shows": "shoreline with driftwood and seaweed", "queries": ["driftwood"]}},
        {"visual": {"shows": "waves washing onto sandy beach", "queries": ["waves beach"]}},
    ]}
    assert not names_subject(published, "Montauk Monster"), \
        "a shared place name is the setting, not the subject"

    shown = {"segments": [
        {"visual": {"shows": "the Montauk Monster carcass on the sand",
                    "queries": ["montauk monster carcass"]}}]}
    assert names_subject(shown, "Montauk Monster")

    # Must never block a video whose subject has no distinctive word to require.
    assert names_subject(published, "The Mystery")


def _grab_validator():
    """Capture the real validate() the writer hands to the LLM, and call it ourselves."""
    import yaml as _y
    from autotube.scriptwriter import ScriptWriter

    cfg = _y.safe_load((ROOT / "config.yaml").read_text())
    box = {}

    class FakeLLM:
        lite = False
        last_used = "fake:model"

        def json(self, system, user, validate=None, **kw):
            box["v"] = validate
            raise RuntimeError("captured")

    class FakeStrat:
        def fit(self, *a, **k):
            return {}

    w = ScriptWriter(cfg, FakeLLM(), FakeStrat())
    try:
        w.write({"topic": "t", "category": "history"},
                {"format": "sounds_fake", "hook_style": "bold_claim", "voice": "v"},
                {"title": "Montauk Monster", "url": "u", "text": "Sourced.", "summary": "s"})
    except Exception:
        pass
    return box.get("v")


def test_validator_actually_rejects_an_all_ambient_script():
    """Run the validator rather than grep for it.

    Disabling the assert in validate() left the word "names_subject" in the source, so
    a grep-based version of this test passed against deliberately broken code.
    """
    import pytest

    v = _grab_validator()
    assert v is not None, "could not capture the script validator"

    ambient = {"title": "T", "segments": [
        {"text": "In 2008 a nightmare washed up on a New York beach.",
         "visual": {"shows": "beach in Montauk with waves", "queries": ["montauk beach"]}},
        {"text": "Experts argued about it for weeks on end.",
         "visual": {"shows": "shoreline with driftwood", "queries": ["driftwood"]}},
        {"text": "Nobody agreed what the creature actually was.",
         "visual": {"shows": "waves washing onto sandy beach", "queries": ["waves"]}},
        {"text": "The carcass vanished before anyone could test it.",
         "visual": {"shows": "sunset over the ocean", "queries": ["sunset"]}},
    ]}
    with pytest.raises(AssertionError, match="no shot shows"):
        v(ambient)


def test_rejects_the_empty_payoff_the_echo_gate_created():
    """Closing one exit opened another.

    With hook-echoing blocked, a real run ended "And that's exactly why we have to
    ask..." - no fact, no number, no answer - with the video's best fact stranded
    mid-script. Same failure as the echo wearing different clothes: a last line
    carrying no information.
    """
    from autotube.scriptwriter import payoff_is_empty

    assert payoff_is_empty("Asia has a 1,500-mile-long scar.",
                           "And that's exactly why we have to ask...")
    assert payoff_is_empty("Asia has a scar.", "So the question remains.")

    # Real payoffs, including two this pipeline actually produced, must survive.
    assert not payoff_is_empty("How do arms reach 25 feet long?",
                               "And nobody has ever caught a single adult specimen.")
    assert not payoff_is_empty("Asia has a scar.",
                               "More than 100 peaks tower past 23,600 feet.")
    assert not payoff_is_empty("A waterfall that runs blood red.",
                               "So the ice down there is still bleeding.")


def test_empty_payoff_is_enforced_in_validation():
    """Run the validator, do not grep for it."""
    import pytest

    v = _grab_validator()
    assert v is not None
    trailing = {"title": "T", "segments": [
        {"text": "Asia has a 1,500-mile-long scar that is still growing.",
         "visual": {"shows": "the Montauk Monster carcass", "queries": ["montauk monster"]}},
        {"text": "It is the slowest heaviest collision in all of history.",
         "visual": {"shows": "mountain range", "queries": ["mountains"]}},
        {"text": "Peaks there keep climbing a little every single year.",
         "visual": {"shows": "summit", "queries": ["summit"]}},
        {"text": "And that's exactly why we have to ask...",
         "visual": {"shows": "plateau", "queries": ["plateau"]}},
    ]}
    with pytest.raises(AssertionError, match="trails off"):
        v(trailing)


# ── pruning the channel ─────────────────────────────────────────────────────
def test_delete_is_dry_run_by_default_and_checks_scope(monkeypatch):
    """Deleting a published video is irreversible, so it defaults to doing nothing.

    And it must say the scope is missing rather than surface a raw 403: the stored
    token holds youtube.upload, which is write-only for new uploads and cannot touch
    anything already on the channel.
    """
    import pytest
    import autotube.youtube as Y

    assert Y.delete(["abc", "def"]) == {"abc": "dry-run", "def": "dry-run"}

    monkeypatch.setattr(Y, "can_delete", lambda: False)
    with pytest.raises(RuntimeError, match="youtube.upload only"):
        Y.delete(["abc"], dry_run=False)

    assert "force-ssl" in Y.DELETE_SCOPE


def test_rejects_the_canned_asides_actually_produced():
    """The prompt offered "Nature, please." as an example; the model returned it verbatim
    in two different scripts.

    Same failure as the echo ending and the trailing payoff: the worked example became
    the output. Canned asides spend words from a 39-52 word budget saying nothing, and
    repeated across uploads they are the template similarity that triggers bulk
    demonetisation.
    """
    from autotube.scriptwriter import aside_is_generic

    for produced in ("Nature, please.", "Rude.", "No airbags.",
                     "Office politics are brutal.", "Okay, this one is unhinged."):
        assert aside_is_generic(produced), f"should be caught: {produced!r}"

    for specific in ("A ghost story with tentacles.", "Twenty-five feet of elbow.",
                     "Cake, at one hundred and ninety."):
        assert not aside_is_generic(specific), f"false positive: {specific!r}"


def test_canned_aside_is_rejected_by_the_real_validator():
    import pytest

    v = _grab_validator()
    assert v is not None
    script = {"title": "T", "segments": [
        {"text": "A nightmare washed up on a New York beach in 2008.",
         "visual": {"shows": "the Montauk Monster carcass", "queries": ["montauk monster"]}},
        {"text": "Biologists argued for weeks about what it actually was.",
         "aside": "Nature, please.",
         "visual": {"shows": "sand", "queries": ["sand"]}},
        {"text": "The carcass vanished before anyone ran a single test.",
         "visual": {"shows": "beach", "queries": ["beach"]}},
        {"text": "Nobody has identified it in eighteen years since.",
         "visual": {"shows": "shore", "queries": ["shore"]}},
    ]}
    with pytest.raises(AssertionError, match="canned filler"):
        v(script)


def test_the_prompt_states_one_hook_instruction_not_two():
    """A CASE episode was told "HOOK STYLE: question" and "never open with a question".

    Both lines were injected - the bandit's hook_style and the lane's hook shape - and
    the model obeyed the first. Every CASE episode therefore opened with a question:
    the weakest form for faceless shorts, and identical across uploads, which is the
    template similarity that gets channels demonetised. hook_style is also frozen in
    the bandit, so it never varied on its own.
    """
    hist = _capture_prompt("history")
    sci = _capture_prompt("space")

    for prompt in (hist, sci):
        assert "HOOK STYLE:" not in prompt, "the generic hook line must not be emitted too"
        assert sum(1 for ln in prompt.splitlines() if ln.startswith("HOOK:")) == 1

    assert "Never open with a question" in hist
    assert "most absurd number" in sci
    assert hist != sci
