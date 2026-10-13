"""CLI entry point.

  python -m autotube run              # full daily run (trends → videos → scheduled uploads)
  python -m autotube run --dry-run    # render only, no upload
  python -m autotube run -n 1         # override number of videos
  python -m autotube trends           # show today's ranked, filtered trend candidates
  python -m autotube analytics        # pull stats, update the learning model
  python -m autotube report           # print what the model has learned
  python -m autotube doctor           # check credentials / providers / ffmpeg
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import run_report
from .common import load_config, setup_logging

log = logging.getLogger("autotube")


def _receipt_run_id(now) -> str:
    return f"{now.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"


def _write_github_summary(produced: int, target: int, failed_uploads: int = 0,
                          rate_limited: set[str] | None = None, dry_run: bool = False) -> None:
    """Make partial/zero-output green runs explicit in GitHub Actions' run summary."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    failed = produced < target or failed_uploads > 0
    lines = ["## AutoTube production result",
             f"- Mode: {'render-only dry run' if dry_run else 'production'}",
             f"- Videos rendered: {produced} / {target}",
             f"- Upload failures: {failed_uploads}"]
    if failed:
        lines.extend(["- Outcome: **incomplete**; no verification or quality gate was bypassed."])
        if rate_limited:
            lines.append("- HTTP 429 responses observed: " + ", ".join(sorted(rate_limited)))
        lines.append("- Follow-up: automated top-up slots remain configured to re-check missing daily output; "
                     "GitHub may delay scheduled events.")
    else:
        lines.append("- Outcome: complete.")
    try:
        with open(path, "a", encoding="utf-8") as summary:
            summary.write("\n".join(lines) + "\n")
    except OSError as exc:
        log.warning("could not write GitHub Actions summary: %s", str(exc)[:180])


def notify(cfg: dict, title: str, msg: str, ok: bool = True) -> None:
    from .common import http
    if (ok and not cfg["notify"].get("on_success")) or (not ok and not cfg["notify"].get("on_failure")):
        return
    topic, hook = os.environ.get("NTFY_TOPIC"), os.environ.get("DISCORD_WEBHOOK_URL")
    try:
        if topic:
            http().post(f"https://ntfy.sh/{topic}", data=msg.encode(), headers={"Title": title,
                        "Tags": "white_check_mark" if ok else "warning"}, timeout=15)
        if hook:
            http().post(hook, json={"content": f"**{title}**\n{msg}"[:1900]}, timeout=15)
    except Exception as e:  # noqa: BLE001
        log.warning("notify failed: %s", e)


def cmd_prune(cfg: dict, ids: list[str], execute: bool = False) -> int:
    """Delete videos from the channel. Dry run unless --execute is passed."""
    from . import youtube
    if not ids:
        print("no video ids given")
        return 1
    if not execute:
        print(f"DRY RUN — {len(ids)} video(s) would be deleted. Re-run with --execute.")
        for v in ids:
            print(f"  https://youtu.be/{v}")
        return 0
    if not youtube.can_delete():
        print("CANNOT DELETE: the stored token holds youtube.upload only, which cannot")
        print("touch existing videos. Re-authorise including")
        print(f"  {youtube.DELETE_SCOPE}")
        print("and update YT_REFRESH_TOKEN, or delete them in YouTube Studio.")
        return 1
    res = youtube.delete(ids, dry_run=False)
    for v, r in res.items():
        print(f"  {v}: {r}")
    return 0 if all(r == "deleted" for r in res.values()) else 1


def cmd_doctor(cfg: dict) -> int:
    ok = True
    from .common import ffmpeg_bin
    from .llm import LLM
    print("ffmpeg:", ffmpeg_bin())
    # Derived, not hardcoded. This list used to be literal and silently fell behind when
    # providers were added: Mistral's key was present and working while doctor's own
    # report did not mention it, which is how a working provider looks identical to a
    # missing one. Anything the router can key off now appears here automatically.
    from .llm import KEYED_PROVIDERS
    keys = sorted(set(KEYED_PROVIDERS.values()) | {
        "CLOUDFLARE_ACCOUNT_ID",      # Cloudflare needs the account id as well as the token
        "PEXELS_API_KEY", "YOUTUBE_API_KEY",
        "YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN",
        "NTFY_TOPIC", "DISCORD_WEBHOOK_URL"})
    for k in keys:
        print(f"  {k:22s} {'set' if os.environ.get(k) else '-'}")
    try:
        r = LLM(cfg).json("Reply with JSON only.", 'Return {"ok": true}', temperature=0)
        print("LLM:", r)
    except Exception as e:  # noqa: BLE001
        print("LLM FAILED:", e)
    # Validate configured model ids against each provider's live catalogue. Three
    # separate failures here have been a model name that does not exist, and each one
    # cost a full production run to find because the fallback chain hid it.
    from .llm import audit_models
    print("\nmodels:")
    for prov, r in audit_models(cfg).items():
        if r["catalogue"] is None:
            from .llm import _provider_key
            why = "no key" if not _provider_key(prov) else "key present but catalogue unreachable (see warning above)"
            print(f"  {prov:11s} (skipped — {why})")
            continue
        for m, present in r["configured"].items():
            print(f"  {prov:11s} {'OK  ' if present else 'GONE'} {m}")
        # Always show the catalogue, not only on failure. A model can be listed and still
        # be uncallable - gemini-2.5-flash answers "no longer available to new users" -
        # so the reachable alternatives need to be visible when choosing a replacement.
        print(f"  {prov:11s} .. {r['catalogue']} available: {', '.join(r['sample'][:60])}")
        if r["missing"]:
            ok = False
            print(f"  {prov:11s} !! {len(r['missing'])} configured model(s) do not exist: {r['missing']}")
            print(f"  {prov:11s} -> available: {', '.join(r['sample'][:14])}")

    # Vision is a hard gate: if nothing answers, finished scripts are discarded. Prove
    # it with a real call rather than trusting the catalogue.
    from .llm import probe_vision, vision_candidates
    print("\nvision (live probe):")
    vp = cfg["llm"].get("vision_providers", [])
    for prov in vp:
        models = cfg["llm"].get(f"{prov}_vision_models") or (
            cfg["llm"].get("gemini_models", []) if prov == "gemini" else [])
        if not models:
            print(f"  {prov:11s} (no vision models configured)")
            continue
        for mdl in models[:2]:
            good, detail = probe_vision(cfg, prov, mdl)
            # A 429 is a spent quota, not a misconfiguration: it clears by itself and
            # says nothing about whether the model is wired correctly. Failing the
            # check on it would leave doctor red most evenings, and a check that is
            # always red is one nobody reads. Structural failures still fail.
            # 503/502/500 are the provider being busy, same category as a 429: transient,
            # self-clearing, and no evidence the model is wired wrongly. Only treat a
            # response that will still be wrong tomorrow as a failure.
            throttled = ("429" in detail or "quota" in detail.lower()
                         or any(f"HTTP {c}" in detail for c in (500, 502, 503, 504)))
            label = "WORKS" if good else ("BUSY " if throttled else "FAILS")
            print(f"  {prov:11s} {label} {mdl}")
            if not good:
                if not throttled:
                    ok = False
                print(f"              -> {detail}")
    print("\nother vision-capable models in the catalogues you have keys for:")
    for prov in ("gemini", "groq", "mistral", "cloudflare", "cerebras", "openrouter"):
        cands = vision_candidates(prov)
        if cands:
            print(f"  {prov:11s} {', '.join(cands[:8])}")

    if os.environ.get("YT_REFRESH_TOKEN"):
        try:
            from .common import http
            from .youtube import DELETE_SCOPE, credentials, service, video_status
            creds = credentials()
            info = http().get("https://oauth2.googleapis.com/tokeninfo",
                              params={"access_token": creds.token}, timeout=15).json()
            granted = set((info.get("scope") or "").split())
            print("Granted scopes:")
            for sc in ("youtube.upload", "youtube.readonly", "yt-analytics.readonly"):
                has = f"https://www.googleapis.com/auth/{sc}" in granted
                print(f"  {'✓' if has else '✗'} {sc}{'' if has else '   ← MISSING: re-authorize with this scope'}")
                ok &= has or sc == "yt-analytics.readonly"
            can_manage_existing = DELETE_SCOPE in granted or "https://www.googleapis.com/auth/youtube" in granted
            print(f"  {'✓' if can_manage_existing else '✗'} youtube.force-ssl (modify existing videos)")
            items = service().channels().list(part="snippet,statistics", mine=True).execute().get("items", [])
            if items:
                it = items[0]
                print(f"✓ YouTube channel: {it['snippet']['title']} (id {it['id']}) {it['statistics']}")
            else:
                ok = False
                print("✗ Token works but this Google account has no YouTube channel. Re-authorize and pick the "
                      "account / Brand Account that owns the channel.")
            status_video_id = os.environ.get("YOUTUBE_STATUS_VIDEO_ID", "").strip()
            if status_video_id:
                status = video_status(status_video_id)
                if status is None:
                    ok = False
                    print(f"✗ No video status returned for {status_video_id}; check the ID and channel permissions.")
                else:
                    print("Read-only video status:")
                    print(f"  id: {status['video_id']} | title: {status['title']}")
                    print(f"  privacy: {status['privacy_status']} | upload: {status['upload_status']} | "
                          f"publishAt: {status['publish_at'] or 'not scheduled'}")
                    if status.get("rejection_reason"):
                        ok = False
                        print(f"  rejection reason: {status['rejection_reason']}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print("✗ YouTube auth FAILED:", e)
            msg = str(e)
            if "unauthorized_client" in msg:
                print("  → The refresh token was created with a DIFFERENT OAuth client than YT_CLIENT_ID/YT_CLIENT_SECRET.\n"
                      "    Fix: in the OAuth Playground ⚙️ tick 'Use your own OAuth credentials', paste the SAME client ID\n"
                      "    and secret that are in your GitHub secrets, re-authorize, and save the new refresh token.")
            elif "invalid_client" in msg:
                print("  → Google doesn't recognise YT_CLIENT_ID / YT_CLIENT_SECRET (typo, extra characters, deleted client,\n"
                      "    or the ID and secret come from two different clients). Re-copy both from Google Auth Platform → Clients.")
            elif "invalid_grant" in msg:
                print("  → The refresh token is expired or revoked (app left in 'Testing' = 7-day expiry, access removed at\n"
                      "    myaccount.google.com/permissions, or token superseded). Publish the app and generate a new token.")
            else:
                print("  Common causes: client ID/secret typo, token created with a different client, or the app "
                      "was left in 'Testing' (tokens expire after 7 days).")
    else:
        print("YouTube secrets not set — skipping YouTube check.")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="autotube")
    ap.add_argument("command", choices=["run", "trends", "analytics", "report", "doctor", "dossier"])
    ap.add_argument("-n", "--count", type=int)
    ap.add_argument("--dry-run", action="store_true", help="render but do not upload")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--top-up", action="store_true",
                    help="only make the videos still missing for today (safe to run several times a day)")
    ap.add_argument("--config")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    setup_logging(a.verbose)
    cfg = load_config(a.config)

    if a.command == "doctor":
        return cmd_doctor(cfg)
    if a.command == "trends":
        from . import trends
        for c in trends.collect(cfg)[:40]:
            print(f"{c['score']:.2f}  {c['topic'][:70]:70s} {','.join(sorted(set(c['sources'])))}")
        return 0
    if a.command == "dossier":
        # Export the provenance record for transparency and internal review; documentation can help explain
        # production but does not guarantee any policy-review or appeal outcome.
        from . import provenance
        print(json.dumps(provenance.dossier(a.count or 0), indent=2))
        return 0
    if a.command == "report":
        from .strategy import Strategy
        print(json.dumps(Strategy(cfg).report(), indent=2))
        return 0
    if a.command == "analytics":
        from . import analytics
        try:
            s = analytics.run(cfg)
            if s.get("rejected"):
                notify(cfg, "AutoTube: videos rejected", f"Rejected: {s['rejected']}", ok=False)
            return 0
        except Exception as e:  # noqa: BLE001
            log.exception("analytics failed")
            notify(cfg, "AutoTube analytics failed", str(e)[:500], ok=False)
            return 1

    from . import llm as llm_mod
    from . import pipeline
    if a.top_up:
        left = pipeline.remaining_today(cfg, upload=not a.dry_run and cfg["upload"]["enabled"])
        if left == 0:
            receipt = run_report.RunReceipt(_receipt_run_id(pipeline.now_utc()), 0,
                                            upload=not a.dry_run and cfg["upload"]["enabled"],
                                            dry_run=a.dry_run)
            receipt.set_stage("preflight")
            receipt.data["top_up"] = True
            receipt.data["outcome"] = "already_satisfied"
            receipt.persist()
            print("Today's videos are already done — nothing to do (top-up run).")
            return 0
        if a.count is not None and a.count <= 0:
            log.error("top-up count must be a positive integer")
            return 2
        # Even an explicitly counted retry is capped by remaining daily capacity. This
        # makes concurrent/delayed scheduled runs safe and prevents manual retries from
        # exceeding the target.
        a.count = left if a.count is None else min(a.count, left)
        log.info("top-up: %d video(s) still needed today (capped to remaining slots)", a.count)
    want = a.count or pipeline.daily_target(cfg)
    receipt = run_report.RunReceipt(_receipt_run_id(pipeline.now_utc()), want,
                                    upload=not a.dry_run and cfg["upload"]["enabled"],
                                    dry_run=a.dry_run)
    receipt.data["top_up"] = bool(a.top_up)
    receipt_token = run_report.activate(receipt)
    try:
        res = pipeline.run(cfg, a.count, upload=False if a.dry_run else None, keep_work=a.keep_work)
        failed_uploads = [r for r in res if str(r["entry"].get("status", "")).startswith("upload_failed")]
        scheduled_count = sum(1 for r in res if r["entry"].get("status") == "scheduled")
        receipt.data["rate_limited_models"] = sorted(llm_mod.rate_limited())[:40]
        receipt.finish(len(res), want, scheduled_count, len(failed_uploads))
    except Exception as e:  # noqa: BLE001
        receipt.fail(e)
        receipt.persist()
        run_report.deactivate(receipt_token)
        log.exception("run failed")
        notify(cfg, "AutoTube run FAILED", str(e)[:800], ok=False)
        return 1
    receipt.persist()
    run_report.deactivate(receipt_token)
    lines = [f"• {r['meta']['title']} → {r['entry'].get('video_id', r['entry']['status'])}" for r in res]
    limited = llm_mod.rate_limited() if len(res) < want else set()
    incomplete = len(res) < want or bool(failed_uploads)
    message = "\n".join(lines) or "no videos produced"
    if incomplete:
        if limited:
            message += "\nHTTP 429 responses were observed from: " + ", ".join(sorted(limited))
        message += "\nNo unverified output was forced through the quality gates; automated top-up retries remain configured."
    notify(cfg, f"AutoTube: {len(res)}/{want} videos", message, ok=not incomplete)
    print("\n".join(lines))
    _write_github_summary(len(res), want, len(failed_uploads), limited, a.dry_run)
    if res and not incomplete:
        return 0
    # A green Actions conclusion must not imply a video exists. Keep a rate-limited
    # run non-red so scheduled recovery can continue, but spell out that 429s are only
    # one observed failure class; topic, other provider, validation and vision failures
    # may also have prevented output. The run summary and failure alert retain the fact
    # that production was incomplete.
    if incomplete and limited:
        log.warning("production incomplete (%d/%d videos rendered); HTTP 429 responses were observed from %s. "
                    "Other provider, topic, script or vision failures may also have contributed. "
                    "Not failing this run so the scheduled top-up can retry.",
                    len(res), want, ", ".join(sorted(limited)))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
