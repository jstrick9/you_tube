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

from .common import load_config, setup_logging

log = logging.getLogger("autotube")


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


def cmd_doctor(cfg: dict) -> int:
    from .common import ffmpeg_bin
    from .llm import LLM
    print("ffmpeg:", ffmpeg_bin())
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "PEXELS_API_KEY", "YOUTUBE_API_KEY",
              "YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN", "NTFY_TOPIC", "DISCORD_WEBHOOK_URL"):
        print(f"  {k:22s} {'set' if os.environ.get(k) else '-'}")
    try:
        r = LLM(cfg).json("Reply with JSON only.", 'Return {"ok": true}', temperature=0)
        print("LLM:", r)
    except Exception as e:  # noqa: BLE001
        print("LLM FAILED:", e)
    ok = True
    if os.environ.get("YT_REFRESH_TOKEN"):
        try:
            from .common import http
            from .youtube import credentials, service
            creds = credentials()
            info = http().get("https://oauth2.googleapis.com/tokeninfo",
                              params={"access_token": creds.token}, timeout=15).json()
            granted = set((info.get("scope") or "").split())
            print("Granted scopes:")
            for sc in ("youtube.upload", "youtube.readonly", "yt-analytics.readonly"):
                has = f"https://www.googleapis.com/auth/{sc}" in granted
                print(f"  {'✓' if has else '✗'} {sc}{'' if has else '   ← MISSING: re-authorize with this scope'}")
                ok &= has or sc == "yt-analytics.readonly"
            items = service().channels().list(part="snippet,statistics", mine=True).execute().get("items", [])
            if items:
                it = items[0]
                print(f"✓ YouTube channel: {it['snippet']['title']} (id {it['id']}) {it['statistics']}")
            else:
                ok = False
                print("✗ Token works but this Google account has no YouTube channel. Re-authorize and pick the "
                      "account / Brand Account that owns the channel.")
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
    ap.add_argument("command", choices=["run", "trends", "analytics", "report", "doctor"])
    ap.add_argument("-n", "--count", type=int)
    ap.add_argument("--dry-run", action="store_true", help="render but do not upload")
    ap.add_argument("--keep-work", action="store_true")
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

    from . import pipeline
    try:
        res = pipeline.run(cfg, a.count, upload=False if a.dry_run else None, keep_work=a.keep_work)
    except Exception as e:  # noqa: BLE001
        log.exception("run failed")
        notify(cfg, "AutoTube run FAILED", str(e)[:800], ok=False)
        return 1
    lines = [f"• {r['meta']['title']} → {r['entry'].get('video_id', r['entry']['status'])}" for r in res]
    want = a.count or cfg["schedule"]["videos_per_day"]
    notify(cfg, f"AutoTube: {len(res)}/{want} videos", "\n".join(lines) or "no videos produced",
           ok=len(res) > 0)
    print("\n".join(lines))
    failed_uploads = [r for r in res if str(r["entry"].get("status", "")).startswith("upload_failed")]
    return 0 if res and not failed_uploads else 1


if __name__ == "__main__":
    sys.exit(main())
