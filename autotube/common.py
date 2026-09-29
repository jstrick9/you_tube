"""Shared helpers: config, paths, logging, JSON state, HTTP session."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
WORK_DIR = ROOT / "work"
OUTPUT_DIR = ROOT / "output"
ASSETS_DIR = ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

USER_AGENT = "AutoTube/1.0 (educational shorts generator; https://github.com/)"

log = logging.getLogger("autotube")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("urllib3", "googleapiclient.discovery_cache", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("AUTOTUBE_CONFIG", ROOT / "config.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def slugify(text: str, max_len: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-") or "video"


# ── JSON state (committed back to the repo by the workflow = free persistence) ─
def read_json(name: str, default: Any) -> Any:
    p = STATE_DIR / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # corrupt file → keep a backup, start fresh
        shutil.copy(p, p.with_suffix(".corrupt.json"))
        return default


def write_json(name: str, data: Any) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = STATE_DIR / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)


# ── HTTP ──────────────────────────────────────────────────────────────────────
_session: requests.Session | None = None


def http() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT})
    return _session


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             timeout: int = 20, retries: int = 4) -> Any:
    last = None
    for attempt in range(retries + 1):
        try:
            r = http().get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in (429, 503):
                # rate-limited (Wikimedia does this under heavy image searching): honour Retry-After, back off
                last = f"HTTP {r.status_code} (rate-limited)"
                try:
                    wait = float(r.headers.get("Retry-After", 0))
                except ValueError:
                    wait = 0
                time.sleep(min(60.0, max(wait, 4 * 2 ** attempt)))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def font_path(name: str = "Anton-Regular.ttf") -> str:
    p = FONTS_DIR / name
    if p.exists():
        return str(p)
    return "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
