"""Nightly clean-up: delete faxes older than the 'Keep' setting.

Runs once a day, at the first quiet moment after 00:05 local time (nothing
being received or decoded).  Only the dated faxes are removed - raw/,
enhanced/, thumbs/, audio/ and their .json.  The station copies, the
learning history and the settings are never touched.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

KEEP_CHOICES = (3, 7, 14, 28)
DEFAULT_KEEP = 28
START_AFTER = (0, 5)          # local time the daily clean-up may start


def keep_days(settings: dict) -> int:
    d = settings.get("keep_days")
    return d if d in KEEP_CHOICES else DEFAULT_KEEP


def purge(out: Path, days: int, log=print) -> tuple[int, int]:
    """Delete faxes received more than `days` days ago. Returns (count, bytes)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    n = freed = 0
    for j in sorted(out.glob("*.json")):
        if j.name == "settings.json":
            continue
        try:
            rec = json.loads(j.read_text())
            when = datetime.strptime(rec["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            continue                          # not a fax record
        if when >= cutoff:
            continue
        stem = rec.get("id") or j.stem
        files = [out / f for f in (rec.get("files") or {}).values() if f]
        files += [f.with_name(f.stem + "_old" + f.suffix) for f in list(files)]   # from reenhance.py
        files += [out / "audio" / f"{stem}.wav", j]
        for f in files:
            try:
                if f.is_file() and out in f.resolve().parents:
                    freed += f.stat().st_size
                    f.unlink()
            except Exception as e:
                log(f"clean-up: could not delete {f.name}: {e}")
        n += 1
    return n, freed


def start(out: Path, hub, is_quiet, log=print):
    """Background thread: once per local day, after START_AFTER, when quiet."""
    def run():
        done_for = None
        while True:
            try:
                now = datetime.now()
                today = now.date()
                if done_for is None:
                    # started after the clean-up time: wait for the next night
                    done_for = today if (now.hour, now.minute) >= START_AFTER else today - timedelta(days=1)
                if done_for != today and (now.hour, now.minute) >= START_AFTER and is_quiet():
                    days = keep_days(hub.load_settings())
                    n, freed = purge(out, days, log)
                    done_for = today
                    log(f"clean-up: kept {days} days; deleted {n} fax(es), {freed / 1e6:.0f} MB freed")
                    if n:
                        hub.publish("fax", {"id": None})      # page reloads its list
            except Exception as e:
                log(f"clean-up error: {e}")
            time.sleep(60)
    threading.Thread(target=run, daemon=True).start()
