"""How long each chart runs, learnt from charts that ended cleanly.

Every chart that ends on its stop tone or its end-of-frame black gives the
time its image ended, measured from its scheduled slot (DWD's 18:00 surface
analysis runs about 20 minutes although the next slot suggests 12). That
gives a deadline for the same chart next time: if the signal fades, or the
stop tone and the black are both missed, the recording is closed a few
seconds after the chart should have ended instead of running on and holding
up the auto-scheduler.

Charts never seen ending cleanly fall back to the schedule: shortly before
the next slot of the same station, so the receiver is free (and on the
right frequency, if the auto-scheduler wants another station) when the next
chart's start tone arrives. Decoding runs separately and never blocks that.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

KEEP = 8                 # most recent measurements kept per chart
CLEAN_ENDS = ("stop tone", "end-of-frame black")
LEARNT_MARGIN_S = 30.0   # after the learnt image end: stop tone (5 s) + black (10 s) + detection
SCHEDULE_LEAD_S = 15.0   # before the next slot's start, when nothing is learnt
PREROLL_S = 12.0         # Detector default; only used to read old fax records
STOP_TAIL_S = 12.5       # stop-tone recordings run on ~10 s + detection lag


def slot_time(slot: str, near: datetime) -> datetime:
    """The datetime of slot 'HHMM' closest to `near` (handles midnight)."""
    h, m = int(slot[:2]), int(slot[2:])
    base = near.replace(hour=h, minute=m, second=0, microsecond=0)
    return min((base + timedelta(days=d) for d in (-1, 0, 1)), key=lambda t: abs((t - near).total_seconds()))


class ChartLengths:
    def __init__(self, out_dir: Path, log=print):
        self.path = Path(out_dir) / "chart_lengths.json"
        self.log = log
        self._lock = threading.Lock()
        self.data: dict[str, list[float]] = {}
        try:
            self.data = json.loads(self.path.read_text())
        except FileNotFoundError:
            self._bootstrap(Path(out_dir))
        except Exception as e:
            log(f"chart lengths: could not read {self.path.name} ({e}) - starting afresh")

    @staticmethod
    def key(short: str, slot: str) -> str:
        return f"{short}|{slot}"

    # ------------------------------------------------------------------
    def learn(self, short: str, slot: str, chart_end: datetime):
        off = (chart_end - slot_time(slot, chart_end)).total_seconds()
        if not 120 <= off <= 3600:             # not a plausible chart length
            return
        k = self.key(short, slot)
        with self._lock:
            v = self.data.setdefault(k, [])
            v.append(round(off, 1))
            del v[:-KEEP]
            self._save()
        self.log(f"chart length: {short} {slot}Z image ended {off / 60:.1f} min after the slot")

    def learnt_s(self, short: str, slot: str) -> float | None:
        """Seconds from slot start to the end of the image (longest recent)."""
        v = self.data.get(self.key(short, slot))
        return max(v[-5:]) if v else None

    def deadline(self, sched, short: str, slot: str, near: datetime):
        """(datetime, 'learnt'|'schedule', expected image end or None) or None."""
        t0 = slot_time(slot, near)
        off = self.learnt_s(short, slot)
        if off is not None:
            end = t0 + timedelta(seconds=off)
            return end + timedelta(seconds=LEARNT_MARGIN_S), "learnt", end
        nxt = next_slot_minutes(sched, short, slot)
        if nxt is not None:
            return t0 + timedelta(minutes=nxt, seconds=-SCHEDULE_LEAD_S), "schedule", None
        return None

    # ------------------------------------------------------------------
    def _save(self):
        tmp = self.path.with_suffix(".json.part")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        tmp.replace(self.path)

    def _bootstrap(self, out: Path):
        """First run: estimate lengths from faxes already saved (start tone to
        stop tone recordings only - their timing is known well enough)."""
        n = 0
        for j in sorted(out.glob("*.json")):
            try:
                r = json.loads(j.read_text())
                p = r.get("product") or {}
                if r.get("end_reason") != "stop tone" or r.get("trigger") != "start tone" \
                        or not p.get("slot") or not p.get("short"):
                    continue
                t = datetime.strptime(r["utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                end = t + timedelta(seconds=float(r["meta"]["duration_s"]) - PREROLL_S - STOP_TAIL_S)
                off = (end - slot_time(p["slot"], end)).total_seconds()
                if 120 <= off <= 3600:
                    v = self.data.setdefault(self.key(p["short"], p["slot"]), [])
                    v.append(round(off, 1))
                    del v[:-KEEP]
                    n += 1
            except Exception:
                continue
        if n:
            self.log(f"chart lengths: estimated from {n} saved fax(es), {len(self.data)} chart(s)")
            try:
                self._save()
            except Exception as e:
                self.log(f"chart lengths: could not save ({e})")


def next_slot_minutes(sched, short: str, slot: str) -> int | None:
    """Minutes from `slot` to the next scheduled slot of the same station."""
    if sched is None:
        return None
    m0 = int(slot[:2]) * 60 + int(slot[2:])
    for st in sched.stations:
        if st.short != short:
            continue
        for _, slots in st.groups:
            mins = {s.minute for s in slots}
            if m0 not in mins:
                continue
            gaps = [(m - m0) % 1440 for m in mins if m != m0]
            return min(gaps) if gaps else None
    return None
