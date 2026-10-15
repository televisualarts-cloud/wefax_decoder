"""Automatic receive scheduler.

Builds a rolling plan of which station/frequency to tune to and when:

* The 1st and 2nd priority stations (Northwood and DWD unless changed on the
  page) are the 'core' stations; their charts are chosen by importance.  A chart that is
  broadcast several times (Northwood repeats its analyses, DWD has
  'Repetition' slots) is only planned until one copy has been decoded well;
  a poor copy makes the next repeat eligible again.
* Clashes are settled by priority, the 1st priority always ahead of the 2nd.
* Gaps that are long enough (chart duration plus tuning/settling guard time)
  are filled with distant ('DX') stations from the schedule database.
* The frequency for each slot is picked from what has worked before at that
  time of day (logged SNR / decode quality), falling back to a simple
  ionospheric prior (path length, solar elevation at the path mid-point).
* Every attempt is logged.  DX station/frequency/time-of-day combinations
  that fail are backed off exponentially (1, 2, 4 ... 28 days) and then tried
  again, so the planner notices when propagation improves.
"""
from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .schedule import Schedules, Station, Slot, DIAL_OFFSET_KHZ

# approximate transmitter sites (lat, lon)
SITES = {
    "Northwood": (51.63, -0.42), "DWD": (53.67, 9.81), "Athens": (37.97, 23.72),
    "Murmansk": (68.97, 33.08), "Boston": (42.08, -70.70), "NewOrleans": (29.88, -89.95),
    "Halifax": (44.65, -63.60), "Sydney": (46.14, -60.20), "Iqaluit": (63.75, -68.50),
    "Resolute": (74.70, -94.80), "Inuvik": (68.36, -133.70), "Kodiak": (57.80, -152.40),
    "PtReyes": (38.10, -122.90), "Honolulu": (21.30, -157.90), "Tokyo": (35.70, 139.80),
    "Seoul": (37.50, 127.00), "Bangkok": (13.75, 100.50), "GuanzouCoastRadioStationXSQ": (23.10, 113.30),
    "RioDeJaneiro": (-22.90, -43.20), "Valparaiso": (-33.00, -71.60), "Charleville": (-26.40, 146.25),
    "Wiluna": (-26.60, 120.20), "Wellington": (-41.30, 174.80),
}
DEFAULT_PRIORITIES = ["Northwood", "DWD"]   # 1st and 2nd priority stations (changeable on the page)


def core_weights(priorities) -> dict:
    """Base score per priority station: the 1st always wins a clash with the 2nd."""
    return {s: w for s, w in zip([p for p in priorities if p], (1000, 800))}


CORE = core_weights(DEFAULT_PRIORITIES)

# chart importance (first matching rule wins); 0 = never plan on its own
IMPORTANCE = [
    (r"gale", 28),
    (r"surface (weather chart|analysis)|sfc analysis", 30),
    (r"surface pressure analysis", 22),
    (r"(prognosis|pressure).*(t\+24|h\+24|24hr)|h ?\+ ?24|t\+24", 24),
    (r"t\+48|h\+48|h\+36", 16),
    (r"t\+72|h\+60|h\+72", 12),
    (r"t\+9|t\+12|h\+8|h\+10", 8),
    (r"swell|wave|sea and swell", 12),
    (r"wind", 10),
    (r"precip|visibility", 10),
    (r"anps", 8),
    (r"tropical", 10),
    (r"ice", 6),
    (r"temperature|fronts", 5),
    (r"schedule|request for comments|notice|test chart|general information", 0),
]
MIN_CORE_IMPORTANCE = 10

LEAD_S = 75          # tune this long before the slot
TAIL_S = 45          # extra time after the expected end
HOLD_MAX_S = 25 * 60  # a recording longer than any single chart is ended for the next planned one
RETRY_BONUS = 2000   # repeat of a priority chart whose first try failed: beats any other chart
GOOD_QUALITY = 0.6   # decode quality that counts as 'collected'
HALF_LIFE_DAYS = 21  # learned results lose half their weight in this time
PROBE_DAYS = 14      # re-check an alternative frequency not tried for this long
DX_GOOD_QUALITY = 0.45


def importance(contents: str) -> int:
    c = contents.lower()
    if "repetition" in c:
        return 0                     # handled through the original chart
    for pat, v in IMPORTANCE:
        if re.search(pat, c):
            return v
    return 6


def chart_key(short: str, contents: str, valid: str) -> str:
    """Same key for a chart and its repeats."""
    c = re.sub(r"\s+", " ", contents.lower()).strip()
    return f"{short}|{c}|{valid}"


# --------------------------------------------------------------------------
# propagation prior
# --------------------------------------------------------------------------
def _solar_elev(lat, lon, when: datetime) -> float:
    doy = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60
    g = 2 * math.pi / 365 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                    - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = hour * 60 + eqt + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1, min(1, cz))))


def _gc(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) and mid-point."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d = 2 * math.asin(math.sqrt(math.sin((p2 - p1) / 2) ** 2
                                + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2))
    bx = math.cos(p2) * math.cos(l2 - l1)
    by = math.cos(p2) * math.sin(l2 - l1)
    pm = math.atan2(math.sin(p1) + math.sin(p2), math.sqrt((math.cos(p1) + bx) ** 2 + by ** 2))
    lm = l1 + math.atan2(by, math.cos(p1) + bx)
    return 6371.0 * d, math.degrees(pm), math.degrees(lm)


def prior(qth, short: str, khz: float, when: datetime) -> float:
    """Rough probability (0..1) that a frequency is usable on this path now.

    foF2 follows solar elevation at the path mid-point (about 3.5 MHz at night
    to 9 MHz at high sun); MUF = foF2 x obliquity factor (single hop, 300 km
    layer, at most 3.4); daytime D-layer absorption raises the LUF on long
    paths.  Crude by design - logged results take over once available.
    """
    site = SITES.get(short)
    if site is None:
        return 0.15
    d, mlat, mlon = _gc(qth[0], qth[1], site[0], site[1])
    el = _solar_elev(mlat, mlon, when)
    s = max(0.0, math.sin(math.radians(el)))
    fof2 = 4.0 + 7.0 * math.sqrt(s)
    hop = min(d, 3500.0) / 2.0
    muf = fof2 * min(3.4, math.sqrt(1 + (hop / 300.0) ** 2))
    luf = 1.5 + 7.0 * s * min(1.0, d / 3000.0)
    f = khz / 1000.0
    if f > muf:
        p = max(0.02, 0.5 - (f / muf - 1) * 2)
    elif f < luf:
        p = max(0.02, 0.5 - (luf / f - 1) * 1.5)
    else:
        p = 0.6 + 0.35 * min(1.0, (muf - f) / max(0.5, 0.3 * muf))
    p *= math.exp(-d / 8000.0)         # multi-hop losses, fading, QRM on long paths
    return max(0.02, min(0.95, p))


# --------------------------------------------------------------------------
# learning log
# --------------------------------------------------------------------------
def hour_bucket(when: datetime) -> int:
    return when.hour // 3                # eight 3-hour blocks


class PropLog:
    """Append-only attempt log + per (station, kHz, 3h block) statistics."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.stats: dict[tuple, dict] = {}
        self.collected: dict[str, float] = {}     # chart key -> best quality
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    self._apply(json.loads(line))
                except Exception:
                    pass

    def _apply(self, r: dict):
        when = datetime.fromisoformat(r["utc"].replace("Z", "+00:00"))
        k = (r["short"], float(r["khz"]), hour_bucket(when))
        st = self.stats.setdefault(k, {"hist": [], "fails_in_row": 0, "last": 0.0, "last_ok": 0.0})
        q = float(r.get("quality") or 0.0)
        core = r.get("kind") == "core" if "kind" in r else r["short"] in CORE
        good = q >= (GOOD_QUALITY if core else DX_GOOD_QUALITY)
        st["hist"] = (st["hist"] + [(when.timestamp(), good, q, r.get("snr_db"))])[-60:]
        st["last"] = when.timestamp()
        if good:
            st["fails_in_row"] = 0
            st["last_ok"] = when.timestamp()
        else:
            st["fails_in_row"] += 1
        if r.get("key") and q > self.collected.get(r["key"], -1):
            self.collected[r["key"]] = q

    @staticmethod
    def weighted(st, at: float) -> dict:
        """Age-weighted summary: results lose half their weight every
        HALF_LIFE_DAYS, so knowledge from a past season fades back towards
        the (season-aware) propagation prior instead of persisting."""
        n = ok = qs = 0.0
        snr = []
        for ts, good, q, s in st["hist"]:
            w = 0.5 ** (max(0.0, at - ts) / 86400.0 / HALF_LIFE_DAYS)
            n += w
            ok += w * good
            qs += w * q
            if s is not None and at - ts < 2 * HALF_LIFE_DAYS * 86400:
                snr.append(s)
        return {"n": n, "ok": ok, "qmean": qs / n if n else None,
                "snr_med": sorted(snr)[len(snr) // 2] if len(snr) >= 2 else None}

    def add(self, r: dict):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            self._apply(r)

    def get(self, short, khz, when):
        return self.stats.get((short, float(khz), hour_bucket(when)))

    def is_collected(self, key: str) -> bool:
        return self.collected.get(key, -1) >= GOOD_QUALITY

    def is_retry(self, key: str) -> bool:
        """Tried already, but only a poor copy or nothing decoded."""
        return -1 < self.collected.get(key, -1) < GOOD_QUALITY

    def backoff_until(self, short, khz, when) -> float:
        st = self.get(short, khz, when)
        if not st or st["fails_in_row"] == 0:
            return 0.0
        days = min(28, 2 ** (st["fails_in_row"] - 1))
        return st["last"] + days * 86400

    def score(self, qth, short, khz, when, sample=False, rng=None) -> float:
        """Probability of a good decode (Beta posterior seeded by the prior)."""
        p0 = prior(qth, short, khz, when)
        st = self.get(short, khz, when)
        a, b = 1 + 3 * p0, 1 + 3 * (1 - p0)
        if st:
            w = self.weighted(st, when.timestamp())
            a += w["ok"]
            b += w["n"] - w["ok"]
        if sample:
            return (rng or random).betavariate(a, b)
        return a / (a + b)

    def value(self, qth, short, khz, when, sample=False, rng=None) -> float:
        """What a frequency is worth for a slot: chance of a good decode, plus
        how good the decodes are and how clean the channel is.  Between two
        frequencies that both work, the cleaner one wins - e.g. a frequency
        that is consistently 6 dB noisier loses about 0.18."""
        v = self.score(qth, short, khz, when, sample, rng)
        st = self.get(short, khz, when)
        if st:
            w = self.weighted(st, when.timestamp())
            if w["n"] >= 1.5:
                v += 0.25 * (w["qmean"] - 0.5)
                if w["snr_med"] is not None:
                    v += 0.03 * max(-10.0, min(20.0, w["snr_med"]))
            # a run of recent failures outweighs a long good history: react
            # within days when a frequency stops working
            if st["fails_in_row"] >= 2:
                v -= 0.15 * min(st["fails_in_row"] - 1, 4)
        return v

    def days_since_tried(self, short, khz, when) -> float:
        st = self.get(short, khz, when)
        return 1e9 if not st else (when.timestamp() - st["last"]) / 86400.0

    def summary(self, qth, now):
        out = []
        for (short, khz, hb), st in sorted(self.stats.items()):
            w = self.weighted(st, now.timestamp())
            out.append({"station": short, "khz": khz, "hours": f"{hb * 3:02d}-{hb * 3 + 3:02d}Z",
                        "attempts": len(st["hist"]), "weight": round(w["n"], 1), "good": round(w["ok"], 1),
                        "avg_quality": None if w["qmean"] is None else round(w["qmean"], 2),
                        "snr_db": w["snr_med"],
                        "last_tried": datetime.fromtimestamp(st["last"], timezone.utc).strftime("%Y-%m-%d"),
                        "retry_after": (datetime.fromtimestamp(
                            st["last"] + min(28, 2 ** (st["fails_in_row"] - 1)) * 86400, timezone.utc)
                            .strftime("%Y-%m-%d %H:%MZ") if st["fails_in_row"] else None)})
        return out


# --------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------
@dataclass
class Item:
    start: datetime
    end: datetime
    short: str
    station: str
    slot: str
    contents: str
    valid: str
    khz: float
    kind: str              # 'core' | 'dx'
    score: float
    key: str
    state: str = "planned"  # planned | tuned | done | skipped
    snr: list = field(default_factory=list)
    result: dict | None = None
    close_t: float = 0.0

    @property
    def dial(self) -> float:
        return round(self.khz - DIAL_OFFSET_KHZ, 2)

    def to_json(self):
        return {"start": self.start.strftime("%H:%MZ"), "end": self.end.strftime("%H:%MZ"),
                "start_ts": self.start.timestamp(),
                "station": self.short, "khz": self.khz, "dial": self.dial, "contents": self.contents,
                "valid": self.valid, "kind": self.kind, "state": self.state, "score": round(self.score, 2)}


def _durations(slots: list[Slot], default: int = 12) -> dict[int, int]:
    """Minutes per slot: from the schedule, else the gap to the next slot,
    limited to 9..default (a 120 LPM chart takes about 10-11 minutes)."""
    mins = sorted({s.minute for s in slots})
    out = {}
    for s in slots:
        if s.duration:
            out[s.minute] = s.duration
            continue
        later = [m for m in mins if (m - s.minute) % 1440 > 0]
        gap = min(((m - s.minute) % 1440 for m in later), default=12)
        out[s.minute] = int(max(9, min(default, gap - 1)))
    return out


class Planner:
    def __init__(self, sched: Schedules, log: PropLog, qth=(50.37, -4.14), dx=True, rng=None,
                 priorities=None):
        self.sched, self.log, self.qth, self.dx = sched, log, qth, dx
        self.rng = rng or random.Random()
        self.core = core_weights(priorities or DEFAULT_PRIORITIES)
        self.lengths = None    # ChartLengths, set by the main program

    def _occurrences(self, st: Station, now: datetime, horizon_h: float):
        """(start, end, slot, [khz options]) for every slot in the window."""
        by_minute: dict[int, tuple[Slot, list]] = {}
        allslots = []
        for khz_set, slots in st.groups:
            for sl in slots:
                ent = by_minute.setdefault(sl.minute, (sl, []))
                ent[1].extend(k for k in khz_set if k not in ent[1])
                allslots.append(sl)
        durs = _durations(allslots, 11 if st.short == "Northwood" else 12)
        day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
        out = []
        for d in (-1, 0, 1, 2):
            for m, (sl, khz) in by_minute.items():
                start = day0 + timedelta(days=d, minutes=m)
                end = start + timedelta(minutes=durs.get(m, 11))
                learnt = self.lengths.learnt_s(st.short, sl.hhmm) if self.lengths else None
                if learnt is not None:     # how long this chart really runs
                    end = start + timedelta(seconds=learnt + 30)
                if end > now and start < now + timedelta(hours=horizon_h):
                    out.append((start, end, sl, sorted(khz)))
        return sorted(out, key=lambda x: x[0])

    def _best_freq(self, short, khzs, when, explore=False):
        """explore=False: best known frequency (important charts).
        explore=True: Thompson sampling, and a frequency that has not been
        tried in this time block for PROBE_DAYS (and is not ruled out by the
        prior) gets a check - this is how a seasonal change is noticed.
        Random draws are seeded by slot so the plan does not flicker."""
        rng = random.Random(f"{short}|{when.isoformat()}") if explore else None
        if explore and len(khzs) > 1:
            ok_prior = [k for k in khzs if prior(self.qth, short, k, when) >= 0.1]
            stale = [k for k in ok_prior if self.log.days_since_tried(short, k, when) > PROBE_DAYS]
            # the usual frequency has started failing: check the others now
            failing = any((st := self.log.get(short, k, when)) and st["fails_in_row"] >= 2 for k in khzs)
            if failing:
                fresh = [k for k in ok_prior if not ((st := self.log.get(short, k, when)) and st["fails_in_row"] >= 2)]
                if fresh:
                    stale = [min(fresh, key=lambda k: -self.log.days_since_tried(short, k, when))]
            if stale:
                k = rng.choice(stale)
                return k, self.log.value(self.qth, short, k, when)
        scored = [(self.log.value(self.qth, short, k, when, sample=explore, rng=rng), k) for k in khzs]
        scored.sort(reverse=True)
        return scored[0][1], scored[0][0]

    def plan(self, now: datetime, horizon_h: float = 24.0, busy: list[Item] | None = None) -> list[Item]:
        """busy: items the radio is on now. Their time stays blocked, so
        nothing is planned over a chart that has already started."""
        busy = list(busy or [])
        cands: list[Item] = []
        # ---- core stations
        for st in self.sched.stations:
            if st.short not in self.core:
                continue
            seen_keys: list[str] = []
            ref = {}           # DWD 'Repetition chart 1050 UTC' -> the 1050 chart
            for start, end, sl, khzs in self._occurrences(st, now, horizon_h):
                contents, valid = sl.contents, sl.valid
                m = re.search(r"repetition (?:of |chart )?(\d{4})", contents.lower())
                if m and m.group(1) in ref:
                    contents, valid = ref[m.group(1)]
                ref[sl.hhmm] = (contents, valid)
                imp = importance(contents)
                if imp < MIN_CORE_IMPORTANCE or start < now - timedelta(seconds=30):
                    continue
                key = chart_key(st.short, contents, valid)
                run_day = (start - timedelta(hours=(start.hour - int(valid)) % 24 if valid.isdigit() else 0)).date()
                key = f"{key}|{run_day}"
                if self.log.is_collected(key):
                    continue              # already have a good copy
                n = seen_keys.count(key)  # later repeats are back-ups for the first
                seen_keys.append(key)
                # the first try at this chart was poor or not decoded: its next
                # repeat goes ahead of everything else (on the best known frequency)
                retry = n == 0 and self.log.is_retry(key)
                khz, p = self._best_freq(st.short, khzs, start, explore=imp < 20 and not retry)
                cands.append(Item(start, end, st.short, st.name, sl.hhmm, contents, valid, khz, "core",
                                  self.core[st.short] + imp * 10 + 50 * p - 25 * n + (RETRY_BONUS if retry else 0),
                                  key))
        chosen = self._select(cands, [b for b in busy if b.kind == "core"])
        if not self.dx:
            return chosen
        # ---- DX stations in the gaps
        dx = []
        for st in self.sched.stations:
            if st.short in self.core:
                continue
            for start, end, sl, khzs in self._occurrences(st, now, horizon_h):
                if start < now + timedelta(seconds=LEAD_S):
                    continue
                elig = [k for k in khzs if self.log.backoff_until(st.short, k, start) <= start.timestamp()]
                if not elig:
                    continue
                khz, p = self._best_freq(st.short, elig, start, explore=True)   # Thompson sample
                if self.log.score(self.qth, st.short, khz, start) < 0.12:
                    continue              # expected value too low to bother
                imp = importance(sl.contents)
                if imp == 0:
                    continue
                key = f"{chart_key(st.short, sl.contents, sl.valid)}|{start.date()}"
                dx.append(Item(start, end, st.short, st.name, sl.hhmm, sl.contents, sl.valid, khz, "dx",
                               100 * p + imp, key))
        return self._select(dx, chosen + [b for b in busy if b.kind != "core"])

    @staticmethod
    def _select(cands: list[Item], fixed: list[Item] | None = None) -> list[Item]:
        """Greedy by score; an item needs its slot plus guard times free."""
        taken = list(fixed or [])
        keys = {t.key for t in taken}

        def free(it):
            a = it.start - timedelta(seconds=LEAD_S)
            b = it.end + timedelta(seconds=TAIL_S)
            for t in taken:
                if t.short == it.short:
                    # same station, e.g. Northwood every 12 minutes: no tuning
                    # margins needed between its charts, just no overlap
                    if it.start < t.end and t.start < it.end:
                        return False
                    continue
                ta = t.start - timedelta(seconds=LEAD_S)
                tb = t.end + timedelta(seconds=TAIL_S)
                if a < tb and ta < b:
                    return False
            return True
        for it in sorted(cands, key=lambda x: -x.score):
            if it.key in keys:
                continue                  # one copy of each chart is enough
            if free(it):
                taken.append(it)
                keys.add(it.key)
        return sorted(taken, key=lambda x: x.start)


# --------------------------------------------------------------------------
# executor
# --------------------------------------------------------------------------
class AutoScheduler:
    """Follows the plan: tunes SDR Console ahead of each slot, samples SNR
    while the slot runs, and logs the outcome."""

    def __init__(self, sched: Schedules, cat, out_dir: Path, qth=(50.37, -4.14),
                 enabled=False, dx=True, log=print, clock=None):
        self.cat, self.logf = cat, log
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.proplog = PropLog(Path(out_dir) / "proplog.jsonl")
        self.settings_path = Path(out_dir) / "settings.json"
        self.priorities = self._load_priorities(sched)
        self.planner = Planner(sched, self.proplog, qth, dx, priorities=self.priorities)
        self.enabled = enabled
        self.paused_until = 0.0
        self.items: list[Item] = []
        self.current: Item | None = None
        self.receiving = False
        self.rec_elapsed = 0.0
        self.force_end = None          # set by the main program: ends the recording in progress
        self._forced_t = 0.0
        self.last_snr = None
        self._last_plan = 0.0
        self._lock = threading.Lock()
        self._stop = False

    # -------- control
    def set_enabled(self, on: bool):
        self.enabled = bool(on)
        self.paused_until = 0.0
        self.logf(f"auto-schedule {'ON' if on else 'OFF'}")
        self.replan()

    def _load_priorities(self, sched):
        try:
            p = json.loads(self.settings_path.read_text()).get("priorities")
            known = {s.short for s in sched.stations}
            if isinstance(p, list) and len(p) == 2 and p[0] in known and (p[1] in known or p[1] is None):
                return p
        except Exception:
            pass
        return list(DEFAULT_PRIORITIES)

    def set_priorities(self, first, second):
        """1st / 2nd priority stations (short names); saved for the next start."""
        if second == first:
            second = None
        self.priorities = [first, second]
        self.planner.core = core_weights(self.priorities)
        try:
            d = json.loads(self.settings_path.read_text()) if self.settings_path.exists() else {}
            d["priorities"] = self.priorities
            self.settings_path.write_text(json.dumps(d, indent=1))
        except Exception as e:
            self.logf(f"could not save settings: {e}")
        self.logf(f"auto: priorities 1st {first}, 2nd {second or '-'}")
        self.replan()

    def manual_override(self, minutes=20):
        if self.enabled:
            self.paused_until = time.time() + minutes * 60
            self.logf(f"auto-schedule paused {minutes} min (manual tune)")

    def replan(self):
        now = self.clock()
        with self._lock:
            busy = [i for i in self.items if i.state == "tuned"
                    and i.end + timedelta(seconds=TAIL_S) > now]
        new = self.planner.plan(now, busy=busy)
        with self._lock:
            keep = {(i.short, i.start): i for i in self.items if i.state != "planned"}
            items = [keep.pop((i.short, i.start), i) for i in new]
            # items still being received / waiting for their decode stay until logged;
            # finished ones stay for an hour so the page can show them
            cutoff = now - timedelta(hours=1)
            items += [i for i in keep.values() if i.state in ("tuned", "closing") or i.end > cutoff]
            self.items = sorted(items, key=lambda x: x.start)
        self._last_plan = time.time()

    # -------- feeds from the decoder
    def on_status(self, s: dict):
        self.receiving = s.get("state") == "receiving"
        self.rec_elapsed = float(s.get("elapsed") or 0) if self.receiving else 0.0
        if s.get("snr_db") is not None:
            self.last_snr = float(s["snr_db"])

    def on_fax(self, product: dict | None, meta: dict, utc: str):
        """A fax was decoded: attach the result to the matching plan item."""
        q = float(meta.get("quality", 0))
        with self._lock:
            it = None
            if product and product.get("slot"):
                it = next((i for i in self.items if i.short == product.get("short")
                           and i.slot == product.get("slot") and i.state in ("tuned", "closing")), None)
        if it is not None:
            it.result = {"quality": q, "lines": meta.get("lines")}
        elif product and product.get("khz"):
            # received outside the plan (manual tuning) - still useful to learn from
            when = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            self.proplog.add({"utc": utc, "short": product["short"], "khz": product["khz"],
                              "slot": product.get("slot"), "snr_db": self.last_snr, "quality": q,
                              "key": None, "kind": "manual", "hour": when.hour})
        if q >= GOOD_QUALITY:
            self.replan()                            # drop repeats of what we now have

    # -------- main loop
    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        self.replan()
        while not self._stop:
            try:
                self.step()
            except Exception as e:
                self.logf(f"auto-schedule error: {e}")
            time.sleep(2)

    def step(self):
        now = self.clock()
        if time.time() - self._last_plan > 120:
            self.replan()
        with self._lock:
            items = list(self.items)
        # close finished items; log once the decode result is in (or given up on)
        for it in items:
            if it.state == "tuned":
                if it.start <= now <= it.end and self.last_snr is not None:
                    it.snr.append(self.last_snr)
                if now > it.end + timedelta(seconds=TAIL_S) and not self.receiving:
                    self._close(it)
            if it.state == "closing" and (it.result is not None or time.time() - it.close_t > 150):
                it.state = "done"
                if it.result is None and getattr(it, "abandoned", False):
                    # left before the chart could finish: says nothing about propagation
                    self.logf(f"auto: {it.short} {it.khz} kHz {it.slot}Z left early - not counted")
                else:
                    self._log_item(it, it.result["quality"] if it.result else 0.0)
        if not self.enabled or time.time() < self.paused_until or self.cat is None or not self.cat.enabled:
            return
        # next item due
        due = next((i for i in items if i.state == "planned"
                    and i.start - timedelta(seconds=LEAD_S) <= now < i.end), None)
        if due is None:
            return
        if self.receiving and time.time() - self._forced_t > 30:
            # a chart is still coming in: never retune while it is being
            # decoded. The one exception is a recording far longer than any
            # single chart (a missed stop tone on a back-to-back broadcast
            # that the phasing check didn't split) - otherwise every later
            # planned chart would be lost.
            el = self.rec_elapsed
            over = el > HOLD_MAX_S
            if over and self.force_end is not None:
                self.logf(f"auto: recording has run {el / 60:.0f} min, longer than any single chart - "
                          f"ending it for {due.start:%H:%M}Z {due.short}")
                self._forced_t = time.time()
                self._hold_logged = False
                self.force_end("overran schedule")
                return
            if not getattr(self, "_hold_logged", False):
                self._hold_logged = True
                self.logf(f"auto: holding - chart still being received; "
                          f"{due.start:%H:%M}Z {due.short} waits")
            return
        self._hold_logged = False
        if now > due.start + timedelta(seconds=60):
            due.state = "skipped"                    # too late to catch the start
            return
        if self.current is not None and self.current.state == "tuned":
            if now < self.current.end:
                self.current.abandoned = True
            self._close(self.current)
        self.cat.tune(due.dial)
        due.state = "tuned"
        self.current = due
        self.logf(f"auto: {due.start:%H:%M}Z {due.short} {due.khz} kHz (dial {due.dial}) - {due.contents}"
                  f" [{due.kind}]")

    def _close(self, it: Item):
        it.state = "closing"
        it.close_t = time.time()
        if it is self.current:
            self.current = None

    def _log_item(self, it: Item, quality: float):
        if getattr(it, "_logged", False):
            return
        it._logged = True
        snr = sorted(it.snr)[len(it.snr) // 2] if it.snr else None
        self.proplog.add({"utc": it.start.strftime("%Y-%m-%dT%H:%M:%SZ"), "short": it.short, "khz": it.khz,
                          "slot": it.slot, "snr_db": snr, "quality": round(quality, 3), "key": it.key,
                          "kind": it.kind, "hour": it.start.hour})
        verdict = "good" if quality >= (GOOD_QUALITY if it.kind == "core" else DX_GOOD_QUALITY) else "poor/none"
        self.logf(f"auto: logged {it.short} {it.khz} kHz {it.slot}Z  SNR "
                  f"{'--' if snr is None else f'{snr:+.0f} dB'}  quality {quality:.2f} ({verdict})")

    def item_at(self, when: datetime):
        """The plan item the scheduler had tuned for at this time (or None)."""
        with self._lock:
            items = list(self.items)
        for it in items:
            if it.state in ("tuned", "closing", "done") and \
                    it.start - timedelta(seconds=180) <= when <= it.end + timedelta(seconds=60):
                return it
        return None

    # -------- for the web page
    def status(self) -> dict:
        now = self.clock()
        with self._lock:
            upcoming = [i.to_json() for i in self.items if i.end > now][:8]
        return {"enabled": self.enabled, "paused_min": max(0, int((self.paused_until - time.time()) / 60)),
                "current": self.current.to_json() if self.current else None, "plan": upcoming,
                "priorities": self.priorities}
