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

import collections
import json
import os
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
DEFAULT_QTH = (50.37, -4.14)               # Plymouth, UK - used until a location is set
DEFAULT_PRIORITIES = ["Northwood", "DWD"]   # 1st and 2nd priority stations (changeable on the page)


def parse_qth(text) -> tuple[float, float]:
    """Receiver location from 'lat, lon' (decimal degrees, S and W negative)
    or a Maidenhead locator ('IO70', 'IO70ri', 'IO70ri45')."""
    t = str(text).strip()
    m = re.fullmatch(r"([A-Ra-r]{2})(\d{2})(?:([A-Xa-x]{2})(?:(\d{2}))?)?", t.replace(" ", ""))
    if m:
        a, n, s, e = m.groups()
        a = a.upper()
        lon = (ord(a[0]) - 65) * 20 - 180 + int(n[0]) * 2
        lat = (ord(a[1]) - 65) * 10 - 90 + int(n[1])
        wl, wa = 2.0, 1.0                                   # size of the square so far
        if s:
            s = s.lower()
            lon += (ord(s[0]) - 97) * 5 / 60
            lat += (ord(s[1]) - 97) * 2.5 / 60
            wl, wa = 5 / 60, 2.5 / 60
            if e:
                lon += int(e[0]) * 0.5 / 60
                lat += int(e[1]) * 0.25 / 60
                wl, wa = 0.5 / 60, 0.25 / 60
        return round(lat + wa / 2, 4), round(lon + wl / 2, 4)     # centre of the square
    parts = [p for p in re.split(r"[,\s;]+", t) if p]
    if len(parts) != 2:
        raise ValueError("give 'lat, lon' or a Maidenhead locator")
    lat, lon = float(parts[0]), float(parts[1])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("latitude -90..90, longitude -180..180")
    return round(lat, 4), round(lon, 4)


def core_weights(priorities) -> dict:
    """Base score per priority station: the 1st always wins a clash with the 2nd."""
    return {s: w for s, w in zip([p for p in priorities if p], (1000, 800))}


CORE = core_weights(DEFAULT_PRIORITIES)

# chart importance (first matching rule wins); 0 = never plan on its own.
# Written to cover the wording of every station in the schedule file, not
# just Northwood and DWD, so any station works as a 1st/2nd priority.
REPEAT_RE = r"repetition|rebroadcast|retransmission|re-?transmit"
IMPORTANCE = [
    (r"not broadcast|schedule|request for comments|notice|test chart|general information|"
     r"observation report|weather report|conditions report|manam|for shipping|sea forecast", 0),
    (r"gale|storm warning|warning", 28),
    (r"sigwx|significant weather", 10),
]
# analyses of the surface / sea-level pressure field (only when no forecast hour)
ANALYSIS_RE = (r"surface (weather chart|analysis|chart|map)|sfc analysis|msl analysis|"
               r"prelim\w* surface|^surface pressure$|surface pressure \(|tasman.*msl analysis")
FORECAST_RE = r"prog|forecast|f'cast|prediction"
UPPER_RE = r"\d{3} ?(mb|hpa)|upper air"
MIN_CORE_IMPORTANCE = 10

LEAD_S = 75          # tune this long before the slot
TAIL_S = 45          # extra time after the expected end
SWITCH_GAP_S = 20    # charts of two stations can both be planned if the second starts this long after the first ends
MIN_LEAD_S = 15      # ... and the receiver leaves the first no earlier than this before the second starts
DECODE_WAIT_MAX_S = 20 * 60  # longest wait for a decode before an item is logged as failed
HOLD_MAX_S = 25 * 60  # a recording longer than any single chart is ended for the next planned one
RETRY_BONUS = 2000   # repeat of a priority chart whose first try failed: beats any other chart
GOOD_QUALITY = 0.6   # decode quality that counts as 'collected'
EMPTY_SLOT_SNR = -6.0    # a failed slot with no signal at all (dB) ...
EMPTY_SLOT_S = 45 * 60   # ... followed this soon by a good decode on the same frequency was
                         # simply not transmitted: it says nothing about propagation
EMPTY_SLOT_COUNT = 3     # a priority slot found empty this often (with the frequency working
                         # either side) is taken as not broadcast and left out of the plan ...
EMPTY_RECHECK_DAYS = 14  # ... but tried again this long after its last attempt
HALF_LIFE_DAYS = 21  # learned results lose half their weight in this time
PROBE_DAYS = 14      # re-check an alternative frequency not tried for this long
DX_GOOD_QUALITY = 0.45
DX_BACKOFF_MAX_DAYS = 7   # longest pause for a DX frequency/time block after failures in a row
DX_MIN_PRIOR = 0.10       # below this the propagation model calls a DX path hopeless
DX_GIVE_UP_S = 240        # silent DX probe: abandoned this long after its slot start ...
DX_GIVE_UP_SNR = 0.0      # ... if the SNR has stayed below this (dB) and nothing was received
SCAN = True              # before a chart sent on several frequencies, compare them on air
SCAN_SETTLE_S = 1.2      # after SDR Console confirms a frequency: audio and SNR settle ...
SCAN_MEASURE_S = 4.0     # ... then the SNR is read for this long
SCAN_DONE_BEFORE_S = 10  # the scan ends this long before the slot, back on the chosen frequency
SCAN_SWITCH_DB = 3.0     # another frequency must be this much better than the planned one
SCAN_MIN_SNR = 3.0       # below this on every frequency: nothing on air yet - keep the plan, log nothing
# Back-to-back charts (Northwood every 12 minutes) leave no signal before the
# next chart to compare.  Then the comparison is made during the chart's own
# start tone and phasing lines (~35 s, sent identically on every frequency):
SCAN_PHASING = False     # off: the receiver stays put from the start tone to the picture.  Retuning
                         # during the phasing damaged it and gave wrong line periods; the pre-chart
                         # scan, the learning and the off-air hunt choose the frequency instead
PH_WAIT_S = 3.5          # after the start tone triggers the recording: let it finish
PH_MEASURE_S = 2.5       # SNR read for this long on each frequency
PH_SETTLE_S = 1.0        # after SDR Console confirms a frequency
PH_BUDGET_S = 25.0       # all done (back on the chosen frequency) this long after the trigger
PH_PHASING_WAIT_S = 0.5  # recording triggered by the phasing itself (start tone missed): it is
PH_PHASING_BUDGET_S = 17.0  # already ~8 s into the ~30 s of phasing - less time, fewer frequencies
# Nothing heard after the start time: the planned frequency may be off air
# (or faded) while the chart goes out on the station's others.
OFFAIR_WAIT_S = 20       # this long after the slot start with no recording ...
OFFAIR_QUIET_S = 10      # ... and the SNR below OFFAIR_SNR for this long ...
OFFAIR_SNR = 0.0
OFFAIR_LATEST_S = 120    # ... (not later than this after the start) - try the other frequencies
AUDIO_SUSPECT_SNR = 5.0   # a slot with a signal this strong (dB) but no fax decoded is suspicious ...
AUDIO_FAULT_COUNT = 3     # ... this many in a row with no decode in between = the audio path is broken
AUDIO_OK_QUALITY = 0.2   # any decode at least this good shows the audio path works
AUDIO_HOLD_H = 6.0        # suspicious failures held back from the learning log at most this long


def lead_hours(c: str) -> int | None:
    """Forecast lead time in hours: 'T+24', 'H+48', '36HR', '72 HR', '3-DAY'."""
    m = re.search(r"[th] ?\+ ?(\d{1,3})", c) or re.search(r"(\d{1,3}) ?-?(?:hr|hrs|h|hour)s?\b", c)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d) ?-?day", c)
    return 24 * int(m.group(1)) if m else None


def importance(contents: str) -> int:
    c = contents.lower().strip()
    if re.search(REPEAT_RE, c):
        return 0                     # handled through the original chart
    for pat, v in IMPORTANCE:
        if re.search(pat, c):
            return v
    lead = lead_hours(c)
    upper = re.search(UPPER_RE, c) is not None
    if lead is None and not re.search(FORECAST_RE, c):
        if "surface pressure analysis" in c:
            return 22
        if re.search(ANALYSIS_RE, c):
            return 30
    if upper:
        # upper-air charts: useful, but behind the surface charts
        return 12 if lead is None and "analysis" in c else 10 if (lead or 0) <= 48 else 6
    if lead is not None:
        # any forecast product: the nearer the lead time the better
        return 24 if lead <= 24 else 20 if lead <= 36 else 16 if lead <= 48 else 12 if lead <= 72 else 8
    if re.search(FORECAST_RE, c) and re.search(r"surface|sfc|msl|pressure", c):
        return 16
    for pat, v in ((r"swell|wave|sea state|sea and swell", 12), (r"wind", 10),
                   (r"precip|visibility", 10), (r"tropical", 10), (r"anps", 8),
                   (r"ice", 10), (r"satellite|goes|msat|infrared", 8),
                   (r"temperature|fronts|sst", 5)):
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


def _gc_point(lat1, lon1, lat2, lon2, frac):
    """Point a fraction `frac` of the way along the great circle."""
    p1, l1, p2, l2 = map(math.radians, (lat1, lon1, lat2, lon2))
    d = 2 * math.asin(math.sqrt(math.sin((p2 - p1) / 2) ** 2
                                + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2))
    if d < 1e-9:
        return lat1, lon1
    A, B = math.sin((1 - frac) * d) / math.sin(d), math.sin(frac * d) / math.sin(d)
    x = A * math.cos(p1) * math.cos(l1) + B * math.cos(p2) * math.cos(l2)
    y = A * math.cos(p1) * math.sin(l1) + B * math.cos(p2) * math.sin(l2)
    z = A * math.sin(p1) + B * math.sin(p2)
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


def prior(qth, short: str, khz: float, when: datetime) -> float:
    """Rough probability (0..1) that a frequency is usable on this path now.

    foF2 follows solar elevation (about 4 MHz at night to 11 MHz at high
    sun); MUF = foF2 x obliquity factor (single hop, 300 km layer, at most
    3.4); daytime D-layer absorption raises the LUF on long paths. Paths up
    to 2000 km are judged at the mid-point. Longer ones at control points
    1000 km in from each end as well: the MUF is set by the darkest of them
    and the LUF by the sunniest, so a 2 MHz path across the Atlantic is not
    rated open at noon just because its mid-point is dark. Crude by design -
    logged results take over once available.
    """
    site = SITES.get(short)
    if site is None:
        return 0.15
    d, mlat, mlon = _gc(qth[0], qth[1], site[0], site[1])
    if d <= 2000.0:
        pts = [(mlat, mlon)]
    else:
        f = 1000.0 / d
        pts = [_gc_point(qth[0], qth[1], site[0], site[1], q) for q in (f, 0.5, 1 - f)]
    sun = [max(0.0, math.sin(math.radians(_solar_elev(la, lo, when)))) for la, lo in pts]
    fof2 = 4.0 + 7.0 * math.sqrt(min(sun))      # the weakest control point limits the MUF
    s = max(sun)                                 # the sunniest one sets the absorption
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
        self._dead: dict[tuple, tuple] = {}         # (station, kHz) -> last silent failure
        self.slot_gaps: dict[tuple, int] = {}       # (station, slot) -> times found empty
        self.slot_last: dict[tuple, float] = {}     # (station, slot) -> last attempt
        self.migrated = self._migrate()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    self._apply(json.loads(line))
                except Exception:
                    pass

    LOG_VERSION = 2

    def _migrate(self) -> str | None:
        """One-time clean-up of logs written by versions before the fixes of
        16 Oct 2026, which logged charts as failed while a good signal was on
        air (stuck recordings, charts credited to the wrong slot) and gave
        off-plan charts an SNR read after the chart had ended. Removes those
        failures and blanks those SNRs; everything else is kept. The original
        is kept as proplog_before_v2.jsonl and a marker stops it running again."""
        marker = self.path.with_name(self.path.stem + f".v{self.LOG_VERSION}")
        if marker.exists():
            return None
        if not self.path.exists():
            marker.write_text("new log\n")
            return None
        lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        out, dropped, blanked = [], 0, 0
        for line in lines:
            try:
                r = json.loads(line)
            except Exception:
                out.append(line)
                continue
            if r.get("kind") == "core" and not r.get("quality") and (r.get("snr_db") or -99) > 3:
                dropped += 1                      # failed with a good signal on air: not propagation
                continue
            if r.get("kind") == "manual" and r.get("snr_db") is not None:
                r["snr_db"] = None                # read after the chart had ended
                blanked += 1
            out.append(json.dumps(r))
        backup = self.path.with_name(self.path.stem + f"_before_v{self.LOG_VERSION}.jsonl")
        if not backup.exists():
            backup.write_bytes(self.path.read_bytes())
        tmp = self.path.with_suffix(".jsonl.part")
        tmp.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
        os.replace(tmp, self.path)
        marker.write_text(f"cleaned: {dropped} false failures removed, {blanked} SNRs blanked\n")
        return f"learning log tidied once: {dropped} false failures removed, {blanked} SNRs blanked " \
               f"(original kept as {backup.name})"

    def _apply(self, r: dict):
        when = datetime.fromisoformat(r["utc"].replace("Z", "+00:00"))
        k = (r["short"], float(r["khz"]), hour_bucket(when))
        st = self.stats.setdefault(k, {"hist": [], "fails_in_row": 0, "last": 0.0, "last_ok": 0.0})
        if r.get("kind") == "scan":
            # SNR of the signal on air just before a chart, read on each of its
            # frequencies: evidence about the channel, not a decode attempt
            if r.get("snr_db") is not None:
                st["scan"] = (st.get("scan", []) + [(when.timestamp(), float(r["snr_db"]))])[-60:]
            return
        q = float(r.get("quality") or 0.0)
        kind = r.get("kind")
        # a priority-station chart received off-plan is judged like a planned one
        core = kind == "core" or (kind == "manual" and r["short"] in CORE) if kind else r["short"] in CORE
        good = q >= (GOOD_QUALITY if core else DX_GOOD_QUALITY)
        ts, snr = when.timestamp(), r.get("snr_db")
        st["hist"] = (st["hist"] + [(ts, good, q, snr)])[-60:]
        st["last"] = ts
        fk = (r["short"], float(r["khz"]))
        sk = (r["short"], r.get("slot"))
        if sk[1]:
            self.slot_last[sk] = max(ts, self.slot_last.get(sk, 0.0))
            if good:
                self.slot_gaps.pop(sk, None)          # it is broadcast after all
        if good:
            st["fails_in_row"] = 0
            st["last_ok"] = ts
            # a silent failure shortly before on this same frequency was a slot
            # the station didn't send (schedule gap), not a propagation failure
            d = self._dead.pop(fk, None)
            if d and 0 < ts - d[0] <= EMPTY_SLOT_S:
                if d[2]:
                    self.slot_gaps[(r["short"], d[2])] = self.slot_gaps.get((r["short"], d[2]), 0) + 1
                dst = self.stats.get(d[1])
                if dst:
                    dst["hist"] = [h for h in dst["hist"] if h[0] != d[0]]
                    if dst is not st and dst["fails_in_row"] > 0:
                        dst["fails_in_row"] -= 1
        else:
            st["fails_in_row"] += 1
            if snr is not None and snr < EMPTY_SLOT_SNR and r.get("kind") != "dx":
                self._dead[fk] = (ts, k, r.get("slot"))
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
        # pre-chart scans add SNR readings (no decode result)
        snr += [s for ts, s in st.get("scan", []) if at - ts < 2 * HALF_LIFE_DAYS * 86400]
        return {"n": n, "ok": ok, "qmean": qs / n if n else None,
                "snr_med": sorted(snr)[len(snr) // 2] if len(snr) >= 2 else None}

    def add(self, r: dict):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
            self._apply(r)

    def slot_empty(self, short: str, slot: str, now: datetime) -> bool:
        """This slot has repeatedly been found empty (schedule gap): skip it,
        except for a recheck every EMPTY_RECHECK_DAYS."""
        if self.slot_gaps.get((short, slot), 0) < EMPTY_SLOT_COUNT:
            return False
        return now.timestamp() - self.slot_last.get((short, slot), 0.0) < EMPTY_RECHECK_DAYS * 86400

    def empty_slots(self, now: datetime) -> list[str]:
        return sorted(f"{s} {sl}Z" for (s, sl) in self.slot_gaps if self.slot_empty(s, sl, now))

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
        days = min(DX_BACKOFF_MAX_DAYS, 2 ** (st["fails_in_row"] - 1))
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
            if w["snr_med"] is not None:      # from decodes and pre-chart scans
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
                        "attempts": len(st["hist"]), "scans": len(st.get("scan", [])), "weight": round(w["n"], 1), "good": round(w["ok"], 1),
                        "avg_quality": None if w["qmean"] is None else round(w["qmean"], 2),
                        "snr_db": w["snr_med"],
                        "last_tried": datetime.fromtimestamp(st["last"], timezone.utc).strftime("%Y-%m-%d"),
                        "retry_after": (datetime.fromtimestamp(
                            st["last"] + min(DX_BACKOFF_MAX_DAYS, 2 ** (st["fails_in_row"] - 1)) * 86400, timezone.utc)
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
    alts: list = field(default_factory=list)   # every frequency the slot is sent on

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
        # never pick a frequency the propagation model rules out (above the MUF,
        # below the LUF) and that has no results of its own, just because the
        # tried ones have been failing - poor conditions on all of them is
        # more likely than an untried 11 MHz working at night
        cand = [k for k in khzs if prior(self.qth, short, k, when) >= 0.1 or self.log.get(short, k, when)] or khzs
        scored = [(self.log.value(self.qth, short, k, when, sample=explore, rng=rng), k) for k in cand]
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
                low = contents.lower()
                if re.search(REPEAT_RE, low):
                    # 'Repetition chart 1050 UTC', '(REBROADCAST OF 0325/1525)':
                    # a back-up of an earlier chart today
                    orig = next((t for t in re.findall(r"\b(\d{4})\b", low) if t in ref), None)
                    if orig:
                        contents, valid = ref[orig]
                ref[sl.hhmm] = (contents, valid)
                imp = importance(contents)
                if imp < MIN_CORE_IMPORTANCE or start < now - timedelta(seconds=30):
                    continue
                if self.log.slot_empty(st.short, sl.hhmm, start):
                    continue              # found empty again and again: not broadcast
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
                                  key, alts=list(khzs)))
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
                # skip what the propagation model calls hopeless (a 2 MHz path
                # across the Atlantic at noon, 18 MHz at night) unless that
                # frequency has actually worked in this time block before
                elig = [k for k in elig if prior(self.qth, st.short, k, start) >= DX_MIN_PRIOR
                        or ((h := self.log.get(st.short, k, start)) and h["last_ok"] > 0)]
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
                               100 * p + imp, key, alts=list(elig)))
        return self._select(dx, chosen + [b for b in busy if b.kind != "core"])

    @staticmethod
    def _select(cands: list[Item], fixed: list[Item] | None = None) -> list[Item]:
        """Greedy by score; an item needs its slot free.

        Same station (Northwood every 12 minutes): just no overlap.  Another
        station: the later chart must start at least SWITCH_GAP_S after the
        earlier one's expected end - enough to retune before its start tone.
        (Requiring the full 75 s tuning lead plus a tail here made e.g. DWD
        18:00 clash with Northwood 17:48; DWD dropped out of the plan, a DX
        probe was fitted into the gap, then DWD came back once Northwood
        was over and took the receiver from the probe.)"""
        taken = list(fixed or [])
        keys = {t.key for t in taken}
        gap = timedelta(seconds=SWITCH_GAP_S)

        def free(it):
            for t in taken:
                g = timedelta(0) if t.short == it.short else gap
                if it.start < t.end + g and t.start < it.end + g:
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
        if self.proplog.migrated:
            log(self.proplog.migrated)
        self.settings_path = Path(out_dir) / "settings.json"
        self.priorities = self._load_priorities(sched)
        self.qth = tuple(qth)
        self.planner = Planner(sched, self.proplog, qth, dx, priorities=self.priorities)
        self.enabled = enabled
        self.paused_until = 0.0
        self.items: list[Item] = []
        self.current: Item | None = None
        self.receiving = False
        self.rec_elapsed = 0.0
        self.decoder_busy = None       # set by the main program: True while faxes wait to be decoded
        self.force_end = None          # set by the main program: ends the recording in progress
        self._forced_t = 0.0
        self.last_snr = None
        # audio-path check: failures held back while it is unclear whether the
        # radio audio reaching the decoder is sound (see _log_item)
        self.held: list[tuple[dict, bool, float]] = []
        self.scanning = False          # comparing a chart's frequencies on air (see _scan)
        self._snr_recent = collections.deque(maxlen=120)   # (time, SNR) from the status feed
        self._scan_snr: list | None = None
        self.audio_fault = False
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

    def set_qth(self, qth):
        """Receiver location changed on the page: the propagation estimates
        (and so the frequency choices and DX probes) follow at once."""
        self.qth = self.planner.qth = tuple(qth)
        self.logf(f"auto: location set to {qth[0]:.2f}, {qth[1]:.2f}")
        self.replan()

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
        empty = set(self.proplog.empty_slots(now))
        new_empty = empty - getattr(self, "_empty_seen", set())
        if new_empty:
            self.logf("auto: slots found empty repeatedly (not broadcast?) - left out of the plan, "
                      f"rechecked every {EMPTY_RECHECK_DAYS} days: {', '.join(sorted(new_empty))}")
        self._empty_seen = empty
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
        was = self.receiving
        self.receiving = s.get("state") == "receiving"
        if self.receiving and not was and s.get("trigger") in ("start tone", "phasing"):
            self._maybe_phasing_scan(s.get("trigger"))
        self.rec_elapsed = float(s.get("elapsed") or 0) if self.receiving else 0.0
        # when the recording in progress started (PC clock)
        self.rec_start_ts = time.time() - self.rec_elapsed if self.receiving else None
        if s.get("snr_db") is not None:
            self.last_snr = float(s["snr_db"])
            self._snr_recent.append((time.time(), self.last_snr))
            if self._scan_snr is not None:
                self._scan_snr.append(self.last_snr)

    def on_fax(self, product: dict | None, meta: dict, utc: str):
        """A fax was decoded: attach the result to the matching plan item."""
        q = float(meta.get("quality", 0))
        # SNR measured while the chart was coming in; the current reading is
        # taken after it ended (often just noise) and would be misleading
        snr = meta.get("snr_db")
        if snr is None:
            snr = self.last_snr
        with self._lock:
            it = None
            if product and product.get("slot"):
                it = next((i for i in self.items if i.short == product.get("short")
                           and i.slot == product.get("slot") and i.state in ("tuned", "closing")), None)
        if it is not None:
            it.result = {"quality": q, "lines": meta.get("lines"), "snr_db": snr}
        elif product and product.get("khz"):
            # not a planned item (e.g. the next chart of the same station while
            # still tuned there) - still useful to learn from
            when = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            self.proplog.add({"utc": utc, "short": product["short"], "khz": product["khz"],
                              "slot": product.get("slot"), "snr_db": None if snr is None else round(snr, 1),
                              "quality": q, "key": None, "kind": "manual", "hour": when.hour})
        if q >= AUDIO_OK_QUALITY:
            self._audio_ok()
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
                if self.receiving and it.start - timedelta(seconds=LEAD_S) <= now:
                    it.heard = True               # something was recorded while on this item
                if self._dx_dead(it, now):
                    continue
                # an earlier recording was still running when this chart began:
                # the receiver was busy, which says nothing about propagation
                rs = getattr(self, "rec_start_ts", None)
                if self.receiving and rs is not None and \
                        it.start + timedelta(seconds=60) <= now <= it.end and \
                        rs < it.start.timestamp() - 90:
                    it.blocked = True
                if now > it.end + timedelta(seconds=TAIL_S) and not self.receiving:
                    self._close(it)
            waited = time.time() - it.close_t if it.state == "closing" else 0
            busy = self.decoder_busy() if self.decoder_busy else False
            # wait for the decode: 150 s, longer while the decoder is still busy
            if it.state == "closing" and (it.result is not None or (waited > 150 and not busy)
                                           or waited > DECODE_WAIT_MAX_S):
                it.state = "done"
                if it.result is None and getattr(it, "abandoned", False):
                    # left before the chart could finish: says nothing about propagation
                    self.logf(f"auto: {it.short} {it.khz} kHz {it.slot}Z left early - not counted")
                elif it.result is None and getattr(it, "blocked", False):
                    self.logf(f"auto: {it.short} {it.khz} kHz {it.slot}Z missed - receiver busy with "
                              f"an earlier recording - not counted")
                else:
                    self._log_item(it, it.result["quality"] if it.result else 0.0)
        if not self.enabled or time.time() < self.paused_until or self.cat is None or not self.cat.enabled:
            return
        if self.scanning:
            return                                   # the scan thread has the receiver
        cur = self.current
        if (cur is not None and cur.state == "tuned" and not self.receiving and SCAN
                and len(set(cur.alts)) >= 2 and not getattr(cur, "hunted", False)
                and cur.start + timedelta(seconds=OFFAIR_WAIT_S) <= now
                <= cur.start + timedelta(seconds=OFFAIR_LATEST_S) and self._quiet()):
            cur.hunted = True
            self.scanning = True
            threading.Thread(target=self._offair_hunt, args=(cur,), daemon=True).start()
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
                # not before the current chart is due to end, unless the next
                # one is about to start (its tuning lead overlaps this chart)
                if now < min(self.current.end, due.start - timedelta(seconds=MIN_LEAD_S)):
                    return
                self.current.abandoned = True
            self._close(self.current)
        due.state = "tuned"
        self.current = due
        if self._scan_list(due, now):
            self.scanning = True
            self.logf(f"auto: {due.start:%H:%M}Z {due.short} - {due.contents} [{due.kind}]: comparing "
                      f"{', '.join(f'{k:g}' for k in self._scan_list(due, now))} kHz on air first")
            threading.Thread(target=self._scan, args=(due,), daemon=True).start()
            return
        self.cat.tune(due.dial)
        self.logf(f"auto: {due.start:%H:%M}Z {due.short} {due.khz} kHz (dial {due.dial}) - {due.contents}"
                  f" [{due.kind}]")

    def _dx_dead(self, it: Item, now: datetime) -> bool:
        """A DX probe that has heard nothing worth recording: stop it early,
        log the failure and free the receiver for the next planned item or
        for another probe fitted into the gap."""
        if it.kind != "dx" or getattr(it, "heard", False) or self.receiving:
            return False
        if now < it.start + timedelta(seconds=DX_GIVE_UP_S) or len(it.snr) < 20:
            return False
        snr = sorted(it.snr)
        if snr[len(snr) // 2] >= DX_GIVE_UP_SNR or snr[int(len(snr) * 0.9)] >= DX_GIVE_UP_SNR + 3:
            return False
        med = round(snr[len(snr) // 2], 1)
        self.logf(f"auto: {it.short} {it.khz} kHz {it.slot}Z - nothing heard after "
                  f"{DX_GIVE_UP_S // 60} min (SNR {med:+.0f} dB), moving on")
        it.result = {"quality": 0.0, "snr_db": med}
        self._close(it)
        threading.Thread(target=self.replan, daemon=True).start()   # use the freed time
        return True

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
        if it.result and it.result.get("snr_db") is not None:
            snr = round(float(it.result["snr_db"]), 1)   # measured during the chart itself
        rec = {"utc": it.start.strftime("%Y-%m-%dT%H:%M:%SZ"), "short": it.short, "khz": it.khz,
               "slot": it.slot, "snr_db": snr, "quality": round(quality, 3), "key": it.key,
               "kind": it.kind, "hour": it.start.hour}
        if not self._audio_check(rec, quality, snr):
            return
        self.proplog.add(rec)
        verdict = "good" if quality >= (GOOD_QUALITY if it.kind == "core" else DX_GOOD_QUALITY) else "poor/none"
        self.logf(f"auto: logged {it.short} {it.khz} kHz {it.slot}Z  SNR "
                  f"{'--' if snr is None else f'{snr:+.0f} dB'}  quality {quality:.2f} ({verdict})")

    # -------- pre-chart frequency scan
    # Northwood, DWD and others send the same chart on several frequencies at
    # once, and keep a signal on air between charts (the previous chart, or a
    # steady carrier).  Before the chart starts, each frequency is tuned in
    # turn and its SNR read; the chart is then received on the best one.  The
    # readings go into the learning log as SNR-only evidence ("scan").
    def _scan_list(self, it: Item, now: datetime) -> list[float]:
        """The frequencies to compare (best expected first), as many as fit
        before the slot; [] if there is nothing to compare or no time."""
        if not SCAN or len(set(it.alts)) < 2:
            return []
        per = SCAN_SETTLE_S + SCAN_MEASURE_S + 1.5          # + CAT poll and confirmation
        room = (it.start - now).total_seconds() - SCAN_DONE_BEFORE_S - 3.0   # - final retune
        n = int(room // per)
        if n < 2:
            return []
        order = sorted(set(it.alts), key=lambda k: -self.proplog.value(self.qth, it.short, k, it.start))
        if it.khz in order:                                   # the planned one always gets measured
            order.remove(it.khz)
            order.insert(0, it.khz)
        return order[:n]

    def _scan(self, it: Item):
        try:
            self._scan_run(it)
        except Exception as e:
            self.logf(f"auto: frequency scan failed ({e}) - receiving on {it.khz} kHz")
            self.cat.tune(it.dial)
        finally:
            self._scan_snr = None
            self.scanning = False

    def _scan_run(self, it: Item):
        freqs = self._scan_list(it, self.clock())
        planned = it.khz
        res = {}
        for khz in freqs:
            if self.receiving or not self.enabled:
                break                     # a chart started (or AUTO switched off): stay put
            dial = round(khz - DIAL_OFFSET_KHZ, 2)
            self.cat.tune(dial, hop=False)
            if hasattr(self.cat, "wait_for") and not self.cat.wait_for(dial, timeout=3.0):
                continue
            time.sleep(SCAN_SETTLE_S)
            self._scan_snr = []
            time.sleep(SCAN_MEASURE_S)
            v, self._scan_snr = self._scan_snr, None
            if v:
                res[khz] = sorted(v)[len(v) // 2]
        if self.receiving:
            # recording on the frequency it is on now: that is where the chart is
            cur = self.cat.dial_khz_at() if hasattr(self.cat, "dial_khz_at") else None
            if cur is not None:
                it.khz = min(it.alts, key=lambda k: abs(k - DIAL_OFFSET_KHZ - cur))
            self.logf(f"auto: scan stopped - reception started on {it.khz} kHz")
            return
        txt = ", ".join(f"{k:g} {s:+.0f} dB" for k, s in res.items())
        best = max(res, key=res.get) if res else planned
        if not res or res[best] < SCAN_MIN_SNR:
            self.logf(f"auto: scan {it.short} {it.slot}Z: {txt or 'no readings'} - nothing on air yet, "
                      f"keeping {planned} kHz")
        else:
            if best != planned and res[best] < res.get(planned, -99) + SCAN_SWITCH_DB:
                best = planned            # not clearly better than the planned frequency
            it.khz = best
            it.scan_done = True           # compared on air: no need to repeat during the phasing
            self.logf(f"auto: scan {it.short} {it.slot}Z: {txt} -> {best} kHz"
                      f"{'' if best == planned else f' (planned {planned})'}")
            for khz, snr in res.items():
                self.proplog.add({"utc": it.start.strftime("%Y-%m-%dT%H:%M:%SZ"), "short": it.short,
                                  "khz": khz, "slot": it.slot, "snr_db": round(snr, 1), "quality": None,
                                  "key": None, "kind": "scan", "hour": it.start.hour})
        if self.current is it and it.state == "tuned":
            self.cat.tune(it.dial)
            self.logf(f"auto: {it.start:%H:%M}Z {it.short} {it.khz} kHz (dial {it.dial}) - {it.contents}"
                      f" [{it.kind}]")

    def _maybe_phasing_scan(self, trigger: str = "start tone"):
        it = self.current
        if not (SCAN and SCAN_PHASING and self.enabled and not self.scanning and it is not None
                and it.state == "tuned" and len(set(it.alts)) >= 2 and not getattr(it, "scan_done", False)
                and self.cat is not None and self.cat.enabled):
            return
        now = self.clock()
        if not (it.start - timedelta(seconds=60) <= now <= it.start + timedelta(seconds=120)):
            return                       # not this item's chart starting
        it.scan_done = True
        self.scanning = True
        threading.Thread(target=self._phasing_scan, args=(it, trigger), daemon=True).start()

    def _phasing_scan(self, it: Item, trigger: str = "start tone"):
        try:
            self._phasing_scan_run(it, trigger)
        except Exception as e:
            self.logf(f"auto: phasing scan failed ({e}) - staying on {it.khz} kHz")
            try:
                self.cat.tune(it.dial, hop=False)
            except Exception:
                pass
        finally:
            self._scan_snr = None
            self.scanning = False

    def _measure(self, seconds: float):
        self._scan_snr = []
        time.sleep(seconds)
        v, self._scan_snr = self._scan_snr, None
        return sorted(v)[len(v) // 2] if v else None

    def _phasing_scan_run(self, it: Item, trigger: str = "start tone"):
        """The chart has just started on it.khz (start tone heard).  Read the
        SNR of its phasing lines here and on its other frequencies, then go
        to the best one before the picture starts.  No centring hop - the
        recording is running."""
        t0 = time.time()
        wait, budget = (PH_WAIT_S, PH_BUDGET_S) if trigger == "start tone" else \
            (PH_PHASING_WAIT_S, PH_PHASING_BUDGET_S)
        planned = it.khz
        others = sorted((k for k in set(it.alts) if k != planned),
                        key=lambda k: -self.proplog.value(self.qth, it.short, k, it.start))
        self.logf(f"auto: {it.short} {it.slot}Z chart starting ({trigger}) - comparing "
                  f"{', '.join(f'{k:g}' for k in [planned] + others)} kHz during its phasing")
        time.sleep(wait)
        res = {}
        snr = self._measure(PH_MEASURE_S)
        if snr is not None:
            res[planned] = snr
        per = PH_SETTLE_S + PH_MEASURE_S + 1.5
        for khz in others:
            if not self.receiving or time.time() - t0 + per + 2.0 > budget:
                break
            dial = round(khz - DIAL_OFFSET_KHZ, 2)
            self.cat.tune(dial, hop=False)
            if hasattr(self.cat, "wait_for") and not self.cat.wait_for(dial, timeout=2.5):
                continue
            time.sleep(PH_SETTLE_S)
            snr = self._measure(PH_MEASURE_S)
            if snr is not None:
                res[khz] = snr
        txt = ", ".join(f"{k:g} {v:+.0f} dB" for k, v in res.items())
        best = max(res, key=res.get) if res else planned
        if best != planned and res[best] < res.get(planned, -99) + SCAN_SWITCH_DB:
            best = planned
        it.khz = best
        if res and res[best] >= SCAN_MIN_SNR:
            for khz, v in res.items():
                self.proplog.add({"utc": it.start.strftime("%Y-%m-%dT%H:%M:%SZ"), "short": it.short,
                                  "khz": khz, "slot": it.slot, "snr_db": round(v, 1), "quality": None,
                                  "key": None, "kind": "scan", "hour": it.start.hour})
        self.cat.tune(it.dial, hop=False)
        self.logf(f"auto: phasing scan {it.short} {it.slot}Z: {txt or 'no readings'} -> {best} kHz"
                  f"{'' if best == planned else f' (planned {planned})'} after {time.time() - t0:.0f} s")

    def _quiet(self) -> bool:
        """Nothing on the tuned frequency for the last OFFAIR_QUIET_S."""
        t = time.time()
        v = [snr for ts, snr in self._snr_recent if t - ts <= OFFAIR_QUIET_S]
        return len(v) >= OFFAIR_QUIET_S * 0.6 and sorted(v)[len(v) // 2] < OFFAIR_SNR and max(v) < SCAN_MIN_SNR

    def _offair_hunt(self, it: Item):
        try:
            self._offair_hunt_run(it)
        except Exception as e:
            self.logf(f"auto: frequency hunt failed ({e}) - back on {it.khz} kHz")
            try:
                self.cat.tune(it.dial, hop=False)
            except Exception:
                pass
        finally:
            self._scan_snr = None
            self.scanning = False

    def _offair_hunt_run(self, it: Item):
        """The chart should have started but nothing is heard on it.khz: look
        for it on the slot's other frequencies and stay on the best one.  Its
        start tone and phasing are missed, so the picture check picks the
        chart up on the new frequency."""
        planned = it.khz
        v = [snr for ts, snr in self._snr_recent if time.time() - ts <= OFFAIR_QUIET_S]
        res = {planned: sorted(v)[len(v) // 2]} if v else {}
        others = sorted((k for k in set(it.alts) if k != planned),
                        key=lambda k: -self.proplog.value(self.qth, it.short, k, it.start))
        self.logf(f"auto: nothing heard on {it.short} {planned:g} kHz {OFFAIR_WAIT_S} s after {it.slot}Z - "
                  f"trying {', '.join(f'{k:g}' for k in others)} kHz")
        for khz in others:
            if self.receiving or not self.enabled or self.current is not it:
                break
            dial = round(khz - DIAL_OFFSET_KHZ, 2)
            self.cat.tune(dial, hop=False)
            if hasattr(self.cat, "wait_for") and not self.cat.wait_for(dial, timeout=2.5):
                continue
            time.sleep(PH_SETTLE_S)
            snr = self._measure(PH_MEASURE_S)
            if snr is not None:
                res[khz] = snr
        txt = ", ".join(f"{k:g} {v:+.0f} dB" for k, v in res.items())
        it.scan_done = True               # just compared: the phasing scan must not repeat it
        best = max(res, key=res.get) if res else planned
        if res.get(best, -99) < SCAN_MIN_SNR:
            best = planned                # nowhere better: the chart is probably not being sent
        it.khz = best
        if res and res.get(best, -99) >= SCAN_MIN_SNR:
            for khz, val in res.items():
                self.proplog.add({"utc": it.start.strftime("%Y-%m-%dT%H:%M:%SZ"), "short": it.short,
                                  "khz": khz, "slot": it.slot, "snr_db": round(val, 1), "quality": None,
                                  "key": None, "kind": "scan", "hour": it.start.hour})
        if self.current is it:
            self.cat.tune(it.dial, hop=False)
        self.logf(f"auto: hunt {it.short} {it.slot}Z: {txt or 'no readings'} -> "
                  + (f"{best} kHz (planned {planned} silent)" if best != planned
                     else f"nothing better, back on {planned} kHz"))

    # -------- audio-path check
    # A strong signal with no fax decoded, slot after slot, says the audio
    # reaching the decoder is broken (a virtual audio cable gone wrong, the
    # wrong input, the receiver's audio routed elsewhere), not that the
    # frequencies are poor.  Such failures are held back from the learning
    # log; AUDIO_FAULT_COUNT of them with no decode in between raise a warning
    # and are dropped, as is every failure until a fax decodes again.  Fewer
    # than that are real failures and are logged once a fax decodes (or after
    # AUDIO_HOLD_H hours).
    def _audio_check(self, rec: dict, quality: float, snr) -> bool:
        """True: log this result now.  False: held back or dropped."""
        if quality >= AUDIO_OK_QUALITY:
            self._audio_ok()
            return True
        if self.audio_fault:
            self.logf(f"auto: {rec['short']} {rec['khz']} kHz {rec['slot']}Z not counted - "
                      f"audio fault suspected")
            return False
        now = time.time()
        if self.held and now - self.held[0][2] > AUDIO_HOLD_H * 3600:
            self._flush_held()
        suspect = snr is not None and snr >= AUDIO_SUSPECT_SNR
        if not suspect and not self.held:
            return True
        self.held.append((rec, suspect, now))
        n = sum(1 for _, sus, _ in self.held if sus)
        if n >= AUDIO_FAULT_COUNT:
            self.audio_fault = True
            self.logf(f"auto: WARNING - {n} slots in a row with a strong signal (SNR >= "
                      f"{AUDIO_SUSPECT_SNR:+.0f} dB) but no fax decoded: the audio reaching the decoder "
                      f"looks broken (check the virtual audio cable / input). Results are not counted "
                      f"against the frequencies until a fax decodes again.")
            self.held.clear()
        else:
            self.logf(f"auto: {rec['short']} {rec['khz']} kHz {rec['slot']}Z  SNR "
                      f"{'--' if snr is None else f'{snr:+.0f} dB'}  no fax - held back until a fax "
                      f"decodes (audio check)")
        return False

    def _audio_ok(self):
        if self.audio_fault:
            self.audio_fault = False
            self.logf("auto: a fax decoded again - audio path OK, results counted again")
        self._flush_held()

    def _flush_held(self):
        held, self.held = self.held, []
        for rec, _, _ in held:
            self.proplog.add(rec)
            self.logf(f"auto: logged {rec['short']} {rec['khz']} kHz {rec['slot']}Z  quality 0.00 "
                      f"(held back, now counted)")

    def item_at(self, when: datetime, dial: float | None = None):
        """The plan item the scheduler had tuned for at this time (or None).
        dial: the frequency the recording was made on - an item tuned
        there beats one the receiver had just left."""
        with self._lock:
            items = list(self.items)
        hits = [it for it in items if it.state in ("tuned", "closing", "done")
                and it.start - timedelta(seconds=180) <= when <= it.end + timedelta(seconds=60)]
        if dial is not None and any(abs(it.dial - dial) <= 1.5 for it in hits):
            hits = [it for it in hits if abs(it.dial - dial) <= 1.5]
        # back-to-back charts overlap these windows: the one whose start is
        # nearest the start of reception is the chart being received (a new
        # start tone a minute after the last chart ended is the next chart)
        return min(hits, key=lambda it: abs((it.start - when).total_seconds()), default=None)

    # -------- for the web page
    def status(self) -> dict:
        now = self.clock()
        with self._lock:
            upcoming = [i.to_json() for i in self.items if i.end > now][:8]
        return {"enabled": self.enabled, "paused_min": max(0, int((self.paused_until - time.time()) / 60)),
                "current": self.current.to_json() if self.current else None, "plan": upcoming,
                "priorities": self.priorities, "audio_fault": self.audio_fault}
