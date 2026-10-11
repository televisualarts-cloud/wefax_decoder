"""Identify which chart is being received from station frequency + start time.

Schedule data comes from the OpenCPN weatherfax plugin's worldwide
WeatherFaxSchedules.xml (GPL-3, github.com/rgleason/weatherfax_pi),
bundled in wefax/data/.  A drop-in replacement can be put next to
wefax_decoder.py to update it without touching the code.  Corrections that
are known to be needed are applied on top (OVERRIDES below).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).with_name("data") / "WeatherFaxSchedules.xml"
DIAL_OFFSET_KHZ = 1.9

SHORT_NAMES = {
    "Hamburg/Pinneburg, Germany": "DWD",
    "Northwood, United Kingdom": "Northwood",
    "Boston, Massachusetts, U.S.A.": "Boston",
    "New Orleans, Louisiana, U.S.A.": "NewOrleans",
    "PT. Reyes, California, U.S.A.": "PtReyes",
    "Kodiak, Alaska, U.S.A.": "Kodiak",
    "Honolulu, Hawaii, USA": "Honolulu",
    "Athens, Greece": "Athens",
    "Murmansk, Russia": "Murmansk",
}

# Northwood (GYA) as broadcast (weatherfax.com listing): the plugin's copy is
# missing the 09-12 UTC block and some slots.  4610 and 8040 carry the full
# programme, 2618.5 the night part, 11086.5 the day part.
_GYA = """
0000 18Z SURFACE ANALYSIS|0012 18Z SURFACE PROGNOSIS T+24|0024 18Z PRECIP AND VISIBILITY T+12
0036 18Z PRECIP AND VISIBILITY T+24|0048 18Z SURFACE ANALYSIS|0100 18Z SURFACE PROGNOSIS T+24
0112 18Z SWELL T+48|0124 18Z SWELL T+72|0136 OCEAN FRONTS|0148 18Z SWELL T+24|0200 18Z SURFACE ANALYSIS
0212 00Z SEA SURFACE TEMPERATURE|0224 18Z SURFACE PROGNOSIS T+24|0236 00Z SURFACE ANALYSIS
0300 00Z SURFACE ANALYSIS|0324 18Z 10M WIND T+24|0336 00Z ANPS PROGNOSIS T+24|0348 00Z ANPS PROGNOSIS T+72
0400 00Z SURFACE ANALYSIS|0424 18Z 10M WIND T+24|0436 00Z SURFACE PROGNOSIS T+24|0448 18Z 10M WIND T+48
0500 00Z SURFACE ANALYSIS|0512 00Z SURFACE PROGNOSIS T+24|0524 00Z SURFACE PROGNOSIS T+48|0548 06Z GALE WARNING
0612 00Z SURFACE ANALYSIS|0624 00Z SURFACE PROGNOSIS T+24|0636 00Z SURFACE ANALYSIS|0648 00Z SURFACE PROGNOSIS T+24
0700 00Z SURFACE ANALYSIS|0712 00Z SURFACE PROGNOSIS T+24|0724 00Z SURFACE PROGNOSIS T+48
0736 00Z SURFACE PROGNOSIS T+72|0748 00Z SURFACE PROGNOSIS T+96|0800 00Z SURFACE PROGNOSIS T+120
0812 00Z SURFACE PROGNOSIS T+48|0824 00Z PRECIP AND VISIBILITY T+12|0836 00Z PRECIP AND VISIBILITY T+24
0848 00Z SURFACE PROGNOSIS T+24|0900 06Z SURFACE ANALYSIS|0924 00Z ANPS PROGNOSIS T+24|0936 00Z ANPS PROGNOSIS T+120
1000 06Z SURFACE ANALYSIS|1012 06Z SURFACE PROGNOSIS T+24|1036 00Z ANPS PROGNOSIS T+24
1100 06Z SURFACE ANALYSIS|1112 06Z SURFACE PROGNOSIS T+24|1124 06Z SWELL T+24|1136 06Z SURFACE PROGNOSIS T+24
1148 06Z SWELL T+24|1200 06Z SURFACE ANALYSIS|1212 06Z SURFACE PROGNOSIS T+24|1224 06Z PRECIP AND VISIBILITY T+12
1236 06Z PRECIP AND VISIBILITY T+24|1248 06Z SURFACE ANALYSIS|1300 06Z SURFACE PROGNOSIS T+24|1312 06Z 10M WIND T+24
1324 06Z GALE WARNING|1336 OCEAN FRONTS|1348 06Z SWELL T+24|1400 00Z SEA SURFACE TEMPERATURE
1412 06Z SURFACE ANALYSIS|1424 06Z SURFACE PROGNOSIS T+24|1436 12Z SURFACE ANALYSIS|1500 12Z SURFACE ANALYSIS
1512 00Z ANPS PROGNOSIS T+24|1524 00Z ANPS PROGNOSIS T+120|1536 06Z SWELL T+48|1600 12Z SURFACE ANALYSIS
1624 06Z 10M WIND T+24|1636 12Z SURFACE PROGNOSIS T+24|1648 06Z 10M WIND T+48|1700 12Z SURFACE ANALYSIS
1712 12Z SURFACE PROGNOSIS T+24|1724 12Z SURFACE PROGNOSIS T+48|1748 06Z SWELL T+72|1800 12Z SURFACE ANALYSIS
1812 12Z SURFACE PROGNOSIS T+24|1824 12Z SURFACE ANALYSIS|1836 12Z SURFACE PROGNOSIS T+24|1848 12Z SURFACE ANALYSIS
1900 12Z SURFACE PROGNOSIS T+24|1924 12Z SURFACE PROGNOSIS T+48|1936 12Z SURFACE PROGNOSIS T+72
1948 12Z SURFACE PROGNOSIS T+96|2012 12Z SURFACE PROGNOSIS T+120|2024 12Z SURFACE PROGNOSIS T+48
2036 12Z PRECIP AND VISIBILITY T+12|2048 12Z SURFACE PROGNOSIS T+24|2100 18Z SURFACE ANALYSIS
2112 12Z PRECIP AND VISIBILITY T+24|2124 00Z ANPS PROGNOSIS T+24|2136 00Z ANPS PROGNOSIS T+120
2200 18Z SURFACE ANALYSIS|2212 18Z SURFACE PROGNOSIS T+24|2236 00Z ANPS PROGNOSIS T+24|2300 18Z SURFACE ANALYSIS
2312 18Z SURFACE PROGNOSIS T+24|2324 18Z SWELL T+24|2336 18Z SURFACE PROGNOSIS T+24|2348 18Z SWELL T+24
"""


@dataclass
class Slot:
    minute: int          # minutes after 00:00 UTC
    contents: str
    valid: str = ""      # model run hour, e.g. "12"
    duration: int = 0    # minutes, 0 = unknown

    @property
    def hhmm(self) -> str:
        return f"{self.minute // 60:02d}{self.minute % 60:02d}"


@dataclass
class Station:
    name: str
    groups: list          # [(set_of_khz, [Slot, ...]), ...]

    @property
    def short(self) -> str:
        return SHORT_NAMES.get(self.name) or re.sub(r"[^A-Za-z]", "", self.name.split(",")[0].split("/")[0])


def _gya_station() -> Station:
    slots = []
    for item in re.split(r"[|\n]", _GYA):
        item = item.strip()
        if not item:
            continue
        t, text = item[:4], item[5:]
        m = re.match(r"(\d\d)Z (.*)", text)
        valid, contents = (m.group(1), m.group(2)) if m else ("", text)
        slots.append(Slot(int(t[:2]) * 60 + int(t[2:]), contents.title().replace("Anps", "ANPS").replace("10M", "10 m"), valid))
    night = [s for s in slots if s.minute >= 22 * 60 or s.minute < 5 * 60]
    day = [s for s in slots if 6 * 60 + 12 <= s.minute <= 19 * 60 + 48]
    return Station("Northwood, United Kingdom",
                   [({4610.0, 8040.0}, slots), ({2618.5}, night), ({11086.5}, day)])


def load(path: Path | None = None) -> list[Station]:
    """Parse the plugin XML (tolerant of its malformed comments)."""
    candidates = [p for p in (path, Path.cwd() / "WeatherFaxSchedules.xml", DATA) if p]
    src = next((p for p in candidates if p.exists()), None)
    stations: list[Station] = []
    if src is not None:
        text = src.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
        root = ET.fromstring(text.encode("utf-8"))
        for st in root.iter("Station"):
            groups = []
            for fr in st.findall("Frequency"):
                khz = {float(k) for k in re.findall(r"[\d.]+", fr.get("khz", ""))}
                slots = []
                for mp in fr.findall("Map"):
                    t = (mp.get("Time") or "").strip()
                    if not re.fullmatch(r"\d{4}", t):
                        continue
                    try:
                        dur = int(float(mp.get("Duration") or 0))
                    except ValueError:
                        dur = 0
                    slots.append(Slot(int(t[:2]) * 60 + int(t[2:]),
                                      (mp.get("Contents") or "").strip(), (mp.get("Valid") or "").strip(), dur))
                if khz and slots:
                    groups.append((khz, slots))
            if groups:
                stations.append(Station(st.get("Name", "?"), groups))
    stations = [s for s in stations if s.name != "Northwood, United Kingdom"]
    stations.append(_gya_station())
    return stations


class Schedules:
    def __init__(self, path: Path | None = None):
        self.stations = load(path)

    def station_by_name(self, text: str) -> Station | None:
        t = text.lower()
        for s in self.stations:
            if t in s.name.lower() or t == s.short.lower():
                return s
        return None

    def station_by_dial(self, dial_khz: float, tol: float = 1.2):
        """(station, slots, assigned_khz) for a USB dial frequency, or None.
        Accepts either the dial (assigned - 1.9 kHz) or the assigned frequency."""
        best = None
        for s in self.stations:
            for khz_set, slots in s.groups:
                for k in khz_set:
                    for d in (abs(dial_khz - (k - DIAL_OFFSET_KHZ)), abs(dial_khz - k)):
                        if d <= tol and (best is None or d < best[0]):
                            best = (d, s, slots, k)
        return None if best is None else best[1:]

    @staticmethod
    def match_slot(slots, when: datetime, early_s: int = 120, late_s: int = 300):
        """Slot whose start is closest to `when` (start tone / phasing time)."""
        t = when.hour * 60 + when.minute + when.second / 60.0
        best = None
        for sl in slots:
            d = (t - sl.minute + 720) % 1440 - 720          # minutes, wraps midnight
            if -early_s / 60 <= d <= late_s / 60 and (best is None or abs(d) < abs(best[0])):
                best = (d, sl)
        return None if best is None else best[1]

    def identify(self, when: datetime, dial_khz: float | None = None, station: str | None = None):
        """Return dict(station, short, slot, contents, valid, khz) or None."""
        st = slots = khz = None
        if dial_khz is not None:
            hit = self.station_by_dial(dial_khz)
            if hit:
                st, slots, khz = hit
        if st is None and station:
            st = self.station_by_name(station)
            if st:
                slots = [s for _, g in st.groups for s in g]
        if st is None:
            return None
        sl = self.match_slot(slots, when)
        out = {"station": st.name, "short": st.short, "khz": khz}
        if sl:
            out.update(slot=sl.hhmm, contents=sl.contents, valid=sl.valid)
        return out

    def next_slots(self, now: datetime, dial_khz: float | None = None, station: str | None = None, n: int = 3):
        st = slots = None
        if dial_khz is not None:
            hit = self.station_by_dial(dial_khz)
            if hit:
                st, slots, _ = hit
        if st is None and station:
            st = self.station_by_name(station)
            slots = [s for _, g in st.groups for s in g] if st else None
        if not slots:
            return st, []
        t = now.hour * 60 + now.minute
        uniq = {s.minute: s for s in slots}
        order = sorted(uniq.values(), key=lambda s: (s.minute - t) % 1440)
        return st, order[:n]


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^A-Za-z0-9+]+", "-", text).strip("-")[:n]


if __name__ == "__main__":
    sc = Schedules()
    print(len(sc.stations), "stations")
    now = datetime.now(timezone.utc)
    for dial in (8038.1, 7878.1, 3853.1, 4608.1):
        st, nxt = sc.next_slots(now, dial)
        print(dial, st.short if st else None, [(s.hhmm, s.contents) for s in nxt])
