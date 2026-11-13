"""Monthly check of the NWS worldwide radiofax schedule list.

NOAA/NWS publishes "Worldwide Marine Radiofacsimile Broadcast Schedules"
(rfax.pdf), with a section per station.  Once a month (at the first quiet
moment) this downloads it and compares every station section with the copy
seen last time - NWS against NWS, so the quirks of reading a PDF table cancel
out and only what NWS actually changed shows up:

* a broadcast time that appears in a station's section is added to that
  station's list (named from the text next to it in the table);
* a broadcast time that has gone from the section altogether is taken off;
* a section that now says the station has stopped (discontinued, stopped
  broadcasting, ...) takes the station off the list.  A station that has
  vanished from the document is only reported (its heading may simply be
  spelt differently now).

Guard rails: a section where more than half the times changed is not applied
(a rewritten table is more likely a new layout than a new schedule) - it is
reported for a person to check instead.  Stations never come back by
themselves, and stations removed on purpose (EXCLUDED) are never touched.
Times that are added but never sent are dropped by the scheduler's
empty-slot learning anyway, so a misread time costs a few listening attempts,
not a wrong chart.

Changes are kept in <out>/schedule/nws_updates.json (applied on top of
WeatherFaxSchedules.xml and the built-in Northwood list when the schedules
load) and every check is summarised in <out>/schedule/nws_report.txt and the
log.  wefax/data/nws_baseline.json is the NWS list the bundled schedules were
last checked against (revision of 7 March 2025); a new installation starts
from it.

The PDF is read with the built-in reader in minipdf.py (standard library
only), so the check needs nothing extra installed.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://www.weather.gov/media/marine/rfax.pdf"
CHECK_DAYS = 30
RETRY_HOURS = 24                 # after a failed download
MAX_CHANGE = 0.5                 # more than this share of a section's times changed -> report only
BASELINE = Path(__file__).with_name("data") / "nws_baseline.json"

# our station name -> the NWS section heading (letters only, upper case).
# Charleville and Wiluna share one table.
SECTIONS = {
    "Tokyo, Japan": "TOKYOJAPAN",
    "Seoul, Republic Of Korea": "SEOULREPUBLICOFKOREA",
    "Bangkok, Thailand": "BANGKOKTHAILAND",
    "Guanzou Coast Radio Station XSQ, China": "GUANGZHOUCOASTRADIOSTATION",
    "Valparaiso, Playa Ancha, Chile": "VALPARAISOPLAYAANCHACHILE/CBV",
    "Iqaluit, N.W.T., Canada": "IQALUITCANADA",
    "Resolute, N.W.T., Canada": "RESOLUTECANADA",
    "Sydney, Nova Scotia, Canada": "SYDNEYNOVASCOTIACANADA",
    "Inuvik, Canada": "INUVIKCANADA",
    "Kodiak, Alaska, U.S.A.": "KODIAKALASKAUSA",
    "PT. Reyes, California, U.S.A.": "PTREYESCALIFORNIAUSA",
    "New Orleans, Louisiana, U.S.A.": "NEWORLEANSLOUISIANAUSA",
    "Boston, Massachusetts, U.S.A.": "BOSTONMASSACHUSETTSUSA",
    "Charleville, Australia": "CHARLEVILLEAUSTRALIA",
    "Wiluna, Australia": "CHARLEVILLEAUSTRALIA/VMW",
    "Wellington, New Zealand": "WELLINGTONNEWZEALAND",
    "Honolulu, Hawaii, USA": "HONOLULUHAWAIIUSA",
    "Athens, Greece": "ATHENSGREECE",
    "Murmansk, Russia": "MURMANSKRUSSIA",
    "Hamburg/Pinneburg, Germany": "HAMBURGPINNEBERGGERMANY",
    "Northwood, United Kingdom": "NORTHWOODUNITEDKINGDOM",
}
# parts of a shared section: (section, where this station's part ends)
PARTS = {
    "VALPARAISOPLAYAANCHACHILE/CBV": ("VALPARAISOPLAYAANCHACHILE", r"TRANSMISSION\s*\(CBM\)"),   # Punta Arenas follows
    "CHARLEVILLEAUSTRALIA/VMW": ("CHARLEVILLEAUSTRALIA", r"repeat broadcasts on 11030"),          # Charleville only
}
# listed, but not weather charts: never added to a station's list
NOT_A_CHART = re.compile(r"TEST ?(?:CHART|PATTERN|KARTE)|SCHEDULE|NEWS|FISH|TUNA|SAURY|SUMO|NOTIFICATION|"
                         r"NAVIGATIONAL WARNING|NOAA INFORMATION|MARITIME SECURITY|PRODUCT NOTICE|SYMBOLS|"
                         r"REQUEST FOR COMMENTS|RADIO PREDICTION|RECOMMENDED FREQUENCIES|INFORMATION NOTICE|"
                         r"VOICE BROADCAST", re.I)
# every section heading in the document, in order, so a section ends where the next begins
HEADINGS = ["TOKYOJAPAN", "KAGOSHIMA", "JFXKAGOSHIMA", "PEVEKCHUKOTKA", "SEOULREPUBLICOFKOREA", "BANGKOKTHAILAND",
            "KYODONEWS", "JJCKYODONEWS", "GUANGZHOUCOASTRADIOSTATION", "VALPARAISOPLAYAANCHACHILE", "IQALUITCANADA",
            "RESOLUTECANADA", "INUVIKCANADA", "SYDNEYNOVASCOTIACANADA", "KODIAKALASKAUSA",
            "PTREYESCALIFORNIAUSA", "POINTREYESCALIFORNIAUSA", "NEWORLEANSLOUISIANAUSA",
            "BOSTONMASSACHUSETTSUSA", "CHARLEVILLEAUSTRALIA", "WELLINGTONNEWZEALAND", "HONOLULUHAWAIIUSA",
            "ATHENSGREECE", "MURMANSKRUSSIA", "HAMBURGPINNEBERGGERMANY", "NORTHWOODUNITEDKINGDOM",
            "APPENDIXA"]
# taken off on purpose (no longer transmitting): never re-added or changed
EXCLUDED = {"Rio De Janeiro, Brazil", "Athens, Greece", "Iqaluit, N.W.T., Canada",
            "Resolute, N.W.T., Canada", "Inuvik, Canada", "Wellington, New Zealand"}
OFF_AIR = re.compile(r"discontinued|stopped broadcasting|ceased|no longer (?:broadcast|transmit)|"
                     r"retirement of radiofax|terminated", re.I)

# a pair, allowing the stray glyphs some tables carry between the halves ("------  g () /1315")
_PAIR = re.compile(r"(?<![\d/])(\d{4}|-{2,8})\s*(?:[a-z()-]{1,2}\s+){0,3}/\s*(?:[a-z()-]{1,2}\s+){0,3}(\d{4}|-{2,8})(?![\d/])")
_HHMM = re.compile(r"(?<![\d./:-])([01]\d|2[0-3])[.:]?([0-5]\d)Z?(?![\d./:-])")
_DATED = re.compile(r"(?:information|info)\s+(?:dated|updated)[:\s]*\(?\s*([^)\n]{3,30})", re.I)


def _letters(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def pdf_text(data: bytes) -> str:
    """The document's text, line by line as it looks on the page."""
    from .minipdf import text
    return text(data)


def _ok_hhmm(h: str, m: str) -> bool:
    return 0 <= int(h) <= 23 and 0 <= int(m) <= 59


def times_in(section: str) -> dict:
    """Broadcast times in a station section -> the text that follows each.

    Recognised: '0230/1400' pairs (either half may be '----'), and a single
    time ('0845', '04.30', '0845Z') that starts a row or stands between
    product names in a two-column table.  Valid times after the RPM/IOC
    column ('120/576 0000'), frequency-table time ranges ('2200Z-0500Z') and
    frequencies are not broadcast times."""
    out: dict[str, list] = {}
    rows = section.splitlines()
    for n, line in enumerate(rows):
        if re.search(r"\bkHz\b", line, re.I):
            continue                                    # frequency table
        if re.search(r"\b(?:DATED|UPDATED?|EFFECTIVE)\b", line, re.I):
            continue                                    # "information dated Feb. 12, 2020"
        if not re.sub(r"(?:\d{4}|\d\d\.\d\d|-{2,8}|[/\s])", "", line):
            continue                                    # only times: an unused slot in the table
        used = []
        for m in _PAIR.finditer(line):
            used.append(m.span())
            if re.search(r"(?:\bOF|F'CAST|RETRANSMISSION|REBROADCAST)\s*$", line[:m.start()], re.I):
                continue                                # 'rebroadcast of 0325/1525': a reference
            for k, part in enumerate(m.groups()):
                if part[0].isdigit() and _ok_hhmm(part[:2], part[2:]):
                    out.setdefault(part, [_after(line, m.end()), _valid(line, m.end(), k)])
        for m in _HHMM.finditer(line):
            if any(a <= m.start() < b for a, b in used):
                continue
            before, after = line[:m.start()], line[m.end():]
            if re.search(r"\d{2,3}\s*/\s*\d{3}\s*$", before):   # valid time after RPM/IOC
                continue
            if re.match(r"\s*[-–]\s*\d", after):            # a time range
                continue
            if re.search(r"(?:\bOF|VT|VALID|\bAT|UPDATED?|DATED|\bBY)\s*$", before, re.I):
                continue
            starts_row = not before.strip() and not re.search(
                r"(?:[A-Za-z)]|T\+\d+)\s{2,}(?:[01]\d|2[0-3])[0-5]\d\s*$", after)   # empty left cell
            # two-column tables (Northwood): 'product  0012  product  1212'
            between = re.search(r"(?:[A-Za-z)]|T\+\d+)\s{2,}$", before) and \
                (not after.strip() or re.match(r"\s{2,}\S", after))
            if between:
                # the product is the cell before the time, after any earlier time
                text = _clean(re.split(r"\s(?:[01]\d|2[0-3])[0-5]\d\s", " " + before + " ")[-1])
            elif starts_row:
                text = _after(line, m.end())
                if not re.search(r"[A-Za-z]{3}", text) and n + 1 < len(rows) \
                        and not re.match(r"\s*(?:\d|-{2})", rows[n + 1]):
                    text = _clean(rows[n + 1])          # the product name wrapped onto the next line
            else:
                continue
            out.setdefault(m.group(1) + m.group(2), [text, _valid(line, m.end(), 0) or _valid_from(text)])
    return out


def _clean(s: str) -> str:
    s = re.sub(r"\s*(?<=\w)\s*-\s+(?=\w)|(?<=\w)\s+-\s*(?=\w)", "-", re.sub(r"\s+", " ", s))
    return s.strip(" -")[:70]


def _after(line: str, i: int) -> str:
    s = line[i:]
    # 'HH.MM  120 / 576  19  00.00  contents' (DWD): the columns come first
    s = re.sub(r"^\s*\d{2,3}\s*/\s*\d{3}(?:\s+(?:\d{1,2}|\d\d[.:]\d\d(?:\s*VT)?|-+))*\s+", "", s)
    s = re.split(r"\s\d{2,3}\s*/\s*\d{3}\b|\s(?:\d{4}|-{2,8})/(?:\d{4}|-{2,8})\b|\s(?:[01]\d|2[0-3])[0-5]\d\s+\d\dZ\s", s)[0]
    return _clean(s)


def parse(text: str) -> dict:
    """{'revision': 'March 7, 2025', 'sections': {heading: {...}}} from the document text."""
    lines = text.splitlines()
    rev = ""
    for l in lines[:40]:
        l = re.sub(r"\s+", " ", re.sub(r"(?<=\d)\s+(?=\d)", "", l)).strip()
        if re.fullmatch(r"[A-Z][a-z]+\.? \d{1,2},? \d{4}", l):
            rev = l
            break
    # the contents page lists every heading once; real sections come after it
    start = 0
    for i, l in enumerate(lines[:400]):
        if _letters(l).startswith("ABOUTTHISPUBLICATION"):
            start = i              # the last one: the contents page lists it too
    found = []                      # (line index, heading)
    for i in range(start, len(lines)):
        if not re.match(r"\s*[A-Za-z]", lines[i]) or "---" in lines[i]:
            continue               # a table row or a contents line, not a heading
        key = _letters(lines[i])
        for h in HEADINGS:
            if (key.startswith(h) or (h in key and len(key) <= len(h) + 12)) and \
                    (h != "APPENDIXA" or len(key) < 12):
                if not any(fh == h for _, fh in found):
                    found.append((i, h))
                break
    found.sort()
    sections = {}
    for n, (i, h) in enumerate(found):
        if h == "APPENDIXA":
            continue
        j = found[n + 1][0] if n + 1 < len(found) else len(lines)
        body = "\n".join(lines[i:j])
        dated = _DATED.search(body)
        sections[h] = {"dated": re.sub(r"\s+([,)])", r"\1", re.sub(r"\s+", " ", dated.group(1))).strip(" -")
                       if dated else "",
                       "off_air": bool(OFF_AIR.search(body)),
                       "off_air_text": (OFF_AIR.search(body).group(0) if OFF_AIR.search(body) else ""),
                       "times": times_in(body), "_body": body}
    for key, (h, stop) in PARTS.items():
        if h in sections:
            body = re.split(stop, sections[h]["_body"], maxsplit=1, flags=re.I)[0]
            sections[key] = dict(sections[h], times=times_in(body))
    for sec in sections.values():
        sec.pop("_body", None)
    return {"revision": rev, "sections": sections}


def _valid_from(text: str) -> str:
    """Model run from a product name like '18Z SFC ANALYSIS' (Northwood)."""
    m = re.match(r"\s*(\d\d)Z\b", text)
    return m.group(1) if m else ""


_VALID = re.compile(r"\d{2,3}\s*/\s*\d{3}\s+(?:\d{1,2}\s+)?(?:(\d\d)\s*/\s*(\d\d)(?![\d.])|(\d\d)[.:]?\d\d(?!\s*/))")


def _valid(line: str, i: int, half: int) -> str:
    """The valid-time column of a row: '120/576 1200' -> '12', '120/576 00/12'
    -> '00' for the first broadcast time and '12' for the second, DWD's
    '120 / 576 11 12.00' -> '12'."""
    m = _VALID.search(line, i)
    if not m:
        return ""
    if m.group(3):
        return m.group(3)
    return m.group(1) if half == 0 else m.group(2)


def _entry(v) -> tuple:
    return (v, "") if isinstance(v, str) else (v[0], v[1])


def compare(old: dict, new: dict) -> dict:
    """What NWS changed between two parses, per our station name."""
    res = {}
    for name, h in SECTIONS.items():
        if name in EXCLUDED:
            continue
        o, n = old["sections"].get(h), new["sections"].get(h)
        if o is None:
            continue                          # never seen: nothing to compare with
        if n is None:
            # could be a station that has gone, or a heading this reader no
            # longer recognises: a person decides
            res[name] = {"missing": True}
            continue
        if n["off_air"] and not o["off_air"]:
            res[name] = {"off_air": f"NWS list now says: {n['off_air_text']}"}
            continue
        ot, nt = set(o["times"]), set(n["times"])
        add, rem = sorted(nt - ot), sorted(ot - nt)
        if not add and not rem and o.get("dated") == n.get("dated"):
            continue
        share = (len(add) + len(rem)) / max(1, len(ot | nt))
        res[name] = {"add": [[t] + list(_entry(n["times"][t])) for t in add], "remove": rem,
                     "dated": [o.get("dated", ""), n.get("dated", "")],
                     "review_only": share > MAX_CHANGE}
    return res


# ----------------------------------------------------------------------
# applying the stored updates to the loaded schedules

def updates_path(out: Path) -> Path:
    return Path(out) / "schedule" / "nws_updates.json"


def load_updates(out: Path | None) -> dict:
    if out is None:
        return {}
    try:
        return json.loads(updates_path(out).read_text(encoding="utf-8")).get("stations", {})
    except Exception:
        return {}


def apply_updates(stations: list, updates: dict, slot_cls) -> list:
    """Stations with the stored NWS changes applied (off-air ones dropped)."""
    keep = []
    for st in stations:
        u = updates.get(st.name)
        if not u:
            keep.append(st)
            continue
        if u.get("off_air"):
            continue
        rem = set(u.get("remove", []))
        adds = u.get("add", [])
        groups = []
        for khz, slots in st.groups:
            s2 = [s for s in slots if s.hhmm not in rem]
            have = {s.hhmm for s in s2}
            for t, contents, valid in adds:
                if t not in have:
                    s2.append(slot_cls(int(t[:2]) * 60 + int(t[2:]), contents or "Chart (new in the NWS list)", valid))
            s2.sort(key=lambda s: s.minute)
            groups.append((khz, s2))
        st.groups = groups
        keep.append(st)
    return keep


def record(out: Path, changes: dict, new_rev: str) -> list[str]:
    """Merge the applied changes into nws_updates.json; return report lines."""
    p = updates_path(out)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {"stations": {}, "history": []}
    lines = []
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for name, c in sorted(changes.items()):
        if c.get("missing"):
            lines.append(f"{name}: no longer found in the NWS list - NOT taken off; please check whether "
                         f"it still transmits (remove it from WeatherFaxSchedules.xml if not)")
            continue
        st = data["stations"].setdefault(name, {"add": [], "remove": []})
        if c.get("off_air"):
            st["off_air"] = c["off_air"]
            lines.append(f"{name}: taken off the list - {c['off_air']}")
            continue
        dated = c.get("dated") or ["", ""]
        tag = f" (section dated {dated[1]}, was {dated[0]})" if dated[0] != dated[1] else ""
        if c.get("review_only"):
            lines.append(f"{name}: section rewritten{tag} - {len(c['add'])} times added, "
                         f"{len(c['remove'])} gone; NOT applied, please check the station by hand")
            continue
        if not c["add"] and not c["remove"]:
            lines.append(f"{name}: section date changed{tag}, broadcast times unchanged")
            continue
        skipped = [a[0] for a in c["add"] if NOT_A_CHART.search(a[1])]
        c["add"] = [a for a in c["add"] if a[0] not in skipped]
        if skipped:
            lines.append(f"{name}: new in the list but not weather charts, not added: {', '.join(skipped)}")
        for t, txt, valid in c["add"]:
            st["remove"] = [r for r in st["remove"] if r != t]
            if not any(a[0] == t for a in st["add"]):
                st["add"].append([t, _tidy(txt), valid or _valid_from(txt)])
        for t in c["remove"]:
            st["add"] = [a for a in st["add"] if a[0] != t]
            if t not in st["remove"]:
                st["remove"].append(t)
        lines.append(f"{name}{tag}: " + "; ".join(
            ([f"added {', '.join(a[0] for a in c['add'])}"] if c["add"] else []) +
            ([f"removed {', '.join(c['remove'])}"] if c["remove"] else [])))
    data["history"] = (data.get("history", []) + [{"date": when, "nws_revision": new_rev, "changes": lines}])[-36:]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return lines


def _tidy(txt: str) -> str:
    t = re.sub(r"^\s*\d\dZ\s+", "", txt).strip()
    return t.title() if t.isupper() else t


# ----------------------------------------------------------------------
# the monthly job

def _state_path(out: Path) -> Path:
    return Path(out) / "schedule" / "nws_state.json"


def check_now(out: Path, log=print, fetch=None) -> list[str] | None:
    """Download, compare with the last copy, record changes.  Returns the
    report lines ([] = no change) or None when the check could not run."""
    out = Path(out)
    try:
        if fetch is None:
            req = urllib.request.Request(URL, headers={"User-Agent": "wefax-decoder schedule check"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
        else:
            data = fetch()
        new = parse(pdf_text(data))
    except Exception as e:
        log(f"schedule check: could not read the NWS schedule list ({e.__class__.__name__}: {e})")
        return None
    if len(new["sections"]) < 10:
        log(f"schedule check: the NWS document only showed {len(new['sections'])} station sections - "
            f"its layout may have changed; nothing applied")
        return None
    sp = _state_path(out)
    try:
        old = json.loads(sp.read_text(encoding="utf-8"))["nws"]
    except Exception:
        old = json.loads(BASELINE.read_text(encoding="utf-8"))
    changes = compare(old, new)
    lines = record(out, changes, new["revision"]) if changes else []
    st = {"checked": time.time(), "nws": new}
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    head = (f"schedule check {datetime.now(timezone.utc):%Y-%m-%d %H:%MZ}: NWS list of {new['revision']} "
            f"(last compared: {old.get('revision', '?')})")
    rep = [head] + (["  " + l for l in lines] if lines else ["  no changes to the stations in this decoder"])
    with open(out / "schedule" / "nws_report.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(rep) + "\n\n")
    for l in rep:
        log(l)
    return lines


def due(out: Path) -> bool:
    try:
        st = json.loads(_state_path(out).read_text(encoding="utf-8"))
        return time.time() - float(st.get("checked", 0)) >= CHECK_DAYS * 86400
    except Exception:
        return True


def start(out: Path, sched, is_quiet, log=print):
    """Background thread: when a check is due and the decoder is quiet, run
    it; if anything changed, reload the schedules in place."""
    def run():
        time.sleep(300)                      # let start-up settle
        while True:
            wait = 3600
            try:
                if due(out) and is_quiet():
                    lines = check_now(out, log)
                    if lines is None:
                        wait = RETRY_HOURS * 3600
                    elif lines:
                        sched.reload()
                        log(f"schedules reloaded with the NWS changes ({len(sched.stations)} stations)")
            except Exception as e:
                log(f"schedule check error: {e}")
            time.sleep(wait)
    threading.Thread(target=run, daemon=True, name="nws-check").start()
