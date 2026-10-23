#!/usr/bin/env python3
"""HF Weather Fax (WEFAX) decoder with automatic AFC, slant correction,
noise removal and line reconstruction.  Decoded faxes are saved to disk and
shown in a local browser page.

Live (radio in USB, dial 1.9 kHz below the assigned frequency):
    python wefax_decoder.py --list-devices
    python wefax_decoder.py --device "Line 8 (Virtual Audio Cable) MME" --channel 0

Decode a recording:
    python wefax_decoder.py --wav recording.wav
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from wefax import dsp, decode, enhance, server
from wefax.schedule import Schedules, slug
from wefax import label
from wefax import housekeep
from wefax.lengths import ChartLengths, CLEAN_ENDS
from wefax.frontend import Detector, LiveCapture

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is required:  pip install opencv-python-headless")

log = logging.getLogger("wefax")
PICTURE_LATE_MAX_S = 15 * 60   # picture-check recording after its chart's time: closed within this
NO_SIGNAL_SCORE = 6.0     # AFC comb score: pure noise ~2, weakest real faxes ~19


def parse_utc(text):
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def identify(sched, cat, station, utc_text, auto=None, quiet=False, end=None, mid=False):
    """Which chart is this?

    The station: the auto-scheduler's plan says what it tuned (unless CAT
    clearly shows the radio somewhere else that has a chart then), else the
    SDR Console frequency (CAT) at the start time, else the station picker.
    The chart: the scheduled slot of that station nearest the start of
    reception - never just "the planned item that was still open", because
    a station sends charts the plan skipped (already collected, low
    importance) straight after the planned one.

    The frequency is the one held for most of the recording (start to `end`,
    default now): a recording that begins a second before a retune is not
    named after the station the receiver was just leaving.
    """
    if sched is None or not utc_text:
        return None
    when = parse_utc(utc_text)
    if when is None:
        return None
    dial = cat.dial_khz_during(when, end) if cat else None
    p = sched.identify(when, dial, station, mid)
    it = auto.item_at(when, dial) if auto is not None else None
    if it is not None and (dial is None or abs(dial - it.dial) <= 1.5 or not (p and p.get("slot"))):
        if dial is not None and abs(dial - it.dial) > 1.5 and not quiet:
            log.info("chart: CAT read %.1f kHz but the scheduler had tuned %s %.1f kHz - using the plan",
                     dial, it.short, it.dial)
        # on the station and frequency the scheduler tuned: the slot by time
        q = sched.identify(when, it.dial, it.station, mid)
        if q and q.get("slot"):
            q["khz"] = q.get("khz") or it.khz
            return q
        return {"station": it.station, "short": it.short, "khz": it.khz, "slot": it.slot,
                "contents": it.contents, "valid": it.valid}
    return p


def expected_now(sched, cat, station, auto, now=None):
    """The chart that should be on air on the tuned frequency right now, or None.

    Auto-scheduler: its current item, from tuning (75 s before) until it is
    closed.  Otherwise: a slot of the tuned station (CAT frequency, else the
    station picker) whose transmission window covers now."""
    if auto is not None and auto.enabled and auto.current is not None and auto.current.state == "tuned":
        it = auto.current
        return {"short": it.short, "slot": it.slot, "contents": it.contents, "khz": it.khz}
    if sched is None:
        return None
    now = now or datetime.now(timezone.utc)
    slots = st = None
    dial = cat.freq_hz / 1000.0 if (cat is not None and cat.freq_hz) else None
    if dial is not None:
        hit = sched.station_by_dial(dial)
        if hit:
            st, slots, _ = hit
    if st is None and station:
        st = sched.station_by_name(station)
        slots = [s for _, g in st.groups for s in g] if st else None
    if not slots:
        return None
    t = now.hour * 60 + now.minute + now.second / 60.0
    ordered = sorted({s.minute: s for s in slots}.values(), key=lambda s: s.minute)
    best = None
    for sl in ordered:
        d = (t - sl.minute + 720) % 1440 - 720       # minutes since the slot start
        if -1.5 <= d <= (sl.duration or 12) and (best is None or d < best[0]):
            best = (d, sl)                           # the most recently started one
    if best is None:
        return None
    sl = best[1]
    return {"short": st.short, "slot": sl.hhmm, "contents": sl.contents}


def describe(p) -> str:
    if not p:
        return ""
    s = p["short"]
    if p.get("slot"):
        s += f" {p['slot']}Z  {p['contents']}"
        if p.get("valid"):
            s += f" ({p['valid']}Z run)"
    return s


def save_image(path: Path, img, params=None):
    """Write an image atomically: anything watching the folder only ever sees
    complete files (written under a temporary name, then renamed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part" + path.suffix)
    if not cv2.imwrite(str(tmp), img, params or []):
        raise IOError(f"could not write {tmp}")
    os.replace(tmp, path)


def run_key(product: dict, utc: str) -> str | None:
    """Same value for a chart and its repeats (the chart, its model run and
    the run's date), so a repeat can be told from a newer issue."""
    try:
        when = datetime.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
        slot = when.replace(hour=int(product["slot"][:2]), minute=int(product["slot"][2:]))
        if slot > when + timedelta(hours=1):
            slot -= timedelta(days=1)
        valid = str(product.get("valid") or "")
        run_day = (slot - timedelta(hours=(slot.hour - int(valid)) % 24)).date() if valid.isdigit() else slot.date()
        c = re.sub(r"\s+", " ", str(product["contents"]).lower()).strip()
        return f"{product['short']}|{c}|{valid}|{run_day}"
    except Exception:
        return None


def station_paths(out: Path, product):
    """stations/<station>/<chart>.png and .json for a chart."""
    folder = out / "stations" / re.sub(r"[^A-Za-z0-9_.-]", "_", product["short"])
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", slug(product["contents"]))
    return folder / f"{name}.png", folder / f"{name}.json"


def keep_station_copy(out: Path, product, img_u8, quality: float, utc: str, source_id: str):
    """stations/<station>/<chart>.png: always the latest copy of each chart.
    A newer issue replaces it; a repeat of the same issue only replaces it
    if it decoded better."""
    if not product or not product.get("slot") or not product.get("contents"):
        return
    png, info = station_paths(out, product)
    key = run_key(product, utc)
    try:
        old = json.loads(info.read_text()) if info.exists() else None
    except Exception:
        old = None
    if old and key and old.get("run") == key and float(old.get("quality", 0)) >= quality:
        log.info("station copy: kept the earlier %s (quality %.2f >= %.2f)",
                 png.relative_to(out), float(old.get("quality", 0)), quality)
        return
    save_image(png, img_u8)
    rec = {"run": key, "quality": round(quality, 3), "utc": utc, "fax": source_id,
           "chart": product.get("contents"), "valid": product.get("valid"), "verified": True}
    tmp = info.with_name(info.name + ".part")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, info)
    why = "better repeat" if old and old.get("run") == key else "latest issue"
    log.info("station copy: %s (%s, quality %.2f)", png.relative_to(out), why, quality)


def organise_existing(out: Path):
    """Move faxes saved by earlier versions (all in one folder) into
    raw/, enhanced/ and thumbs/ and update their index files."""
    moved = 0
    for j in sorted(out.glob("*.json")):
        try:
            rec = json.loads(j.read_text())
        except Exception:
            continue
        files = rec.get("files", {})
        changed = False
        for key, sub in (("raw", "raw"), ("enhanced", "enhanced"), ("thumb", "thumbs")):
            name = files.get(key)
            if not name or "/" in name:
                continue
            src = out / name
            stem = rec["id"]
            dst_rel = f"{sub}/{stem}{src.suffix}"
            if src.exists():
                (out / sub).mkdir(exist_ok=True)
                os.replace(src, out / dst_rel)
                old = src.with_name(src.stem + "_old" + src.suffix)     # from reenhance.py
                if old.exists():
                    os.replace(old, out / sub / f"{stem}_old{src.suffix}")
            files[key] = dst_rel
            changed = True
        if changed:
            rec["files"] = files
            j.write_text(json.dumps(rec, indent=1, default=float))
            moved += 1
    if moved:
        log.info("moved %d earlier fax(es) into raw/, enhanced/ and thumbs/", moved)


class Processor:
    """Decodes finished recordings on a worker thread and saves the results."""

    def __init__(self, out_dir: Path, hub: server.Hub, args, sched=None, cat=None):
        self.out = out_dir
        self.hub = hub
        self.args = args
        self.sched = sched
        self.cat = cat
        self.q: queue.Queue = queue.Queue()
        self.busy = 0
        self.done = 0
        self.learnt = self._recent_clocks()
        threading.Thread(target=self._run, daemon=True).start()

    CLOCK_KEEP = 30

    def _recent_clocks(self) -> dict:
        """Clock error (ppm) and tuning offset (Hz) measured on this
        installation's own charts decoded with phasing, per station (the
        transmitters' clocks differ: DWD about +5 ppm, Northwood about -33
        ppm here) and for all stations together ("*").  Learnt, never
        assumed; both barely change, so they anchor the decode."""
        recs = []
        for j in self.out.glob("*.json"):
            try:
                r = json.loads(j.read_text())
                m = r.get("meta") or {}
                if m.get("phasing") and m.get("clock_ppm") is not None and abs(m["clock_ppm"]) < 1000:
                    recs.append((j.stat().st_mtime, (r.get("product") or {}).get("short"),
                                 float(m["clock_ppm"]), float(m.get("afc_hz") or 0.0)))
            except Exception:
                continue
        learnt: dict = {}
        for _, short, c, a in sorted(recs):
            for k in ("*", short):
                if k:
                    d = learnt.setdefault(k, {"clock": [], "afc": []})
                    d["clock"] = (d["clock"] + [c])[-self.CLOCK_KEEP:]
                    d["afc"] = (d["afc"] + [a])[-self.CLOCK_KEEP:]
        return learnt

    def _learn(self, short, m):
        """Learn only from what a chart measured itself: its period from
        accepted phasing (never one the learnt clock imposed).  If the
        phasing keeps disagreeing with the learnt clock but agrees with
        itself, the clock really has changed (new sound card, other cable):
        adopt it."""
        if not m.get("phasing") or m.get("clock_ppm") is None or abs(m["clock_ppm"]) >= 1000:
            return
        if m.get("period_from_phasing", True):
            for k in ("*", short):
                if k:
                    d = self.learnt.setdefault(k, {"clock": [], "afc": []})
                    d["clock"] = (d["clock"] + [float(m["clock_ppm"])])[-self.CLOCK_KEEP:]
                    d["afc"] = (d["afc"] + [float(m.get("afc_hz") or 0.0)])[-self.CLOCK_KEEP:]
                    d["rejected"] = []
            return
        if m.get("phasing_ppm") is None:
            return
        key = short if short in self.learnt else "*"
        d = self.learnt.setdefault(key, {"clock": [], "afc": []})
        rj = d["rejected"] = (d.get("rejected", []) + [float(m["phasing_ppm"])])[-5:]
        if len(rj) == 5:
            med = sorted(rj)[2]
            if max(abs(r - med) for r in rj) <= 30:
                log.info("clock: the last 5 %s charts' phasing all say %+.0f ppm, not the learnt %+.0f - "
                         "the clock has changed, learning the new value", key, med, self._hint("clock", short) or 0)
                d["clock"] = rj[:]
                d["rejected"] = []

    def _hint(self, what, short=None):
        """Median for this station (5+ charts), else for all stations, else None."""
        for k in (short, "*"):
            v = (self.learnt.get(k) or {}).get(what, []) if k else []
            if len(v) >= 5:
                v = sorted(v)
                return v[len(v) // 2]
        return None

    def _reference(self, product):
        """The raw picture of the last good copy of this chart (its station
        copy's source), used to check where the frame should be; None if
        there is none yet or it was not a verified good copy."""
        if not product or not product.get("slot") or not product.get("contents"):
            return None
        try:
            png, info = station_paths(self.out, product)
            st = json.loads(info.read_text())
            if st.get("verified") is not True:
                return None     # only a copy that passed every check (older copies may be misframed)
            rec = json.loads((self.out / f"{st['fax']}.json").read_text())
            if (rec.get("meta") or {}).get("align") == "uncertain":
                return None
            img = cv2.imread(str(self.out / rec["files"]["raw"]), cv2.IMREAD_GRAYSCALE)
            return None if img is None else img.astype(np.float32) / 255.0
        except Exception:
            return None

    PUBLISH_MIN_QUALITY = 0.5
    CHECK_PPM = 15.0      # line period this close to the station's learnt clock (measured spread
                          # is +/-2 ppm; 15 ppm is about 25 px of shear over a whole chart)
    CHECK_HZ = 80.0       # tuning offset this close to its learnt offset

    def _publish_check(self, m: dict, short) -> list:
        """Reasons not to publish this chart ([] = publish).  Every check is
        independent of how the decode reached its answer."""
        why = []
        if not m.get("align_ok", True):
            why.append("framing not confirmed (" + ", ".join(f"{k} {v}" for k, v in
                                                          (m.get("align_votes") or {}).items()) + ")")
        ch = self.clock_hint(short)
        if ch is not None and m.get("clock_ppm") is not None and abs(m["clock_ppm"] - ch) > self.CHECK_PPM:
            why.append(f"slant: clock {m['clock_ppm']:+.0f} ppm, station usually {ch:+.0f}")
        ah = self.afc_hint(short)
        if ah is not None and m.get("afc_hz") is not None and abs(m["afc_hz"] - ah) > self.CHECK_HZ:
            why.append(f"tuning: {m['afc_hz']:+.0f} Hz, station usually {ah:+.0f}")
        if m.get("tears") and not m.get("gaps_filled_ms"):
            why.append("tear corrected without lost audio")
        if float(m.get("quality", 0)) < self.PUBLISH_MIN_QUALITY:
            why.append(f"quality {float(m.get('quality', 0)):.0%}")
        return why

    def clock_hint(self, short=None):
        return self._hint("clock", short)

    def afc_hint(self, short=None):
        return self._hint("afc", short)

    def submit(self, audio, fs, meta):
        self.busy += 1
        self.q.put((audio, fs, meta))
        self.hub.publish("status", dict(self.hub.last_status, state="decoding"))

    def _run(self):
        while True:
            audio, fs, meta = self.q.get()
            try:
                self._process(audio, fs, meta)
            except Exception as e:
                log.exception("decode failed: %s", e)
            finally:
                self.busy -= 1
                self.done += 1
                self.q.task_done()

    def _process(self, audio, fs, meta):
        t0 = time.time()
        a = self.args
        timeline = meta.pop("timeline", None)     # flywheel data, not saved
        # which chart this is, first: the station's own clock and offset, and
        # the last good copy of the same chart, anchor the decode
        t_start = parse_utc(meta.get("utc") or "")
        t_end = t_start + timedelta(seconds=len(audio) / fs) if t_start is not None else None
        product = identify(self.sched, self.cat, self.hub.settings.get("station"), meta.get("utc"),
                           getattr(self, "auto", None), end=t_end, mid=meta.get("trigger") == "picture")
        short = (product or {}).get("short")
        res = decode.decode_recording(audio, fs, ioc=a.ioc or meta.get("ioc"), lpm=a.lpm,
                                      align=a.align, log=log.info, timeline=timeline,
                                      clock_hint_ppm=self.clock_hint(short), afc_hint_hz=self.afc_hint(short),
                                      reference=self._reference(product))
        self._learn(short, res["meta"])
        if res["meta"].get("afc_score", 99) < NO_SIGNAL_SCORE and not meta.get("name"):
            # no fax tones anywhere in the audio: a false trigger on noise
            log.info("no fax signal in this recording (score %.1f) - not saved", res["meta"]["afc_score"])
            return
        rm = res["meta"]
        if not (rm.get("start_tone") or rm.get("phasing") or rm.get("stop_tone")) \
                and rm.get("align") == "uncertain" and not meta.get("name"):
            # no start tone, phasing or stop tone, and no line structure in the
            # picture either: noise or another kind of signal, not a chart
            log.info("no fax structure in this recording (no tones, no phasing, no line seam; quality %.2f) "
                     "- not saved", rm.get("quality", 0))
            return
        img, conf = res["image"], res["conf"]
        enh, einfo = enhance.enhance(img, conf, scale=a.scale, use_ml=not a.no_ml,
                                     mode=a.mode, log=log.info, find_tears_again=False)
        if product:
            log.info("chart: %s", describe(product))
        if meta.get("name"):
            base = meta["name"]
        elif product and product.get("slot"):
            # e.g. 20261010_1636Z_DWD_Surface-weather-chart
            base = f"{meta['utc'][:10].replace('-', '')}_{product['slot']}Z_{product['short']}_{slug(product['contents'])}"
        else:
            base = "wefax_" + meta["utc"].replace("-", "").replace(":", "").replace("T", "_")
        base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
        n, stem = 1, base
        while (self.out / f"{stem}.json").exists():
            n += 1
            stem = f"{base}_{n}"
        # finished images go straight into their own folders
        raw_f, enh_f, th_f = f"raw/{stem}.png", f"enhanced/{stem}.png", f"thumbs/{stem}.jpg"
        u8 = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
        e8 = u8(enh)
        if product and product.get("contents"):
            try:            # title block, top left of the enhanced chart
                e8 = label.stamp(e8, label.title_lines(product, meta.get("utc", ""),
                                                       res["meta"].get("quality")))
            except Exception as e:
                log.info("title block failed: %s", e)
        tw = 300
        th = cv2.resize(e8, (tw, max(1, int(e8.shape[0] * tw / e8.shape[1]))), interpolation=cv2.INTER_AREA)
        save_image(self.out / raw_f, u8(img))
        save_image(self.out / enh_f, e8)
        save_image(self.out / th_f, th, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if a.save_audio:
            from scipy.io import wavfile
            (self.out / "audio").mkdir(exist_ok=True)
            wavfile.write(str(self.out / "audio" / f"{stem}.wav"), int(round(fs)),
                          (np.clip(audio / (np.max(np.abs(audio)) + 1e-12), -1, 1) * 32000).astype(np.int16))
        review = self._publish_check(res["meta"], short)
        rec = {"id": stem, "utc": meta.get("utc"), "trigger": meta.get("trigger"),
               "verified": not review, "review": review,
               "end_reason": meta.get("end_reason"), "product": product,
               "audio_overflows": meta.get("audio_overflows", 0),
               "files": {"raw": raw_f, "enhanced": enh_f, "thumb": th_f},
               "meta": res["meta"], "enhance": einfo,
               "processing_s": round(time.time() - t0, 1)}
        # the .json is written last: it is what makes the fax appear in the gallery
        tmp = self.out / f"{stem}.json.part"
        tmp.write_text(json.dumps(rec, indent=1, default=float))
        os.replace(tmp, self.out / f"{stem}.json")
        log.info("saved %s  (%d x %d, %.1f s)", self.out / enh_f, e8.shape[1], e8.shape[0], time.time() - t0)
        if review:
            # kept in the gallery for you to see, but not published: the station
            # copy (what the website takes) is left as it was, and the chart
            # counts as not collected so a repeat is tried
            log.info("NOT PUBLISHED (%s) - station copy unchanged", "; ".join(review))
        else:
            try:
                keep_station_copy(self.out, product, e8, float(res["meta"].get("quality", 0)),
                                  meta.get("utc", ""), stem)
            except Exception as e:
                log.info("station copy failed: %s", e)
        self.hub.publish("fax", {"id": stem})
        lengths = getattr(self, "lengths", None)
        end_ts = meta.get("chart_end_ts")
        if lengths is not None and end_ts and product and product.get("slot") \
                and str(meta.get("end_reason", "")).startswith(CLEAN_ENDS) \
                and meta.get("trigger") in ("start tone", "phasing"):
            # only a recording that began at the chart's start measures its
            # length (a tail found by the picture check taught "2.4 min")
            lengths.learn(product["short"], product["slot"], datetime.fromtimestamp(end_ts, timezone.utc))
        if getattr(self, "auto", None) is not None:
            fm = dict(res["meta"], snr_db=meta.get("snr_db"))
            if review:
                fm["quality"] = min(float(fm.get("quality", 0)), 0.59)   # not a collected copy
            self.auto.on_fax(product, fm, meta.get("utc", ""))


def station_info(sched, cat, station, s, auto=None) -> dict:
    """Frequency / station / current-or-next chart for the status line."""
    now = datetime.now(timezone.utc)
    info = cat.status() if cat else {}
    dial = info.get("dial_khz")
    if dial is None and auto is not None and auto.current is not None:
        station = auto.current.station      # frequency unknown: what the scheduler tuned
    if s.get("state") == "receiving":
        start = now.timestamp() - float(s.get("elapsed", 0))
        # same rules as naming the saved fax: the plan, unless CAT clearly says otherwise
        p = identify(sched, cat, station, datetime.fromtimestamp(start, timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ"), auto, quiet=True)
        if p:
            info["station"] = p["short"]
            if p.get("slot"):
                info["chart"] = f"{p['slot']}Z {p['contents']}"
    else:
        st, nxt = sched.next_slots(now, dial, station, 1)
        if st:
            info["station"] = st.short
        if nxt:
            mins = (nxt[0].minute - (now.hour * 60 + now.minute)) % 1440
            info["next"] = f"{nxt[0].hhmm}Z {nxt[0].contents}"
            info["next_in"] = mins
    return info


def read_wav(path: str, channel: int):
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if x.ndim > 1:
        x = x[:, min(channel, x.shape[1] - 1)]
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    elif x.dtype.kind == "u":
        x = (x.astype(np.float64) - 128) / 128
    return float(sr), x.astype(np.float64)


def run_file(args, proc: Processor, hub: server.Hub):
    sr, x = read_wav(args.wav, args.channel)
    dec = dsp.Decimator(sr)
    found = []
    stem = Path(args.wav).stem
    file_utc = datetime.fromtimestamp(Path(args.wav).stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def on_rec(audio, fs, meta):
        meta = dict(meta, utc=file_utc, name=f"{stem}_{len(found) + 1}" if found else stem)
        found.append(meta)
        proc.submit(audio, fs, meta)

    det = Detector(dec.fs, on_rec, lambda s: None, max_minutes=args.max_minutes, log=log.info)
    hub.publish("status", {"state": "file"})
    step = int(sr)
    resampled = []
    for i in range(0, len(x), step):
        y = dec.process(x[i:i + step])
        resampled.append(y)
        det.feed(y)
    det.flush()
    if not found:
        log.info("no start tone / phasing found - decoding the whole file")
        proc.submit(np.concatenate(resampled), dec.fs,
                    {"utc": file_utc, "trigger": "forced", "name": stem})
    proc.q.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--device", default="Line 8 (Virtual Audio Cable) MME",
                    help="input device name (substring) or index")
    ap.add_argument("--channel", type=int, default=0, help="input channel (0 = left)")
    ap.add_argument("--samplerate", type=float, help="force capture sample rate")
    ap.add_argument("--wav", help="decode a WAV recording instead of live audio")
    ap.add_argument("--out", default="faxes", help="output folder (default ./faxes)")
    ap.add_argument("--port", type=int, default=8730)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to view from other devices")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--exit", action="store_true", help="with --wav: exit when done instead of serving")
    ap.add_argument("--no-ml", action="store_true", help="disable the FSRCNN upscaler (Lanczos instead)")
    ap.add_argument("--scale", type=int, choices=(1, 2), default=2, help="enhanced image scale")
    ap.add_argument("--mode", choices=("auto", "clean", "chart", "greyscale"), default="auto",
                    help="auto: rebuild chart lines (default); clean: de-streak only; "
                         "chart: hard black/white; greyscale: satellite pictures")
    ap.add_argument("--ioc", type=int, choices=(288, 576), help="override IOC")
    ap.add_argument("--lpm", type=int, choices=(60, 90, 120, 240), help="override lines/minute")
    ap.add_argument("--align", choices=("center", "start", "end"), default="end",
                    help="line start at the end of the phasing pulse (default; what stations transmit), "
                         "its centre, or its start")
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--save-audio", action="store_true", help="also keep each transmission as WAV")
    ap.add_argument("--cat", metavar="COMx", help="SDR Console CAT (TS-2000) port, to read the frequency")
    ap.add_argument("--cat-rx", type=int, default=1, help="SDR Console receiver number (1-6)")
    ap.add_argument("--cat-baud", type=int, default=57600)
    ap.add_argument("--station", help="station when no CAT, e.g. DWD or Northwood")
    ap.add_argument("--auto", action="store_true", help="start with the auto-scheduler on (needs --cat)")
    ap.add_argument("--no-dx", action="store_true", help="auto-scheduler: 1st/2nd priority stations only, no DX gap filling")
    ap.add_argument("--qth", default=None,
                    help="your location for the propagation model: 'lat,lon' or a Maidenhead locator "
                         "(default Plymouth, UK; a location set on the page takes precedence)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # keep a log file as well as the console (faxes/wefax.log, 3 x 2 MB)
    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(out / "wefax.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)
    organise_existing(out)
    hub = server.Hub(out)
    try:
        server.start(hub, args.host, args.port)
    except OSError as e:
        sys.exit(f"cannot open web server on port {args.port}: {e}  (try --port)")
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    log.info("display: %s   output: %s  (raw\\ and enhanced\\)", url, out.resolve())
    if not args.no_browser and not (args.wav and args.exit):
        webbrowser.open(url)

    sched = Schedules()
    log.info("schedules: %d stations loaded", len(sched.stations))
    cat = None
    if not args.wav:
        # the port chosen on the page (saved in faxes/settings.json) wins over --cat
        from wefax.cat import CatReader
        saved = hub.load_settings().get("cat_port", "unset")
        port = args.cat if saved == "unset" else saved
        if saved != "unset" and args.cat and saved != args.cat:
            log.info("CAT: using %s chosen on the page (start option said %s)", saved or "off", args.cat)
        cat = CatReader(port, args.cat_baud, args.cat_rx, log=log.info)
    from wefax.autosched import AutoScheduler, parse_qth, DEFAULT_QTH
    # the location set on the page (faxes/settings.json) wins over --qth
    saved_qth = hub.load_settings().get("qth")
    qth = DEFAULT_QTH
    for src, val in (("--qth", args.qth), ("page", saved_qth)):
        if val:
            try:
                qth = parse_qth(val)
            except ValueError as e:
                log.info("location from %s ignored: %s (%s)", src, val, e)
    if saved_qth and args.qth and saved_qth != args.qth:
        log.info("location: using %s set on the page (start option said %s)", saved_qth, args.qth)
    log.info("location: %.2f, %.2f%s", qth[0], qth[1], "" if (args.qth or saved_qth) else
             " (default - set yours in the QTH box on the page)")
    auto = AutoScheduler(sched, cat, out, qth, enabled=bool(args.auto and cat and cat.enabled),
                         dx=not args.no_dx, log=log.info)
    if args.auto and not (cat and cat.enabled):
        log.info("auto-schedule needs a CAT port to tune SDR Console - left off")
    hub.sched, hub.cat, hub.auto = sched, cat, auto
    hub.settings["station"] = args.station
    proc = Processor(out, hub, args, sched, cat)
    proc.auto = auto
    lengths = None
    if not args.wav:                      # live only: chart ends are timed by the PC clock
        lengths = ChartLengths(out, log=log.info)
        proc.lengths = lengths
        auto.planner.lengths = lengths
    if not args.wav:
        auto.start()

    if args.wav:
        run_file(args, proc, hub)
        hub.publish("status", {"state": "idle"})
        if args.exit:
            return
        log.info("done - still serving the display, Ctrl+C to quit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)
    saved_audio = hub.load_settings().get("audio_device")
    if saved_audio:
        device = saved_audio              # the input chosen on the page wins over --device
        log.info("audio: using %s (%s) chosen on the page", saved_audio.get("name"), saved_audio.get("api"))

    def status_cb(s):
        try:        # the picture check needs this installation's usual tuning offset
            if cap.detector is not None:
                cap.detector.afc_usual = proc.afc_hint()
        except NameError:
            pass    # capture not set up yet
        if s.get("state") == "idle" and proc.busy:
            s = dict(s, state="decoding")
        s = dict(s, **station_info(sched, cat, hub.settings.get("station"), s, auto))
        auto.on_status(s)
        hub.publish("status", s)

    cap = LiveCapture(lambda fs: Detector(fs, proc.submit, status_cb,
                                          max_minutes=args.max_minutes, log=log.info,
                                          on_preview=hub.set_live,
                                          on_spectrum=lambda d: hub.publish("spec", d)),
                      device=device, channel=args.channel, samplerate=args.samplerate, log=log.info)
    hub.capture = cap
    auto.decoder_busy = lambda: proc.busy > 0
    auto.force_end = lambda reason: cap.detector and cap.detector.request_finish(reason)
    if cat is not None:
        # every retune (scheduler, Tune button, chart chips) tells the detector,
        # so the picture check never reaches back into the previous frequency
        _tune = cat.tune

        def tune_and_mark(khz, usb=True, _retry=0, **kw):
            if cap.detector is not None:
                cap.detector.retuned()
            return _tune(khz, usb, _retry, **kw)
        cat.tune = tune_and_mark

    def set_deadline(det):
        """Once a recording has started: when should this chart be over?"""
        meta = dict(det.rec_meta)
        rec_id = meta.get("t0_samples")
        p = identify(sched, cat, hub.settings.get("station"), meta.get("utc"), auto,
                     mid=meta.get("trigger") == "picture")
        if not (p and p.get("slot")):
            return rec_id
        dl = lengths.deadline(sched, p["short"], p["slot"], parse_utc(meta["utc"]))
        if dl is None:
            return rec_id
        when, how, end = dl
        if when.timestamp() < time.time() + 60:
            # already (nearly) over: the chart was misidentified, or this is a
            # late repeat - never cut a recording that has only just begun
            if meta.get("trigger") == "picture" and det.state == "receiving" \
                    and det.rec_meta.get("t0_samples") == rec_id:
                # found by the picture check alone, after its chart's time: never
                # let it run on unchecked (a 32 min recording blocked the plan)
                lim = time.time() + PICTURE_LATE_MAX_S
                det.deadline = (lim, None, f"{p['short']} {p['slot']}Z, picture found late")
                log.info("receiving %s %sZ: its expected end has passed - closing within %d min at the latest",
                         p["short"], p["slot"], PICTURE_LATE_MAX_S // 60)
                return rec_id
            log.info("receiving %s %sZ: its expected end has passed - no length limit for this one",
                     p["short"], p["slot"])
            return rec_id
        if det.state == "receiving" and det.rec_meta.get("t0_samples") == rec_id:
            det.deadline = (when.timestamp(), end.timestamp() if end else None,
                            f"{p['short']} {p['slot']}Z, {how}")
            log.info("receiving %s %sZ: will close by %s UTC at the latest (%s length)",
                     p["short"], p["slot"], when.strftime("%H:%M:%S"), how)
        return rec_id

    def watcher():          # tells the live panel whether a chart is expected now
        last = None
        done_rec = None
        while True:
            try:
                det = cap.detector
                if det is not None and det.state == "receiving" and det.deadline is None \
                        and det.rec_meta.get("t0_samples") != done_rec:
                    done_rec = set_deadline(det)
            except Exception as e:
                log.info("deadline error: %s", e)
            try:
                w = expected_now(sched, cat, hub.settings.get("station"), auto)
                cap.detector.watch = w
                key = (w or {}).get("short"), (w or {}).get("slot")
                if key != last:
                    last = key
                    log.info("live panel: %s", f"expecting {w['short']} {w['slot']}Z {w['contents']}"
                             if w else "nothing scheduled on this frequency (IDLE)")
            except Exception as e:
                log.info("watch error: %s", e)
            time.sleep(5)
    threading.Thread(target=watcher, daemon=True).start()
    # nightly clean-up of old faxes (Keep box on the page), at a quiet moment
    housekeep.start(out, hub, lambda: (cap.detector is None or cap.detector.state == "idle") and not proc.busy,
                    log.info)
    log.info("listening - Ctrl+C to stop")
    try:
        cap.run()
    except KeyboardInterrupt:
        cap.stop()
        det = cap.detector
        if det.state == "receiving":
            log.info("stopping: decoding the partial transmission")
            det.flush()
            proc.q.join()


if __name__ == "__main__":
    main()
