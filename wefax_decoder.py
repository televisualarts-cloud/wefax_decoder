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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wefax import dsp, decode, enhance, server
from wefax.schedule import Schedules, slug
from wefax.frontend import Detector, LiveCapture

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is required:  pip install opencv-python-headless")

log = logging.getLogger("wefax")
NO_SIGNAL_SCORE = 6.0     # AFC comb score: pure noise ~2, weakest real faxes ~19


def parse_utc(text):
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def identify(sched, cat, station, utc_text):
    """Which chart is this?  Station from the SDR Console frequency (CAT) or
    --station, chart from the schedule slot nearest the start time."""
    if sched is None or not utc_text:
        return None
    when = parse_utc(utc_text)
    if when is None:
        return None
    dial = cat.dial_khz_at(when) if cat else None
    return sched.identify(when, dial, station)


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
        threading.Thread(target=self._run, daemon=True).start()

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
        res = decode.decode_recording(audio, fs, ioc=a.ioc or meta.get("ioc"), lpm=a.lpm,
                                      align=a.align, log=log.info, timeline=timeline)
        if res["meta"].get("afc_score", 99) < NO_SIGNAL_SCORE and not meta.get("name"):
            # no fax tones anywhere in the audio: a false trigger on noise
            log.info("no fax signal in this recording (score %.1f) - not saved", res["meta"]["afc_score"])
            return
        img, conf = res["image"], res["conf"]
        enh, einfo = enhance.enhance(img, conf, scale=a.scale, use_ml=not a.no_ml,
                                     mode=a.mode, log=log.info)
        product = identify(self.sched, self.cat, self.hub.settings.get("station"), meta.get("utc"))
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
        rec = {"id": stem, "utc": meta.get("utc"), "trigger": meta.get("trigger"),
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
        self.hub.publish("fax", {"id": stem})
        if getattr(self, "auto", None) is not None:
            self.auto.on_fax(product, res["meta"], meta.get("utc", ""))


def station_info(sched, cat, station, s) -> dict:
    """Frequency / station / current-or-next chart for the status line."""
    now = datetime.now(timezone.utc)
    info = cat.status() if cat else {}
    dial = info.get("dial_khz")
    if s.get("state") == "receiving":
        start = now.timestamp() - float(s.get("elapsed", 0))
        p = sched.identify(datetime.fromtimestamp(start, timezone.utc), dial, station)
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
    ap.add_argument("--align", choices=("center", "start", "end"), default="center",
                    help="x=0 at the centre/start/end of the phasing pulse")
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--save-audio", action="store_true", help="also keep each transmission as WAV")
    ap.add_argument("--cat", metavar="COMx", help="SDR Console CAT (TS-2000) port, to read the frequency")
    ap.add_argument("--cat-rx", type=int, default=1, help="SDR Console receiver number (1-6)")
    ap.add_argument("--cat-baud", type=int, default=57600)
    ap.add_argument("--station", help="station when no CAT, e.g. DWD or Northwood")
    ap.add_argument("--auto", action="store_true", help="start with the auto-scheduler on (needs --cat)")
    ap.add_argument("--no-dx", action="store_true", help="auto-scheduler: Northwood/DWD only, no DX gap filling")
    ap.add_argument("--qth", default="50.37,-4.14", help="your lat,lon for the propagation model")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
    from wefax.autosched import AutoScheduler
    qth = tuple(float(v) for v in args.qth.split(","))
    auto = AutoScheduler(sched, cat, out, qth, enabled=bool(args.auto and cat and cat.enabled),
                         dx=not args.no_dx, log=log.info)
    if args.auto and not (cat and cat.enabled):
        log.info("auto-schedule needs a CAT port to tune SDR Console - left off")
    hub.sched, hub.cat, hub.auto = sched, cat, auto
    hub.settings["station"] = args.station
    proc = Processor(out, hub, args, sched, cat)
    proc.auto = auto
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
        if s.get("state") == "idle" and proc.busy:
            s = dict(s, state="decoding")
        s = dict(s, **station_info(sched, cat, hub.settings.get("station"), s))
        auto.on_status(s)
        hub.publish("status", s)

    cap = LiveCapture(lambda fs: Detector(fs, proc.submit, status_cb,
                                          max_minutes=args.max_minutes, log=log.info,
                                          on_preview=hub.set_live),
                      device=device, channel=args.channel, samplerate=args.samplerate, log=log.info)
    hub.capture = cap

    def watcher():          # tells the live panel whether a chart is expected now
        last = None
        while True:
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
