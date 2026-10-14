"""Live audio capture and APT tone state machine.

The front-end only has to *find* transmissions: a cheap causal demodulator
watches for the start tone (300 Hz -> IOC 576, 675 Hz -> IOC 288), the
phasing pattern (for stations heard without a start tone), the 450 Hz stop
tone and loss of signal.  Raw audio of each transmission (with pre-roll) is
handed to the offline decoder, which can then use zero-phase filters and look
at the whole picture at once.
"""
from __future__ import annotations

import collections
import threading
import time
from datetime import datetime, timezone

import numpy as np

from . import dsp
from .preview import LivePreview


class Detector:
    BLOCK_S = 0.25
    PRESENT_RATIO = 1.4

    def __init__(self, fs: float, on_recording, on_status=None, max_minutes: float = 40.0,
                 preroll_s: float = 12.0, lost_s: float = 45.0, log=print, on_preview=None):
        self.fs = fs
        self.nb = int(round(self.BLOCK_S * fs))
        self.on_recording = on_recording
        self.on_status = on_status or (lambda s: None)
        self.max_samples = int(max_minutes * 60 * fs)
        self.lost_blocks = int(lost_s / self.BLOCK_S)
        self.log = log
        self.demod = dsp.StreamDemod(fs)
        self.pending = np.zeros(0)
        self.pre = collections.deque(maxlen=int(preroll_s / self.BLOCK_S))
        self.fhist = collections.deque(maxlen=int(6.0 / self.BLOCK_S))
        self.state = "idle"
        self.rec: list[np.ndarray] = []
        self.rec_n = 0
        self.rec_meta: dict = {}
        self.start_run = 0
        self.start_kind = None
        self.stop_run = 0
        self.invalid_run = 0
        self.phase_hits = 0
        self.block_count = 0
        self.level_db = -120.0
        self.last_label = "noise"
        self.samples_seen = 0
        self._tail = None
        self.ratio_hist = collections.deque(maxlen=4)
        self.stop_hist = collections.deque(maxlen=10)
        self.band_ratio = 1.0
        # live preview: discriminator output kept alongside the audio blocks
        self.on_preview = on_preview
        self.preview: LivePreview | None = None
        self.fbuf = collections.deque(maxlen=max(self.pre.maxlen, int(8 / self.BLOCK_S)))
        # flywheel: PC-clock arrival time of each sound-card buffer, against
        # the running sample count.  Lost samples show up as a step.
        self.fed_total = 0
        self.timeline: "collections.deque[tuple]" = collections.deque(maxlen=40000)
        self.rec_start_abs = 0
        # set from outside: the chart we expect to be on air now (or None).
        # While it is set and nothing is being received, the live panel shows
        # the incoming noise ('NO SIGNAL') - nothing is recorded or saved.
        self.watch: dict | None = None
        self.npreview: LivePreview | None = None
        # set from another thread (the auto-scheduler) to end the recording
        self._finish_req: str | None = None

    # ------------------------------------------------------------------
    def feed(self, x: np.ndarray, t: float | None = None, overflow: bool = False):
        x = np.asarray(x, np.float64)
        self.fed_total += len(x)
        if t is not None:
            # (sample index at the end of this buffer, arrival time, overflow flag)
            self.timeline.append((self.fed_total, t, bool(overflow)))
        x = np.concatenate([self.pending, x])
        n = len(x) // self.nb
        for i in range(n):
            self._block(x[i * self.nb:(i + 1) * self.nb])
        self.pending = x[n * self.nb:]

    def flush(self):
        """End of input (file mode): finish any recording in progress."""
        if self.state == "receiving":
            self._finish("end of input")

    # ------------------------------------------------------------------
    def _block(self, a: np.ndarray):
        self.block_count += 1
        self.samples_seen += len(a)
        f, mag = self.demod.process(a)
        feats = dsp.block_features(f, mag, self.fs)
        label = dsp.classify_block(feats)
        # spectral presence test, smoothed over ~1 s: keeps a weak or
        # fading fax counted as 'signal' long after the demodulator
        # output has become too noisy to classify
        self.ratio_hist.append(dsp.band_ratio(a, self.fs))
        self.band_ratio = float(np.mean(self.ratio_hist))
        if label == "noise" and self.band_ratio > self.PRESENT_RATIO:
            label = "signal"
        self.last_label = label
        rms = float(np.sqrt(np.mean(a * a))) + 1e-12
        self.level_db = 20 * np.log10(rms)
        self.fhist.append(f)
        self.fbuf.append(f)

        if label.startswith("start"):
            if label == self.start_kind:
                self.start_run += 1
            else:
                self.start_kind, self.start_run = label, 1
        else:
            self.start_run = 0
            self.start_kind = None

        if self.state == "idle":
            self.pre.append(a)
            self._noise_view(f)
            if self.start_run >= 6:
                self._begin("start tone", 576 if self.start_kind == "start576" else 288)
            elif self.block_count % 4 == 0 and len(self.fhist) == self.fhist.maxlen:
                if phasing_like(np.concatenate(self.fhist), self.fs):
                    self.phase_hits += 1
                else:
                    self.phase_hits = 0
                if self.phase_hits >= 3:
                    self._begin("phasing", None)
        else:
            self.rec.append(a)
            self.rec_n += len(a)
            if self.preview is not None:
                self.preview.add(f)
                if self.block_count % 8 == 0:
                    self._update_preview()
            self.invalid_run = self.invalid_run + 1 if label == "noise" else 0
            # tolerate the odd misclassified block in a noisy stop tone
            self.stop_hist.append(label == "stop")
            self.stop_run = sum(self.stop_hist)
            elapsed = self.rec_n / self.fs
            if self.stop_run >= 6 and elapsed > 60:
                # keep ~10 s after the stop tone for the end-of-frame black
                self._tail = int(10 / self.BLOCK_S)
            if self._tail is not None:
                self._tail -= 1
                if self._tail <= 0:
                    self._tail = None
                    self._finish("stop tone")
            elif self.start_run >= 8 and elapsed > 90:
                seed = self.rec_tail_for_restart
                ioc = 576 if self.start_kind == "start576" else 288
                self.rec = self.rec[:-len(seed)]
                self._finish("new start tone")
                self._begin("start tone", ioc, seed=seed)
            elif self._finish_req is not None:
                reason, self._finish_req = self._finish_req, None
                self._finish(reason)
            elif elapsed > self.SPLIT_MIN_S and self.block_count % 4 == 0 and self._phasing_now():
                # the next chart's phasing has started, but its start tone (and
                # this chart's stop tone) were missed: close this chart and
                # start a new recording from shortly before the phasing
                k = int(self.SPLIT_SEED_S / self.BLOCK_S)
                seed = self.rec[-k:]
                self.rec = self.rec[:-k]
                self._finish("new phasing")
                self._begin("phasing", None, seed=seed)
            elif self.invalid_run >= self.lost_blocks:
                # drop the trailing noise, keep a few seconds of margin
                keep = len(self.rec) - max(0, self.invalid_run - int(4 / self.BLOCK_S))
                self.rec = self.rec[:keep]
                self._finish("signal lost")
            elif self.rec_n >= self.max_samples:
                self._finish("maximum length")

        if self.block_count % 4 == 0:
            self.on_status(self.status())

    # ------------------------------------------------------------------
    def status(self) -> dict:
        s = {"state": self.state, "level_db": round(self.level_db, 1), "signal": self.last_label,
             "snr_db": round(10 * np.log10(max(self.band_ratio - 1.0, 1e-3)), 1)}
        s["watch"] = self.watch
        if self.state == "receiving":
            s["elapsed"] = round(self.rec_n / self.fs, 1)
            # recording continues, but nothing usable has been heard for a while
            s["faded"] = self.invalid_run * self.BLOCK_S >= self.FADED_S
            s["ioc"] = self.rec_meta.get("ioc")
            s["trigger"] = self.rec_meta.get("trigger")
        return s

    SPLIT_MIN_S = 150     # earliest point in a recording to look for the next chart's phasing
    SPLIT_HITS = 5        # consecutive 1 s phasing checks needed to split
    SPLIT_SEED_S = 20.0   # audio before the detection carried into the new recording

    def request_finish(self, reason: str = "ended by scheduler"):
        """Ask for the current recording to end (safe from another thread)."""
        if self.state == "receiving":
            self._finish_req = reason

    def _phasing_now(self) -> bool:
        if len(self.fhist) < self.fhist.maxlen:
            return False
        if phasing_like(np.concatenate(self.fhist), self.fs):
            self.phase_hits += 1
        else:
            self.phase_hits = 0
        return self.phase_hits >= self.SPLIT_HITS

    def _begin(self, trigger: str, ioc, seed=None):
        self.state = "receiving"
        self._finish_req = None
        self.npreview = None
        self.rec = list(seed) if seed is not None else list(self.pre)
        self.rec_n = sum(len(b) for b in self.rec)
        self.rec_start_abs = self.samples_seen - self.rec_n
        self.pre.clear()
        self.stop_run = 0
        self.stop_hist.clear()
        self.invalid_run = 0
        self.phase_hits = 0
        self._tail = None
        self.rec_meta = {"trigger": trigger, "ioc": ioc,
                         "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "t0_samples": self.samples_seen}
        if self.on_preview is not None:
            self.preview = LivePreview(self.fs)
            for fb in list(self.fbuf)[-len(self.rec):] if self.rec else []:
                self.preview.add(fb)
        self.log(f"[{time.strftime('%H:%M:%S')}] receiving (trigger: {trigger}, IOC {ioc or '?'})")
        self.on_status(self.status())

    FADED_S = 5.0         # receiving, but no signal for this long -> 'NO SIGNAL'
    NOISE_ROWS = 160      # rows kept in the scrolling 'no signal' view

    def _noise_view(self, f):
        if self.watch is None or self.on_preview is None:
            self.npreview = None
            return
        if self.npreview is None or self.npreview.n > self.npreview.fs * 300:
            self.npreview = LivePreview(self.fs)
        self.npreview.add(f)
        if self.block_count % 8 == 0:
            try:
                png, info = self.npreview.render(max_rows=self.NOISE_ROWS)
                if png is not None:
                    self.on_preview(png, dict(info, mode="nosignal"))
            except Exception as e:
                self.log(f"preview error: {e}")

    def _update_preview(self):
        try:
            # coarse AFC from the last 20 s so black/white levels are right
            k = int(20 / self.BLOCK_S)
            off, score = dsp.psd_comb(np.concatenate(self.rec[-k:]), self.fs)
            if score > 6:
                self.preview.set_offset(off)
            png, info = self.preview.render()
            if png is not None:
                self.on_preview(png, dict(info, mode="receiving"))
        except Exception as e:  # preview must never stop reception
            self.log(f"preview error: {e}")

    @property
    def rec_tail_for_restart(self):
        k = int(8 / self.BLOCK_S)
        return self.rec[-k:]

    def _finish(self, reason: str):
        audio = np.concatenate(self.rec) if self.rec else np.zeros(0)
        meta = dict(self.rec_meta, end_reason=reason)
        if self.timeline and len(audio):
            a0, a1 = self.rec_start_abs, self.rec_start_abs + len(audio)
            tl = [(i - a0, t, o) for i, t, o in self.timeline if a0 <= i <= a1]
            if len(tl) > 50:
                meta["timeline"] = tl
        self.state = "idle"
        self.rec, self.rec_n = [], 0
        self._tail = None
        self.preview = None
        dur = len(audio) / self.fs
        self.log(f"[{time.strftime('%H:%M:%S')}] transmission ended ({reason}), {dur:.0f} s")
        if dur >= 60:
            self.on_recording(audio, self.fs, meta)
        else:
            self.log("  too short to be a fax - discarded")
        self.on_status(self.status())


def phasing_like(f: np.ndarray, fs: float) -> bool:
    """Mostly black with a narrow, regularly repeating white pulse."""
    lo = np.percentile(f, 30)
    hi = np.percentile(f, 99.5)
    if hi - lo < 450:
        return False
    v = (f - lo) / (hi - lo)
    black = np.mean(v < 0.35)
    white = np.mean(v > 0.65)
    if not (0.72 < black < 0.98 and 0.015 < white < 0.15):
        return False
    w = (v > 0.5).astype(np.float32)
    w -= w.mean()
    e = float(np.dot(w, w)) + 1e-9
    for lpm in (120, 60, 90, 240):
        lag = int(round(fs * 60 / lpm))
        if 2 * lag >= len(w):
            continue
        r = max(float(np.dot(w[:-l], w[l:])) / e for l in (lag - 2, lag, lag + 2))
        if r > 0.5:
            return True
    return False


# --------------------------------------------------------------------------
_UNSET = object()


def list_inputs(sd) -> list[dict]:
    """All audio inputs, with the Windows host API (MME, WASAPI, ...)."""
    apis = sd.query_hostapis()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append({"index": i, "name": d["name"], "api": apis[d["hostapi"]]["name"],
                        "channels": d["max_input_channels"], "default": i == default_in})
    return out


def resolve_input(sd, spec):
    """Device index for a spec: None (system default), an index, a name
    (substring, as on the command line) or {"name", "api"} as saved by the page."""
    if spec is None:
        return None
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        return int(spec)
    ins = list_inputs(sd)
    if isinstance(spec, dict):
        for d in ins:
            if d["name"] == spec.get("name") and d["api"] == spec.get("api"):
                return d["index"]
        for d in ins:                        # same device under another API
            if d["name"] == spec.get("name"):
                return d["index"]
        raise ValueError(f"audio input '{spec.get('name')}' ({spec.get('api')}) not found")
    want = str(spec).lower()
    for d in ins:
        if f"{d['name']} {d['api']}".lower() == want or d["name"].lower() == want:
            return d["index"]
    for d in ins:
        if want in f"{d['name']} {d['api']}".lower():
            return d["index"]
    raise ValueError(f"audio input '{spec}' not found")


class LiveCapture:
    """Sound-card input -> decimator -> detector, on its own thread.

    The input can be changed while running (request_device); the stream is
    reopened and a fresh detector made for the new sample rate.
    """

    def __init__(self, detector_factory, device=None, channel: int = 0,
                 samplerate: float | None = None, log=print):
        import sounddevice as sd
        self.sd = sd
        self.factory = detector_factory
        self.want_channel = channel
        self.want_sr = samplerate
        self.log = log
        self.q: "collections.deque[tuple]" = collections.deque()
        self.ev = threading.Event()
        self.running = False
        self.overflows = 0
        self.overflow_events: "collections.deque[str]" = collections.deque(maxlen=50)
        self.detector = None
        self.dev_index = None
        self.info: dict | None = None
        self.error: str | None = None
        self.streaming = False
        self._req = _UNSET
        if not self._configure(device) and device is not None:
            self.log("audio: falling back to the system default input")
            self._configure(None)

    # ------------------------------------------------------------------
    def _configure(self, spec) -> bool:
        sd = self.sd
        try:
            idx = resolve_input(sd, spec)
            info = sd.query_devices(idx, "input") if idx is not None else sd.query_devices(kind="input")
            if idx is None:
                idx = sd.default.device[0] if sd.default.device[0] is not None and sd.default.device[0] >= 0 else None
        except Exception as e:
            self.error = f"{e}"
            self.log(f"audio: {e}")
            return False
        nin = int(info["max_input_channels"])
        self.channel = min(self.want_channel, max(0, nin - 1))
        self.channels = self.channel + 1
        self.sr = float(self.want_sr or info["default_samplerate"])
        self.dec = dsp.Decimator(self.sr)
        old = self.detector
        if old is not None and old.state == "receiving":
            old.flush()                       # decode what was received so far
        self.detector = self.factory(self.dec.fs)
        if old is not None:
            self.detector.watch = old.watch
        self.dev_index, self.info, self.error = idx, info, None
        try:
            api = self.sd.query_hostapis()[info["hostapi"]]["name"]
        except Exception:
            api = ""
        self.current = {"name": info["name"], "api": api}
        self.log(f"input: {info['name']} ({api}) @ {self.sr:.0f} Hz, channel {self.channel} "
                 f"-> internal {self.dec.fs:.0f} Hz")
        return True

    def request_device(self, spec):
        """Switch input (done on the capture thread)."""
        self._req = spec
        self.ev.set()

    def _cb(self, indata, frames, t, status):
        now = time.perf_counter()          # high-resolution PC clock (the flywheel)
        ovf = bool(status and status.input_overflow)
        if ovf:
            self.overflows += 1
            self.overflow_events.append(time.strftime("%H:%M:%S"))
        self.q.append((indata[:, self.channel].copy(), now, ovf))
        self.ev.set()

    def run(self):
        self.running = True
        warned = None
        while self.running:
            if self._req is not _UNSET:
                spec, self._req = self._req, _UNSET
                self.q.clear()
                self._configure(spec)
                warned = None
            if self.info is None:
                time.sleep(0.5)
                continue
            try:
                with self.sd.InputStream(device=self.dev_index, channels=self.channels,
                                         samplerate=self.sr, dtype="float32",
                                         blocksize=int(self.sr * 0.1), latency="high", callback=self._cb):
                    self.streaming, self.error, warned = True, None, None
                    while self.running and self._req is _UNSET:
                        self.ev.wait(0.5)
                        self.ev.clear()
                        while self.q:
                            data, tstamp, ovf = self.q.popleft()
                            self.detector.feed(self.dec.process(data), tstamp, ovf)
                        while self.overflow_events:
                            ts = self.overflow_events.popleft()
                            self.log(f"[{ts}] WARNING: audio input overflow - samples lost (this tears the fax)")
                            if self.detector.state == "receiving":
                                self.detector.rec_meta["audio_overflows"] = \
                                    self.detector.rec_meta.get("audio_overflows", 0) + 1
            except Exception as e:
                self.error = f"{e.__class__.__name__}: {e}"
                if warned != self.error:
                    self.log(f"audio: cannot open {self.info['name']} ({self.error}) - retrying")
                    warned = self.error
                for _ in range(15):
                    if self._req is not _UNSET or not self.running:
                        break
                    time.sleep(0.2)
            finally:
                self.streaming = False

    def status(self) -> dict:
        return {"current": getattr(self, "current", None), "streaming": self.streaming, "error": self.error}

    def stop(self):
        self.running = False
