"""Live audio capture and APT tone state machine.

The front-end only has to *find* transmissions: a cheap causal demodulator
watches for the start tone (300 Hz -> IOC 576, 675 Hz -> IOC 288), the
phasing pattern (for stations heard without a start tone), the 450 Hz stop
tone and loss of signal.  Raw audio of each transmission (with pre-roll) is
handed to the offline decoder, which can then use zero-phase filters and look
at the whole picture at once.
"""
from __future__ import annotations

import base64
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
                 preroll_s: float = 12.0, lost_s: float = 45.0, log=print, on_preview=None,
                 on_spectrum=None):
        self.fs = fs
        self.on_spectrum = on_spectrum      # live spectrum for the page (see _spectrum)
        self.spec_afc = None
        self.nb = int(round(self.BLOCK_S * fs))
        self.on_recording = on_recording
        self.on_status = on_status or (lambda s: None)
        self.max_samples = int(max_minutes * 60 * fs)
        self.lost_blocks = int(lost_s / self.BLOCK_S)
        self.log = log
        self.demod = dsp.StreamDemod(fs)
        self.pending = np.zeros(0)
        self.pre = collections.deque(maxlen=int(preroll_s / self.BLOCK_S))
        # idle lookback: audio, demodulated frequency and signal presence of
        # every block since the last recording, so a chart whose start tone
        # and phasing were both missed can still be recorded from its start
        self.look = collections.deque(maxlen=int(self.LOOKBACK_S / self.BLOCK_S))
        self.img_votes = collections.deque(maxlen=self.IMG_VOTES)
        self.idle_since = 0.0
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
        # set from outside once the chart is known: (epoch deadline, expected
        # image end epoch or None, description). Past the deadline the
        # recording is closed even if no stop tone or black was seen.
        self.deadline: tuple | None = None
        self.eof_hist = collections.deque(maxlen=int(self.EOF_WIN_S / self.BLOCK_S))
        self.paper_hist = collections.deque(maxlen=int((self.EOF_PAPER_S + self.EOF_WIN_S) / self.BLOCK_S))

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
        if self.on_spectrum is not None:
            try:
                self._spectrum(a)
            except Exception as e:          # the display must never stop reception
                if self.block_count % 400 == 1:
                    self.log(f"spectrum error: {e}")
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
            self.look.append((a.astype(np.float32), f, label != "noise"))
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
                elif self._image_now():
                    self._begin_from_lookback()
        else:
            self.rec.append(a)
            self.rec_n += len(a)
            if self.preview is not None:
                self.preview.add(f)
                if self.block_count % 8 == 0:
                    self._update_preview()
            self.invalid_run = self.invalid_run + 1 if label == "noise" else 0
            if label != "noise" and self.block_count % 4 == 0:
                self.rec_snr.append(10 * np.log10(max(self.band_ratio - 1.0, 1e-3)))
            # tolerate the odd misclassified block in a noisy stop tone
            self.stop_hist.append(label == "stop")
            self.stop_run = sum(self.stop_hist)
            elapsed = self.rec_n / self.fs
            # a recording started from the picture check began mid-chart: any
            # start tone or phasing after that is the next chart, split at once
            mid = self.rec_meta.get("trigger") == "picture"
            split_tone_s = self.PICTURE_SPLIT_S if mid else 90
            split_phase_s = self.PICTURE_SPLIT_S if mid else self.SPLIT_MIN_S
            if self.stop_run >= 6 and elapsed > 60:
                # keep ~10 s after the stop tone for the end-of-frame black
                if self._tail is None:
                    self.rec_meta["chart_end_ts"] = time.time() - self.stop_run * self.BLOCK_S
                self._tail = int(10 / self.BLOCK_S)
            if self._tail is not None:
                self._tail -= 1
                if self._tail <= 0:
                    self._tail = None
                    self._finish("stop tone")
            elif self.start_run >= 8 and elapsed > split_tone_s:
                seed = self.rec_tail_for_restart
                ioc = 576 if self.start_kind == "start576" else 288
                self.rec = self.rec[:-len(seed)]
                self._finish("new start tone")
                self._begin("start tone", ioc, seed=seed)
            elif self._finish_req is not None:
                reason, self._finish_req = self._finish_req, None
                self._finish(reason)
            elif self.deadline is not None and time.time() > self.deadline[0]:
                self._end_by_deadline()
            elif elapsed > self.SPLIT_MIN_S and self._end_of_frame(f):
                pass                          # finished inside _end_of_frame
            elif elapsed > split_phase_s and self.block_count % 4 == 0 and self._phasing_now():
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

    SPEC_F0, SPEC_F1, SPEC_BIN = 800, 3000, 10   # Hz: the receiver audio shown on the page

    def _spectrum(self, a: np.ndarray):
        """Power spectrum of this 0.25 s block, 800-3000 Hz in 10 Hz bins, sent
        to the page as bytes (dB x2 + 260). Every 2 s also the tuning offset,
        measured as the decoder does (psd_comb) over the last 12-20 s."""
        n = len(a)
        X = np.abs(np.fft.rfft(a * np.hanning(n))) ** 2 / n
        f = np.fft.rfftfreq(n, 1.0 / self.fs)
        edges = np.arange(self.SPEC_F0, self.SPEC_F1 + self.SPEC_BIN, self.SPEC_BIN)
        idx = np.searchsorted(f, edges)
        p = np.array([X[i0:max(i1, i0 + 1)].mean() for i0, i1 in zip(idx[:-1], idx[1:])])
        u8 = np.clip((10 * np.log10(p + 1e-14) + 130) * 2, 0, 255).astype(np.uint8)
        if self.block_count % 8 == 0:
            blocks = self.rec[-80:] if self.state == "receiving" else list(self.pre)
            if len(blocks) >= 16:
                off, score = dsp.psd_comb(np.concatenate(blocks), self.fs)
                self.spec_afc = round(float(off), 1) if score > 6 else None
        self.on_spectrum({"f0": self.SPEC_F0, "bin": self.SPEC_BIN,
                          "db": base64.b64encode(u8.tobytes()).decode("ascii"), "afc": self.spec_afc})

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
            if self.deadline is not None:
                s["ends_by"] = round(self.deadline[0])
        return s

    SPLIT_MIN_S = 150     # earliest point in a recording to look for the next chart's phasing
    SPLIT_HITS = 5        # consecutive 1 s phasing checks needed to split
    SPLIT_SEED_S = 20.0   # audio before the detection carried into the new recording

    # End of a chart without a usable stop tone. Stations that keep the
    # carrier up between charts (DWD sends white between broadcasts) never
    # trigger 'signal lost', so a missed stop tone would otherwise leave the
    # recording open until the next start tone or the length limit. Every
    # chart ends with ~10 s of black after the stop tone; that is caught here.
    EOF_WIN_S = 8.0       # look at the last 8 s ...
    EOF_BLACK_FRAC = 0.8  # ... and end the chart when this much of it is solid black
    EOF_KEEP_S = 1.0      # of the black kept at the end of the recording
    EOF_PAPER_S = 60.0    # the minute before the black ...
    EOF_PAPER_FRAC = 0.5  # ... must have been mostly white paper

    def _end_of_frame(self, f: np.ndarray) -> bool:
        # 5 ms averages: noise drops ~7x, chart detail and the phasing pulse remain
        k = max(1, int(self.fs * 0.005))
        m = f[:len(f) // k * k].reshape(-1, k).mean(axis=1)
        med, q75 = np.percentile(m, (50, 75))
        # weak signals read compressed towards mid-grey, so stay generous
        self.eof_hist.append(med < -150.0 and q75 < 0.0)
        self.paper_hist.append(med > 0.0)
        h = self.eof_hist
        if len(h) < h.maxlen or not h[-1] or sum(h) < self.EOF_BLACK_FRAC * len(h):
            return False
        # only after a chart on white paper: a dark image (white on black,
        # satellite pictures) must not be cut off at its first dark patch
        before = list(self.paper_hist)[:-len(h)]
        if not before or sum(before) < self.EOF_PAPER_FRAC * len(before):
            return False
        # phasing is black too, but with a white pulse on every line: leave
        # that to the phasing split (and do not end a chart on its own phasing)
        if phasing_like(np.concatenate(self.fhist), self.fs):
            return False
        # trim back to just after the first black block of the window
        first = next(i for i, b in enumerate(h) if b)
        cut = len(h) - first - int(self.EOF_KEEP_S / self.BLOCK_S)
        self.rec_meta["chart_end_ts"] = time.time() - (len(h) - first) * self.BLOCK_S
        if cut > 0:
            self.rec = self.rec[:-cut]
        self._finish("end-of-frame black (no stop tone)")
        return True

    DEADLINE_KEEP_S = 6.0  # after the expected image end, when ending by deadline

    def _end_by_deadline(self):
        _, end_ts, what = self.deadline
        if end_ts is not None:
            # drop what came after the chart should have ended (idle carrier,
            # noise) but keep a few seconds in case it ran a little long
            extra = int((time.time() - end_ts - self.DEADLINE_KEEP_S) / self.BLOCK_S)
            if 0 < extra < len(self.rec) - int(60 / self.BLOCK_S):
                self.rec = self.rec[:-extra]
        self._finish(f"expected length ({what})")

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

    # ------------------------------------------------------------------
    # A chart already on air (start tone and phasing missed): a fax picture
    # repeats from line to line - borders, grid lines, isobars - so the
    # demodulated frequency correlates at exactly one line period, which
    # noise, an idle carrier and fading do not.
    LOOKBACK_S = 300.0    # how far back a chart found this way may be recorded from
    IMG_VOTES = 25        # 1 s checks considered ...
    IMG_NEED = 15         # ... and how many must look like a picture
    IMG_SCORE = 0.12      # line-period correlation above the off-period level
    IMG_QUIET_S = 60.0    # not straight after a recording (its own tail)
    GAP_S = 15.0          # this long without signal separates two transmissions

    def _image_now(self) -> bool:
        if self.block_count % 4 or len(self.look) < int(8 / self.BLOCK_S):
            return False
        if self.samples_seen / self.fs - self.idle_since < self.IMG_QUIET_S:
            return False
        f = np.concatenate([b[1] for b in list(self.look)[-int(8 / self.BLOCK_S):]])
        self.img_votes.append(line_score(f, self.fs) > self.IMG_SCORE)
        return sum(self.img_votes) >= self.IMG_NEED

    def _begin_from_lookback(self):
        look = list(self.look)
        # back to where the signal started: the end of the last gap of
        # GAP_S with (almost) no signal, then the usual pre-roll before it
        gap = int(self.GAP_S / self.BLOCK_S)
        start, quiet = 0, 0
        for i in range(len(look) - 1, -1, -1):
            quiet = 0 if look[i][2] else quiet + 1
            if quiet >= gap:
                start = i + quiet
                break
        start = max(0, start - self.pre.maxlen)
        seed = look[start:]
        back = len(seed) * self.BLOCK_S
        self.log(f"[{time.strftime('%H:%M:%S')}] picture on air with no start tone or phasing heard - "
                 f"recording from {back:.0f} s back")
        self._begin("picture", None, seed=[b[0] for b in seed], fseed=[b[1] for b in seed], back_s=back)

    def _begin(self, trigger: str, ioc, seed=None, fseed=None, back_s=0.0):
        self.state = "receiving"
        self._finish_req = None
        self.deadline = None
        self.npreview = None
        self.rec = list(seed) if seed is not None else list(self.pre)
        self.rec_n = sum(len(b) for b in self.rec)
        self.rec_start_abs = self.samples_seen - self.rec_n
        self.pre.clear()
        self.look.clear()
        self.img_votes.clear()
        self.stop_run = 0
        self.rec_snr = []
        self.stop_hist.clear()
        self.invalid_run = 0
        self.phase_hits = 0
        self.eof_hist = collections.deque(maxlen=int(self.EOF_WIN_S / self.BLOCK_S))
        self.paper_hist = collections.deque(maxlen=int((self.EOF_PAPER_S + self.EOF_WIN_S) / self.BLOCK_S))
        self._tail = None
        self.rec_meta = {"trigger": trigger, "ioc": ioc,
                         "utc": datetime.fromtimestamp(time.time() - back_s, timezone.utc)
                                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "t0_samples": self.samples_seen}
        if self.on_preview is not None:
            self.preview = LivePreview(self.fs)
            for fb in fseed if fseed is not None else (list(self.fbuf)[-len(self.rec):] if self.rec else []):
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

    PICTURE_SPLIT_S = 15.0    # earliest split in a recording started from the picture check
    TAIL_FRAGMENT_S = 180.0   # picture-triggered recording cut short by the next chart

    def retuned(self):
        """The receiver has just been retuned (called from the CAT thread):
        audio from before belongs to another frequency, so the picture check
        must not reach back into it, and waits a minute before looking."""
        if self.state == "idle":
            self.look.clear()
            self.img_votes.clear()
            self.idle_since = self.samples_seen / self.fs

    def _finish(self, reason: str):
        audio = np.concatenate(self.rec) if self.rec else np.zeros(0)
        meta = dict(self.rec_meta, end_reason=reason)
        if len(getattr(self, "rec_snr", [])) >= 5:
            meta["snr_db"] = round(float(np.median(self.rec_snr)), 1)   # while the chart was on air
        if self.timeline and len(audio):
            a0, a1 = self.rec_start_abs, self.rec_start_abs + len(audio)
            tl = [(i - a0, t, o) for i, t, o in self.timeline if a0 <= i <= a1]
            if len(tl) > 50:
                meta["timeline"] = tl
        self.state = "idle"
        self.deadline = None
        self.idle_since = self.samples_seen / self.fs
        self.look.clear()
        self.img_votes.clear()
        self.rec, self.rec_n = [], 0
        self._tail = None
        self.preview = None
        dur = len(audio) / self.fs
        self.log(f"[{time.strftime('%H:%M:%S')}] transmission ended ({reason}), {dur:.0f} s")
        if meta.get("trigger") == "picture" and reason in ("new start tone", "new phasing") \
                and dur < self.TAIL_FRAGMENT_S:
            # picked up the last part of the chart before (e.g. just after a
            # retune) and the next chart has begun: only a tail - not saved
            self.log("  only the tail of the previous chart - discarded")
        elif dur >= 60:
            self.on_recording(audio, self.fs, meta)
        else:
            self.log("  too short to be a fax - discarded")
        self.on_status(self.status())


def line_score(f: np.ndarray, fs: float, lpms=(120, 60, 90, 240)) -> float:
    """How strongly the demodulated frequency repeats at one line period,
    over and above its correlation at other lags (fades, drift, an idle
    carrier and noise score ~0; a chart 0.15-0.9)."""
    k = max(1, int(fs * 0.005))
    m = f[:len(f) // k * k].reshape(-1, k).mean(axis=1)
    m = m - np.convolve(m, np.ones(41) / 41, "same")     # drop slow fades / level drift

    def r(lag):
        a, b = m[:-lag], m[lag:]
        return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b) + 1e-9))

    best = -1.0
    for lpm in lpms:
        lag = int(round(60.0 / lpm / 0.005))
        if 3 * lag > len(m):
            continue
        on = max(r(l) for l in (lag - 1, lag, lag + 1))
        off = float(np.median([abs(r(int(lag * q))) for q in (0.61, 0.73, 0.87, 1.17, 1.31, 1.43)]))
        best = max(best, on - off)
    return best


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
