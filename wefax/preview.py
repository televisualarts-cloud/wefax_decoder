"""Live 'building up' preview of the fax currently being received.

Uses the live detector's causal discriminator output, so it is a quick,
uncleaned raw view: no noise removal and only coarse geometry.  Once enough
phasing lines have arrived the line period (sound-card clock error, i.e.
slant) and line start phase are locked from them, and the preview is redrawn
straight.  The full decoder still produces the final image afterwards.
"""
from __future__ import annotations

import math

import numpy as np

from . import decode, dsp

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DECIM = 4            # 12 kHz -> 3 kHz is plenty for a 600 px preview
WIDTH = 600          # preview pixels per line
FAX_WIDTH = 1810     # IOC 576 line width, sets the preview aspect ratio


class LivePreview:
    def __init__(self, fs: float, lpm: int = 120):
        self.fs = fs / DECIM
        self.buf = np.zeros(int(self.fs * 120), np.float32)
        self.n = 0
        self._carry = np.zeros(0, np.float32)
        self.lpm = lpm
        self.T = self.fs * 60.0 / lpm
        self.s0 = 0.0
        self.locked = False
        self.offset = 0.0
        self.rows: list[np.ndarray] = []
        self._last_lock_try = 0

    # ------------------------------------------------------------------
    def add(self, f: np.ndarray):
        """Append discriminator output (Hz relative to 1900) at the full rate."""
        x = np.concatenate([self._carry, np.asarray(f, np.float32)])
        m = len(x) // DECIM
        self._carry = x[m * DECIM:]
        if m == 0:
            return
        y = x[:m * DECIM].reshape(m, DECIM).mean(axis=1)
        if self.n + m > len(self.buf):
            nb = np.zeros(max(2 * len(self.buf), self.n + m), np.float32)
            nb[:self.n] = self.buf[:self.n]
            self.buf = nb
        self.buf[self.n:self.n + m] = y
        self.n += m

    def set_offset(self, hz: float):
        if abs(hz - self.offset) > 8.0:
            self.offset = hz
            self.rows = []          # levels changed: redraw

    def _v(self, a, b):
        return np.clip((self.buf[a:b] - self.offset) / (2 * dsp.DEV_HZ) + 0.5, 0, 1)

    def _try_lock(self):
        secs = self.n / self.fs
        if self.locked or secs < 25 or secs > 150 or self.n - self._last_lock_try < 5 * self.fs:
            return
        self._last_lock_try = self.n
        v = self._v(0, self.n)
        ph = decode.find_phasing(v, self.fs, 0, self.n)
        if ph and ph["n"] >= 25:
            self.lpm = ph["lpm"]
            self.T = ph["T"]
            # pulse centre = line start (the decoder's default alignment)
            self.s0 = ph["a"] - math.floor(ph["a"] / self.T) * self.T
            self.locked = True
            self.rows = []

    # ------------------------------------------------------------------
    def render(self, max_rows: int | None = None):
        """Return (png_bytes, info) or (None, info) if nothing to show yet.
        max_rows: only draw the most recent rows (scrolling noise view)."""
        self._try_lock()
        T = self.T
        k = len(self.rows)
        K = int((self.n - self.s0) // T)
        if K > k:
            a = int(math.floor(self.s0 + k * T))
            b = min(self.n, int(math.ceil(self.s0 + K * T)) + 1)
            v = self._v(a, b).astype(np.float64)
            cs = np.concatenate([[0.0], np.cumsum(v)])
            idx = np.arange(len(cs)) + a
            ks = np.arange(k, K)
            edges = self.s0 + ks[:, None] * T + np.arange(WIDTH + 1)[None, :] * (T / WIDTH)
            c = np.interp(edges, idx, cs)
            px = np.diff(c, axis=1) / (T / WIDTH)     # box average over each pixel
            self.rows.extend((np.clip(px, 0, 1) * 255 + 0.5).astype(np.uint8))
        info = {"lines": len(self.rows), "lpm": self.lpm, "locked": self.locked,
                "offset_hz": round(self.offset, 1)}
        if not self.rows or cv2 is None:
            return None, info
        img = np.stack(self.rows[-max_rows:] if max_rows else self.rows)
        h = max(1, int(round(len(img) * WIDTH / FAX_WIDTH)))
        img = cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", img)
        return (png.tobytes() if ok else None), info
