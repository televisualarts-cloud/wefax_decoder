"""Core DSP building blocks shared by the live detector and the offline decoder.

Signal model (ITU-R F.343 / WMO, receiver in USB with dial 1.9 kHz below the
assigned carrier):  black = 1500 Hz, white = 2300 Hz, centre = 1900 Hz,
deviation +/-400 Hz.  All frequencies returned by the demodulators here are
*relative to the mixing frequency* (normally 1900 Hz), so black is about
-400 Hz and white about +400 Hz.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

BLACK_HZ = 1500.0
WHITE_HZ = 2300.0
CENTRE_HZ = 1900.0
DEV_HZ = 400.0
IQ_CUTOFF_HZ = 1000.0          # complex baseband low-pass (passes +/-400 Hz + video)

START_TONES = {300.0: 576, 675.0: 288}   # APT start tone (Hz) -> IOC
STOP_TONE = 450.0
PEAK_THRESH = 70.0      # tone line / spectral floor in the demodulated signal


def lpf_sos(fs: float, cutoff: float, order: int = 6):
    return signal.butter(order, cutoff, fs=fs, output="sos")


def demod(audio: np.ndarray, fs: float, fc: float = CENTRE_HZ,
          drift: np.ndarray | None = None, post_cut: float = 1100.0,
          cutoff: float = IQ_CUTOFF_HZ, chunk_s: float = 60.0, pad_s: float = 0.5):
    """Zero-phase correlation FM discriminator.

    Mixes the audio down by ``fc`` (+ optional per-sample ``drift`` in Hz, so
    AFC tracking is applied before demodulation), low-passes I/Q with a
    forward-backward Butterworth (no group delay -> no horizontal smear), then
    forms d[n] = z[n] z*[n-1] and low-passes *that complex product* before
    taking its angle.  Averaging the product rather than the angle weights each
    sample by its power, so noise-dominated samples during fades and QRN barely
    contribute: far fewer FM 'clicks' than a plain phase-difference detector.

    Returns (freq_rel_Hz float32, strength float32) where strength = |filtered d|.
    """
    audio = np.asarray(audio)
    n = len(audio)
    sos = lpf_sos(fs, cutoff)
    sos_p = lpf_sos(fs, post_cut, 4)
    freq = np.empty(n, np.float32)
    strength = np.empty(n, np.float32)
    C = int(chunk_s * fs)
    P = int(pad_s * fs)
    if drift is None:
        phase = None
        w = 2.0 * np.pi * fc / fs
    else:
        phase = np.cumsum(2.0 * np.pi * (fc + np.asarray(drift, np.float64)) / fs)
    for s in range(0, n, C):
        a = max(0, s - P)
        b = min(n, s + C + P)
        if phase is None:
            ph = w * np.arange(a, b, dtype=np.float64)
        else:
            ph = phase[a:b]
        z = audio[a:b].astype(np.float64) * np.exp(-1j * ph)
        z = signal.sosfiltfilt(sos, z)
        d = z[1:] * np.conj(z[:-1])
        d = np.concatenate([d[:1], d])
        d = signal.sosfiltfilt(sos_p, d)
        e = min(n, s + C)
        freq[s:e] = (np.angle(d[s - a:e - a]) * fs / (2.0 * np.pi))
        strength[s:e] = np.abs(d[s - a:e - a])
    return freq, strength


def psd_comb(audio: np.ndarray, fs: float, centre: float = 0.0, search: float = 300.0):
    """Unbiased tuning-offset estimate from the audio power spectrum.

    WEFAX spends most of its time on black (1500 Hz) or white (2300 Hz), which
    show as spectral lines.  White noise is flat, so unlike a frequency
    histogram this does not get pulled towards 1900 Hz at low SNR.
    Returns (offset_Hz, score) where score ~ peak-to-floor ratio.
    """
    if len(audio) < 4096:
        return centre, 0.0
    f, P = signal.welch(audio, fs, nperseg=4096, noverlap=2048, detrend="constant")
    band = (f > 1000) & (f < 2800)
    base = np.median(P[band]) + 1e-20
    Pn = np.convolve(P / base, [0.25, 0.5, 0.25], "same")
    os_ = np.arange(centre - search, centre + search + 0.25, 0.25)
    score = np.interp(BLACK_HZ + os_, f, Pn) + np.interp(WHITE_HZ + os_, f, Pn)
    i = int(np.argmax(score))
    return float(os_[i]), float(score[i])


class StreamDemod:
    """Causal (stateful) correlation discriminator for live detection.

    Same idea as :func:`demod`: the product z[n] z*[n-1] is low-passed before
    its angle is taken, so noise-dominated samples carry little weight and
    the output is far less 'clicky' on weak signals.
    """

    def __init__(self, fs: float, fc: float = CENTRE_HZ, cutoff: float = IQ_CUTOFF_HZ,
                 post_cut: float = 1100.0):
        self.fs = fs
        self.sos = lpf_sos(fs, cutoff, 4)
        self.zi = np.zeros((self.sos.shape[0], 2), complex)
        self.sos_p = lpf_sos(fs, post_cut, 4)
        self.zi_p = np.zeros((self.sos_p.shape[0], 2), complex)
        self.w = 2.0 * np.pi * fc / fs
        self.ph = 0.0
        self.last = 1.0 + 0j

    def process(self, x: np.ndarray):
        n = len(x)
        ph = self.ph + self.w * np.arange(n)
        self.ph = (self.ph + self.w * n) % (2.0 * np.pi)
        z = x * np.exp(-1j * ph)
        z, self.zi = signal.sosfilt(self.sos, z, zi=self.zi)
        zz = np.concatenate([[self.last], z])
        self.last = z[-1]
        d = zz[1:] * np.conj(zz[:-1])
        d, self.zi_p = signal.sosfilt(self.sos_p, d, zi=self.zi_p)
        f = np.angle(d) * self.fs / (2.0 * np.pi)
        return f.astype(np.float32), np.abs(z).astype(np.float32)


class Decimator:
    """Stateful integer-ratio decimator to an internal rate of ~11-12 kHz."""

    def __init__(self, sr: float):
        self.k = max(1, int(sr // 11000))
        self.fs = sr / self.k
        self.off = 0
        if self.k > 1:
            self.b = signal.firwin(48 * self.k + 1, 0.42 * self.fs, fs=sr)
            self.zi = np.zeros(len(self.b) - 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if self.k == 1:
            return x
        y, self.zi = signal.lfilter(self.b, 1.0, x, zi=self.zi)
        out = y[self.off::self.k]
        self.off = (self.off - len(x)) % self.k
        return out


def tone_fraction(x: np.ndarray, fs: float, f: float) -> float:
    """Fraction of the AC energy of ``x`` at frequency ``f`` (1.0 = pure sine).

    Applied to the *demodulated frequency* signal, a 300 Hz black/white square
    wave gives ~0.81.
    """
    x = x - x.mean()
    n = len(x)
    e = float(np.dot(x, x)) + 1e-12
    k = np.exp(-2j * np.pi * f * np.arange(n) / fs)
    X = np.dot(x, k)
    return float(2.0 * abs(X) ** 2 / (n * e))


def block_features(f: np.ndarray, mag: np.ndarray, fs: float, offset: float = 0.0) -> dict:
    fr = f - offset
    feats = {
        "std": float(np.std(fr)),
        "inrange": float(np.mean(np.abs(fr) < 520.0)),
        "envcv": float(np.std(mag) / (np.mean(mag) + 1e-12)),
        "mean": float(np.mean(fr)),
    }
    for tf in list(START_TONES) + [STOP_TONE]:
        feats[tf] = tone_fraction(fr, fs, tf)
    # spectral-line test: a keyed tone is a sharp line in the demodulated
    # spectrum, noise is broadband, so this survives far worse SNR
    x = (fr - fr.mean()) * np.hanning(len(fr))
    P = np.abs(np.fft.rfft(x)) ** 2
    fq = np.fft.rfftfreq(len(fr), 1.0 / fs)
    floor = np.median(P[(fq > 60) & (fq < 1500)]) + 1e-12
    df = fq[1] - fq[0]
    for tf in list(START_TONES) + [STOP_TONE]:
        k = int(round(tf / df))
        feats["pk%d" % tf] = float(P[max(0, k - 1):k + 2].max() / floor)
    return feats


def classify_block(feats: dict, thresh: float = 0.42) -> str:
    """'start576' | 'start288' | 'stop' | 'signal' | 'noise'."""
    tones = [300.0, 675.0, STOP_TONE]
    best = None
    if feats["std"] > 180.0 and feats["inrange"] > 0.6:
        t = max(tones, key=lambda t: feats[t])
        if feats[t] > thresh:
            best = t
    if best is None and "pk300" in feats:
        t = max(tones, key=lambda t: feats["pk%d" % t])
        others = [feats["pk%d" % o] for o in tones if o != t]
        if feats["pk%d" % t] > PEAK_THRESH and feats["pk%d" % t] > 4 * max(others):
            best = t
    if best is not None:
        if best == STOP_TONE:
            return "stop"
        return "start%d" % START_TONES[best]
    if feats["inrange"] > 0.86 or (feats["envcv"] < 0.3 and feats["inrange"] > 0.7):
        return "signal"
    return "noise"


def band_ratio(x: np.ndarray, fs: float) -> float:
    """Signal-presence measure robust to fading and low SNR.

    WEFAX energy sits between black (1500 Hz) and white (2300 Hz).  Compare
    the mean power density there with that of the noise either side of it
    (900-1300 and 2550-2900 Hz).  Pure noise gives ~1; a fax signal well
    below readable SNR still gives clearly more.
    """
    n = len(x)
    if n < 256:
        return 1.0
    X = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    sig = X[(f >= 1440) & (f <= 2360)].mean()
    ref = np.concatenate([X[(f >= 900) & (f <= 1300)], X[(f >= 2550) & (f <= 2900)]])
    return float(sig / (np.median(ref) / 0.693 + 1e-20))
