"""Offline WEFAX decode of a complete recorded transmission.

Pipeline
--------
1. Coarse demod at 1900 Hz -> global AFC from a black/white comb fit.
2. Re-demod at the corrected centre (zero-phase), then track slow frequency
   drift (rig drift / Doppler) in 20 s windows and remove it.
3. FM click and fade detection -> per-sample confidence; clicks are
   interpolated out before post-detection filtering.
4. APT tone segmentation (start 300/675 Hz -> IOC 576/288, stop 450 Hz).
5. Phasing analysis: per-line white-pulse tracking gives LPM, an accurate
   line period and the line-start phase.
6. Slant correction: global period refinement by maximising vertical-edge
   sharpness (column-profile energy), then a smooth per-block residual fit.
7. Band-limited resampling of every line onto pi*IOC pixels.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import signal, ndimage

from . import dsp

LPM_CANDIDATES = (120, 60, 90, 240)


# --------------------------------------------------------------------------
# AFC / drift
# --------------------------------------------------------------------------
def _rolling_median(x: np.ndarray, fs: float, win_s: float = 2.0) -> np.ndarray:
    """Cheap rolling median: block medians, linearly interpolated."""
    B = max(16, int(win_s * fs / 4))
    n = len(x)
    nb = max(1, n // B)
    meds = np.array([np.median(x[i * B:(i + 1) * B]) for i in range(nb)])
    if nb == 1:
        return np.full(n, meds[0], np.float32)
    meds = ndimage.median_filter(meds, size=5, mode="nearest")
    centres = (np.arange(nb) + 0.5) * B
    return np.interp(np.arange(n), centres, meds).astype(np.float32)


def track_drift(audio: np.ndarray, fs: float, centre: float,
                win_s: float = 20.0, hop_s: float = 10.0, search: float = 60.0):
    """Per-sample tuning offset (Hz, absolute) from windowed spectral comb fits."""
    n = len(audio)
    W, H = int(win_s * fs), int(hop_s * fs)
    centres, offs, scores = [], [], []
    for s in range(0, max(1, n - W // 2), H):
        seg = audio[s:s + W]
        if len(seg) < 2 * fs:
            continue
        o, sc = dsp.psd_comb(seg, fs, centre=centre, search=search)
        centres.append(s + len(seg) / 2)
        offs.append(o)
        scores.append(sc)
    if not offs:
        return np.full(n, centre, np.float32), 0.0
    offs, scores = np.array(offs), np.array(scores)
    ok = scores > max(6.0, 0.25 * np.max(scores))      # clear spectral line present
    if ok.sum() < 1:
        return np.full(n, centre, np.float32), 0.0
    c = np.array(centres)[ok]
    o = offs[ok]
    if len(o) >= 5:
        o = ndimage.median_filter(o, size=5, mode="nearest")
    if len(o) >= 3:
        o = np.convolve(np.pad(o, 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
    d = np.interp(np.arange(n), c, o).astype(np.float32)
    return d, float(o.max() - o.min())


def level_scale(fr: np.ndarray, fs: float, block_s: float = 8.0) -> np.ndarray:
    """Per-sample black/white level magnitude (Hz).

    Near FM threshold the discriminator output is compressed towards the
    centre frequency.  Track where the black and white histogram modes actually
    sit so contrast is restored to full scale.
    """
    n = len(fr)
    B = int(block_s * fs)
    edges = np.arange(-700, 701, 10.0)
    cen = 0.5 * (edges[:-1] + edges[1:])
    xs, ss = [], []
    for s in range(0, n, B):
        seg = fr[s:s + B]
        if len(seg) < fs:
            continue
        h, _ = np.histogram(seg, edges)
        h = np.convolve(h.astype(float), [0.25, 0.5, 0.25], "same")
        neg = (cen < -150) & (cen > -650)
        pos = (cen > 150) & (cen < 650)
        vals, wts = [], []
        tot = h.sum() + 1e-9
        for m in (neg, pos):
            i = np.argmax(np.where(m, h, -1))
            # prominence: mode must hold a decent share of samples
            if h[i] / tot > 0.012:
                vals.append(abs(cen[i]))
                wts.append(h[i])
        if vals:
            xs.append(s + len(seg) / 2)
            ss.append(np.average(vals, weights=wts))
    if not ss:
        return np.full(n, dsp.DEV_HZ, np.float32)
    ss = np.clip(np.array(ss), 250.0, 430.0)
    if len(ss) >= 5:
        ss = ndimage.median_filter(ss, size=5, mode="nearest")
    return np.interp(np.arange(n), xs, ss).astype(np.float32)


# --------------------------------------------------------------------------
# Tone segmentation
# --------------------------------------------------------------------------
def segment_tones(fr: np.ndarray, mag: np.ndarray, fs: float, block_s: float = 0.25):
    nb = int(block_s * fs)
    labels = []
    for i in range(len(fr) // nb):
        sl = slice(i * nb, (i + 1) * nb)
        labels.append(dsp.classify_block(dsp.block_features(fr[sl], mag[sl], fs)))
    labels = np.array(labels)

    def runs(prefix):
        m = np.array([l.startswith(prefix) for l in labels], bool)
        # close 1-block gaps
        m2 = m.copy()
        m2[1:-1] |= m[:-2] & m[2:]
        out, i = [], 0
        while i < len(m2):
            if m2[i]:
                j = i
                while j < len(m2) and m2[j]:
                    j += 1
                out.append((i, j))
                i = j
            else:
                i += 1
        return out

    res = {"start": None, "start_end": 0, "ioc": None, "stop": None, "labels": labels}
    st = [r for r in runs("start") if r[1] - r[0] >= 4 and r[0] * block_s < 90]
    if st:
        a, b = st[0]
        lab = labels[a:b]
        n576 = np.sum(lab == "start576")
        n288 = np.sum(lab == "start288")
        res["start"] = a * nb
        res["start_end"] = b * nb
        res["ioc"] = 576 if n576 >= n288 else 288
    min_stop = res["start_end"] + int(60 * fs)
    sp = [r for r in runs("stop") if r[1] - r[0] >= 4 and r[0] * nb > min_stop]
    if sp:
        res["stop"] = sp[0][0] * nb
    return res


# --------------------------------------------------------------------------
# Phasing
# --------------------------------------------------------------------------
def _box_circ(x: np.ndarray, w: int) -> np.ndarray:
    """Centred circular moving average along the last axis."""
    w = max(1, int(w))
    n = x.shape[-1]
    xx = np.concatenate([x[..., n - w:], x, x[..., :w]], axis=-1)
    c = np.cumsum(xx, axis=-1)
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), c], axis=-1)
    s = (c[..., w:] - c[..., :-w]) / w
    off = w // 2
    return s[..., w - off: w - off + n]


def find_phasing(v: np.ndarray, fs: float, s0: int, s1: int, lpms=LPM_CANDIDATES):
    """Locate the phasing run and fit pulse position = a + T*k.

    Returns dict(lpm, T, a, first, last, n) or None.  ``a`` is the absolute
    sample index of the white-pulse centre of line ``first``.
    """
    s1 = min(s1, len(v))
    # Rank LPM candidates by the true repetition period of the white pulses:
    # a 240 LPM window grid splits a 120 LPM phasing pulse into two halves
    # that look like valid pulses, but the stream itself only repeats at the
    # true period.  Shortest period with strong autocorrelation wins.
    seg = (v[s0:s1] > 0.5).astype(np.float32)
    # high-pass (remove phasing->image step) and keep only phasing-like stretches
    seg -= ndimage.uniform_filter1d(seg, int(2.2 * fs), mode="nearest")
    blk = ndimage.uniform_filter1d((v[s0:s1] < 0.4).astype(np.float32), int(2 * fs))
    seg *= (blk > 0.7)
    e0 = float(np.dot(seg, seg)) + 1e-9
    ranked = []
    for lpm in lpms:
        T = fs * 60.0 / lpm
        r = 0.0
        for lag in np.arange(int(T * 0.997), int(T * 1.003) + 1, max(1, int(T * 0.0005))):
            if lag * 3 < len(seg):
                r = max(r, float(np.dot(seg[:-lag], seg[lag:])) / e0)
        ranked.append((r, T, lpm))
    order = [l for r, T, l in sorted(ranked, key=lambda x: x[1]) if r > 0.4]
    order += [l for r, T, l in sorted(ranked, reverse=True) if l not in order]
    for lpm in order:
        T = fs * 60.0 / lpm
        L = int((s1 - s0) // T)
        if L < 8:
            continue
        Ti = int(T)
        starts = (s0 + np.arange(L) * T).astype(int)
        lines = np.stack([v[s:s + Ti] for s in starts])
        pw = max(3, int(0.05 * T))
        box = _box_circ(lines, pw)
        pos = np.argmax(box, axis=1)
        peak = box[np.arange(L), pos]
        black = np.mean(lines < 0.4, axis=1)
        med = np.median(lines, axis=1)
        isph = (black > 0.7) & (peak > 0.65) & (peak - med > 0.45)
        if isph.sum() < 8:
            continue
        # longest run of phasing lines (allow single-line gaps)
        m = isph.copy()
        m[1:-1] |= isph[:-2] & isph[2:]
        bi, bj, i = 0, 0, 0
        while i < L:
            if m[i]:
                j = i
                while j < L and m[j]:
                    j += 1
                if j - i > bj - bi:
                    bi, bj = i, j
                i = j
            else:
                i += 1
        if bj - bi < 8:
            continue
        ks = np.arange(bi, bj)
        ks = ks[isph[ks]]
        # absolute pulse centres, numbered by pulse index m
        absy = starts[ks] + pos[ks].astype(float)
        mi = np.round((absy - absy[0]) / T)
        keep = np.ones(len(ks), bool)
        for _ in range(4):
            A = np.vstack([np.ones(keep.sum()), mi[keep]]).T
            coef, *_ = np.linalg.lstsq(A, absy[keep], rcond=None)
            r = absy - (coef[0] + coef[1] * mi)
            mad = np.median(np.abs(r[keep])) + 0.5
            keep = np.abs(r) < 4 * mad
            if keep.sum() < 6:
                break
        if keep.sum() < 6 or keep.sum() < 0.6 * (bj - bi):
            continue      # e.g. 240 LPM candidate on a 120 LPM signal: half the windows empty
        a0, Tfit = coef
        cand = {"lpm": lpm, "T": float(Tfit), "a": float(a0),
                "first": int(bi), "last": int(bj), "n": int(keep.sum()),
                "s0": s0, "Tnom": T,
                "end": float(a0 + Tfit * (mi[keep].max() + 1))}
        # sanity: fitted period within 0.2% of nominal
        if abs(Tfit / T - 1) > 2e-3:
            continue
        return cand
    return None


# --------------------------------------------------------------------------
# Slant (sample-clock) correction
# --------------------------------------------------------------------------
def _gather(v: np.ndarray, pos: np.ndarray) -> np.ndarray:
    i = np.floor(pos).astype(np.int64)
    np.clip(i, 0, len(v) - 2, out=i)
    fr = pos - i
    return v[i] * (1 - fr) + v[i + 1] * fr


def _sharpness(v, s_first, T, rows, Wr, smooth):
    pos = s_first + rows[:, None] * T + np.arange(Wr)[None, :] * (T / Wr)
    c = _gather(v, pos).mean(axis=0)
    if smooth:
        c = ndimage.gaussian_filter1d(c, smooth, mode="wrap")
    d = np.diff(np.concatenate([c, c[:1]]))
    return float(np.sum(d * d))


def refine_period(v, s_first, T0, K, ppm_range, log=None):
    """Find the line period that makes vertical structure sharpest."""
    if K < 40:
        return T0, 0.0
    rows_c = np.arange(0, K, 2)
    Wr = 256
    step_ppm = 2.0 / (Wr * K) * 1e6 * 2      # ~1 px total drift per step (smoothed)
    step_ppm = max(step_ppm, 0.5)
    ppms = np.arange(-ppm_range, ppm_range + step_ppm, step_ppm)
    J = np.array([_sharpness(v, s_first, T0 * (1 + p * 1e-6), rows_c, Wr, 1.5) for p in ppms])
    i = int(np.argmax(J))
    contrast = J[i] / (np.median(J) + 1e-12)
    # fine golden-section search at full row density
    rows = np.arange(K)
    Wf = 1024
    lo, hi = ppms[max(0, i - 2)], ppms[min(len(ppms) - 1, i + 2)]
    g = (math.sqrt(5) - 1) / 2
    f = lambda p: -_sharpness(v, s_first, T0 * (1 + p * 1e-6), rows, Wf, 0)
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(18):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    p = (a + b) / 2
    return T0 * (1 + p * 1e-6), float(contrast)


def residual_slant(v, s_first, T, K, W, block=160):
    """Smooth per-line start correction from block-vs-global profile shifts."""
    if K < 3 * block:
        return np.zeros(K)
    Wp = 2048
    rows = np.arange(K)
    pos = s_first + rows[:, None] * T + np.arange(Wp)[None, :] * (T / Wp)
    img = _gather(v, pos)
    g = img.mean(0)
    g = np.diff(np.concatenate([g, g[:1]]))
    G = np.fft.rfft(g)
    xs, ys, ws = [], [], []
    for b0 in range(0, K - block // 2, block // 2):
        blk = img[b0:b0 + block]
        if len(blk) < block // 2:
            continue
        p = blk.mean(0)
        p = np.diff(np.concatenate([p, p[:1]]))
        xc = np.fft.irfft(np.fft.rfft(p) * np.conj(G), Wp)
        k = int(np.argmax(xc))
        # sub-sample parabolic peak
        y0, y1, y2 = xc[k - 1], xc[k], xc[(k + 1) % Wp]
        den = (y0 - 2 * y1 + y2)
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        sh = k + frac
        if sh > Wp / 2:
            sh -= Wp
        conf = y1 / (np.sqrt(np.dot(p, p) * np.dot(g, g)) + 1e-12)
        xs.append(b0 + len(blk) / 2)
        ys.append(sh * T / Wp)
        ws.append(max(conf, 0))
    xs, ys, ws = map(np.array, (xs, ys, ws))
    ok = ws > 0.3
    if ok.sum() < 4:
        return np.zeros(K)
    # quadratic fit (thermal drift of the soundcard clock is smooth)
    deg = 2 if ok.sum() >= 6 else 1
    cf = np.polyfit(xs[ok], ys[ok], deg, w=ws[ok])
    corr = np.polyval(cf, rows)
    # cap at 2 % of a line - anything bigger is not clock drift
    if np.max(np.abs(corr)) > 0.02 * T:
        return np.zeros(K)
    return corr


def find_margin(img: np.ndarray) -> int:
    """Column index of the quietest band (the margin) for images w/o phasing."""
    act = img.std(axis=0)
    W = img.shape[1]
    act = ndimage.uniform_filter1d(act, max(3, int(0.03 * W)), mode="wrap")
    return int(np.argmin(act))


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# flywheel: repair lost audio samples from the PC-clock arrival timeline
GAP_MIN_S = 0.003       # smallest loss acted on (~11 px at 120 LPM / IOC 576)
GAP_WIN_S = 4.0         # lower-envelope window either side of a step
GAP_HOLD_S = 12.0       # the step must persist this long (real losses never recover)


def _lower_env(x, w):
    from scipy.ndimage import minimum_filter1d
    return minimum_filter1d(x, size=max(1, w), mode="nearest")


def fill_gaps(audio: np.ndarray, fs: float, timeline, log=print):
    """Insert silence where samples were lost between the sound card and us.

    timeline: [(sample index at end of buffer, PC time s, overflow flag), ...]
    A buffer can be delivered late but never early, so the *lower envelope*
    of (arrival time - index/fs) is the true clock; scheduling jitter only
    adds positive spikes.  Lost samples raise that envelope by a step equal
    to the lost duration.  Returns (audio, gaps) with gaps = [(index, n)],
    index in the *output* audio.
    """
    if timeline is None or len(timeline) < 50:
        return audio, []
    tl = np.asarray([(i, t) for i, t, *_ in timeline], np.float64)
    ovf = np.asarray([bool(e[2]) if len(e) > 2 else False for e in timeline])
    idx, t = tl[:, 0], tl[:, 1]
    keep = np.r_[True, np.diff(idx) > 0]
    idx, t, ovf = idx[keep], t[keep], ovf[keep]
    if len(idx) < 50 or idx[-1] - idx[0] < 30 * fs:
        return audio, []
    r = t - idx / fs
    dt_blk = float(np.median(np.diff(t)))
    w = int(round(GAP_WIN_S / max(dt_blk, 1e-3)))
    hold = int(round(GAP_HOLD_S / max(dt_blk, 1e-3)))
    # remove the sound-card vs PC clock rate difference: median of local
    # slopes over 20 s chunks, so the chunks holding a step are outliers
    env = _lower_env(r, w)
    cn = max(2, int(round(20.0 / max(dt_blk, 1e-3))))
    ch = np.arange(0, len(r) - cn, cn)
    if len(ch) >= 3:
        slope = float(np.median((env[ch + cn] - env[ch]) / (idx[ch + cn] - idx[ch])))
    else:
        slope = float((env[-1] - env[0]) / (idx[-1] - idx[0]))
    r = r - slope * idx
    # noise of the envelope itself sets the threshold
    env = _lower_env(r, w)
    noise = float(np.median(np.abs(np.diff(env[::max(1, w // 2)])))) * 1.4826
    thr = max(GAP_MIN_S, 6 * noise)
    log(f"flywheel: {len(r)} buffers, clock {slope * fs * 1e6:+.0f} ppm vs PC, threshold {thr * 1000:.1f} ms")

    gaps = []
    k = w
    n_tl = len(r)
    while k < n_tl - w:
        before = r[max(0, k - 3 * w):k].min()
        after = r[k:min(n_tl, k + hold)].min()
        step = after - before
        # the sound card's own overflow flag nearby: accept a smaller step
        flagged = ovf[max(0, k - w):k + w].any()
        if step > (max(0.0015, 3 * noise) if flagged else thr) and k + hold // 2 <= n_tl:
            # exact buffer: the first one from which every arrival is late by >= 70 % of the step
            lo = max(1, k - w)
            cand = [j for j in range(lo, min(n_tl, k + w))
                    if r[j:min(n_tl, j + hold)].min() - before > 0.7 * step]
            j = cand[0] if cand else k
            # an overflow flag nearby pins the location exactly
            near = np.flatnonzero(ovf[max(1, j - 3):j + 3])
            if len(near):
                j = max(1, j - 3) + int(near[0])
            gaps.append((int(idx[j - 1]), step / max(1.0 + slope * fs, 0.05), bool(flagged)))
            r[j:] -= step                     # treat the rest as re-synchronised
            k = j + w
            continue
        k += max(1, w // 4)
    unflagged = sum(1 for g in gaps if not g[2])
    if unflagged > 3:
        # a real sound card clock is steady; many unexplained steps mean the
        # timestamps themselves are wandering - trust only flagged losses
        log(f"flywheel: timing too unsteady ({unflagged} unflagged steps) - ignored")
        gaps = [g for g in gaps if g[2]]
    if not gaps:
        return audio, []
    out, done, last, shift = [], [], 0, 0
    for pos, step, _ in gaps:
        n_ins = int(round(step * fs))
        pos = min(max(pos, 0), len(audio))
        out.append(audio[last:pos])
        out.append(np.zeros(n_ins, audio.dtype))
        done.append((pos + shift, n_ins))
        shift += n_ins
        last = pos
        log(f"flywheel: {step * 1000:.1f} ms of audio lost at {pos / fs:.1f} s - filled")
    out.append(audio[last:])
    return np.concatenate(out), done


def trailing_signal_end(audio, fs, s_end, log=print, win_s=10.0, thresh=6.0):
    """No stop tone: cut off trailing noise (fade-out, or the receiver retuned).
    Walks back from the end in 10 s windows until the fax tones (1500/2300 Hz
    spectral lines) are clearly present again."""
    w = int(win_s * fs)
    e = s_end
    while e - w > 0:
        _, sc = dsp.psd_comb(audio[e - w:e], fs, search=300.0)
        if sc > thresh:
            break
        e -= w // 2
    if e < s_end - w // 2:
        log(f"trailing noise: {(s_end - e) / fs:.0f} s removed from the end")
        return e
    return s_end


def trim_edges(img, cimg, max_top=40, max_bot=12):
    """Remove phasing lines left at the top and stop-tone lines at the bottom.

    Phasing line: black across the picture with the white phasing pulse at
    the line edges (the decoder puts the line start at the pulse centre).
    Stop tone (450 Hz) shows as a fine regular black/white stripe pattern.
    """
    K, W = img.shape
    centre = img[:, int(0.06 * W):int(0.94 * W)]
    dark = (centre < 0.4).mean(1)
    # the phasing pulse is 5 % of a line, centred on the line start: a white
    # run of ~2.5 % of the width at the left and/or right edge (45 px of 1810).
    # A chart's white margin is a different width, so frame lines stay.
    pw = 0.025 * W

    def edge_run(row, rev):
        b = row[::-1] if rev else row
        w = np.argmin(b > 0.5) if not (b > 0.5).all() else len(b)
        return w

    def pulsed(i):
        return dark[i] >= 0.8 and any(0.75 * pw <= edge_run(img[i], r) <= 1.25 * pw for r in (False, True))

    top = 0
    for i in range(min(max_top, K // 4)):
        if pulsed(i):
            top = i + 1
        elif dark[i] < 0.5 and i - top > 1:
            break
    # stop tone: many black/white transitions along the line, mid-grey on average
    trans = (np.abs(np.diff((centre > 0.5).astype(np.int8), axis=1)).sum(1) / centre.shape[1])
    bot = 0
    for j in range(K - 1, max(K - 1 - max_bot, top), -1):
        if trans[j] > 0.15 and 0.25 < centre[j].mean() < 0.75:
            bot = K - j
        else:
            break
    return img[top:K - bot], cimg[top:K - bot], top, bot


def decode_recording(audio: np.ndarray, fs: float, ioc=None, lpm=None,
                     align: str = "center", log=print, timeline=None) -> dict:
    audio = np.asarray(audio, np.float64)
    audio, gaps = fill_gaps(audio, fs, timeline, log)
    audio = audio - audio.mean()
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-12
    audio = audio / rms
    n = len(audio)
    log(f"decode: {n / fs:.1f} s of audio at {fs:.0f} Hz")

    # 1-2. AFC: global spectral comb, then 20 s windowed drift track
    off, sc = dsp.psd_comb(audio, fs, search=300.0)
    drift, drift_pp = track_drift(audio, fs, off)
    afc_hz = float(np.median(drift))
    log(f"AFC: offset {afc_hz:+.1f} Hz (score {sc:.0f}), drift p-p {drift_pp:.1f} Hz")
    fr, strength = dsp.demod(audio, fs, dsp.CENTRE_HZ, drift=drift, post_cut=1100.0)

    # 3. confidence from discriminator strength; clicks interpolated out
    smed = _rolling_median(strength, fs)
    conf = np.clip(strength / (smed + 1e-12), 0, 1).astype(np.float32)
    # QRN crashes are *loud*: flag over-amplitude bursts as well as fades
    loud = strength > 2.2 * smed
    loud = ndimage.binary_opening(loud, structure=np.ones(max(3, int(0.002 * fs)), bool))
    crash = ndimage.binary_dilation(loud, iterations=int(0.003 * fs))
    click = (np.abs(fr) > 620.0) | (conf < 0.15) | crash
    click = ndimage.binary_dilation(click, iterations=2)
    conf[click] = 0.0
    if click.any() and (~click).sum() > 10:
        idx = np.flatnonzero(~click)
        bi = np.flatnonzero(click)
        fr = fr.copy()
        fr[bi] = np.interp(bi, idx, fr[idx])
    click_rate = float(click.mean())
    mag = strength
    lvl = level_scale(fr, fs)
    v_coarse = (fr / lvl + 1.0) * 0.5

    # 4. tones
    tones = segment_tones(fr, mag, fs)
    ioc_found = tones["ioc"]
    if ioc is None:
        ioc = ioc_found or 576
    W = int(round(math.pi * ioc))
    s_img0 = tones["start_end"] if tones["start"] is not None else 0
    s_end = tones["stop"] if tones["stop"] is not None else n
    if tones["stop"] is None:
        s_end = trailing_signal_end(audio, fs, s_end, log)
    log(f"tones: start={'yes' if tones['start'] is not None else 'no'} "
        f"(IOC {ioc_found}), stop={'yes' if tones['stop'] is not None else 'no'}")

    # 5. phasing
    vs = signal.sosfiltfilt(dsp.lpf_sos(fs, 800.0, 4), v_coarse)
    lpms = (lpm,) if lpm else LPM_CANDIDATES
    ph = find_phasing(vs, fs, s_img0, min(n, s_img0 + int(75 * fs)), lpms)
    if ph:
        lpm = ph["lpm"]
        T = ph["T"]
        log(f"phasing: {ph['n']} lines, LPM {lpm}, period {T:.3f} smp "
            f"({(T / ph['Tnom'] - 1) * 1e6:+.0f} ppm)")
    else:
        lpm = lpm or 120
        T = fs * 60.0 / lpm
        log(f"phasing: not found, assuming {lpm} LPM")
    Tnom = fs * 60.0 / lpm

    # final post-detection filter, matched to pixel Nyquist
    px_rate = W * lpm / 60.0
    post_cut = min(1100.0, 0.45 * px_rate)
    if post_cut < 1000.0:
        v = signal.sosfiltfilt(dsp.lpf_sos(fs, post_cut, 4), v_coarse).astype(np.float32)
    else:
        v = v_coarse.astype(np.float32)

    # image start / line phase
    if ph:
        a = ph["a"]
        if align == "start":
            a -= 0.025 * T
        elif align == "end":
            a += 0.025 * T
        # start a couple of lines early so nothing of the chart is lost; the
        # leftover phasing lines are trimmed from the picture afterwards
        k_end = int(math.ceil((ph["end"] - a) / T)) - 2
        s_first = a + k_end * T
    else:
        s_first = float(s_img0)
    K = int((s_end - s_first) // T) + 1     # trailing stop-tone rows trimmed later
    if K < 10:
        raise ValueError("too few image lines decoded")

    # 6. slant
    # few phasing lines (fade during phasing) -> period estimate is rough: search wider
    rng = 60.0 if (ph and ph["n"] >= 30) else (300.0 if ph else 700.0)
    T2, contrast = refine_period(v, s_first, T, K, rng)
    if contrast > 1.15 or not ph:
        T = T2
    corr = residual_slant(v, s_first, T, K, W)
    starts = s_first + np.arange(K) * T + corr
    if not ph:
        # put the quietest band (margin) at the edges
        pos = starts[::4, None] + np.arange(W)[None, :] * (T / W)
        m = find_margin(_gather(v, pos))
        starts = starts + m * T / W
    K = int(np.sum(starts + T < n))
    starts = starts[:K]
    clock_ppm = (T / Tnom - 1) * 1e6
    log(f"slant: period {T:.3f} smp, clock {clock_ppm:+.1f} ppm, "
        f"residual {np.ptp(corr):.1f} smp, {K} lines")

    # 7. resample lines
    pos = starts[:, None] + np.arange(W)[None, :] * (T / W)
    img = ndimage.map_coordinates(v, [pos.ravel()], order=3, mode="nearest").reshape(K, W)
    cimg = ndimage.map_coordinates(conf, [pos.ravel()], order=1, mode="nearest").reshape(K, W)
    img = np.clip(img, 0, 1).astype(np.float32)
    cimg = np.clip(cimg, 0, 1).astype(np.float32)

    # 8. tears: sudden sideways jumps from lost / extra audio samples
    from .realign import find_tears, apply_tears
    img, cimg, top, bot = trim_edges(img, cimg)
    if top or bot:
        log(f"edges: {top} phasing line(s) removed at the top, {bot} stop-tone/blank line(s) at the bottom")
    K = img.shape[0]
    hint_rows = [int(round((p - s_first) / T)) - top for p, _ in gaps]
    hint_rows = [h for h in hint_rows if 0 < h < K - 1]
    tears = find_tears(img, hints=hint_rows)
    if tears:
        img, cimg = apply_tears(img, tears), apply_tears(cimg, tears)
        log("tears corrected: " + ", ".join(f"line {r} ({s:+d} px)" for r, s in tears))

    quality = float(np.mean(cimg > 0.5))
    return {
        "image": img, "conf": cimg,
        "meta": {
            "ioc": int(ioc), "lpm": int(lpm), "width": W, "lines": int(K),
            "afc_hz": round(afc_hz, 1), "drift_pp_hz": round(drift_pp, 1),
            "clock_ppm": round(clock_ppm, 1), "phasing": bool(ph),
            "start_tone": tones["start"] is not None, "stop_tone": tones["stop"] is not None,
            "tears": [[int(r), int(s)] for r, s in tears],
            "gaps_filled_ms": [round(n_ / fs * 1000, 1) for _, n_ in gaps],
            "click_rate": round(click_rate, 4), "quality": round(quality, 3),
            "duration_s": round(n / fs, 1),
            "afc_score": round(float(sc), 1),
        },
    }
