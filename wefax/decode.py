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
        # the stop tone ends the chart, so it is the last such run - and a real
        # one lasts ~5 s.  Fine hatching in a chart (WAFC SIGWX cloud areas)
        # can look like the stop tone for a second or two mid-picture; taking
        # the first run cut such charts off after a few dozen lines.
        long_ = [r for r in sp if r[1] - r[0] >= 8]
        res["stop"] = (long_ or sp)[-1][0] * nb
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
                "end": float(a0 + Tfit * (mi[keep].max() + 1)),
                "pulses": absy[keep], "pulse_idx": mi[keep]}
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


SEAM_MIN_CONF = 4.5   # line-start seam this distinct (std devs) is trusted
SEAM_ALONE_CONF = 8.0  # ... and this distinct to be trusted with nothing else to confirm it
REF_MIN_RATIO = 8.0    # frame match with the last good copy: peak this far above the rest is trusted
REF_ALONE_RATIO = 12.0  # ... and this far to be trusted on its own
REF_MIN_R = 0.5         # ... with the column profiles at least this alike (0.7 on its own)
REF_ALONE_R = 0.7
AGREE_PX = 40          # two line-start estimates within this many pixels agree (2 % of the line:
                       # stations differ a little in where the picture starts after the phasing
                       # pulse; the misframes this guards against are hundreds of pixels)


def reference_shift(pic: np.ndarray, ref: np.ndarray):
    """Where the frame of this picture lies compared with the last good copy
    of the same chart (same station and chart: same frame, grid, title box).
    Returns (m, ratio, r): the line start is m columns to the right of the
    current one; ratio = how far the match stands above every other offset;
    r = how alike the two column profiles are at that offset."""
    if ref is None or ref.shape[1] != pic.shape[1]:
        return None, 0.0, 0.0
    W = pic.shape[1]
    prof = lambda im: ndimage.uniform_filter1d(1.0 - im, 1, axis=0).mean(0)
    pp, pr = prof(pic), prof(ref)
    hp = lambda x: x - ndimage.uniform_filter1d(x, 31, mode="wrap")
    pp, pr = hp(pp), hp(pr)
    c = np.fft.irfft(np.fft.rfft(pr) * np.conj(np.fft.rfft(pp)), n=W)
    k = int(np.argmax(c))
    excl = np.abs(((np.arange(W) - k + W // 2) % W) - W // 2) > 10
    ratio = float(c[k] / (np.median(np.abs(c[excl])) + 1e-12))
    r = float(np.corrcoef(np.roll(pp, k), pr)[0, 1])
    return int((-k) % W), ratio, r


def _circ_dist(a: int, b: int, W: int) -> int:
    d = abs(a - b) % W
    return min(d, W - d)


def find_line_start(img: np.ndarray, gap: int = 2):
    """Column where each line really starts, for charts heard without phasing.

    Within a picture, neighbouring columns are strongly alike (isobars, coasts
    and grid lines run across them). Where one transmitted line ends and the
    next begins they are unrelated, so the correlation between columns a
    couple of pixels apart drops sharply at exactly that column. Unlike a
    margin search this works on charts printed edge to edge (Northwood).
    Returns (column, confidence in standard deviations)."""
    x = img.astype(np.float64)
    x = x - x.mean(axis=0)
    W = x.shape[1]
    b = np.roll(x, -gap, axis=1)
    r = (x * b).sum(axis=0) / (np.sqrt((x * x).sum(axis=0) * (b * b).sum(axis=0)) + 1e-9)
    straddle = np.convolve(np.r_[r[-gap:], r], np.ones(gap), "valid")[:W] / gap
    c = int(np.argmin(straddle))
    conf = float((np.median(straddle) - straddle[c]) / (np.std(straddle) + 1e-9))
    start = (c + 1) % W
    return start, conf


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


def trim_edges(img, cimg, max_top=40, max_bot=12, align="end"):
    """Remove phasing lines left at the top and stop-tone lines at the bottom.

    Phasing line: black across the picture with the white phasing pulse at
    the line edges (the decoder puts the line start at the pulse centre).
    Stop tone (450 Hz) shows as a fine regular black/white stripe pattern.
    """
    K, W = img.shape
    centre = img[:, int(0.06 * W):int(0.94 * W)]
    dark = (centre < 0.4).mean(1)
    # the phasing pulse is 5 % of a line just before the line start: a white
    # run of ~5 % at the right edge (or ~2.5 % at each edge with --align center).
    # A chart's white margin is a different width, so frame lines stay.
    pw = 0.025 * W

    def edge_run(row, rev):
        """Width of the white run at one edge (a couple of soft edge pixels allowed)."""
        w = (row[::-1] if rev else row) > 0.5
        s0 = next((i for i in range(4) if w[i]), None)
        if s0 is None:
            return 0
        rest = w[s0:]
        return s0 + (int(np.argmin(rest)) if not rest.all() else len(rest))

    def pulsed(i):
        if dark[i] < 0.8:
            return False
        lr, rr = edge_run(img[i], False), edge_run(img[i], True)
        if align == "end":        # whole pulse at the right end, nothing at the left
            return 1.6 * pw <= rr <= 2.5 * pw and lr <= 0.5 * pw
        if align == "start":
            return 1.6 * pw <= lr <= 2.5 * pw and rr <= 0.5 * pw
        return any(0.75 * pw <= r <= 1.25 * pw for r in (lr, rr))

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


SNR_FULL_DB = 12.0    # at or above this SNR the quality is not reduced ...
SNR_ZERO_FACTOR = 0.6  # ... at 0 dB it is multiplied by this (linear in between)
SNR_MIN_FACTOR = 0.5   # ... and never by less than this


def snr_factor(snr_db) -> float:
    """Quality multiplier for the signal-to-noise ratio during the picture:
    1 at SNR_FULL_DB and above, SNR_ZERO_FACTOR at 0 dB, at least SNR_MIN_FACTOR."""
    if snr_db is None:
        return 1.0
    f = SNR_ZERO_FACTOR + (1 - SNR_ZERO_FACTOR) * snr_db / SNR_FULL_DB
    return float(min(1.0, max(SNR_MIN_FACTOR, f)))


def image_snr(audio: np.ndarray, fs: float, a: int, b: int, win_s: float = 1.0):
    """Median in-band SNR (dB) over the picture: the fax band (1440-2360 Hz)
    against the noise either side, in 1 s windows - the same measure as the
    live SNR on the page."""
    w = int(win_s * fs)
    if b - a < 5 * w:
        a, b = 0, len(audio)
    r = [dsp.band_ratio(audio[i:i + w], fs) for i in range(a, b - w, w)]
    if len(r) < 5:
        return None
    return float(10 * np.log10(max(float(np.median(r)) - 1.0, 1e-3)))


AFC_NEAR_HZ = 80.0      # tuning offsets are searched this close to the usual offset first ...
AFC_FAR_RATIO = 1.6     # ... a farther one is taken only if its spectral lines are this much stronger
CLOCK_SEARCH_PPM = 25.0  # line period searched this close to the station's learnt clock (and phasing
                         # further from it than this is not trusted); measured spread is +/-2 ppm


def decode_recording(audio: np.ndarray, fs: float, ioc=None, lpm=None,
                     align: str = "end", log=print, timeline=None, clock_hint_ppm=None,
                     afc_hint_hz=None, reference=None) -> dict:
    """clock_hint_ppm / afc_hint_hz: this installation's sound-card clock error
    and tuning offset, measured on its own recent charts with phasing (both
    barely change).  The clock steadies charts without phasing; the offset
    is where the tuning search starts.  None (a new installation): 0 Hz and
    the full-range searches."""
    audio = np.asarray(audio, np.float64)
    audio, gaps = fill_gaps(audio, fs, timeline, log)
    audio = audio - audio.mean()
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-12
    audio = audio / rms
    n = len(audio)
    log(f"decode: {n / fs:.1f} s of audio at {fs:.0f} Hz")

    # 1-2. AFC: global spectral comb, then 20 s windowed drift track
    # The offset of a receiver barely changes from chart to chart; charts with
    # fine hatching (WAFC SIGWX) can show stronger false lines 200 Hz away,
    # which wrecks the levels and hides the phasing.  Look near the usual
    # offset first (0 Hz until this installation has measured its own).
    c0 = float(afc_hint_hz or 0.0)
    off, sc = dsp.psd_comb(audio, fs, centre=c0, search=AFC_NEAR_HZ)
    off_w, sc_w = dsp.psd_comb(audio, fs, search=300.0)
    if abs(off_w - c0) > AFC_NEAR_HZ and sc_w > AFC_FAR_RATIO * sc:
        off, sc = off_w, sc_w
    elif abs(off_w - off) > 20:
        log(f"AFC: ignored a stronger-looking offset of {off_w:+.0f} Hz (score {sc_w:.0f}) "
            f"far from the usual {c0:+.0f} Hz - using {off:+.0f} Hz (score {sc:.0f})")
    drift, drift_pp = track_drift(audio, fs, off)
    if abs(off - c0) <= AFC_NEAR_HZ:
        # the 20 s drift track must not wander off to a stronger non-fax
        # signal either (it walked from +64 to +104 Hz on interference)
        drift = np.clip(drift, c0 - AFC_NEAR_HZ, c0 + AFC_NEAR_HZ).astype(drift.dtype)
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
        if clock_hint_ppm is not None:
            T *= 1 + clock_hint_ppm * 1e-6
            log(f"phasing: not found, assuming {lpm} LPM and the usual clock ({clock_hint_ppm:+.0f} ppm)")
        else:
            log(f"phasing: not found, assuming {lpm} LPM")
    Tnom = fs * 60.0 / lpm
    # A few broken phasing lines (a fade, or the receiver stepping through
    # frequencies) can give a period far from this installation's own clock,
    # which barely changes: then the clock is trusted instead (a +800 ppm
    # phasing estimate sheared a chart by a pixel a line).
    ph_period_rejected = False
    ppm_ph = (T / Tnom - 1) * 1e6 if ph else None
    if ph and clock_hint_ppm is not None:
        if abs(ppm_ph - clock_hint_ppm) > CLOCK_SEARCH_PPM:
            T = Tnom * (1 + clock_hint_ppm * 1e-6)
            ph_period_rejected = True
            # the pulses themselves still mark where each line starts: re-fit
            # only their position, with the period fixed at the usual clock
            pk, mi = ph["pulses"], ph["pulse_idx"]
            ph["a"] = float(np.median(pk - T * mi))
            ph["end"] = float(ph["a"] + T * (mi.max() + 1))
            ph["T"] = T
            spread = float(np.median(np.abs(pk - (ph["a"] + T * mi))))
            ph["spread"] = spread
            log(f"phasing period {ppm_ph:+.0f} ppm is far from the usual clock ({clock_hint_ppm:+.0f} ppm) - "
                f"using the usual clock; line start re-fitted to the {len(pk)} phasing pulses "
                f"(spread {spread:.1f} samples)")

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
    T_search = T
    if clock_hint_ppm is not None:
        # the clock barely changes: search near the station's learnt value, with
        # or without phasing.  A wide search can lock onto a wrong period that
        # lines up hatching or a diagonal, shearing the chart (~1 px a line at 550 ppm)
        T_search, rng = Tnom * (1 + clock_hint_ppm * 1e-6), CLOCK_SEARCH_PPM
    T2, contrast = refine_period(v, s_first, T_search, K, rng)
    if contrast > 1.15 or not ph or ph_period_rejected:
        T = T2
    corr = residual_slant(v, s_first, T, K, W)
    starts = s_first + np.arange(K) * T + corr
    # line start (framing): up to three independent estimates - the phasing
    # pulses, the picture's own seam, and the frame of the last good copy of
    # the same chart - and the decision rests on two of them agreeing
    pos = starts[:, None] + np.arange(W)[None, :] * (T / W)
    pic = _gather(v, pos)
    m_seam, align_conf = find_line_start(pic)
    m_ref, ref_ratio, ref_r = reference_shift(pic, reference)
    cands = {}
    if ph:
        cands["phasing"] = 0
    if align_conf >= SEAM_MIN_CONF:
        cands["seam"] = int(m_seam)
    if m_ref is not None and ref_ratio >= REF_MIN_RATIO and ref_r >= REF_MIN_R:
        cands["reference"] = int(m_ref)
    near = lambda a, b: a in cands and b in cands and _circ_dist(cands[a], cands[b], W) <= AGREE_PX
    # strong: the phasing pulses (enough of them, fitting well) and a clear frame
    # match with the last good copy.  Weak: the seam, which a faint line in the
    # chart can fool - it confirms, but never overrules a strong estimate.
    ph_ok = bool(ph) and ph["n"] >= 10 and (not ph_period_rejected or ph.get("spread", 0) < 3.0)
    if ph_ok and "reference" in cands:
        if near("phasing", "reference"):
            pick, align_ok, verdict = "phasing", True, "phasing and reference agree"
        else:
            pick, align_ok, verdict = "reference", False, "phasing and reference disagree"
    elif ph_ok:
        pick, align_ok, verdict = "phasing", True, "phasing (seam not needed)"
    elif near("reference", "seam"):
        pick, align_ok, verdict = "seam", True, "reference and seam agree"
    elif "phasing" in cands and (near("phasing", "reference") or near("phasing", "seam")):
        other = "reference" if near("phasing", "reference") else "seam"
        pick, align_ok, verdict = "phasing", True, f"weak phasing confirmed by {other}"
    elif "reference" in cands and ref_ratio >= REF_ALONE_RATIO and ref_r >= REF_ALONE_R \
            and not ("seam" in cands and align_conf >= SEAM_ALONE_CONF):
        pick, align_ok, verdict = "reference", True, "clear match with the last good copy"
    elif list(cands) == ["seam"] and align_conf >= SEAM_ALONE_CONF:
        pick, align_ok, verdict = "seam", True, "very clear seam, nothing else to go on"
    elif cands:
        pick = next(n for n in ("reference", "phasing", "seam") if n in cands)
        align_ok, verdict = False, "not confirmed: " + ("estimates disagree" if len(cands) > 1
                                                         else f"{pick} alone is not enough")
    else:
        pick, align_ok, verdict = None, False, "no estimate"
    if pick is None:
        m = find_margin(pic[::4])              # last resort: the quietest band
        align_how = "uncertain"
    else:
        m, align_how = cands[pick], pick
    log("line start: " + ", ".join(f"{k} {v}" for k, v in cands.items()) +
        f" (seam confidence {align_conf:.1f}, reference match {ref_ratio:.1f}/{ref_r:.2f}) -> {align_how}: {verdict}")
    if m:
        starts = starts + m * T / W
    align_votes = {k: int(v) for k, v in cands.items()}

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
    img, cimg, top, bot = trim_edges(img, cimg, align=align)
    if top or bot:
        log(f"edges: {top} phasing line(s) removed at the top, {bot} stop-tone/blank line(s) at the bottom")
    K = img.shape[0]
    hint_rows = [int(round((p - s_first) / T)) - top for p, _ in gaps]
    hint_rows = [h for h in hint_rows if 0 < h < K - 1]
    # live recordings carry the flywheel's record of lost audio: tears can
    # only be where it found gaps.  Files without it are searched blind.
    tears = find_tears(img, hints=hint_rows, blind=not (timeline is not None and len(timeline) > 50))
    if tears:
        img, cimg = apply_tears(img, tears), apply_tears(cimg, tears)
        log("tears corrected: " + ", ".join(f"line {r} ({s:+d} px)" for r, s in tears))

    # quality: the share of the picture received on a steady signal, scaled
    # by how far the signal stood above the noise while the picture was sent
    steady = float(np.mean(cimg > 0.5))
    snr_db = image_snr(audio, fs, int(max(0, s_first)), int(min(n, s_end)))
    quality = steady * snr_factor(snr_db)
    log(f"quality: {100 * steady:.0f}% steady x {snr_factor(snr_db):.2f} "
        f"(SNR {'--' if snr_db is None else f'{snr_db:+.0f} dB'}) = {100 * quality:.0f}%")
    if align_how == "uncertain":
        # the chart may be split sideways: never let it count as a good copy
        # (it would replace the station copy and mark the chart as collected)
        quality *= 0.5
        log(f"WARNING: line start not found reliably - chart may be shifted sideways; "
            f"quality reduced to {quality:.2f}")
    return {
        "image": img, "conf": cimg,
        "meta": {
            "ioc": int(ioc), "lpm": int(lpm), "width": W, "lines": int(K),
            "afc_hz": round(afc_hz, 1), "drift_pp_hz": round(drift_pp, 1),
            "clock_ppm": round(clock_ppm, 1), "phasing": bool(ph), "align": align_how,
            "align_ok": bool(align_ok), "align_votes": align_votes,
            "ref_match": round(ref_ratio, 1), "period_from_phasing": bool(ph) and not ph_period_rejected,
            "phasing_ppm": None if ppm_ph is None else round(ppm_ph, 1),
            "align_conf": None if align_conf is None else round(align_conf, 1),
            "start_tone": tones["start"] is not None, "stop_tone": tones["stop"] is not None,
            "tears": [[int(r), int(s)] for r, s in tears],
            "gaps_filled_ms": [round(n_ / fs * 1000, 1) for _, n_ in gaps],
            "click_rate": round(click_rate, 4), "quality": round(quality, 3),
            "steady": round(steady, 3), "image_snr_db": None if snr_db is None else round(snr_db, 1),
            "duration_s": round(n / fs, 1),
            "afc_score": round(float(sc), 1),
        },
    }
