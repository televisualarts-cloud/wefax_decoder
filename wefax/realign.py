"""Find and undo sudden sideways jumps ('tearing') in a decoded fax.

A few lost or extra audio samples (sound-card / virtual-cable glitch, SDR
software hiccup) move every following line sideways by the same amount: the
chart looks torn, with the lower part shifted.  Lines in a fax are periodic,
so the fix is a circular shift of all rows below the tear.

Detection compares each 24-line block's column profile with the blocks just
above it (after streak removal and a 5-line vertical median, so only detail
that continues from line to line counts).  A tear is only accepted when the
shifted match is clearly better than no shift AND the next blocks confirm the
same shift - tested on real noisy charts with injected tears it made no false
corrections, at the cost of missing some small or noisy-region tears.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

B = 24          # rows per block
NREF = 4        # reference = previous blocks (already corrected)
MIN_SHIFT = 6   # px; smaller jumps are left alone
PK_MIN = 0.15
RATIO = 2.0     # shifted match must beat the unshifted one by this factor
LOOK, NEED = 3, 2   # of the next 3 blocks, 2 must agree (+-3 px)
PK2_MIN = 0.12


def _prep(img):
    from .enhance import destreak
    d, _ = destreak(img)
    return 1.0 - ndimage.median_filter(d, size=(5, 1), mode="nearest")


def _hp(x, axis=-1):
    return x - ndimage.uniform_filter1d(x, 41, axis=axis, mode="wrap")


def _xcorr(a, b):
    c = np.fft.irfft(np.fft.rfft(a) * np.conj(np.fft.rfft(b)), len(a))
    n = np.sqrt(float(a @ a) * float(b @ b)) + 1e-12
    k = int(np.argmax(c))
    s = k if k <= len(a) // 2 else k - len(a)
    return s, float(c[k] / n), float(c[0] / n)


HINT_SEARCH = 25   # px either side, at rows where the flywheel filled a gap
HINT_BAND = 48     # rows compared above / below the hint row


def _hint_fixes(ink, hints):
    """Small residual shifts at rows where audio was known to be lost.

    The flywheel has already put the timing back to within a few pixels, so
    only a narrow search is needed and a lower confidence is acceptable.
    """
    K, W = ink.shape
    fixes = []
    for h in sorted(set(hints)):
        if h < 12 or h > K - 12:
            continue
        a = _hp(ink[max(0, h - HINT_BAND):h - 1].mean(0))
        b = _hp(ink[h + 1:min(K, h + HINT_BAND)].mean(0))
        c = np.fft.irfft(np.fft.rfft(b) * np.conj(np.fft.rfft(a)), W)
        nrm = np.sqrt(float(a @ a) * float(b @ b)) + 1e-12
        lags = np.r_[np.arange(0, HINT_SEARCH + 1), np.arange(W - HINT_SEARCH, W)]
        k = lags[int(np.argmax(c[lags]))]
        s = int(k if k <= W // 2 else k - W)
        pk, z = float(c[k] / nrm), float(c[0] / nrm)
        if abs(s) >= 2 and pk > 0.10 and pk > 1.15 * max(z, 0.02):
            fixes.append((int(h), s))
    return fixes


def find_tears(img: np.ndarray, hints=()):
    """Return [(row, shift_px), ...]; shift > 0 means content below moved right.

    hints: rows where lost audio was filled (flywheel) - checked first with a
    narrow, more sensitive search; then the whole image is searched blind.
    """
    ink = _prep(img.astype(np.float32))
    K, W = ink.shape
    pre = _hint_fixes(ink, hints) if len(hints) else []
    if pre:
        ink = apply_tears(ink, pre)
    nb = K // B
    if nb < 6:
        return pre
    P = _hp(ink[:nb * B].reshape(nb, B, W).mean(1), axis=1)
    cur, aligned, tears = 0, [], []
    for i in range(nb):
        p = np.roll(P[i], -cur)
        if len(aligned) >= 2 and i + LOOK < nb:
            R = np.mean(aligned[-NREF:], 0)
            s, pk, z = _xcorr(p, R)
            if abs(s) >= MIN_SHIFT and pk > PK_MIN and pk > RATIO * max(z, 0.04):
                ok = 0
                for j in range(i + 1, i + 1 + LOOK):
                    s2, pk2, _ = _xcorr(np.roll(P[j], -cur), R)
                    ok += abs(s2 - s) <= 3 and pk2 > PK2_MIN
                if ok >= NEED and _persists(P, i, s, cur, R):
                    row = _refine_row(ink, cur, s, R, i)
                    tears.append((row, int(s)))
                    cur += s
                    p = np.roll(P[i], -cur)
        aligned.append(p)
    return _merge(pre, tears)


def _merge(a, b):
    """Combine two tear lists (b found after applying a) into one list."""
    if not a:
        return b
    if not b:
        return a
    # shifts are cumulative from each row down, so the union applies correctly
    out = {}
    for r, s in list(a) + list(b):
        out[r] = out.get(r, 0) + s
    return sorted((r, s) for r, s in out.items() if s)


LATER = 6   # blocks after the confirmation window used for the persistence test


def _cc(a, b, s):
    a2 = np.roll(a, -s)
    return float(a2 @ b) / (np.sqrt(float(a @ a) * float(b @ b)) + 1e-12)


def _persists(P, i, s, cur, R):
    """A real tear never 'heals': further down, the picture keeps matching the
    lines above at the new shift, not at the old one.  A chance match (busy or
    noisy chart) fades and the unshifted alignment comes back."""
    js = range(i + 1 + LOOK, min(len(P), i + 1 + LOOK + LATER))
    if len(js) < 3:
        return True
    back = fwd = 0
    for j in js:
        d = _cc(np.roll(P[j], -cur), R, s) - _cc(np.roll(P[j], -cur), R, 0)
        if d > 0.05:
            fwd += 1
        elif d < -0.05:
            back += 1          # blocks matching neither (e.g. after a further tear) don't vote
    return not (back >= 2 and back > fwd)


def _refine_row(ink, cur, s, R, i):
    """Exact row of the tear within the previous/current block."""
    lo, hi = max(1, (i - 1) * B), min(ink.shape[0], (i + 1) * B)
    rows = _hp(np.roll(ink[lo:hi], -cur, axis=1), axis=1)
    gain = rows @ np.roll(R, s) - rows @ R          # >0: row looks 'after the tear'
    cum = np.cumsum(gain[::-1])[::-1]
    return lo + int(np.argmax(cum))


def apply_tears(img: np.ndarray, tears) -> np.ndarray:
    out = img.copy()
    total = 0
    bounds = [r for r, _ in tears] + [img.shape[0]]
    for (r, s), r2 in zip(tears, bounds[1:]):
        total += s
        out[r:r2] = np.roll(img[r:r2], -total, axis=1)
    return out
