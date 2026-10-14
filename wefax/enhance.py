"""Noise removal and line reconstruction for decoded WEFAX images.

Stages
------
1. Repair: whole-line dropouts (fades) and FM-click / QRN streak pixels are
   flagged from the demodulator confidence map and a vertical-median test,
   then filled by vertical interpolation (WEFAX noise is uncorrelated
   line-to-line, image content is not).
2. Denoise: non-local means with strength set from a robust noise estimate.
3. Upscale x2: FSRCNN (ONNX, bundled) or Lanczos fallback.
4. Default ('clean'): HF streak removal, light denoise, contrast; output
   stays greyscale so faint lines survive.  Optional 'chart' mode renders
   hard black/white line art for strong signals.
   Greyscale mode (satellite imagery): levels preserved, mild unsharp mask.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

MODEL_PATH = Path(__file__).with_name("models") / "fsrcnn_x2.onnx"
_session = None
_session_failed = False


# --------------------------------------------------------------------------
def _ml_session(log):
    global _session, _session_failed
    if _session is not None or _session_failed:
        return _session
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _session = ort.InferenceSession(str(MODEL_PATH), so, providers=["CPUExecutionProvider"])
        log("ML upscaler: FSRCNN x2 loaded (onnxruntime)")
    except Exception as e:  # missing package or model
        _session_failed = True
        log(f"ML upscaler unavailable ({e.__class__.__name__}: {e}); using Lanczos")
    return _session


def upscale2(img: np.ndarray, use_ml: bool, log=print, tile: int = 384, ov: int = 12):
    """x2 upscale of a float [0,1] image."""
    sess = _ml_session(log) if use_ml else None
    H, W = img.shape
    if sess is None:
        return _lanczos2(img), "lanczos"
    out = np.zeros((2 * H, 2 * W), np.float32)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y0, x0 = max(0, y - ov), max(0, x - ov)
            y1, x1 = min(H, y + tile + ov), min(W, x + tile + ov)
            t = img[y0:y1, x0:x1].astype(np.float32)[None, None]
            o = sess.run(None, {"input": t})[0][0, 0]
            oy, ox = 2 * (y - y0), 2 * (x - x0)
            hh, ww = 2 * (min(H, y + tile) - y), 2 * (min(W, x + tile) - x)
            out[2 * y:2 * y + hh, 2 * x:2 * x + ww] = o[oy:oy + hh, ox:ox + ww]
    return np.clip(out, 0, 1), "fsrcnn"


def _lanczos2(img):
    if cv2 is not None:
        return np.clip(cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4), 0, 1)
    return np.clip(ndimage.zoom(img, 2, order=3), 0, 1)


# --------------------------------------------------------------------------
def fill_vertical(img: np.ndarray, bad: np.ndarray) -> np.ndarray:
    out = img.copy()
    rows = np.arange(img.shape[0])
    for j in np.flatnonzero(bad.any(axis=0)):
        b = bad[:, j]
        g = ~b
        if g.sum() < 2:
            continue
        out[b, j] = np.interp(rows[b], rows[g], img[g, j])
    return out


def noise_sigma(img: np.ndarray) -> float:
    r = img - ndimage.median_filter(img, size=3)
    return float(1.4826 * np.median(np.abs(r - np.median(r))) * 1.3)


def remove_specks(soft: np.ndarray, min_area: int) -> np.ndarray:
    if cv2 is None or min_area < 1:
        return soft
    out = soft.copy()
    for polarity in (0, 1):
        b = (soft < 0.5) if polarity == 0 else (soft >= 0.5)
        n, lab, st, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
        small = np.flatnonzero(st[:, cv2.CC_STAT_AREA] < min_area)
        small = small[small > 0]
        if len(small):
            m = np.isin(lab, small)
            out[m] = 1.0 if polarity == 0 else 0.0
    return out


# --------------------------------------------------------------------------
def halfdepth_threshold(x: np.ndarray, win: int, min_contrast: float = 0.3) -> np.ndarray:
    """Per-pixel threshold halfway between local white and local black.

    Thin lines blurred by the limited video bandwidth never reach full black;
    a fixed 50 % threshold drops them.  Thresholding at half the *local* depth
    keeps them at their true (FWHM) width.  Where there is no real local
    contrast the threshold falls to 0.25, so noise on blank paper stays white.
    """
    bg = ndimage.uniform_filter(ndimage.maximum_filter(x, size=win), 3)
    fg = ndimage.uniform_filter(ndimage.minimum_filter(x, size=win), 3)
    return np.where(bg - fg > min_contrast, 0.5 * (bg + fg), 0.25).astype(np.float32)


def short_runs(mask: np.ndarray, max_len: int) -> np.ndarray:
    """Keep only vertical runs of True no longer than max_len rows."""
    out = np.zeros_like(mask)
    lab, n = ndimage.label(mask, structure=[[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    if n == 0:
        return out
    sl = ndimage.find_objects(lab)
    lengths = np.array([s[0].stop - s[0].start for s in sl])
    ok = np.zeros(n + 1, bool)
    ok[1:] = lengths <= max_len
    return ok[lab]


def destreak(x: np.ndarray, tau: float = 0.18) -> np.ndarray:
    """Remove HF noise streaks.

    Impulsive noise and fading hit one scan line at a time, and the video
    filter smears each hit sideways into a short horizontal dash.  Chart
    detail continues from line to line, so a pixel whose difference from its
    3-line vertical median is *horizontally coherent* is noise.
    """
    m3 = ndimage.median_filter(x, size=(3, 1), mode="nearest")
    dev = x - m3
    coh = ndimage.uniform_filter1d(dev, 9, axis=1)
    mask = (np.abs(coh) > 0.5 * tau) | (np.abs(dev) > 2 * tau)
    y = np.where(mask, m3, x)
    # gentle extra vertical cleaning, half strength so thin lines survive
    return 0.5 * y + 0.5 * ndimage.median_filter(y, size=(3, 1), mode="nearest"), float(mask.mean())


# below this noise level (clean, strong signal) the chart gets only light
# processing; measured: clean DWD ~0.008, typical noisy copies 0.02 - 0.1
CLEAN_SIGMA = 0.015
# Paper is shown at this level rather than 100 % white: easier on the eye,
# and faint marks near white aren't lost on screens that clip early.
WHITE_LEVEL = 0.92
# background greys lighter than PAPER_KNEE[0] fade smoothly to paper white by
# PAPER_KNEE[1] (was a hard snap at 0.6, which erased faint or noisy text)
PAPER_KNEE = (0.62, 0.88)
NLM_STRENGTH = 1.35   # background denoising strength (x noise level; was 1.6)


def _soft_paper(bg: np.ndarray) -> np.ndarray:
    a, b = PAPER_KNEE
    t = np.clip((bg - a) / (b - a), 0, 1)
    t = t * t * (3 - 2 * t)
    return bg + (1 - bg) * t

_KERNELS = {}


def _line_kernel(th: float, sig_w: float = 1.0, sig_l: float = 7.0) -> np.ndarray:
    """Zero-mean matched filter for a thin dark line at angle th (~28 px long)."""
    key = (round(th, 4), sig_w, sig_l)
    if key not in _KERNELS:
        R = int(3 * sig_l)
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
        u = xx * np.cos(th) + yy * np.sin(th)
        v = -xx * np.sin(th) + yy * np.cos(th)
        k = (np.exp(-v ** 2 / (2 * sig_w ** 2)) - np.exp(-v ** 2 / (2 * (3 * sig_w) ** 2)) / 3) \
            * np.exp(-u ** 2 / (2 * sig_l ** 2))
        k -= k.mean()
        _KERNELS[key] = (k / np.sqrt((k ** 2).sum())).astype(np.float32)
    return _KERNELS[key]


def reconstruct_lines(base: np.ndarray, n_angles: int = 24, z0: float = 4.5):
    """Rebuild broken chart lines from a de-streaked image.

    Each pixel is tested against 24 oriented line detectors ~28 px long.  A
    real isobar or front lines up with one of them and integrates over many
    scan lines, so it stands out far above the noise even where it is broken
    up; dots, text and noise respond to all orientations equally and are
    ignored.  Near-horizontal detectors look at a 5-line vertical median so
    that single-line HF streaks cannot pose as lines.  Dense areas (text,
    station plots) are left as received.  Returns (image, line_map).
    """
    vm = ndimage.median_filter(base, size=(5, 1), mode="nearest")
    resp = []
    for th in np.arange(n_angles) * np.pi / n_angles:
        horiz = min(th, np.pi - th) < np.radians(25)
        src = 1.0 - (vm if horiz else base)
        r = cv2.filter2D(src, -1, _line_kernel(th), borderType=cv2.BORDER_REFLECT)
        resp.append(r * (0.75 if horiz else 1.0))
    resp = np.stack(resp)
    k = np.argmax(resp, 0)
    rmax = resp.max(0)
    rorth = np.take_along_axis(resp, ((k + n_angles // 2) % n_angles)[None], 0)[0]
    med = np.median(resp, axis=0)
    # noise scale measured locally (64 px neighbourhood), so a busy or
    # streaky part of the chart does not make the rest over-sensitive
    dev = np.abs(med - np.median(med))
    sig = ndimage.uniform_filter(dev, 65) * 1.4826 * 1.6 + 1e-3
    z = (rmax - med) / sig
    # a real line responds along its direction and NOT across it; text,
    # corners and noise respond across as well -> no 'spokes' around them
    aniso = (rmax - rorth) / (np.abs(rmax) + np.abs(rorth) + 1e-6)
    p = np.clip((z - z0) / 4.0, 0, 1) * np.clip((aniso - 0.55) / 0.25, 0, 1)
    p = ndimage.gaussian_filter(p, 0.6) ** 1.5
    ink = ndimage.uniform_filter(
        (ndimage.median_filter(base, size=(3, 1), mode="nearest") < 0.45).astype(np.float32), 15)
    p *= np.clip(1 - (ink - 0.12) / 0.08, 0, 1)
    # background: denoised, paper pushed to white
    sigma = noise_sigma(base)
    u8 = (np.clip(base, 0, 1) * 255 + 0.5).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(u8, None, float(np.clip(sigma * 255 * NLM_STRENGTH, 3, 18)), 5, 13) / 255.0
    lo, hi = np.percentile(den, 1), np.percentile(den, 55)
    if hi - lo < 0.2:
        lo, hi = 0.0, 1.0
    bg = np.clip((den - lo) / (hi - lo), 0, 1)
    bg = _soft_paper(bg) ** 1.3
    return np.minimum(bg, 1 - 0.9 * p).astype(np.float32), p


def enhance(img: np.ndarray, conf: np.ndarray, scale: int = 2,
            use_ml: bool = True, mode: str = "auto", log=print):
    """Return (enhanced float image, info dict).

    mode 'auto' (default): de-streak, then rebuild broken chart lines.
    mode 'clean': de-streak, light denoise, contrast only (no line rebuilding).
    mode 'chart': hard black/white line art - only for strong signals.
    mode 'greyscale': satellite pictures; light denoise and sharpening.
    """
    img = img.astype(np.float32)
    info = {}
    u8 = lambda a: (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)

    # 0. tears (already done by the decoder for new faxes; this catches older ones)
    from .realign import find_tears, apply_tears
    tears = find_tears(img)
    if tears:
        img, conf = apply_tears(img, tears), apply_tears(conf, tears)
        info["tears_fixed"] = [[int(r), int(s)] for r, s in tears]

    # 1. repair only *short* dropouts (fades of a few lines); longer gaps are
    #    left as received - interpolating across them invents vertical smears
    line_bad = conf.mean(axis=1) < 0.3
    bad = short_runs(conf < 0.3, 2) | short_runs(np.repeat(line_bad[:, None], img.shape[1], 1), 3)
    info["repaired_pct"] = round(100.0 * float(bad.mean()), 2)
    rep = fill_vertical(img, bad) if bad.any() else img

    # 2. streak removal
    if mode in ("auto", "clean", "chart"):
        rep, frac = destreak(rep)
        info["destreaked_pct"] = round(100 * frac, 1)

    if mode == "auto" and cv2 is not None and noise_sigma(img) < CLEAN_SIGMA:
        # already a clean copy: heavy cleaning would only distort it.  Just
        # streak removal, levels and the 2x upscale.
        lo, hi = np.percentile(rep, 1), np.percentile(rep, 55)
        if hi - lo < 0.2:
            lo, hi = 0.0, 1.0
        light = np.clip((rep - lo) / (hi - lo), 0, 1).astype(np.float32)
        if scale == 2:
            up2, how = upscale2(light, use_ml, log)
        else:
            up2, how = light, "none"
        info["upscaler"] = how
        info["mode"] = "light"
        log(f"enhance: {info}")
        return (np.clip(up2, 0, 1) * WHITE_LEVEL).astype(np.float32), info

    if mode == "auto" and cv2 is not None:
        rec, pmap = reconstruct_lines(rep)
        info["lines_rebuilt_pct"] = round(100 * float((pmap > 0.5).mean()), 2)
        if scale == 2:
            up2, how = upscale2(rec, use_ml, log)
        else:
            up2, how = rec, "none"
        info["upscaler"] = how
        info["mode"] = "reconstruct"
        log(f"enhance: {info}")
        return (np.clip(up2, 0, 1) * WHITE_LEVEL).astype(np.float32), info

    # 3. light non-local means (strength from a robust noise estimate)
    sigma = noise_sigma(rep)
    info["noise_sigma"] = round(sigma, 4)
    if cv2 is not None and sigma > 0.01:
        h = float(np.clip(sigma * 255 * 0.9, 3, 20))
        den = cv2.fastNlMeansDenoising(u8(rep), None, h, 5, 13).astype(np.float32) / 255.0
    else:
        den = rep

    # 4. upscale
    if scale == 2:
        up2, how = upscale2(np.clip(den, 0, 1), use_ml, log)
    else:
        up2, how = np.clip(den, 0, 1), "none"
    info["upscaler"] = how

    # 5. finish
    if mode == "chart":
        t = halfdepth_threshold(up2, 9 * scale)
        soft = 1.0 / (1.0 + np.exp(-(up2 - t) / 0.04))
        out = remove_specks(soft.astype(np.float32), int(2 * scale * scale))
        info["mode"] = "chart"
    elif mode == "greyscale":
        blur = ndimage.gaussian_filter(up2, 1.0 * scale)
        out = np.clip(up2 + 0.5 * (up2 - blur), 0, 1)
        info["mode"] = "greyscale"
    else:
        # paper -> white, ink -> black, slight gamma to strengthen faint lines
        lo, hi = np.percentile(up2, 1), np.percentile(up2, 55)
        if hi - lo < 0.2:
            lo, hi = 0.0, 1.0
        out = np.clip((up2 - lo) / (hi - lo), 0, 1) ** 1.3
        info["mode"] = "clean"
    log(f"enhance: {info}")
    return (np.clip(out, 0, 1) * WHITE_LEVEL).astype(np.float32), info
