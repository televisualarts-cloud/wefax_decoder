#!/usr/bin/env python3
"""Synthesise a realistic, impaired HF WEFAX recording for testing.

Impairments: tuning offset + slow drift, sound-card clock error (slant),
AWGN, impulsive QRN crashes, selective fades.  Also writes the ground-truth
chart so decoder output can be scored.

    python tools/make_test_signal.py out.wav --lines 600 --snr 8
"""
import argparse
import math

import numpy as np
from scipy.io import wavfile

try:
    import cv2
except ImportError:
    cv2 = None


def make_chart(W=1810, H=600, seed=1):
    rng = np.random.default_rng(seed)
    img = np.ones((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # pressure-like field -> isobars
    field = np.zeros((H, W), np.float32)
    for _ in range(6):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        s = rng.uniform(150, 400)
        field += rng.choice([-1, 1]) * 30 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    field += 0.004 * xx
    lv = field / 4.0
    gy, gx = np.gradient(lv)
    gm = np.hypot(gx, gy) + 1e-6
    dist = np.abs(lv - np.round(lv)) / gm
    img[dist < 0.9] = 0.0            # ~2 px isobars
    # lat/lon grid (1 px)
    img[:, ::150] = 0.0
    img[::100, :] = 0.0
    # border
    img[:4, :] = 0
    img[-4:, :] = 0
    img[:, 60:63] = 0
    img[:, W - 63:W - 60] = 0
    # left margin white
    img[:, :60] = 1.0
    img[:, W - 60:] = 1.0
    if cv2 is not None:
        u8 = (img * 255).astype(np.uint8)
        for i in range(25):
            x, y = int(rng.uniform(80, W - 200)), int(rng.uniform(30, H - 10))
            cv2.putText(u8, f"{int(rng.uniform(960, 1040))}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_8)
        cv2.putText(u8, "NORTHWOOD TEST CHART 1200Z", (100, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_8)
        img = (u8 > 127).astype(np.float32)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--lines", type=int, default=600)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--snr", type=float, default=10.0, help="dB in 3 kHz")
    ap.add_argument("--offset", type=float, default=37.0)
    ap.add_argument("--drift", type=float, default=12.0, help="peak drift Hz")
    ap.add_argument("--ppm", type=float, default=140.0, help="clock error")
    ap.add_argument("--qrn", type=float, default=0.6, help="crashes per second")
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--lead", type=float, default=12.0, help="seconds of noise before")
    ap.add_argument("--invert", action="store_true", help="white-on-black chart (dark imagery)")
    ap.add_argument("--no-stop", action="store_true", help="omit the 450 Hz stop tone")
    ap.add_argument("--no-phasing", action="store_true", help="omit the phasing lines")
    ap.add_argument("--no-black", action="store_true", help="omit the end-of-frame black")
    ap.add_argument("--idle", type=float, default=0.0,
                    help="seconds of idle carrier after the end-of-frame black (DWD style)")
    ap.add_argument("--idle-level", type=float, default=1.0, help="idle carrier level, 0 black .. 1 white")
    args = ap.parse_args()

    W, lpm = 1810, 120
    chart = make_chart(W, args.lines)
    if args.invert:
        chart = 1.0 - chart
    np.save(args.out.rsplit(".", 1)[0] + "_truth.npy", chart)
    Tl = 60.0 / lpm
    sr = args.sr

    def vfun_segments():
        segs = []
        segs.append(("noise", args.lead))
        if not args.no_start:
            segs.append(("start", 5.0))
        if not args.no_phasing:
            segs.append(("phasing", 30.0))
        segs.append(("image", args.lines * Tl))
        if not args.no_stop:
            segs.append(("stop", 5.0))
        if not args.no_black:
            segs.append(("black", 10.0))
        if args.idle > 0:
            segs.append(("idle", args.idle))
        segs.append(("noise", 5.0))
        return segs

    segs = vfun_segments()
    total = sum(d for _, d in segs)
    n = int(total * sr)
    t_rx = np.arange(n) / sr
    t = t_rx * (1 + args.ppm * 1e-6)          # transmitter time as seen by our clock
    v = np.zeros(n, np.float32)
    present = np.ones(n, bool)
    t0 = 0.0
    for kind, d in segs:
        m = (t >= t0) & (t < t0 + d)
        tt = t[m] - t0
        if kind == "noise":
            present[m] = False
        elif kind == "start":
            v[m] = (np.floor(tt * 300 * 2) % 2 == 0)
        elif kind == "stop":
            v[m] = (np.floor(tt * 450 * 2) % 2 == 0)
        elif kind == "black":
            v[m] = 0
        elif kind == "idle":
            v[m] = args.idle_level
        elif kind == "phasing":
            ph = (tt / Tl) % 1.0
            v[m] = (ph > 0.95)                         # 5 % white pulse ending at the line start
        elif kind == "image":
            line = np.floor(tt / Tl).astype(int)
            px = np.floor(((tt / Tl) % 1.0) * W).astype(int)
            line = np.clip(line, 0, args.lines - 1)
            v[m] = chart[line, px]
        t0 += d
    f = 1500 + 800 * v + args.offset + args.drift * np.sin(2 * np.pi * t_rx / 400.0)
    # transmitter keying is not infinitely fast: ~0.25 ms rise
    k = int(sr * 0.00025)
    f = np.convolve(f, np.ones(k) / k, "same")
    x = np.cos(2 * np.pi * np.cumsum(f) / sr)
    rng = np.random.default_rng(7)
    # fading: slow Rayleigh-ish + a few deep fades
    fad = 0.7 + 0.3 * np.sin(2 * np.pi * t_rx / 23.0) * np.sin(2 * np.pi * t_rx / 7.3)
    for c in rng.uniform(0, total, 6):
        fad *= 1 - 0.97 * np.exp(-0.5 * ((t_rx - c) / 0.15) ** 2)
    x = x * fad * present
    # AWGN in 3 kHz
    noise = rng.normal(size=n)
    from scipy import signal
    b = signal.butter(6, [300, 3000], "band", fs=sr, output="sos")
    noise = signal.sosfilt(b, noise)
    sig_p = 0.5
    noise *= math.sqrt(sig_p / 10 ** (args.snr / 10) / np.mean(noise ** 2))
    y = x + noise
    # QRN crashes
    for c in rng.uniform(0, total, int(args.qrn * total)):
        i = int(c * sr)
        L = int(sr * rng.uniform(0.003, 0.03))
        y[i:i + L] += rng.normal(size=len(y[i:i + L])) * rng.uniform(3, 15)
    y = np.tanh(y / 3.0) * 3.0      # receiver/ADC soft limiting
    y = (y / np.max(np.abs(y)) * 0.8 * 32767).astype(np.int16)
    wavfile.write(args.out, sr, y)
    print(f"wrote {args.out}: {total:.0f} s, offset {args.offset} Hz, {args.ppm} ppm, SNR {args.snr} dB")


if __name__ == "__main__":
    main()
