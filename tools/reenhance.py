#!/usr/bin/env python3
"""Re-run the enhancement on faxes you have already received.

Rebuilds each enhanced image (and its thumbnail) from the saved *_raw.png,
using the current enhancement settings.  The original enhanced image is kept
as *_old.png the first time.

    python tools/reenhance.py                 # every fax in ./faxes
    python tools/reenhance.py faxes --mode chart
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wefax import enhance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="faxes")
    ap.add_argument("--mode", choices=("auto", "clean", "chart", "greyscale"), default="auto")
    ap.add_argument("--no-ml", action="store_true")
    args = ap.parse_args()
    folder = Path(args.folder)
    jobs = sorted(folder.glob("*.json"))
    if not jobs:
        sys.exit(f"no faxes found in {folder.resolve()}")
    for j in jobs:
        rec = json.loads(j.read_text())
        raw = folder / rec["files"]["raw"]
        enh = folder / rec["files"]["enhanced"]
        if not raw.exists():
            continue
        img = cv2.imread(str(raw), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        old = enh.with_name(enh.stem + "_old.png")
        if enh.exists() and not old.exists():
            enh.replace(old)
        out, info = enhance.enhance(img, np.ones_like(img), use_ml=not args.no_ml,
                                    mode=args.mode, log=lambda *a: None)
        e8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
        enh.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(enh), e8)
        th = cv2.resize(e8, (300, max(1, int(e8.shape[0] * 300 / e8.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(folder / rec["files"]["thumb"]), th, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rec["enhance"] = info
        j.write_text(json.dumps(rec, indent=1, default=float))
        print(f"{enh.name}: {info['mode']}, de-streaked {info.get('destreaked_pct', 0)} %")
    print("done - reload the browser page")


if __name__ == "__main__":
    main()
