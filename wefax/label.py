"""Title block stamped on the enhanced chart: black text on white, black
border, top left.

    NORTHWOOD UK | 4610 kHz | 2026-10-14 08:11z
    Surface Prognosis T+48 | 12z run | Quality 70%
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FONT = Path(__file__).parent / "fonts" / "DejaVuSans-Bold.ttf"
TEXT_FRAC = 0.0125      # text height as a fraction of the image width (45 px on a 3620 px chart)
PAD_FRAC = 0.45         # padding inside the box, as a fraction of the text height
LINE_GAP = 0.35         # gap between the two lines, as a fraction of the text height
BORDER_FRAC = 0.10      # border thickness, as a fraction of the text height


def _white() -> int:
    """Box white matches the chart's paper white (enhance.WHITE_LEVEL)."""
    from .enhance import WHITE_LEVEL
    return int(round(255 * WHITE_LEVEL))


_COUNTRY = {"united kingdom": "UK", "u.s.a.": "USA", "usa": "USA", "republic of korea": "South Korea"}
_CITY = {"DWD": "DWD Hamburg", "PtReyes": "Point Reyes", "GuanzouCoastRadioStationXSQ": "Guanzou",
         "RioDeJaneiro": "Rio de Janeiro", "NewOrleans": "New Orleans"}


def station_title(product: dict) -> str:
    """'Northwood, United Kingdom' -> 'NORTHWOOD UK'."""
    parts = [p.strip() for p in str(product.get("station") or product.get("short", "")).split(",")]
    city = _CITY.get(product.get("short"), parts[0])
    country = parts[-1] if len(parts) > 1 else ""
    country = _COUNTRY.get(country.lower(), country)
    return f"{city} {country}".strip().upper()


def title_lines(product: dict, utc: str, quality: float | None) -> list[str]:
    first = [station_title(product)]
    if product.get("khz"):
        first.append(f"{float(product['khz']):g} kHz")
    if utc:
        # date and the time reception started, e.g. 2026-10-18 08:11z
        first.append(f"{utc[:10]} {utc[11:16]}z" if len(utc) >= 16 else utc[:10])
    second = [str(product.get("contents", "")).strip()]
    if str(product.get("valid") or "").strip():
        second.append(f"{product['valid']}z run")
    if quality is not None:
        second.append(f"Quality {round(100 * float(quality))}%")
    return [" | ".join(first), " | ".join(p for p in second if p)]


def stamp(img_u8: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw the title block onto a greyscale uint8 image (returns a new image)."""
    h, w = img_u8.shape[:2]
    size = max(14, int(round(w * TEXT_FRAC)))
    pad = int(round(size * PAD_FRAC))
    gap = int(round(size * LINE_GAP))
    border = max(2, int(round(size * BORDER_FRAC)))
    try:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype(str(FONT), size)
        im = Image.fromarray(img_u8)
        d = ImageDraw.Draw(im)
        boxes = [d.textbbox((0, 0), t, font=font) for t in lines]
        asc, desc = font.getmetrics()
        lh = asc + desc
        tw = max(b[2] for b in boxes)
        bw = tw + 2 * pad + 2 * border
        bh = len(lines) * lh + (len(lines) - 1) * gap + 2 * pad + 2 * border
        d.rectangle([0, 0, bw - 1, bh - 1], fill=0)
        d.rectangle([border, border, bw - 1 - border, bh - 1 - border], fill=_white())
        y = border + pad
        for t in lines:
            d.text((border + pad, y), t, font=font, fill=0)
            y += lh + gap
        return np.asarray(im).copy()
    except ImportError:
        # no Pillow: OpenCV's built-in font (plainer, but readable)
        import cv2
        out = img_u8.copy()
        scale = size / 22.0
        th = max(1, int(round(scale * 1.6)))
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, th)[0] for t in lines]
        lh = max(s[1] for s in sizes) + int(size * 0.35)
        bw = max(s[0] for s in sizes) + 2 * pad + 2 * border
        bh = len(lines) * lh + (len(lines) - 1) * gap + 2 * pad + 2 * border
        cv2.rectangle(out, (0, 0), (bw - 1, bh - 1), 0, -1)
        cv2.rectangle(out, (border, border), (bw - 1 - border, bh - 1 - border), _white(), -1)
        y = border + pad
        for t, s in zip(lines, sizes):
            cv2.putText(out, t, (border + pad, y + s[1]), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, th, cv2.LINE_AA)
            y += lh + gap
        return out
