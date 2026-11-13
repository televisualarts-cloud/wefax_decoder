"""Minimal PDF text reader (standard library only).

Enough for reading text tables such as the NWS radiofax schedule list: it
finds every object in the file (including those packed in object streams),
inflates Flate streams, follows the page tree, decodes text through each
font's ToUnicode map (or WinAnsi), and tracks the text and graphics matrices
so every piece of text has its position on the page.  page_lines() then
rebuilds the lines as they appear on the page, left to right, with two
spaces between separate pieces.

It does not render, and it ignores what text extraction does not need
(images, colours, glyph widths).  Text inside form XObjects is followed.
"""
from __future__ import annotations

import re
import zlib

_WS = b" \t\r\n\f\x00"
_DELIM = b"()<>[]{}/%"


class Ref:
    __slots__ = ("num",)

    def __init__(self, num):
        self.num = num

    def __repr__(self):
        return f"Ref({self.num})"


class Name(str):
    pass


class Op(str):
    pass


class _Lexer:
    def __init__(self, data: bytes, pos: int = 0):
        self.d, self.i, self.n = data, pos, len(data)

    def _skip(self):
        d, n = self.d, self.n
        while self.i < n:
            c = d[self.i]
            if c in _WS:
                self.i += 1
            elif c == 0x25:                              # % comment
                j = d.find(b"\n", self.i)
                self.i = n if j < 0 else j + 1
            else:
                break

    def token(self):
        """Next token: numbers, Name, bytes (strings), Op, or '[' ']' '<<' '>>'."""
        self._skip()
        d, i = self.d, self.i
        if i >= self.n:
            return None
        c = d[i]
        if c == 0x2F:                                    # /Name
            j = i + 1
            while j < self.n and d[j] not in _WS and d[j] not in _DELIM:
                j += 1
            raw = d[i + 1:j]
            raw = re.sub(rb"#([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]), raw)
            self.i = j
            return Name(raw.decode("latin-1"))
        if c == 0x28:                                    # (string)
            return self._string()
        if c == 0x3C:
            if d[i + 1:i + 2] == b"<":
                self.i = i + 2
                return "<<"
            j = d.find(b">", i)
            hx = re.sub(rb"\s", b"", d[i + 1:j])
            if len(hx) % 2:
                hx += b"0"
            self.i = j + 1
            return bytes.fromhex(hx.decode("latin-1"))
        if c == 0x3E and d[i + 1:i + 2] == b">":
            self.i = i + 2
            return ">>"
        if c in b"[]{}":
            self.i = i + 1
            return chr(c)
        j = i
        while j < self.n and d[j] not in _WS and d[j] not in _DELIM:
            j += 1
        if j == i:
            self.i = i + 1
            return Op(chr(c))
        w = d[i:j]
        self.i = j
        if re.fullmatch(rb"[+-]?(\d+\.?\d*|\.\d+)", w):
            return float(w) if b"." in w else int(w)
        return Op(w.decode("latin-1"))

    def _string(self) -> bytes:
        d, i = self.d, self.i + 1
        out, depth = bytearray(), 1
        esc = {0x6E: 10, 0x72: 13, 0x74: 9, 0x62: 8, 0x66: 12, 0x28: 0x28, 0x29: 0x29, 0x5C: 0x5C}
        while i < self.n:
            c = d[i]
            if c == 0x5C:                                # backslash
                i += 1
                c = d[i]
                if c in esc:
                    out.append(esc[c])
                    i += 1
                elif 0x30 <= c <= 0x37:
                    j = i
                    while j < i + 3 and j < self.n and 0x30 <= d[j] <= 0x37:
                        j += 1
                    out.append(int(d[i:j], 8) & 0xFF)
                    i = j
                elif c in (0x0D, 0x0A):                  # line continuation
                    i += 2 if d[i:i + 2] == b"\r\n" else 1
                else:
                    out.append(c)
                    i += 1
                continue
            if c == 0x28:
                depth += 1
            elif c == 0x29:
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            out.append(c)
            i += 1
        self.i = i
        return bytes(out)

    def obj(self, tok=None):
        """One complete object (dicts and arrays assembled, 'n g R' -> Ref)."""
        t = self.token() if tok is None else tok
        if t == "<<":
            dct = {}
            while True:
                k = self.token()
                if k == ">>" or k is None:
                    return dct
                dct[k] = self.obj()
        if t == "[":
            arr = []
            while True:
                k = self.token()
                if k == "]" or k is None:
                    return arr
                arr.append(self.obj(k))
        if isinstance(t, int):
            save = self.i
            t2 = self.token()
            if isinstance(t2, int):
                t3 = self.token()
                if t3 == "R":
                    return Ref(t)
            self.i = save
        if isinstance(t, Op):
            return {"true": True, "false": False, "null": None}.get(t, t)
        return t


class Document:
    def __init__(self, data: bytes):
        self.data = data
        self.objs: dict[int, object] = {}
        self.streams: dict[int, bytes] = {}            # raw (still encoded) stream data
        self._scan()

    # -- reading the file ------------------------------------------------
    def _scan(self):
        d = self.data
        for m in re.finditer(rb"(?<![0-9])(\d+)\s+(\d+)\s+obj\b", d):
            num = int(m.group(1))
            lx = _Lexer(d, m.end())
            try:
                val = lx.obj()
            except Exception:
                continue
            self.objs[num] = val
            if isinstance(val, dict):
                k = d.find(b"stream", lx.i, lx.i + 40)
                if k >= 0 and d[lx.i:k].strip() == b"":
                    s = k + 6
                    if d[s:s + 2] == b"\r\n":
                        s += 2
                    elif d[s:s + 1] in (b"\n", b"\r"):
                        s += 1
                    ln = val.get("Length")
                    if isinstance(ln, int) and d[s + ln:s + ln + 30].lstrip().startswith(b"endstream"):
                        e = s + ln
                    else:
                        e = d.find(b"endstream", s)
                    self.streams[num] = d[s:e]
        for num, val in list(self.objs.items()):         # objects packed in object streams
            if isinstance(val, dict) and val.get("Type") == "ObjStm":
                try:
                    self._objstm(val, self.stream(num))
                except Exception:
                    continue

    def _objstm(self, dct, data: bytes):
        n, first = int(dct.get("N", 0)), int(dct.get("First", 0))
        hdr = _Lexer(data[:first])
        pairs = [(hdr.token(), hdr.token()) for _ in range(n)]
        for onum, off in pairs:
            if isinstance(onum, int) and isinstance(off, int) and onum not in self.streams:
                lx = _Lexer(data, first + off)
                self.objs.setdefault(onum, lx.obj())

    def get(self, v):
        while isinstance(v, Ref):
            v = self.objs.get(v.num)
        return v

    def stream(self, num: int) -> bytes:
        raw = self.streams.get(num, b"")
        dct = self.objs.get(num) or {}
        f = self.get(dct.get("Filter"))
        filters = f if isinstance(f, list) else ([f] if f else [])
        for fl in filters:
            fl = self.get(fl)
            if fl == "FlateDecode":
                try:
                    raw = zlib.decompress(raw)
                except zlib.error:
                    raw = zlib.decompressobj().decompress(raw)
            else:
                return b""                              # not needed for text tables
        return raw

    def _stream_of(self, v) -> bytes:
        return self.stream(v.num) if isinstance(v, Ref) else b""

    # -- pages -------------------------------------------------------------
    def pages(self) -> list[dict]:
        cat = next((v for v in self.objs.values() if isinstance(v, dict) and v.get("Type") == "Catalog"), None)
        out = []

        def walk(node, inherited, seen):
            node_d = self.get(node)
            if not isinstance(node_d, dict) or id(node_d) in seen:
                return
            seen.add(id(node_d))
            res = node_d.get("Resources", inherited)
            if node_d.get("Type") == "Pages" or "Kids" in node_d:
                for k in self.get(node_d.get("Kids")) or []:
                    walk(k, res, seen)
            else:
                out.append(dict(node_d, Resources=res))
        if cat is not None:
            walk(cat.get("Pages"), None, set())
        return out

    # -- text --------------------------------------------------------------
    def _font(self, fdict, cache):
        key = id(fdict)
        if key in cache:
            return cache[key]
        fd = self.get(fdict) or {}
        cmap, width = {}, 1
        tu = fd.get("ToUnicode")
        if isinstance(tu, Ref):
            cmap, width = _parse_cmap(self.stream(tu.num))
        width = 2 if fd.get("Subtype") == "Type0" else 1     # simple fonts: one byte per code
        diffs = {}
        enc = self.get(fd.get("Encoding"))
        if isinstance(enc, dict):
            code = 0
            for v in self.get(enc.get("Differences")) or []:
                if isinstance(v, int):
                    code = v
                elif isinstance(v, str):
                    diffs[code] = _GLYPHS.get(v, v if len(v) == 1 else "")
                    code += 1
        cache[key] = (cmap, width, diffs)
        return cache[key]

    def page_items(self, page) -> list[tuple]:
        """(x, y, text, size, order) for every piece of text on the page."""
        res = self.get(page.get("Resources")) or {}
        contents = self.get(page.get("Contents"))
        refs = contents if isinstance(contents, list) else [page.get("Contents")]
        data = b"\n".join(self._stream_of(r) for r in refs)
        items: list = []
        self._run(data, res, [1, 0, 0, 1, 0, 0], items, {}, 0)
        return items

    def _run(self, data, res, ctm, items, fcache, depth):
        if depth > 4:
            return
        fonts = self.get(res.get("Font")) or {}
        xobjs = self.get(res.get("XObject")) or {}
        lx = _Lexer(data)
        stack, gstack = [], []
        tm = tlm = [1, 0, 0, 1, 0, 0]
        font, size, lead = None, 10.0, 0.0

        def place(text, kern=0.0):
            """Record text at the current point, then move the point along by
            an estimate of its width (glyph widths are not read: half the font
            size per character keeps later pieces of the same line to the right)."""
            nonlocal tm
            if text:
                m = _mul(tm, ctm)
                items.append((m[4], m[5], text, abs(size * (m[3] or m[0])), len(items)))
            tm = _mul([1, 0, 0, 1, len(text) * 0.5 * size - kern / 1000.0 * size, 0], tm)

        while True:
            t = lx.token()
            if t is None:
                break
            if not isinstance(t, Op):
                stack.append(lx.obj(t) if t in ("[", "<<") else t)
                continue
            a = stack
            try:
                if t == "q":
                    gstack.append(ctm)
                elif t == "Q":
                    ctm = gstack.pop() if gstack else ctm
                elif t == "cm" and len(a) >= 6:
                    ctm = _mul([float(v) for v in a[-6:]], ctm)
                elif t == "BT":
                    tm = tlm = [1, 0, 0, 1, 0, 0]
                elif t == "Tf" and len(a) >= 2:
                    font = self._font(fonts.get(a[-2]), fcache) if a[-2] in fonts else None
                    size = float(a[-1])
                elif t == "TL" and a:
                    lead = float(a[-1])
                elif t in ("Td", "TD") and len(a) >= 2:
                    tx, ty = float(a[-2]), float(a[-1])
                    if t == "TD":
                        lead = -ty
                    tm = tlm = _mul([1, 0, 0, 1, tx, ty], tlm)
                elif t == "Tm" and len(a) >= 6:
                    tm = tlm = [float(v) for v in a[-6:]]
                elif t == "T*":
                    tm = tlm = _mul([1, 0, 0, 1, 0, -lead], tlm)
                elif t in ("Tj", "'", '"') and a:
                    if t != "Tj":
                        tm = tlm = _mul([1, 0, 0, 1, 0, -lead], tlm)
                    if isinstance(a[-1], bytes):
                        place(_decode(a[-1], font))
                elif t == "TJ" and a and isinstance(a[-1], list):
                    parts, kern = [], 0.0
                    for el in a[-1]:
                        if isinstance(el, bytes):
                            parts.append(_decode(el, font))
                        elif isinstance(el, (int, float)):
                            kern += el
                            if el < -250:
                                parts.append(" ")    # a wide gap inside one text run
                    place("".join(parts), kern)
                elif t == "Do" and a and a[-1] in xobjs:
                    ref = xobjs[a[-1]]
                    xo = self.get(ref)
                    if isinstance(xo, dict) and xo.get("Subtype") == "Form" and isinstance(ref, Ref):
                        mtx = [float(v) for v in (self.get(xo.get("Matrix")) or [1, 0, 0, 1, 0, 0])]
                        self._run(self.stream(ref.num), self.get(xo.get("Resources")) or res,
                                  _mul(mtx, ctm), items, fcache, depth + 1)
            except (ValueError, TypeError, IndexError):
                pass
            stack = []


def _mul(a, b):
    """3x3 affine product a*b, matrices as [a b c d e f]."""
    return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3],
            a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]


def _decode(s: bytes, font) -> str:
    if font is None:
        return s.decode("cp1252", "replace")
    cmap, width, diffs = font
    out = []
    for i in range(0, len(s) - width + 1, width):
        code = int.from_bytes(s[i:i + width], "big")
        if code in cmap:
            out.append(cmap[code])
        elif code in diffs:
            out.append(diffs[code])
        elif width == 1:
            out.append(bytes([code]).decode("cp1252", "replace"))
    return "".join(out)


def _hexstr(h: bytes) -> str:
    b = bytes.fromhex(h.decode("latin-1"))
    try:
        return b.decode("utf-16-be")
    except UnicodeDecodeError:
        return ""


def _parse_cmap(data: bytes):
    cmap, width = {}, 1
    m = re.search(rb"begincodespacerange\s*<([0-9A-Fa-f]+)>", data)
    if m:
        width = max(1, len(m.group(1)) // 2)
    for blk in re.findall(rb"beginbfchar(.*?)endbfchar", data, re.S):
        for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", blk):
            cmap[int(src, 16)] = _hexstr(dst)
    for blk in re.findall(rb"beginbfrange(.*?)endbfrange", data, re.S):
        for lo, hi, rest in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<[0-9A-Fa-f]*>|\[[^\]]*\])", blk):
            lo, hi = int(lo, 16), int(hi, 16)
            if rest.startswith(b"["):
                for k, h in enumerate(re.findall(rb"<([0-9A-Fa-f]*)>", rest)):
                    cmap[lo + k] = _hexstr(h)
            else:
                base = bytes.fromhex(rest[1:-1].decode("latin-1"))
                for k in range(hi - lo + 1):
                    v = base[:-1] + bytes([(base[-1] + k) & 0xFF]) if base else b""
                    try:
                        cmap[lo + k] = v.decode("utf-16-be")
                    except UnicodeDecodeError:
                        pass
    return cmap, width


_GLYPHS = {"space": " ", "hyphen": "-", "period": ".", "comma": ",", "slash": "/", "colon": ":",
           "parenleft": "(", "parenright": ")", "plus": "+", "minus": "-", "endash": "–", "emdash": "—",
           "quoteright": "'", "quoteleft": "'", "quotesingle": "'", "bullet": "•", "degree": "°"}
_GLYPHS.update({d: str(i) for i, d in enumerate(["zero", "one", "two", "three", "four", "five",
                                                  "six", "seven", "eight", "nine"])})


def page_lines(items: list) -> list[str]:
    """Group a page's text pieces into lines, top to bottom, left to right."""
    items = sorted(items, key=lambda r: (-r[1], r[0], r[4]))
    lines, cur, cy = [], [], None
    for x, y, s, h, seq in items:
        tol = max(2.5, (h or 10) * 0.4)
        if cy is not None and abs(y - cy) <= tol:
            cur.append((x, seq, s))
        else:
            if cur:
                lines.append(_join(cur))
            cur, cy = [(x, seq, s)], y
    if cur:
        lines.append(_join(cur))
    return lines


def _join(cells) -> str:
    return "  ".join(re.sub(r"\s+", " ", s).strip() for _, _, s in sorted(cells) if s.strip())


def text(data: bytes) -> str:
    """The whole document as lines of text, page by page."""
    doc = Document(data)
    return "\n".join("\n".join(page_lines(doc.page_items(p))) for p in doc.pages())
