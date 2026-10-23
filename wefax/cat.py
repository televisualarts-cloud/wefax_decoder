"""Read (and optionally set) the receiver frequency from SDR Console.

SDR Console's "Serial Port (CAT)" emulates a subset of the Kenwood TS-2000
protocol on a virtual COM port (com0com / VSP Manager pair):
    FA; .. FF;   frequency of receivers 1..6 in Hz  -> "FA00008038100;"
    MD;          mode (1 LSB, 2 USB, ...)
    SM0;         S-meter
Set: "FA00008038100;" / "MD2;".
"""
from __future__ import annotations

import collections
import threading
import time
from datetime import datetime, timezone

RX_LETTERS = "ABCDEF"
_UNSET = object()


def list_ports() -> list[dict]:
    """COM ports Windows (or Linux) knows about, including virtual pairs."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        return []
    out = []
    for p in sorted(lp.comports(), key=lambda p: (len(p.device), p.device)):
        out.append({"port": p.device, "desc": (p.description or "").strip()})
    return out


class CatReader:
    def __init__(self, port: str | None, baud: int = 57600, rx: int = 1, poll_s: float = 1.0, log=print):
        self.port, self.baud, self.poll_s, self.log = port, baud, poll_s, log
        self.letter = RX_LETTERS[max(1, min(6, rx)) - 1]
        self.freq_hz: int | None = None
        self.mode: int | None = None
        self.smeter: int | None = None
        self.connected = False
        self.history = collections.deque(maxlen=3600)      # (unix time, freq_hz)
        self.last_ok = 0.0          # time of the last good frequency reply
        self.stale = False          # connected, but no frequency reply for STALE_S
        self._want: tuple | None = None   # (hz, time sent, retries) of the last tune
        self._ser = None
        self._lock = threading.Lock()
        self._stop = False
        self._pending: list = []          # commands (bytes) and pauses (seconds)
        self._new_port = _UNSET
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def enabled(self) -> bool:
        return bool(self.port)

    def set_port(self, port: str | None):
        """Switch to another COM port (None = CAT off); done on the polling thread."""
        self._new_port = port or None

    # ------------------------------------------------------------------
    def _open(self):
        import serial
        self._ser = serial.Serial(self.port, self.baud, timeout=0.4, write_timeout=0.5)
        self._ser.reset_input_buffer()
        self.connected = True
        self.log(f"CAT: connected to SDR Console on {self.port} (RX {RX_LETTERS.index(self.letter) + 1})")

    STALE_S = 10.0       # no frequency reply this long -> frequency unknown
    TUNE_CHECK_S = 4.0   # after a tune, the radio should report the new frequency by then

    def _query(self, cmd: str, prefix: str) -> str | None:
        """Send a query and return its reply. Anything left over from an
        earlier command (a late reply, an echo of a set command) is thrown
        away first and any reply that is not for this query is skipped, so
        one stray reply can never shift every later reading by one."""
        self._ser.reset_input_buffer()
        self._ser.write(cmd.encode("ascii"))
        buf = b""
        t0 = time.time()
        while time.time() - t0 < 0.6:
            c = self._ser.read(1)
            if not c:
                break
            buf += c
            if c == b";":
                s = buf.decode("ascii", "replace").strip()
                if s.startswith(prefix):
                    return s
                buf = b""               # someone else's reply: keep reading
        return None

    def _close(self):
        self.connected = False
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _run(self):
        warned = False
        while not self._stop:
            if self._new_port is not _UNSET:
                port, self._new_port = self._new_port, _UNSET
                self._close()
                self.port, self.freq_hz, self.mode, self.smeter = port, None, None, None
                warned = False
                self.log(f"CAT: port set to {port}" if port else "CAT: off")
            if not self.port:
                time.sleep(0.3)
                continue
            try:
                if self._ser is None:
                    self._open()
                    warned = False
                with self._lock:
                    pend, self._pending = self._pending, []
                for p in pend:
                    if isinstance(p, (int, float)):
                        time.sleep(p)            # a pause between tuning hops
                        continue
                    self._ser.write(p)
                    time.sleep(0.05)
                if pend:
                    time.sleep(0.2)             # let SDR Console act on the set command
                r = self._query(f"F{self.letter};", "F" + self.letter)
                if r and r[2:-1].isdigit():
                    self.freq_hz = int(r[2:-1])
                    self.history.append((time.time(), self.freq_hz))
                    self.last_ok = time.time()
                    if self.stale:
                        self.stale = False
                        self.log(f"CAT: frequency replies are back ({self.freq_hz / 1000:.2f} kHz)")
                    self._check_tune()
                elif time.time() - self.last_ok > self.STALE_S and not self.stale:
                    # do not keep showing (and identifying charts by) an old reading
                    self.stale = True
                    self.freq_hz = None
                    self.log(f"CAT: no frequency reply from SDR Console for {self.STALE_S:.0f} s - "
                             f"frequency unknown until it answers again")
                r = self._query("MD;", "MD")
                if r and r[2:-1].isdigit():
                    self.mode = int(r[2:-1])
                r = self._query("SM0;", "SM")
                if r and r[2:-1].isdigit():
                    self.smeter = int(r[2:-1][-4:])
            except Exception as e:
                self._close()
                self.freq_hz, self.stale = None, False    # unknown while disconnected
                if not warned:
                    self.log(f"CAT: no connection on {self.port} ({e.__class__.__name__}: {e}) - retrying")
                    warned = True
                for _ in range(25):            # retry in 5 s, sooner if the port is changed
                    if self._new_port is not _UNSET or self._stop:
                        break
                    time.sleep(0.2)
                continue
            time.sleep(self.poll_s)

    # ------------------------------------------------------------------
    def dial_khz_at(self, when: datetime | float | None = None) -> float | None:
        """Dial frequency (kHz) at a given time, from the polling history."""
        if when is None:
            return None if self.freq_hz is None else self.freq_hz / 1000.0
        t = when.timestamp() if isinstance(when, datetime) else float(when)
        best = None
        for ts, f in self.history:
            if best is None or abs(ts - t) < abs(best[0] - t):
                best = (ts, f)
        if best is None or abs(best[0] - t) > 120:
            return None          # no reading near that time: don't guess from the current one
        return best[1] / 1000.0

    def dial_khz_during(self, t0: datetime | float, t1: datetime | float | None = None) -> float | None:
        """The dial frequency (kHz) the receiver held for most of t0..t1
        (t1 = now).  A recording that starts a second before a retune - the
        picture check reaching back to the retune, or audio arriving a little
        late - belongs to the frequency it was tuned to afterwards, not the
        one read at its very first second."""
        a = t0.timestamp() if isinstance(t0, datetime) else float(t0)
        b = time.time() if t1 is None else (t1.timestamp() if isinstance(t1, datetime) else float(t1))
        hist = list(self.history)
        before = [(ts, f) for ts, f in hist if ts <= a]
        inside = [(ts, f) for ts, f in hist if a < ts <= b]
        if not inside:
            return self.dial_khz_at(t0)
        pts = ([(a, before[-1][1])] if before and a - before[-1][0] <= 120 else []) + inside
        held: dict[int, float] = {}
        for (ts, f), nxt in zip(pts, pts[1:] + [(b, None)]):
            held[f] = held.get(f, 0.0) + max(0.0, min(nxt[0], b) - ts)
        return max(held, key=held.get) / 1000.0

    # Every retune first hops HOP_KHZ above and below the target, HOP_S on each,
    # then settles on it.  SDR Console moves its spectrum span when the tune
    # frequency leaves it, so the final frequency lands near the middle of the
    # span instead of near an edge, where the SDR's own filtering weakens the
    # signal.  HOP_KHZ = 0 turns this off.
    HOP_KHZ = 200.0
    HOP_S = 0.5

    def tune(self, khz: float, usb: bool = True, _retry: int = 0, hop: bool = True):
        """Set the receiver (sent on the polling thread)."""
        hz = int(round(khz * 1000))
        fa = lambda f: f"F{self.letter}{int(f):011d};".encode()
        cmds: list = []
        hop = int(self.HOP_KHZ * 1000) if hop else 0     # hop=False: straight there (frequency scans)
        if hop and _retry == 0:
            cmds += [fa(hz + hop), self.HOP_S, fa(max(100_000, hz - hop)), self.HOP_S]
        cmds += [fa(hz)] + ([b"MD2;"] if usb else [])
        settle = 2 * self.HOP_S if hop and _retry == 0 else 0.0
        with self._lock:
            self._pending.extend(cmds)
            self._want = (hz, time.time() + settle, _retry, usb)
        hops = f" (via {(hz + hop) / 1000:.0f} and {max(100_000, hz - hop) / 1000:.0f} kHz to centre the span)" \
            if settle else ""
        self.log(f"CAT: sent {fa(hz).decode()}{'MD2;' if usb else ''}{hops}")

    def _check_tune(self):
        """After a tune: confirm the radio went there; resend once, then say so."""
        w = self._want
        if w is None:
            return
        hz, t, retry, usb = w
        if abs(self.freq_hz - hz) <= 20:
            self._want = None
            return
        if time.time() - t < self.TUNE_CHECK_S:
            return
        if retry == 0:
            self.log(f"CAT: SDR Console reports {self.freq_hz / 1000:.2f} kHz, not {hz / 1000:.2f} - tuning again")
            self.tune(hz / 1000.0, usb, _retry=1)
        else:
            self._want = None
            self.log(f"CAT: WARNING - SDR Console stays on {self.freq_hz / 1000:.2f} kHz after tuning to "
                     f"{hz / 1000:.2f} kHz (receiver {RX_LETTERS.index(self.letter) + 1})")

    def wait_for(self, khz: float, timeout: float = 3.0) -> bool:
        """True once SDR Console reports the requested frequency."""
        target = int(round(khz * 1000))
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.freq_hz is not None and abs(self.freq_hz - target) <= 20:
                return True
            time.sleep(0.2)
        return False

    def status(self) -> dict:
        return {"cat": self.connected, "cat_port": self.port, "cat_stale": self.stale,
                "dial_khz": None if self.freq_hz is None else round(self.freq_hz / 1000.0, 2),
                "usb": self.mode == 2 if self.mode is not None else None}

    def close(self):
        self._stop = True


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
