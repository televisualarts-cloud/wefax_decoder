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


class CatReader:
    def __init__(self, port: str, baud: int = 57600, rx: int = 1, poll_s: float = 1.0, log=print):
        self.port, self.baud, self.poll_s, self.log = port, baud, poll_s, log
        self.letter = RX_LETTERS[max(1, min(6, rx)) - 1]
        self.freq_hz: int | None = None
        self.mode: int | None = None
        self.smeter: int | None = None
        self.connected = False
        self.history = collections.deque(maxlen=3600)      # (unix time, freq_hz)
        self._ser = None
        self._lock = threading.Lock()
        self._stop = False
        self._pending: list[bytes] = []
        threading.Thread(target=self._run, daemon=True).start()

    # ------------------------------------------------------------------
    def _open(self):
        import serial
        self._ser = serial.Serial(self.port, self.baud, timeout=0.4, write_timeout=0.5)
        self._ser.reset_input_buffer()
        self.connected = True
        self.log(f"CAT: connected to SDR Console on {self.port} (RX {RX_LETTERS.index(self.letter) + 1})")

    def _query(self, cmd: str) -> str | None:
        self._ser.write(cmd.encode("ascii"))
        buf = b""
        t0 = time.time()
        while time.time() - t0 < 0.6:
            c = self._ser.read(1)
            if not c:
                break
            buf += c
            if c == b";":
                break
        s = buf.decode("ascii", "replace").strip()
        if not s or s.startswith("?"):
            return None
        return s

    def _run(self):
        warned = False
        while not self._stop:
            try:
                if self._ser is None:
                    self._open()
                    warned = False
                with self._lock:
                    pend, self._pending = self._pending, []
                for p in pend:
                    self._ser.write(p)
                    time.sleep(0.05)
                r = self._query(f"F{self.letter};")
                if r and r.startswith("F" + self.letter) and r[2:-1].isdigit():
                    self.freq_hz = int(r[2:-1])
                    self.history.append((time.time(), self.freq_hz))
                r = self._query("MD;")
                if r and r.startswith("MD") and r[2:-1].isdigit():
                    self.mode = int(r[2:-1])
                r = self._query("SM0;")
                if r and r.startswith("SM") and r[2:-1].isdigit():
                    self.smeter = int(r[2:-1][-4:])
            except Exception as e:
                self.connected = False
                if self._ser is not None:
                    try:
                        self._ser.close()
                    except Exception:
                        pass
                    self._ser = None
                if not warned:
                    self.log(f"CAT: no connection on {self.port} ({e.__class__.__name__}: {e}) - retrying")
                    warned = True
                time.sleep(5)
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
            return None if self.freq_hz is None else self.freq_hz / 1000.0
        return best[1] / 1000.0

    def tune(self, khz: float, usb: bool = True):
        """Set the receiver (sent on the polling thread)."""
        hz = int(round(khz * 1000))
        cmds = [f"F{self.letter}{hz:011d};".encode()] + ([b"MD2;"] if usb else [])
        with self._lock:
            self._pending.extend(cmds)
        self.log(f"CAT: sent {b''.join(cmds).decode()}")

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
        return {"cat": self.connected,
                "dial_khz": None if self.freq_hz is None else round(self.freq_hz / 1000.0, 2),
                "usb": self.mode == 2 if self.mode is not None else None}

    def close(self):
        self._stop = True


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
