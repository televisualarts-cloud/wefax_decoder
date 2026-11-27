"""Read (and set) the receiver frequency.

Two ways of talking to the receiver, chosen by the "port" setting:

* SDR Console (Windows): its "Serial Port (CAT)" emulates a subset of the
  Kenwood TS-2000 protocol on a virtual COM port (com0com / VSP Manager pair):
      FA; .. FF;   frequency of receivers 1..6 in Hz  -> "FA00008038100;"
      MD;          mode (1 LSB, 2 USB, ...)
      SM0;         S-meter
  Set: "FA00008038100;" / "MD2;".  Port: "COM4", "/dev/ttyUSB0", ...

* Network receivers speaking Hamlib's rigctl text protocol over TCP - Gqrx
  (Tools > Remote control, port 7356), rigctld (any Hamlib radio, port 4532)
  and SDR++ (rigctl server module, port 4532):
      f            -> "8038100"          frequency in Hz
      F 8038100    -> "RPRT 0"           set the frequency
      m            -> "USB" / "2800"     mode and passband
      M USB 0      -> "RPRT 0"           set the mode (0: passband unchanged / default)
  Port: "gqrx" (127.0.0.1:7356), "rigctld" (127.0.0.1:4532) or "tcp:host:port".

* Port "auto" (the default when nothing is chosen): every 30 s, look for a
  network receiver on this PC - Gqrx on 7356, then rigctld / SDR++ on 4532 -
  and use the first one that answers.  A COM port or network receiver chosen
  on the page or with --cat always wins; "(off)" turns all of this off.
"""
from __future__ import annotations

import collections
import socket
import threading
import time
from datetime import datetime, timezone

RX_LETTERS = "ABCDEF"
_UNSET = object()

AUTO = "auto"
GQRX_PORT = 7356
RIGCTLD_PORT = 4532
# what "auto" looks for, in order: (host, port, name)
PROBES = [("127.0.0.1", GQRX_PORT, "Gqrx"), ("127.0.0.1", RIGCTLD_PORT, "rigctld / SDR++")]
PROBE_EVERY_S = 30.0

# rigctl mode names -> the Kenwood mode numbers used everywhere else (2 = USB)
_MODES = {"LSB": 1, "USB": 2, "CW": 3, "CWU": 3, "CWL": 3, "CWR": 7, "FM": 4, "WFM": 4,
          "WFM_ST": 4, "WFM_ST_OIRT": 4, "AM": 5, "AMS": 5, "RAW": 0, "OFF": 0}


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


def parse_target(port: str | None):
    """None | ("auto",) | ("serial", device) | ("tcp", host, port, name)."""
    if not port:
        return None
    p = str(port).strip()
    low = p.lower()
    if low == AUTO:
        return (AUTO,)
    if low == "gqrx":
        return ("tcp", "127.0.0.1", GQRX_PORT, "Gqrx")
    if low in ("rigctld", "sdrpp", "sdr++"):
        return ("tcp", "127.0.0.1", RIGCTLD_PORT, "rigctld / SDR++")
    if low.startswith("tcp:"):
        rest = p[4:]
        host, _, num = rest.rpartition(":")
        if not host:                       # "tcp:7356" -> this PC
            host, num = "127.0.0.1", rest
        try:
            n = int(num)
        except ValueError:
            raise ValueError(f"bad network receiver '{p}' - use tcp:host:port, e.g. tcp:127.0.0.1:7356")
        host = host.strip("[]") or "127.0.0.1"
        name = "Gqrx" if n == GQRX_PORT else ("rigctld / SDR++" if n == RIGCTLD_PORT else "rigctl")
        return ("tcp", host, n, name)
    return ("serial", p)


def describe(port: str | None) -> str:
    t = parse_target(port) if port else None
    if t is None:
        return "off"
    if t[0] == AUTO:
        return "auto (Gqrx / rigctld / SDR++ on this PC)"
    if t[0] == "tcp":
        return f"{t[3]} on {t[1]}:{t[2]}"
    return t[1]


# --------------------------------------------------------------------------
# links: one per protocol, all used from the polling thread only
# --------------------------------------------------------------------------
class _SerialLink:
    """SDR Console's Kenwood TS-2000 emulation on a COM port."""
    hops = True             # SDR Console moves its span only when the frequency leaves it

    def __init__(self, device: str, baud: int, letter: str):
        self.device, self.baud, self.letter = device, baud, letter
        self.name = f"SDR Console on {device} (RX {RX_LETTERS.index(letter) + 1})"
        self.radio = "SDR Console"
        self._ser = None

    def open(self):
        import serial
        self._ser = serial.Serial(self.device, self.baud, timeout=0.4, write_timeout=0.5)
        self._ser.reset_input_buffer()

    def close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

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

    def freq(self) -> int | None:
        r = self._query(f"F{self.letter};", "F" + self.letter)
        return int(r[2:-1]) if r and r[2:-1].isdigit() else None

    def mode(self) -> int | None:
        r = self._query("MD;", "MD")
        return int(r[2:-1]) if r and r[2:-1].isdigit() else None

    def smeter(self) -> int | None:
        r = self._query("SM0;", "SM")
        return int(r[2:-1][-4:]) if r and r[2:-1].isdigit() else None

    def set_freq(self, hz: int):
        self._ser.write(f"F{self.letter}{int(hz):011d};".encode())
        time.sleep(0.05)

    def set_usb(self, current_mode: int | None):
        self._ser.write(b"MD2;")
        time.sleep(0.05)

    def show(self, hz: int, usb: bool) -> str:
        return f"F{self.letter}{int(hz):011d};" + ("MD2;" if usb else "")


class _RigctlLink:
    """Hamlib rigctl text protocol over TCP (Gqrx, rigctld, SDR++)."""
    hops = False            # Gqrx / SDR++ recentre themselves; a radio has no span

    def __init__(self, host: str, port: int, name: str):
        self.host, self.port = host, port
        self.radio = name
        self.name = f"{name} on {host}:{port}"
        self._sock: socket.socket | None = None
        self._buf = b""

    def open(self, timeout: float = 2.0):
        s = socket.create_connection((self.host, self.port), timeout=timeout)
        s.settimeout(0.6)
        try:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self._sock, self._buf = s, b""
        # a receiver that accepts the connection but does not speak rigctl
        # (or refuses this PC: Gqrx's allowed-hosts list) is not a receiver
        if self.freq() is None:
            self.close()
            raise ConnectionError("connected, but no frequency reply (rigctl 'f')")

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _drain(self):
        """Throw away anything still waiting (a late reply)."""
        self._buf = b""
        self._sock.setblocking(False)
        try:
            while True:
                d = self._sock.recv(4096)
                if not d:
                    raise ConnectionError("closed by the receiver")
        except (BlockingIOError, InterruptedError):
            pass
        finally:
            self._sock.settimeout(0.6)

    def _lines(self, n: int) -> list[str]:
        out = []
        t0 = time.time()
        while len(out) < n and time.time() - t0 < 1.0:
            if b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                out.append(line.decode("ascii", "replace").strip())
                continue
            try:
                d = self._sock.recv(4096)
            except socket.timeout:
                break
            if not d:
                raise ConnectionError("closed by the receiver")
            self._buf += d
        return out

    def _cmd(self, line: str, n: int = 1) -> list[str]:
        self._drain()
        self._sock.sendall((line + "\n").encode("ascii"))
        return self._lines(n)

    def freq(self) -> int | None:
        r = self._cmd("f")
        if r and r[0].lstrip("-").replace(".", "", 1).isdigit():
            return int(round(float(r[0])))
        return None

    def mode(self) -> int | None:
        r = self._cmd("m", 2)
        return _MODES.get(r[0].upper()) if r and r[0] and not r[0].startswith("RPRT") else None

    def smeter(self) -> int | None:
        return None

    def set_freq(self, hz: int):
        r = self._cmd(f"F {int(hz)}")
        if r and r[0].startswith("RPRT") and r[0] != "RPRT 0":
            raise ConnectionError(f"{self.radio} refused the frequency ({r[0]})")

    def set_usb(self, current_mode: int | None):
        if current_mode == 2:
            return                   # already USB: leave the user's filter alone
        self._cmd("M USB 0")         # 0: passband unchanged (Gqrx) / default (rigctld)

    def show(self, hz: int, usb: bool) -> str:
        return f"F {int(hz)}" + (" / M USB" if usb else "")


def _probe() -> tuple | None:
    """Look for a network receiver on this PC: (host, port, name) or None."""
    for host, port, name in PROBES:
        link = _RigctlLink(host, port, name)
        try:
            link.open(timeout=0.5)
        except Exception:
            continue
        link.close()
        return host, port, name
    return None


# --------------------------------------------------------------------------
class CatReader:
    def __init__(self, port: str | None, baud: int = 57600, rx: int = 1, poll_s: float = 1.0, log=print):
        self.port, self.baud, self.poll_s, self.log = port, baud, poll_s, log
        self.letter = RX_LETTERS[max(1, min(6, rx)) - 1]
        self.freq_hz: int | None = None
        self.mode: int | None = None
        self.smeter: int | None = None
        self.connected = False
        self.found: str | None = None       # what 'auto' found ("Gqrx on 127.0.0.1:7356")
        self.on_connect = None              # called (no arguments) on each new connection
        self.history = collections.deque(maxlen=3600)      # (unix time, freq_hz)
        self.last_ok = 0.0          # time of the last good frequency reply
        self.stale = False          # connected, but no frequency reply for STALE_S
        self._want: tuple | None = None   # (hz, time sent, retries) of the last tune
        self._link = None
        self._lock = threading.Lock()
        self._stop = False
        self._pending: list = []          # ("F", hz), ("USB",) and pauses (seconds)
        self._new_port = _UNSET
        self._next_probe = 0.0
        self._probe_logged = False
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def enabled(self) -> bool:
        """A receiver to control: a COM port or network receiver chosen, or one
        found by 'auto'."""
        if not self.port:
            return False
        if str(self.port).lower() == AUTO:
            return self.found is not None
        return True

    @property
    def radio(self) -> str:
        return self._link.radio if self._link is not None else "the receiver"

    def set_port(self, port: str | None):
        """Switch to another COM port / network receiver / 'auto' (None = off);
        done on the polling thread."""
        self._new_port = port or None

    def _hops(self) -> bool:
        t = parse_target(self.port) if self.port else None
        return bool(t and t[0] == "serial")

    # ------------------------------------------------------------------
    def _make_link(self):
        t = parse_target(self.port)
        if t is None:
            return None
        if t[0] == "serial":
            return _SerialLink(t[1], self.baud, self.letter)
        if t[0] == "tcp":
            return _RigctlLink(t[1], t[2], t[3])
        # auto: look for one now and then
        if time.time() < self._next_probe:
            return None
        self._next_probe = time.time() + PROBE_EVERY_S
        hit = _probe()
        if hit is None:
            if not self._probe_logged:
                self._probe_logged = True
                self.log("CAT: auto - no receiver control found yet (looking every 30 s for Gqrx on port "
                         f"{GQRX_PORT}, rigctld / SDR++ on port {RIGCTLD_PORT}; in Gqrx turn on "
                         "Tools > Remote control)")
            return None
        self.found = f"{hit[2]} on {hit[0]}:{hit[1]}"
        self.log(f"CAT: auto - found {self.found}")
        return _RigctlLink(*hit)

    STALE_S = 10.0       # no frequency reply this long -> frequency unknown
    TUNE_CHECK_S = 4.0   # after a tune, the radio should report the new frequency by then

    def _close(self):
        self.connected = False
        if self._link is not None:
            self._link.close()
            self._link = None

    def _run(self):
        warned = False
        while not self._stop:
            if self._new_port is not _UNSET:
                port, self._new_port = self._new_port, _UNSET
                self._close()
                self.port, self.freq_hz, self.mode, self.smeter = port, None, None, None
                self.found, self._next_probe, self._probe_logged = None, 0.0, False
                warned = False
                self.log(f"CAT: set to {describe(port)}" if port else "CAT: off")
            if not self.port:
                time.sleep(0.3)
                continue
            try:
                if self._link is None:
                    try:
                        link = self._make_link()
                    except ValueError as e:
                        if not warned:
                            self.log(f"CAT: {e}")
                            warned = True
                        time.sleep(1.0)
                        continue
                    if link is None:                   # auto, nothing found yet
                        time.sleep(0.5)
                        continue
                    self._link = link
                    link.open()
                    self.connected = True
                    warned = False
                    self.log(f"CAT: connected to {link.name}")
                    if self.on_connect is not None:
                        try:
                            self.on_connect()
                        except Exception as e:
                            self.log(f"CAT: on-connect action failed: {e}")
                with self._lock:
                    pend, self._pending = self._pending, []
                for p in pend:
                    if isinstance(p, (int, float)):
                        time.sleep(p)            # a pause between tuning hops
                    elif p[0] == "F":
                        self._link.set_freq(p[1])
                    elif p[0] == "USB":
                        self._link.set_usb(self.mode)
                if pend:
                    time.sleep(0.2)             # let the receiver act on the set command
                f = self._link.freq()
                if f is not None:
                    self.freq_hz = f
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
                    self.log(f"CAT: no frequency reply from {self.radio} for {self.STALE_S:.0f} s - "
                             f"frequency unknown until it answers again")
                m = self._link.mode()
                if m is not None:
                    self.mode = m
                s = self._link.smeter()
                if s is not None:
                    self.smeter = s
            except Exception as e:
                name = self._link.name if self._link is not None else describe(self.port)
                self._close()
                self.freq_hz, self.stale = None, False    # unknown while disconnected
                if str(self.port).lower() == AUTO:
                    # the receiver found by 'auto' went away: look again
                    if self.found:
                        self.log(f"CAT: lost {self.found} ({e.__class__.__name__}: {e}) - looking again")
                    self.found, self._next_probe, self._probe_logged = None, time.time() + 5.0, False
                    continue
                if not warned:
                    self.log(f"CAT: no connection to {name} ({e.__class__.__name__}: {e}) - retrying")
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

    # Every retune on SDR Console first hops HOP_KHZ above and below the target,
    # HOP_S on each, then settles on it.  SDR Console moves its spectrum span
    # when the tune frequency leaves it, so the final frequency lands near the
    # middle of the span instead of near an edge, where the SDR's own filtering
    # weakens the signal.  HOP_KHZ = 0 turns this off.  Not used on network
    # receivers (Gqrx and SDR++ recentre by themselves).
    HOP_KHZ = 200.0
    HOP_S = 0.5

    def tune(self, khz: float, usb: bool = True, _retry: int = 0, hop: bool = True):
        """Set the receiver (sent on the polling thread)."""
        hz = int(round(khz * 1000))
        cmds: list = []
        hop = int(self.HOP_KHZ * 1000) if (hop and self._hops()) else 0   # hop=False: straight there
        if hop and _retry == 0:
            cmds += [("F", hz + hop), self.HOP_S, ("F", max(100_000, hz - hop)), self.HOP_S]
        cmds += [("F", hz)] + ([("USB",)] if usb else [])
        settle = 2 * self.HOP_S if hop and _retry == 0 else 0.0
        with self._lock:
            self._pending.extend(cmds)
            self._want = (hz, time.time() + settle, _retry, usb)
        hops = f" (via {(hz + hop) / 1000:.0f} and {max(100_000, hz - hop) / 1000:.0f} kHz to centre the span)" \
            if settle else ""
        link = self._link
        shown = link.show(hz, usb) if link is not None else f"{hz} Hz" + (" USB" if usb else "")
        self.log(f"CAT: sent {shown}{hops}")

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
            self.log(f"CAT: {self.radio} reports {self.freq_hz / 1000:.2f} kHz, not {hz / 1000:.2f} - tuning again")
            self.tune(hz / 1000.0, usb, _retry=1)
        else:
            self._want = None
            self.log(f"CAT: WARNING - {self.radio} stays on {self.freq_hz / 1000:.2f} kHz after tuning to "
                     f"{hz / 1000:.2f} kHz")

    def wait_for(self, khz: float, timeout: float = 3.0) -> bool:
        """True once the receiver reports the requested frequency."""
        target = int(round(khz * 1000))
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.freq_hz is not None and abs(self.freq_hz - target) <= 20:
                return True
            time.sleep(0.2)
        return False

    def status(self) -> dict:
        return {"cat": self.connected, "cat_port": self.port, "cat_stale": self.stale,
                "cat_link": self._link.name if (self._link is not None and self.connected) else None,
                "dial_khz": None if self.freq_hz is None else round(self.freq_hz / 1000.0, 2),
                "usb": self.mode == 2 if self.mode is not None else None}

    def close(self):
        self._stop = True


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
