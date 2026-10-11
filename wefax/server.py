"""Tiny local web UI: serves the gallery page, images, and a Server-Sent Events
stream that pushes decoder status and 'new fax' notifications to the browser.
Standard library only."""
from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB_DIR = Path(__file__).with_name("web")
SUBDIRS = {"raw", "enhanced", "thumbs"}


class Hub:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()
        self.last_status = {"state": "idle"}
        self.live_png: bytes | None = None
        self.sched = None          # schedule.Schedules
        self.cat = None            # cat.CatReader
        self.settings = {"station": None}
        self.auto = None           # autosched.AutoScheduler
        self.live_info: dict = {}

    def publish(self, kind: str, data: dict):
        if kind == "status":
            self.last_status = data
        msg = f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode()
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def set_live(self, png: bytes, info: dict):
        self.live_png = png
        self.live_info = info
        self.publish("live", info)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def list_faxes(self) -> list[dict]:
        items = []
        for p in sorted(self.out_dir.glob("*.json"), reverse=True):
            try:
                items.append(json.loads(p.read_text()))
            except Exception:
                continue
        return items


def make_handler(hub: Hub):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                return self._send(400, b'{"error":"bad json"}', "application/json")
            if path == "/api/tune":
                if hub.cat is None:
                    return self._send(409, json.dumps({"error": "CAT not enabled - start with --cat COMx"}).encode(),
                                      "application/json")
                try:
                    khz = float(body["khz"])            # assigned frequency
                except Exception:
                    return self._send(400, b'{"error":"khz missing"}', "application/json")
                dial = khz - 1.9
                if body.get("station"):
                    hub.settings["station"] = body["station"]
                if not hub.cat.connected:
                    return self._send(503, json.dumps({"error": f"CAT not connected on {hub.cat.port}"}).encode(),
                                      "application/json")
                hub.cat.tune(dial)
                if hub.auto is not None and not body.get("from_auto"):
                    hub.auto.manual_override(20)
                if hub.cat.wait_for(dial):
                    return self._send(200, json.dumps({"ok": True, "dial_khz": round(dial, 2)}).encode(),
                                      "application/json")
                now = hub.cat.freq_hz
                msg = (f"sent, but SDR Console still reports {now / 1000:.1f} kHz" if now
                       else "sent, but SDR Console is not reporting a frequency")
                return self._send(502, json.dumps({"error": msg}).encode(), "application/json")
            if path == "/api/auto":
                if hub.auto is None:
                    return self._send(409, b'{"error":"no scheduler"}', "application/json")
                on = bool(body.get("on"))
                if on and hub.cat is None:
                    return self._send(409, json.dumps({"error": "needs CAT - start with --cat COMx"}).encode(),
                                      "application/json")
                hub.auto.set_enabled(on)
                return self._send(200, json.dumps(hub.auto.status()).encode(), "application/json")
            if path == "/api/priorities":
                if hub.auto is None:
                    return self._send(409, b'{"error":"no scheduler"}', "application/json")
                first, second = body.get("first"), body.get("second") or None
                known = {st.short for st in (hub.sched.stations if hub.sched else [])}
                if first not in known or (second is not None and second not in known):
                    return self._send(400, b'{"error":"unknown station"}', "application/json")
                hub.auto.set_priorities(first, second)
                return self._send(200, json.dumps(hub.auto.status()).encode(), "application/json")
            if path == "/api/station":
                hub.settings["station"] = body.get("station") or None
                return self._send(200, json.dumps({"ok": True, **hub.settings}).encode(), "application/json")
            self._send(404, b"not found", "text/plain")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/list":
                self._send(200, json.dumps(hub.list_faxes()).encode(), "application/json")
            elif path == "/live.png":
                if hub.live_png is None:
                    return self._send(404, b"no live image", "text/plain")
                self._send(200, hub.live_png, "image/png")
            elif path == "/api/stations":
                out = []
                for st in (hub.sched.stations if hub.sched else []):
                    out.append({"name": st.name, "short": st.short,
                                "groups": [{"khz": sorted(k), "slots": [[sl.hhmm, sl.contents, sl.valid] for sl in sorted(g, key=lambda x: x.minute)]}
                                           for k, g in st.groups]})
                pri = hub.auto.priorities if hub.auto else ["Northwood", "DWD"]
                out.sort(key=lambda d: (d["short"] not in pri, d["name"]))
                self._send(200, json.dumps({"stations": out, "cat": hub.cat is not None,
                                            "priorities": hub.auto.priorities if hub.auto else None,
                                            "station": hub.settings.get("station")}).encode(), "application/json")
            elif path == "/api/plan":
                st = hub.auto.status() if hub.auto else {"enabled": False, "plan": []}
                st["cat"] = hub.cat is not None
                self._send(200, json.dumps(st).encode(), "application/json")
            elif path == "/api/proplog":
                from datetime import datetime, timezone
                rows = hub.auto.proplog.summary(hub.auto.planner.qth, datetime.now(timezone.utc)) if hub.auto else []
                self._send(200, json.dumps(rows).encode(), "application/json")
            elif path == "/api/status":
                self._send(200, json.dumps(hub.last_status).encode(), "application/json")
            elif path.startswith("/faxes/"):
                name = path[len("/faxes/"):]
                parts = name.split("/")
                if ("\\" in name or any(p in ("", ".", "..") or p.startswith(".") for p in parts)
                        or len(parts) > 2 or (len(parts) == 2 and parts[0] not in SUBDIRS)):
                    return self._send(404, b"not found", "text/plain")
                f = hub.out_dir / name
                if not f.is_file():
                    return self._send(404, b"not found", "text/plain")
                ctype = {"png": "image/png", "jpg": "image/jpeg",
                         "json": "application/json"}.get(f.suffix[1:], "application/octet-stream")
                self._send(200, f.read_bytes(), ctype, {"Cache-Control": "max-age=3600"})
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                q = hub.subscribe()
                try:
                    first = f"event: status\ndata: {json.dumps(hub.last_status)}\n\n".encode()
                    self.wfile.write(first)
                    self.wfile.flush()
                    while True:
                        try:
                            msg = q.get(timeout=15)
                        except queue.Empty:
                            msg = b": keepalive\n\n"
                        self.wfile.write(msg)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    hub.unsubscribe(q)
            else:
                self._send(404, b"not found", "text/plain")

    return H


def start(hub: Hub, host: str = "127.0.0.1", port: int = 8730) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(hub))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
