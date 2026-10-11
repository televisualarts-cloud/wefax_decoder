#!/usr/bin/env python3
"""Diagnose serial / CAT ports: tries to open every COM port, and on the ones
that open, asks for the frequency in TS-2000 protocol (as SDR Console expects).

    python tools/comtest.py
"""
import ctypes
import sys
import time

import serial
import serial.tools.list_ports


def raw_open(name):
    """Open the port directly with the Windows API and report the error code."""
    if sys.platform != "win32":
        return "n/a"
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = ctypes.c_void_p
    h = k32.CreateFileW("\\\\.\\" + name, 0xC0000000, 0, None, 3, 0, None)
    if h in (None, ctypes.c_void_p(-1).value):
        err = ctypes.get_last_error()
        return f"FAILED, Windows error {err} ({ctypes.FormatError(err).strip()})"
    k32.CloseHandle(ctypes.c_void_p(h))
    return "ok"


print(f"Python {sys.version.split()[0]} ({8 * ctypes.sizeof(ctypes.c_void_p)}-bit), pyserial {serial.__version__}\n")
for p in serial.tools.list_ports.comports():
    print(f"{p.device:6s} {p.description}")
    if "bluetooth" in p.description.lower():
        print("       (Bluetooth - skipped)\n")
        continue
    print(f"       Windows open: {raw_open(p.device)}")
    try:
        s = serial.Serial(p.device, 57600, timeout=0.5, write_timeout=0.5)
    except Exception as e:
        print(f"       pyserial open: FAILED - {e}\n")
        continue
    try:
        s.reset_input_buffer()
        s.write(b"FA;")
        time.sleep(0.3)
        reply = s.read(64).decode("ascii", "replace")
        print(f"       pyserial open: ok, reply to FA; = {reply!r}"
              + ("   <-- SDR Console answering" if reply.startswith("FA") else "   (no CAT program on the other end)"))
    except Exception as e:
        print(f"       pyserial open: ok, but write/read failed - {e}")
    finally:
        s.close()
    print()
