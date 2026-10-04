"""Global hotkeys via Win32 RegisterHotKey (no admin rights, no keyboard hooks)."""

from __future__ import annotations

import ctypes
import logging
import threading
from ctypes import wintypes
from typing import Callable

log = logging.getLogger("lacyan.hotkeys")
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
MODS = {"alt": 0x1, "ctrl": 0x2, "control": 0x2, "shift": 0x4, "win": 0x8}
MOD_NOREPEAT = 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012


def parse(combo: str) -> tuple[int, int] | None:
    mods, vk = 0, 0
    for part in combo.lower().replace(" ", "").split("+"):
        if part in MODS:
            mods |= MODS[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit():
            vk = 0x6F + int(part[1:])
    return (mods | MOD_NOREPEAT, vk) if vk else None


class Hotkeys:
    def __init__(self, bindings: dict[str, Callable[[], None]]) -> None:
        self.bindings = bindings
        self._thread_id = 0
        threading.Thread(target=self._loop, name="lacyan-hotkeys", daemon=True).start()

    def _loop(self) -> None:
        self._thread_id = kernel32.GetCurrentThreadId()
        actions = {}
        for i, (combo, fn) in enumerate(self.bindings.items(), start=1):
            parsed = parse(combo)
            if parsed and user32.RegisterHotKey(None, i, *parsed):
                actions[i] = fn
            else:
                log.warning("Could not register hotkey %s (in use by another app?)", combo)
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam in actions:
                actions[msg.wParam]()
        for i in actions:
            user32.UnregisterHotKey(None, i)

    def stop(self) -> None:
        if self._thread_id:
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
