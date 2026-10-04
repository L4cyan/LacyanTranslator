"""Small Win32 helpers: DPI awareness, window bounds, click-through and capture exclusion."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

user32 = ctypes.windll.user32
dwmapi = ctypes.windll.dwmapi

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11
DWMWA_EXTENDED_FRAME_BOUNDS = 9

user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]


def make_dpi_aware() -> None:
    """Per-monitor v2 awareness so screen capture and window rects use physical pixels."""
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            user32.SetProcessDPIAware()


def make_overlay_window(hwnd: int, hide_from_capture: bool) -> bool:
    """Click-through, never-focused, hidden from alt-tab; optionally invisible to screen capture.

    Returns True if capture exclusion is active.
    """
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)
    if hide_from_capture:
        return bool(user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_EXCLUDEFROMCAPTURE))
    user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), WDA_NONE)
    return False


def foreground_rect(ignore: set[int]) -> tuple[int, int, int, int] | None:
    """Physical-pixel bounds (left, top, right, bottom) of the foreground window, or None."""
    hwnd = user32.GetForegroundWindow()
    if not hwnd or hwnd in ignore or user32.IsIconic(hwnd):
        return None
    rect = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect)
    ) != 0:
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
    if rect.right - rect.left < 200 or rect.bottom - rect.top < 150:
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def foreground_title() -> str:
    hwnd = user32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(hwnd, buf, 256)
    return buf.value
