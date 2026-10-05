"""Screen capture. DXGI Desktop Duplication (dxcam) when available: ~0.1 ms per grab and it only hands
back a frame when the screen actually changed. Falls back to mss (GDI, ~40 ms at 2560x1600)."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("lacyan.capture")


class Capturer:
    def __init__(self, monitor: dict, monitor_index: int) -> None:
        self.monitor = monitor
        self.backend = "mss"
        self._cam = None
        self._sct = None
        self._last: np.ndarray | None = None
        self._last_region: tuple[int, int, int, int] | None = None
        try:
            import dxcam

            self._cam = dxcam.create(output_idx=max(0, monitor_index - 1), output_color="BGR")
            self.backend = "dxcam"
        except Exception:  # noqa: BLE001 - no DXGI (remote desktop, old drivers): use GDI
            log.warning("dxcam unavailable, using mss", exc_info=True)
            import mss

            self._sct = mss.mss()
        log.info("Capture backend: %s", self.backend)

    def grab(self, region: tuple[int, int, int, int]) -> tuple[np.ndarray | None, bool]:
        """Returns (frame, changed). `frame` is BGR for `region` (absolute screen pixels);
        `changed` is False when the screen hasn't changed since the last grab of the same region."""
        same_region = region == self._last_region
        self._last_region = region
        if self._cam is not None:
            m = self.monitor
            rel = (region[0] - m["left"], region[1] - m["top"], region[2] - m["left"], region[3] - m["top"])
            try:
                img = self._cam.grab(region=rel)
            except Exception:  # noqa: BLE001 - display mode change, lost device: recreate next time
                log.warning("dxcam grab failed", exc_info=True)
                img = None
            if img is None:
                if same_region and self._last is not None:
                    return self._last, False
                # first frame for a new region: dxcam only returns *changed* frames, so ask again shortly
                return (self._last if same_region else None), False
            self._last = np.ascontiguousarray(img)
            return self._last, True
        shot = self._sct.grab({"left": region[0], "top": region[1], "width": region[2] - region[0], "height": region[3] - region[1]})
        frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])
        changed = not (same_region and self._last is not None and frame.shape == self._last.shape
                       and np.array_equal(frame[::16, ::16], self._last[::16, ::16]))
        self._last = frame
        return frame, changed

    def close(self) -> None:
        try:
            if self._cam is not None:
                self._cam.release()
            if self._sct is not None:
                self._sct.close()
        except Exception:  # noqa: BLE001
            pass
