"""The pipeline thread: capture → change check → OCR → blocks → tracking → translation → render items."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import cv2
import mss
import numpy as np
from PySide6.QtCore import QObject, Signal

from . import winutil
from .cache import TranslationCache
from .config import Config, data_path
from .langs import has_source_text
from .layout import group
from .ocr import OCR
from .render import Item, estimate_text_color
from .tracker import Tracker
from .translator import Translator

log = logging.getLogger("veil.engine")


@dataclass
class Frame:
    """One captured frame plus where it sits on the overlay's monitor (physical pixels)."""

    image: np.ndarray
    off_x: int
    off_y: int
    items: list[Item]
    obstacles: list[tuple[int, int, int, int]]  # every OCR'd line; grown translations must not cover them


class Engine(QObject):
    frame_ready = Signal(object)  # Frame
    status = Signal(str)

    def __init__(self, cfg: Config, monitor: dict, ignore_hwnds: set[int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.monitor = monitor
        self.ignore = ignore_hwnds
        self.paused = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cache = TranslationCache(str(data_path("translations.sqlite")))
        self.translator = Translator(cfg)
        self.tracker = Tracker(cfg.stable_passes, cfg.keep_missing_passes)
        self._pending: dict[Future, tuple[int, str]] = {}
        self._region: tuple[int, int, int, int] | None = None
        self._obstacles: list[tuple[int, int, int, int]] = []
        self._last_log = 0.0
        self._last_full = 0.0
        self._last_refresh = 0.0
        self._lines: list | None = None
        self.stats = {"ocr_ms": 0.0, "blocks": 0, "translated": 0, "cached": 0}

    # --- control -----------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="veil-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.translator.shutdown()

    def set_target(self, language: str) -> None:
        self.cfg.target_language = language
        self.tracker.clear()

    # --- loop --------------------------------------------------------------
    def _capture_region(self) -> tuple[int, int, int, int]:
        m = self.monitor
        mon = (m["left"], m["top"], m["left"] + m["width"], m["top"] + m["height"])
        if self.cfg.capture == "foreground":
            r = winutil.foreground_rect(self.ignore)
            if r:
                x0, y0 = max(r[0], mon[0]), max(r[1], mon[1])
                x1, y1 = min(r[2], mon[2]), min(r[3], mon[3])
                if x1 - x0 > 200 and y1 - y0 > 150:
                    return x0, y0, x1, y1
            if self._region:
                return self._region
        return mon

    def _run(self) -> None:
        try:
            ocr = OCR()
        except Exception:
            log.exception("OCR failed to start")
            self.status.emit("OCR failed to start (see data/veil.log)")
            return
        self.translator.warm_up()
        self.status.emit(f"Running · OCR on {ocr.device}")
        prev_thumb = None
        last_ocr = 0.0
        self.stats["device"] = ocr.device
        with mss.mss() as sct:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                region = self._capture_region()
                if region != self._region:
                    self.tracker.clear()
                    self._obstacles = []
                    self._lines = None
                    prev_thumb = None
                    self._region = region
                x0, y0, x1, y1 = region
                shot = sct.grab({"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0})
                frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])

                now = time.time()
                if not self.paused and (now - last_ocr) * 1000 >= self.cfg.ocr_interval_ms:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    thumb = cv2.resize(gray, (max(1, gray.shape[1] // 8), max(1, gray.shape[0] // 8)), interpolation=cv2.INTER_AREA)
                    dirty = self._dirty_rect(prev_thumb, thumb)
                    unsettled = [t.block.rect for t in self.tracker.tracks if t.stable_count < self.cfg.stable_passes]
                    for r in unsettled:
                        dirty = r if dirty is None else _union(dirty, r)
                    # Safety nets: every 3 s re-detect boxes (reusing known text), every 20 s re-read everything.
                    refresh_due = now - self._last_refresh > 3.0
                    deep_due = now - self._last_full > 20.0
                    if dirty is not None or refresh_due:
                        prev_thumb = thumb
                        last_ocr = now
                        self._last_refresh = now
                        if deep_due:
                            dirty = None
                        elif dirty is None:
                            dirty = (0, 0, 0, 0)  # nothing changed: only boxes that are new get read
                        self._ocr_pass(ocr, frame, dirty)
                self._collect()
                if now - self._last_log > 30:
                    self._last_log = now
                    log.info("stats %s cache=%d", self.stats, self.cache.size())
                self.frame_ready.emit(Frame(frame, x0 - self.monitor["left"], y0 - self.monitor["top"], self._items(now), self._obstacles))
                spent = time.perf_counter() - t0
                self._stop.wait(max(0.0, self.cfg.frame_interval_ms / 1000 - spent))

    def _dirty_rect(self, prev: np.ndarray | None, thumb: np.ndarray) -> tuple[int, int, int, int] | None:
        """Bounding box (full-res pixels) of what changed since the last OCR pass, or None if nothing did."""
        if prev is None or prev.shape != thumb.shape:
            h, w = thumb.shape
            return 0, 0, w * 8, h * 8
        diff = cv2.absdiff(thumb, prev)
        if float(diff.mean()) < self.cfg.change_threshold * 0.15 and int(diff.max()) < 24:
            return None
        ys, xs = np.nonzero(diff > 12)
        if len(xs) == 0:
            return None
        return int(xs.min()) * 8, int(ys.min()) * 8, (int(xs.max()) + 1) * 8, (int(ys.max()) + 1) * 8

    def _ocr_pass(self, ocr: OCR, frame: np.ndarray, dirty: tuple[int, int, int, int] | None) -> None:
        """Detect on the whole frame; re-read only lines inside the changed area (dirty=None reads everything)."""
        t = time.perf_counter()
        H, W = frame.shape[:2]
        if dirty is not None and dirty[2] > dirty[0]:
            dirty = _grow(dirty, 8, W, H)
        every = ocr.read(frame, self.cfg.min_confidence, dirty, self._lines)
        if dirty is None:
            self._last_full = time.time()
        self.stats["mode"] = "full" if dirty is None else ("refresh" if dirty[2] <= dirty[0] else "partial")
        self._lines = every
        self._obstacles = [(l.x0, l.y0, l.x1, l.y1) for l in every]
        lines = [l for l in every if has_source_text(l.text, self.cfg.source_language)]
        blocks = group(lines)
        ready = self.tracker.update(blocks)
        self.stats["ocr_ms"] = (time.perf_counter() - t) * 1000
        self.stats["blocks"] = len(blocks)
        lang, model = self.cfg.target_language, self.cfg.model
        for tr in ready:
            text = tr.block.text
            tr.text_color = estimate_text_color(frame, tr.block.rect) or tr.text_color
            hit = self.cache.get(text, lang, model)
            if hit is not None:
                self._apply(tr, text, hit)
                self.stats["cached"] += 1
            elif tr.requested_text != text:
                tr.requested_text = text
                self._pending[self.translator.submit(text)] = (tr.id, text)

    def _collect(self) -> None:
        for fut in [f for f in self._pending if f.done()]:
            track_id, text = self._pending.pop(fut)
            try:
                result = fut.result()
            except Exception as e:  # network / model errors shouldn't kill the loop
                log.warning("Translation failed: %s", e)
                self.status.emit(f"Translation error: {e}")
                for t in self.tracker.tracks:
                    if t.id == track_id:
                        t.requested_text = ""
                continue
            if not result:
                continue
            self.cache.put(text, self.cfg.target_language, self.cfg.model, result)
            self.stats["translated"] += 1
            for t in self.tracker.tracks:
                if t.id == track_id and t.stable_text == text:
                    self._apply(t, text, result)

    @staticmethod
    def _apply(track, text: str, translation: str) -> None:
        if track.translation is None:
            track.shown_at = time.time()
        track.translation = translation
        track.translated_text = text

    def _items(self, now: float) -> list[Item]:
        items = []
        for t in self.tracker.tracks:
            if not t.visible:
                continue
            fade_in = min(1.0, (now - t.shown_at) / 0.15) if t.shown_at else 1.0
            fade_out = max(0.0, 1.0 - (now - t.lost_at) / 0.25) if t.lost_at else 1.0
            b = t.block
            items.append(Item(t.id, b.rect, b.line_h, b.n_lines, t.translation, t.text_color, fade_in * fade_out))
        return items


def _union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _touches(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _grow(r: tuple[int, int, int, int], m: int, W: int, H: int) -> tuple[int, int, int, int]:
    return max(0, r[0] - m), max(0, r[1] - m), min(W, r[2] + m), min(H, r[3] + m)
