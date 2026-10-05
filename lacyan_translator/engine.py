"""The pipeline.

Capture thread (~30 fps while something is shown, ~10 fps idle):
    grab → follow every visible translation (move with scrolling, hide the instant its text is gone)
    → apply finished OCR / translations → emit what to draw.
OCR thread: detect + read text on the frames the capture thread hands it, so following never waits on OCR.
Translation pool: the local model, behind the on-disk translation memory.
"""

from __future__ import annotations

import logging
import queue
import statistics
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, replace

import cv2
import mss
import numpy as np
from PySide6.QtCore import QObject, Signal

from . import winutil
from .cache import TranslationCache
from .config import Config, data_path
from .langs import has_source_text
from .layout import group
from .motion import GONE, MOVE, Pyramid, locate, make_ref
from .ocr import OCR, Line
from .render import Item, estimate_text_color
from .tracker import Tracker, show
from .translator import Translator

log = logging.getLogger("lacyan.engine")

ACTIVE_FPS = 30
IDLE_FPS = 10


@dataclass
class Frame:
    """One captured frame plus where it sits on the overlay's monitor (physical pixels)."""

    image: np.ndarray
    off_x: int
    off_y: int
    items: list[Item]
    obstacles: list[tuple[int, int, int, int]]  # every OCR'd line; grown translations must not cover them


@dataclass
class _Job:
    frame: np.ndarray
    pyr: Pyramid
    dirty: tuple[int, int, int, int] | None
    prev: list[Line] | None
    motion_mark: tuple[float, float]


@dataclass
class _Result:
    lines: list[Line]
    pyr: Pyramid
    full: bool
    ms: float
    motion_mark: tuple[float, float]


class OCRWorker:
    def __init__(self, cfg: Config, on_ready, on_fail) -> None:
        self.cfg = cfg
        self.inbox: queue.Queue[_Job | None] = queue.Queue(maxsize=1)
        self.outbox: queue.Queue[_Result] = queue.Queue()
        self.busy = False
        self.device = ""
        self._on_ready, self._on_fail = on_ready, on_fail
        threading.Thread(target=self._run, name="lacyan-ocr", daemon=True).start()

    def submit(self, job: _Job) -> None:
        self.busy = True
        self.inbox.put(job)

    def stop(self) -> None:
        try:
            self.inbox.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        try:
            ocr = OCR()
        except Exception:
            log.exception("OCR failed to start")
            self._on_fail()
            return
        self.device = ocr.device
        self._on_ready(ocr.device)
        while True:
            job = self.inbox.get()
            if job is None:
                return
            t = time.perf_counter()
            try:
                lines = ocr.read(job.frame, self.cfg.min_confidence, job.dirty, job.prev)
            except Exception:
                log.exception("OCR pass failed")
                lines = job.prev or []
            self.outbox.put(_Result(lines, job.pyr, job.dirty is None, (time.perf_counter() - t) * 1000, job.motion_mark))
            self.busy = False


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
        self.cache = TranslationCache(str(data_path("translations.sqlite")))
        self.translator = Translator(cfg)
        self.tracker = Tracker(cfg.stable_passes, cfg.keep_missing_passes)
        self._pending: dict[Future, tuple[int, str]] = {}
        self._region: tuple[int, int, int, int] | None = None
        self._lines: list[Line] | None = None
        self._obstacles: list[tuple[int, int, int, int]] = []
        self._prev_thumb: np.ndarray | None = None
        self._last_submit = self._last_refresh = self._last_full = self._last_log = 0.0
        self._vel = (0.0, 0.0)  # per-frame motion of followed text (scroll speed), for prediction
        self._motion = (0.0, 0.0)  # accumulated motion, to place OCR results that arrive late
        self.stats = {"ocr_ms": 0.0, "blocks": 0, "translated": 0, "cached": 0, "fps": 0.0}
        self.worker: OCRWorker | None = None

    # --- control -----------------------------------------------------------
    def start(self) -> None:
        self.worker = OCRWorker(self.cfg, self._ocr_ready, lambda: self.status.emit("OCR failed to start (see data/lacyan.log)"))
        threading.Thread(target=self._run, name="lacyan-engine", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self.worker:
            self.worker.stop()
        self.translator.shutdown()

    def set_target(self, language: str) -> None:
        self.cfg.target_language = language
        self.tracker.clear()

    def _ocr_ready(self, device: str) -> None:
        self.stats["device"] = device
        self.status.emit(f"Running · OCR on {device}")

    # --- capture loop ------------------------------------------------------
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
        self.translator.warm_up()
        frames, fps_t = 0, time.time()
        with mss.mss() as sct:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                region = self._capture_region()
                if region != self._region:
                    same_size = self._region and (region[2] - region[0], region[3] - region[1]) == (
                        self._region[2] - self._region[0], self._region[3] - self._region[1])
                    if not same_size:  # a different window (or resized): start fresh
                        self.reset()
                    self._region = region  # merely moved: coordinates are window-relative, keep everything
                x0, y0, x1, y1 = region
                shot = sct.grab({"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0})
                frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])
                items = self.tick(frame, time.time())
                self.frame_ready.emit(Frame(frame, x0 - self.monitor["left"], y0 - self.monitor["top"], items, self._obstacles))

                frames += 1
                now = time.time()
                if now - fps_t >= 1:
                    self.stats["fps"], frames, fps_t = frames / (now - fps_t), 0, now
                if now - self._last_log > 30:
                    self._last_log = now
                    log.info("stats %s cache=%d", self.stats, self.cache.size())
                fps = IDLE_FPS if self.paused or not self._busy() else ACTIVE_FPS
                self._stop.wait(max(0.0, 1 / fps - (time.perf_counter() - t0)))

    def _busy(self) -> bool:
        return bool(self._pending) or (self.worker is not None and self.worker.busy) or any(
            t.translation is not None or t.stable_count < self.cfg.stable_passes for t in self.tracker.tracks
        )

    def reset(self) -> None:
        self.tracker.clear()
        self._lines = None
        self._obstacles = []
        self._prev_thumb = None
        self._vel = (0.0, 0.0)

    # --- one frame ---------------------------------------------------------
    def tick(self, frame: np.ndarray, now: float) -> list[Item]:
        if self.paused:
            return []
        pyr = Pyramid(frame)
        self._follow(pyr)
        if self.worker is not None:
            while not self.worker.outbox.empty():
                self._apply_ocr(self.worker.outbox.get_nowait(), pyr, frame)
            if not self.worker.busy and self.worker.device:
                self._maybe_submit(frame, pyr, now)
        self._collect(frame)
        return self._items(now)

    def _follow(self, pyr: Pyramid) -> None:
        """Move each tracked block with its text; hide its translation the moment the text is gone."""
        moves = []
        for t in self.tracker.tracks:
            ref = t.ref if t.translation is not None else t.cand_ref
            if ref is None:
                continue
            b = t.block
            score, nx, ny = locate(pyr, ref, b.x0, b.y0, b.y1 - b.y0, self._vel)
            if score >= MOVE:
                t.bad = 0
                t.hidden = False
                if (nx, ny) != (b.x0, b.y0):
                    dx, dy = nx - b.x0, ny - b.y0
                    self._shift_lines(b.rect, dx, dy)
                    t.move(dx, dy)
                    moves.append((dx, dy))
            else:
                t.bad += 1
                # Clearly gone (covered by a window, scrolled away, replaced): hide this very frame.
                # Borderline match (e.g. an animated background behind the text): wait one more frame.
                if score < GONE or t.bad >= 2:
                    t.hidden = True
        if moves:
            self._vel = (statistics.median(m[0] for m in moves), statistics.median(m[1] for m in moves))
        else:
            self._vel = (0.0, 0.0)
        self._motion = (self._motion[0] + self._vel[0], self._motion[1] + self._vel[1])

    def _shift_lines(self, rect: tuple[int, int, int, int], dx: int, dy: int) -> None:
        if not self._lines:
            return
        x0, y0, x1, y1 = rect
        moved = []
        for l in self._lines:
            if l.x0 >= x0 - 4 and l.y0 >= y0 - 4 and l.x1 <= x1 + 4 and l.y1 <= y1 + 4:
                l = replace(l, x0=l.x0 + dx, y0=l.y0 + dy, x1=l.x1 + dx, y1=l.y1 + dy)
            moved.append(l)
        self._lines = moved
        self._obstacles = [l.rect for l in moved]

    def _maybe_submit(self, frame: np.ndarray, pyr: Pyramid, now: float) -> None:
        if (now - self._last_submit) * 1000 < self.cfg.ocr_interval_ms:
            return
        thumb = pyr.thumb()
        dirty = _dirty_rect(self._prev_thumb, thumb, self.cfg.change_threshold)
        for t in self.tracker.tracks:  # keep reading text that's still settling (typewriter effect)
            if t.stable_count < self.cfg.stable_passes:
                dirty = t.block.rect if dirty is None else _union(dirty, t.block.rect)
        refresh_due = now - self._last_refresh > 3.0
        if dirty is None and not refresh_due:
            return
        if now - self._last_full > 20.0 or self._lines is None:
            dirty = None  # occasional full re-read as a safety net
            self._last_full = now
        elif dirty is None:
            dirty = (0, 0, 0, 0)  # nothing changed: re-detect boxes, only brand-new ones get read
        else:
            H, W = frame.shape[:2]
            dirty = (max(0, dirty[0] - 8), max(0, dirty[1] - 8), min(W, dirty[2] + 8), min(H, dirty[3] + 8))
        self._prev_thumb = thumb
        self._last_submit = self._last_refresh = now
        self.worker.submit(_Job(frame, pyr, dirty, self._lines, self._motion))

    def _apply_ocr(self, res: _Result, pyr: Pyramid, frame: np.ndarray) -> None:
        """OCR ran on an older frame; place each block where its text is *now* before tracking it."""
        self.stats["ocr_ms"] = res.ms
        self.stats["mode"] = "full" if res.full else "partial"
        lag = (self._motion[0] - res.motion_mark[0], self._motion[1] - res.motion_mark[1])
        lines = res.lines
        if lag != (0, 0):
            lines = [replace(l, x0=l.x0 + int(lag[0]), y0=l.y0 + int(lag[1]), x1=l.x1 + int(lag[0]), y1=l.y1 + int(lag[1])) for l in lines]
        self._lines = lines
        self._obstacles = [l.rect for l in lines]
        src = [l for l in res.lines if has_source_text(l.text, self.cfg.source_language)]
        blocks, refs = [], []
        for b in group(src):
            ref = make_ref(res.pyr, b.rect)
            if ref is not None:
                score, nx, ny = locate(pyr, ref, b.x0 + int(lag[0]), b.y0 + int(lag[1]), b.y1 - b.y0, (0, 0))
                if score < MOVE:
                    continue  # already gone from the screen
                b = replace(b, x0=nx, y0=ny, x1=nx + (b.x1 - b.x0), y1=ny + (b.y1 - b.y0))
                ref = make_ref(pyr, b.rect)
            blocks.append(b)
            refs.append(ref)
        self.stats["blocks"] = len(blocks)
        lang, model = self.cfg.target_language, self.cfg.model
        ready = self.tracker.update(blocks, refs)
        # Text already in the translation memory doesn't need to settle first: show it on first sight.
        for tr in self.tracker.tracks:
            if tr not in ready and tr.translated_text != tr.block.text:
                hit = self.cache.get(tr.block.text, lang, model)
                if hit is not None:
                    tr.stable_text = tr.block.text
                    tr.text_color = estimate_text_color(frame, tr.block.rect) or tr.text_color
                    show(tr, tr.block.text, hit)
                    self.stats["cached"] += 1
        for tr in ready:
            text = tr.block.text
            tr.text_color = estimate_text_color(frame, tr.block.rect) or tr.text_color
            hit = self.cache.get(text, lang, model)
            if hit is not None:
                show(tr, text, hit)
                self.stats["cached"] += 1
            elif tr.requested_text != text:
                tr.requested_text = text
                self._pending[self.translator.submit(text)] = (tr.id, text)
                if tr.translation is not None and tr.translated_text != text:
                    tr.hidden = True  # old translation no longer matches what's on screen

    def _collect(self, frame: np.ndarray) -> None:
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
                    show(t, text, result)

    def _items(self, now: float) -> list[Item]:
        items = []
        for t in self.tracker.tracks:
            if not t.visible:
                continue
            fade_in = min(1.0, (now - t.shown_at) / 0.12) if t.shown_at else 1.0
            b = t.block
            items.append(Item(t.id, b.rect, b.line_h, b.n_lines, t.translation, t.text_color, fade_in))
        return items


def _dirty_rect(prev: np.ndarray | None, thumb: np.ndarray, threshold: float) -> tuple[int, int, int, int] | None:
    """Bounding box (full-res pixels) of what changed since the last OCR submission, or None."""
    h, w = thumb.shape
    if prev is None or prev.shape != thumb.shape:
        return 0, 0, w * 8, h * 8
    diff = cv2.absdiff(thumb, prev)
    if float(diff.mean()) < threshold * 0.15 and int(diff.max()) < 24:
        return None
    ys, xs = np.nonzero(diff > 12)
    if len(xs) == 0:
        return None
    return int(xs.min()) * 8, int(ys.min()) * 8, (int(xs.max()) + 1) * 8, (int(ys.max()) + 1) * 8


def _union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])



