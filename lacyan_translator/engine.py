"""The pipeline.

Capture thread (up to the profile's fps; no work at all on frames where nothing changed):
    grab → follow every translation (move with scrolling, hide the instant its text is gone)
    → apply finished OCR / translations → publish the latest frame for the overlay.
OCR thread: detect + read text on frames the capture thread hands it, so following never waits on OCR.
Translation threads: batched, streamed calls to the local model, top of the screen first, behind the
exact-match translation memory and phrasebook.
"""

from __future__ import annotations

import logging
import queue
import statistics
import threading
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from . import winutil
from .cache import TranslationCache
from .capture import Capturer
from .config import Config, data_path
from .langs import has_source_text
from .layout import group
from .motion import Pyramid, locate, make_ref, score_at, thresholds
from .ocr import OCR, Line
from .render import Item, estimate_text_color
from .tracker import Tracker, show
from .translator import LONG, Translator

log = logging.getLogger("lacyan.engine")
HIDDEN_TTL = 0.8  # seconds a hidden translation waits for its text to come back before it's dropped


@dataclass
class Frame:
    """The latest captured frame plus everything the overlay needs (physical pixels)."""

    image: np.ndarray
    off_x: int
    off_y: int
    items: list[Item]
    obstacles: list[tuple[int, int, int, int]]  # every OCR'd line; grown translations must not cover them
    t: float = 0.0  # capture time
    vel: tuple[float, float] = (0.0, 0.0)  # scroll speed in px/s, for drawing slightly ahead of the capture


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


class TranslationQueue:
    """Texts waiting for the model. Workers take the topmost texts still on screen, in batches."""

    def __init__(self, translator: Translator, cfg: Config) -> None:
        self.tr = translator
        self.cfg = cfg
        self.results: queue.Queue[tuple[str, str]] = queue.Queue()
        self.errors: queue.Queue[str] = queue.Queue()
        self._wanted: dict[str, int] = {}  # text -> y on screen
        self._inflight: set[str] = set()
        self._cv = threading.Condition()
        self._stop = False
        for i in range(max(1, cfg.translation_workers)):
            threading.Thread(target=self._run, name=f"lacyan-mt{i}", daemon=True).start()

    def want(self, wanted: dict[str, int]) -> None:
        """Replace the set of texts that still need translating (texts no longer on screen are dropped)."""
        with self._cv:
            self._wanted = {t: y for t, y in wanted.items() if t not in self._inflight}
            if self._wanted:
                self._cv.notify_all()

    def busy(self) -> bool:
        return bool(self._wanted) or bool(self._inflight)

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def _take(self) -> list[str]:
        order = sorted(self._wanted.items(), key=lambda kv: kv[1])
        first = order[0][0]
        if len(first) > LONG:
            batch = [first]
        else:
            batch = [t for t, _ in order if len(t) <= LONG][: max(1, self.cfg.batch_size)]
        for t in batch:
            self._wanted.pop(t, None)
            self._inflight.add(t)
        return batch

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._wanted and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                batch = self._take()
            try:
                self.tr.translate_batch(batch, lambda src, out: self.results.put((src, out)))
            except Exception as e:  # noqa: BLE001 - model offline etc.; texts get re-requested later
                log.warning("Translation failed: %s", e)
                self.errors.put(str(e))
            finally:
                with self._cv:
                    self._inflight.difference_update(batch)


class Engine(QObject):
    frame_ready = Signal()  # "a new frame is waiting"; the overlay pulls only the latest one
    status = Signal(str)

    def __init__(self, cfg: Config, monitor: dict, monitor_index: int, ignore_hwnds: set[int]) -> None:
        super().__init__()
        self.cfg = cfg
        self.monitor = monitor
        self.monitor_index = monitor_index
        self.ignore = ignore_hwnds
        self.paused = False
        self._stop = threading.Event()
        self.cache = TranslationCache(str(data_path("translations.sqlite")))
        self.translator = Translator(cfg)
        self.queue = TranslationQueue(self.translator, cfg)
        self.tracker = Tracker(cfg.stable_passes, cfg.keep_missing_passes)
        self._region: tuple[int, int, int, int] | None = None
        self._lines: list[Line] | None = None
        self._obstacles: list[tuple[int, int, int, int]] = []
        self._pyr: Pyramid | None = None
        self._prev_thumb: np.ndarray | None = None
        self._last_submit = self._last_refresh = self._last_full = self._last_log = self._last_change = 0.0
        self._last_tick = 0.0
        self._vel = (0.0, 0.0)  # per-frame motion of followed text, for prediction
        self._vel_s = (0.0, 0.0)  # the same in px/s, for the overlay's look-ahead
        self._motion = (0.0, 0.0)  # accumulated motion, to place OCR results that arrive late
        self._force_ocr = False
        self._latest: Frame | None = None
        self._latest_lock = threading.Lock()
        self._gui_waiting = False
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
        self.queue.stop()

    def set_target(self, language: str) -> None:
        self.cfg.target_language = language
        self.reset()

    def take_latest(self) -> Frame | None:
        with self._latest_lock:
            self._gui_waiting = False
            return self._latest

    def _publish(self, frame: Frame) -> None:
        with self._latest_lock:
            self._latest = frame
            notify = not self._gui_waiting
            self._gui_waiting = True
        if notify:  # never queue more than one: the overlay always draws the newest frame
            self.frame_ready.emit()

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
        cap = Capturer(self.monitor, self.monitor_index)
        self.stats["capture"] = cap.backend
        self.translator.warm_up()
        frames, fps_t = 0, time.time()
        while not self._stop.is_set():
            t0 = time.perf_counter()
            region = self._capture_region()
            if region != self._region:
                old = self._region
                same_size = old and (region[2] - region[0], region[3] - region[1]) == (old[2] - old[0], old[3] - old[1])
                if not same_size:  # a different window (or resized): start fresh
                    self.reset()
                self._region = region  # merely moved: coordinates are window-relative, keep everything
            frame, changed = cap.grab(region)
            if frame is not None:
                now = time.time()
                items = self.tick(frame, changed, now)
                self._publish(Frame(frame, region[0] - self.monitor["left"], region[1] - self.monitor["top"], items,
                                    self._obstacles, now, self._vel_s))
                frames += 1
            now = time.time()
            if now - fps_t >= 1:
                self.stats["fps"], frames, fps_t = frames / (now - fps_t), 0, now
            if now - self._last_log > 30:
                self._last_log = now
                log.info("stats %s cache=%d", self.stats, self.cache.size())
            # Full speed while things move or are being worked on; relaxed when the screen is still.
            active = not self.paused and (now - self._last_change < 0.5 or self.queue.busy()
                                          or (self.worker is not None and self.worker.busy))
            fps = self.cfg.max_fps if active else min(15, self.cfg.max_fps)
            self._stop.wait(max(0.0, 1 / fps - (time.perf_counter() - t0)))
        cap.close()

    def reset(self) -> None:
        self.tracker.clear()
        self.queue.want({})
        self._lines = None
        self._obstacles = []
        self._prev_thumb = None
        self._vel = self._vel_s = (0.0, 0.0)
        self._force_ocr = True

    # --- one frame ---------------------------------------------------------
    def tick(self, frame: np.ndarray, changed: bool, now: float) -> list[Item]:
        if self.paused:
            return []
        if changed or self._pyr is None or self._pyr.w != frame.shape[1] or self._pyr.h != frame.shape[0]:
            self._last_change = now
            dt = max(1e-3, now - self._last_tick) if self._last_tick else 1 / 30
            self._pyr = Pyramid(frame)
            self._follow(self._pyr, now)
            self._vel_s = (self._vel[0] / dt, self._vel[1] / dt)
        else:
            self._vel = self._vel_s = (0.0, 0.0)
            for t in self.tracker.tracks:
                t.moved_now = False
        self._last_tick = now
        pyr = self._pyr
        if self.worker is not None:
            while not self.worker.outbox.empty():
                self._apply_ocr(self.worker.outbox.get_nowait(), pyr, frame)
            if not self.worker.busy and self.worker.device:
                self._maybe_submit(frame, pyr, now)
        self._collect()
        return self._items(now)

    def _follow(self, pyr: Pyramid, now: float) -> None:
        """Move each tracked block with its text; hide its translation the moment the text is gone."""
        moves, lost, tracked = [], [], 0
        for t in self.tracker.tracks:
            t.moved_now = False
            ref = t.ref if t.translation is not None else t.cand_ref
            if ref is None:
                continue
            tracked += 1
            b = t.block
            stay, move, gone = thresholds(ref)
            if t.hidden:
                # Hidden: only come back if the text is back exactly where it was (or where it scrolled to);
                # never search widely, which could latch onto similar-looking text on a new page.
                s = max(score_at(pyr, ref, b.x0, b.y0), score_at(pyr, ref, b.x0 + self._vel[0], b.y0 + self._vel[1]))
                if s >= stay:
                    t.hidden, t.bad, t.hidden_at = False, 0, 0.0
                continue
            # Scrolling off the edge of the screen: hide now. A half-visible block must never be searched
            # for, or it could latch onto similar text elsewhere (e.g. the next product card).
            px0, py0 = b.x0 + self._vel[0], b.y0 + self._vel[1]
            if px0 < 0 or py0 < 0 or px0 + (b.x1 - b.x0) > pyr.w or py0 + (b.y1 - b.y0) > pyr.h:
                t.hidden, t.hidden_at = True, now
                lost.append(t)
                continue
            score, nx, ny = locate(pyr, ref, b.x0, b.y0, b.y1 - b.y0, self._vel, stay, move)
            if score >= move:
                t.bad = 0
                if (nx, ny) != (b.x0, b.y0):
                    dx, dy = nx - b.x0, ny - b.y0
                    self._shift_lines(b.rect, dx, dy)
                    t.move(dx, dy)
                    t.moved_now = True
                    moves.append((dx, dy))
            else:
                t.bad += 1
                # Clearly gone (covered, scrolled away, replaced): hide this very frame.
                # Borderline (e.g. animated background behind the text): wait one more frame.
                if score < gone or t.bad >= 2:
                    t.hidden, t.hidden_at = True, now
                    lost.append(t)
        self._vel = (statistics.median(m[0] for m in moves), statistics.median(m[1] for m in moves)) if moves else (0.0, 0.0)
        self._motion = (self._motion[0] + self._vel[0], self._motion[1] + self._vel[1])
        # Most of the screen's text vanished at once and nothing scrolled: it's a new page / screen.
        if tracked >= 3 and len(lost) >= 0.6 * tracked and not moves and self._vel == (0.0, 0.0):
            self._new_screen()
        # Hidden translations whose text never came back are dropped.
        self.tracker.tracks = [t for t in self.tracker.tracks if not (t.hidden and now - t.hidden_at > HIDDEN_TTL)]

    def _new_screen(self) -> None:
        self.tracker.tracks = [t for t in self.tracker.tracks if not t.hidden]
        self._lines = None
        self._force_ocr = True  # read the new screen right away, not on the next interval

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
        if not self._force_ocr and (now - self._last_submit) * 1000 < self.cfg.ocr_interval_ms:
            return
        thumb = pyr.thumb()
        dirty = _dirty_rect(self._prev_thumb, thumb, self.cfg.change_threshold)
        for t in self.tracker.tracks:  # keep reading text that's still settling (typewriter effect)
            if t.stable_count < self.cfg.stable_passes:
                dirty = t.block.rect if dirty is None else _union(dirty, t.block.rect)
        refresh_due = now - self._last_refresh > 3.0
        if dirty is None and not refresh_due and not self._force_ocr:
            return
        if self._force_ocr or now - self._last_full > 20.0 or self._lines is None:
            dirty = None  # whole-frame read: new screen, or the occasional safety net
            self._last_full = now
        elif dirty is None:
            dirty = (0, 0, 0, 0)  # nothing changed: re-detect boxes, only brand-new ones get read
        else:
            H, W = frame.shape[:2]
            dirty = (max(0, dirty[0] - 8), max(0, dirty[1] - 8), min(W, dirty[2] + 8), min(H, dirty[3] + 8))
        self._force_ocr = False
        self._prev_thumb = thumb
        self._last_submit = self._last_refresh = now
        self.worker.submit(_Job(np.array(frame), pyr, dirty, self._lines, self._motion))

    def _apply_ocr(self, res: _Result, pyr: Pyramid, frame: np.ndarray) -> None:
        """OCR ran on an older frame; place each block where its text is *now* before tracking it."""
        self.stats["ocr_ms"] = res.ms
        self.stats["mode"] = "full" if res.full else "partial"
        lag = (int(self._motion[0] - res.motion_mark[0]), int(self._motion[1] - res.motion_mark[1]))
        lines = res.lines
        if lag != (0, 0):
            lines = [replace(l, x0=l.x0 + lag[0], y0=l.y0 + lag[1], x1=l.x1 + lag[0], y1=l.y1 + lag[1]) for l in lines]
        self._lines = lines
        self._obstacles = [l.rect for l in lines]
        src = [l for l in res.lines if has_source_text(l.text, self.cfg.source_language)]
        blocks, refs = [], []
        for b in group(src):
            ref = make_ref(res.pyr, b.rect)
            if ref is not None:
                stay, move, _ = thresholds(ref)
                score, nx, ny = locate(pyr, ref, b.x0 + lag[0], b.y0 + lag[1], b.y1 - b.y0, (0, 0), stay, move)
                if score < move:
                    continue  # already gone from the screen
                b = replace(b, x0=nx, y0=ny, x1=nx + (b.x1 - b.x0), y1=ny + (b.y1 - b.y0))
                ref = make_ref(pyr, b.rect)
            else:
                b = replace(b, x0=b.x0 + lag[0], y0=b.y0 + lag[1], x1=b.x1 + lag[0], y1=b.y1 + lag[1])
            blocks.append(b)
            refs.append(ref)
        self.stats["blocks"] = len(blocks)
        ready = {id(t) for t in self.tracker.update(blocks, refs)}
        for t in self.tracker.tracks:
            text = t.block.text
            if t.translated_text == text:
                continue
            hit = self.translator.phrase(text) or self.cache.get(text, self.cfg.target_language, self.cfg.model)
            if hit is not None:  # exact known phrase: show it on first sight, no need to wait for it to settle
                t.stable_text = text
                t.text_color = estimate_text_color(frame, t.block.rect) or t.text_color
                show(t, text, hit)
                self.stats["cached"] += 1
            elif id(t) in ready:
                t.text_color = estimate_text_color(frame, t.block.rect) or t.text_color
                if t.translation is not None:
                    t.hidden, t.hidden_at = True, time.time()  # old translation no longer matches the screen
        self._request_translations()

    def _request_translations(self) -> None:
        wanted: dict[str, int] = {}
        for t in self.tracker.tracks:
            if t.stable_text and t.translated_text != t.stable_text and t.stable_text == t.block.text:
                wanted[t.stable_text] = min(wanted.get(t.stable_text, 1 << 30), t.block.y0)
        self.queue.want(wanted)

    def _collect(self) -> None:
        got = False
        while not self.queue.results.empty():
            text, result = self.queue.results.get_nowait()
            if not result:
                continue
            got = True
            self.cache.put(text, self.cfg.target_language, self.cfg.model, result)
            self.stats["translated"] += 1
            for t in self.tracker.tracks:
                if t.stable_text == text and t.translated_text != text:
                    show(t, text, result)
        while not self.queue.errors.empty():
            self.status.emit(f"Translation error: {self.queue.errors.get_nowait()}")
        if got:
            self._request_translations()

    def _items(self, now: float) -> list[Item]:
        items = []
        for t in self.tracker.tracks:
            if not t.visible:
                continue
            fade_in = min(1.0, (now - t.shown_at) / 0.12) if t.shown_at else 1.0
            b = t.block
            items.append(Item(t.id, b.rect, b.line_h, b.n_lines, t.translation, t.text_color, fade_in, t.moved_now))
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
