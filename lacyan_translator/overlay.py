"""Full-screen, transparent, click-through window that paints translations over the game."""

from __future__ import annotations

import logging
import math
import time

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QFont, QGuiApplication, QPainter, QRegion, QScreen
from PySide6.QtWidgets import QWidget

from . import render, winutil
from .engine import Frame

log = logging.getLogger("lacyan.overlay")


def screen_for(monitor: dict) -> QScreen:
    """Match an mss monitor (physical pixels) to the Qt screen showing it."""
    best = QGuiApplication.primaryScreen()
    for s in QGuiApplication.screens():
        g, dpr = s.geometry(), s.devicePixelRatio()
        if abs(g.width() * dpr - monitor["width"]) < 4 and abs(g.height() * dpr - monitor["height"]) < 4:
            if abs(g.x() * dpr - monitor["left"]) < 4 or s == best:
                best = s
    return best


class Overlay(QWidget):
    def __init__(self, cfg, monitor: dict) -> None:
        super().__init__(None)
        self.cfg = cfg
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.screen_ = screen_for(monitor)
        self.setGeometry(self.screen_.geometry())
        self.font_ = QFont(cfg.font_family)
        self.font_.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self.font_.setWeight(QFont.Weight.DemiBold)
        self.capture_hidden = False
        self._cache = render.RenderCache()
        self._plans: list[render.Plan] = []
        self._offset = (0, 0)
        self._painted: list[QRect] = []  # logical rects drawn last time, cleared on the next update
        self._sig = None

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        self.capture_hidden = winutil.make_overlay_window(int(self.winId()), self.cfg.hide_from_capture)
        g = self.geometry()
        log.info("Overlay on %s (%dx%d @ %.2fx), hidden from capture: %s",
                 self.screen_.name(), g.width(), g.height(), self.screen_.devicePixelRatio(), self.capture_hidden)

    def on_frame(self, frame: Frame) -> None:
        """Plan this frame; repaint only the areas that changed (nothing at all if nothing did)."""
        plans = render.plan_all(frame.image, frame.items, self.font_, self.cfg, self._cache, frame.obstacles, time.time())
        sig = (frame.off_x, frame.off_y, tuple(
            (p.item.id, p.rect, p.item.text, round(p.item.opacity, 2), p.img.cacheKey()) for p in plans))
        if sig == self._sig:
            return
        self._sig = sig
        dpr = self.screen_.devicePixelRatio()
        ox, oy = frame.off_x, frame.off_y
        rects = []
        for p in plans:
            x0, y0, x1, y1 = p.bounds
            rects.append(QRect(
                math.floor((x0 + ox) / dpr) - 2, math.floor((y0 + oy) / dpr) - 2,
                math.ceil((x1 - x0) / dpr) + 5, math.ceil((y1 - y0) / dpr) + 5,
            ))
        region = QRegion()
        for r in self._painted + rects:
            region = region.united(r)
        self._plans, self._offset, self._painted = plans, (ox, oy), rects
        if not region.isEmpty():
            self.update(region)

    def clear(self) -> None:
        self._plans, self._sig = [], None
        region = QRegion()
        for r in self._painted:
            region = region.united(r)
        self._painted = []
        self.update(region)

    def paintEvent(self, _e) -> None:  # noqa: N802
        if not self._plans:
            return
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        dpr = self.screen_.devicePixelRatio()
        p.scale(1 / dpr, 1 / dpr)
        p.translate(*self._offset)
        render.paint_all(p, self._plans, self.cfg)
        p.end()
