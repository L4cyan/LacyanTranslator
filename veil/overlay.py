"""Full-screen, transparent, click-through window that paints translations over the game."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication, QPainter, QScreen
from PySide6.QtWidgets import QWidget

from . import render, winutil
from .engine import Frame


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
        self.frame: Frame | None = None
        self.capture_hidden = False
        self._layouts: dict = {}

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        self.capture_hidden = winutil.make_overlay_window(int(self.winId()), self.cfg.hide_from_capture)

    def on_frame(self, frame: Frame) -> None:
        self.frame = frame
        if len(self._layouts) > 400:
            self._layouts.clear()
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        f = self.frame
        if not f or not f.items:
            return
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        dpr = self.screen_.devicePixelRatio()
        p.scale(1 / dpr, 1 / dpr)
        p.translate(f.off_x, f.off_y)
        render.draw_all(p, f.image, f.items, self.font_, self.cfg, self._layouts, f.obstacles)
        p.end()
