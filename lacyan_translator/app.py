"""Tray app: starts everything, owns the menu, and keeps the user informed."""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler

import mss
from PySide6.QtCore import QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QDialog, QMenu, QSystemTrayIcon

from . import __version__, model_setup
from .config import CONFIG_PATH, DATA_DIR, Config, data_path, open_in_explorer
from .engine import Engine
from .hotkeys import Hotkeys
from .langs import TARGETS
from . import hardware
from .launcher import LaunchDialog, PerformanceDialog, logo
from .overlay import Overlay

log = logging.getLogger("lacyan")


def setup_logging() -> None:
    handler = RotatingFileHandler(data_path("lacyan.log"), maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def make_icon(paused: bool = False) -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(70, 70, 78) if paused else QColor(36, 116, 98))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(2, 2, 60, 60), 14, 14)
    f = QFont("Microsoft YaHei")
    f.setPixelSize(40)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(245, 245, 240))
    p.drawText(QRectF(0, 0, 64, 62), Qt.AlignmentFlag.AlignCenter, "译")
    p.end()
    return QIcon(pm)


class Bridge(QObject):
    """Moves callbacks from background threads onto the GUI thread."""

    call = Signal(object)


class TranslatorApp:
    def __init__(self, app: QApplication, cfg: Config) -> None:
        self.app = app
        self.cfg = cfg
        self.bridge = Bridge()
        self.bridge.call.connect(lambda fn: fn())
        with mss.mss() as sct:
            mons = sct.monitors
        self.monitor_index = self.cfg.monitor if 0 < self.cfg.monitor < len(mons) else 1
        self.monitor = mons[self.monitor_index]

        self.overlay = Overlay(self.cfg, self.monitor)
        self.engine: Engine | None = None
        self.overlay_on = True

        self.tray = QSystemTrayIcon(make_icon(), app)
        self.tray.setToolTip(f"Lacyan Translator {__version__} · starting")
        self._build_menu()
        self.tray.show()

        self.hotkeys = Hotkeys({
            self.cfg.hotkey_toggle: lambda: self.bridge.call.emit(self.toggle_overlay),
            self.cfg.hotkey_pause: lambda: self.bridge.call.emit(self.toggle_pause),
            self.cfg.hotkey_quit: lambda: self.bridge.call.emit(self.quit),
        })
        threading.Thread(target=self._prepare, daemon=True).start()

    # --- startup -----------------------------------------------------------
    def _prepare(self) -> None:
        local_ollama = "127.0.0.1:11434" in self.cfg.endpoint or "localhost:11434" in self.cfg.endpoint
        if local_ollama and model_setup.variant_of(self.cfg.model) is not None:
            if not model_setup.start_ollama():
                self._notify("Ollama isn't running", "Start Ollama, then Lacyan Translator will connect automatically.")
                while model_setup.ollama_models() is None:
                    threading.Event().wait(5)
            if not model_setup.has_model(self.cfg.model):
                self._notify("Setting up", "Downloading the translation model (first time only)…")
                result = model_setup.ensure_model(self.cfg.model)
                if result not in ("ok", "created"):
                    self._notify("Model setup failed", "See data/lacyan.log for details.")
                    return
        self.bridge.call.emit(self._start_engine)

    def _start_engine(self) -> None:
        self.engine = Engine(self.cfg, self.monitor, self.monitor_index, {int(self.overlay.winId())})
        self.overlay.attach(self.engine)
        self.engine.status.connect(self._status)
        self.overlay.show()
        self.engine.start()
        self._notify("Lacyan Translator is running", f"Translating {self.cfg.source_language} → {self.cfg.target_language}. "
                     f"{self.cfg.hotkey_toggle.upper()} shows the original, {self.cfg.hotkey_pause.upper()} pauses.")
        QTimer.singleShot(4000, self._check_capture_hidden)

    def _check_capture_hidden(self) -> None:
        if self.cfg.hide_from_capture and not self.overlay.capture_hidden:
            self._notify("Heads up", "Windows couldn't hide the overlay from capture; translations may flicker.")

    # --- menu --------------------------------------------------------------
    def _build_menu(self) -> None:
        m = QMenu()
        self.status_action = m.addAction("Starting…")
        self.status_action.setEnabled(False)
        m.addSeparator()
        m.addAction("Languages && options…", self.open_options)
        self.pause_action = m.addAction("Pause", self.toggle_pause)
        self.toggle_action = m.addAction("Show original text", self.toggle_overlay)

        lang_menu = m.addMenu("Translate to")
        group = QActionGroup(lang_menu)
        self.lang_actions = {}
        for name in TARGETS:
            a = QAction(name, lang_menu, checkable=True, checked=name == self.cfg.target_language)
            a.triggered.connect(lambda _=False, n=name: self.set_target(n))
            group.addAction(a)
            lang_menu.addAction(a)
            self.lang_actions[name] = a

        cap_menu = m.addMenu("Capture")
        cgroup = QActionGroup(cap_menu)
        self.cap_actions = {}
        for key, label in (("foreground", "Active window"), ("monitor", "Whole screen")):
            a = QAction(label, cap_menu, checkable=True, checked=self.cfg.capture == key)
            a.triggered.connect(lambda _=False, k=key: self.set_capture(k))
            cgroup.addAction(a)
            cap_menu.addAction(a)
            self.cap_actions[key] = a

        m.addSeparator()
        m.addAction("Open settings file", lambda: open_in_explorer(CONFIG_PATH))
        m.addAction("Open data folder", lambda: open_in_explorer(DATA_DIR))
        m.addAction("Clear translation memory", self.clear_cache)
        m.addSeparator()
        m.addAction("Quit", self.quit)
        self.menu = m
        self.tray.setContextMenu(m)
        self.tray.activated.connect(lambda r: self.open_options() if r == QSystemTrayIcon.ActivationReason.Trigger else None)
        timer = QTimer(self.app)
        timer.timeout.connect(self._refresh_status)
        timer.start(1000)

    def _refresh_status(self) -> None:
        if not self.engine:
            return
        s = self.engine.stats
        state = "Paused" if self.engine.paused else f"OCR {s['ocr_ms']:.0f} ms · {s['blocks']} blocks"
        self.status_action.setText(f"{state} · {s['translated']} new / {s['cached']} from memory")

    # --- actions -----------------------------------------------------------
    def open_options(self) -> None:
        if getattr(self, "_dialog", None) and self._dialog.isVisible():
            self._dialog.raise_()
            return
        before = self.cfg.model
        self._dialog = LaunchDialog(self.cfg, running=True)
        if self._dialog.exec():
            self.set_target(self.cfg.target_language)
            self.set_capture(self.cfg.capture)
            if self.cfg.model != before:
                threading.Thread(target=self._switch_model, args=(before,), daemon=True).start()
        self._dialog = None

    def _switch_model(self, previous: str) -> None:
        """Download the newly chosen model size if needed, then switch to it (keep the old one meanwhile)."""
        chosen = self.cfg.model
        if not model_setup.has_model(chosen):
            self.cfg.model = previous
            self._notify("Downloading", "Getting the new translation model; it switches over when ready.")
            if model_setup.ensure_model(chosen) not in ("ok", "created"):
                self._notify("Download failed", "Kept the previous model. See data/lacyan.log.")
                self.cfg.save()
                return
            self.cfg.model = chosen
        self.cfg.save()
        if self.engine:
            self.bridge.call.emit(self.engine.reset)
        self._notify("Model switched", f"Now translating with {chosen}.")

    def toggle_overlay(self) -> None:
        self.overlay_on = not self.overlay_on
        self.overlay.setVisible(self.overlay_on)
        self.toggle_action.setText("Show original text" if self.overlay_on else "Show translations")

    def toggle_pause(self) -> None:
        if not self.engine:
            return
        self.engine.paused = not self.engine.paused
        if self.engine.paused:
            self.engine.tracker.clear()
        self.pause_action.setText("Resume" if self.engine.paused else "Pause")
        self.tray.setIcon(make_icon(self.engine.paused))

    def set_target(self, name: str) -> None:
        self.cfg.target_language = name
        self.cfg.save()
        if name in self.lang_actions:
            self.lang_actions[name].setChecked(True)
        if self.engine:
            self.engine.set_target(name)

    def set_capture(self, key: str) -> None:
        self.cfg.capture = key
        self.cfg.save()
        if key in self.cap_actions:
            self.cap_actions[key].setChecked(True)

    def clear_cache(self) -> None:
        if self.engine:
            self.engine.cache.clear()
            self.engine.tracker.clear()

    def quit(self) -> None:
        if self.engine:
            self.engine.stop()
        self.hotkeys.stop()
        self.tray.hide()
        self.app.quit()

    def _status(self, text: str) -> None:
        self.tray.setToolTip(f"Lacyan Translator {__version__} · {text}")
        if "error" in text.lower() or "failed" in text.lower():
            self.status_action.setText(text[:80])

    def _notify(self, title: str, body: str) -> None:
        self.bridge.call.emit(lambda: self.tray.showMessage(f"Lacyan Translator · {title}", body, make_icon(), 6000))


def run() -> int:
    setup_logging()
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\LacyanTranslator")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        return 0
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("Lacyan Translator")
    app.setWindowIcon(QIcon(logo(64)))
    cfg = Config.load()
    if not cfg.profile_confirmed:  # first launch: check the hardware and confirm a performance profile
        if PerformanceDialog(cfg, hardware.detect()).exec() != QDialog.DialogCode.Accepted:
            ctypes.windll.kernel32.CloseHandle(mutex)
            return 0
    if cfg.show_launcher and LaunchDialog(cfg).exec() != QDialog.DialogCode.Accepted:
        ctypes.windll.kernel32.CloseHandle(mutex)
        return 0
    translator_app = TranslatorApp(app, cfg)  # noqa: F841 (kept alive by the event loop)
    code = app.exec()
    ctypes.windll.kernel32.CloseHandle(mutex)
    return code
