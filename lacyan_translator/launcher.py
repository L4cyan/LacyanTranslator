"""The window shown at launch: pick languages, see that everything's ready, press Start."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QVBoxLayout,
)

from . import __version__, hardware, model_setup
from .config import Config
from .langs import TARGETS

STYLE = """
QDialog { background: #15191a; }
QLabel { color: #d9e2de; font-size: 13px; }
QLabel#title { color: #f2f5f3; font-size: 22px; font-weight: 600; }
QLabel#subtitle { color: #8fa39b; font-size: 12px; }
QLabel#field { color: #8fa39b; font-size: 11px; font-weight: 600; letter-spacing: 1px; }
QLabel#status { font-size: 12px; padding: 8px 10px; border-radius: 8px; background: #1d2426; }
QLabel#hint { color: #7b8d86; font-size: 11px; }
QComboBox { background: #1f2729; color: #eef2f0; border: 1px solid #2f3b3d; border-radius: 8px;
            padding: 7px 10px; font-size: 14px; min-width: 150px; }
QComboBox:hover, QComboBox:focus { border-color: #3fa184; }
QComboBox QAbstractItemView { background: #1f2729; color: #eef2f0; selection-background-color: #24745f; }
QComboBox:disabled { color: #a9b8b2; }
QRadioButton, QCheckBox { color: #c9d4cf; font-size: 13px; spacing: 7px; }
QPushButton { border-radius: 9px; padding: 10px 18px; font-size: 14px; font-weight: 600; }
QPushButton#start { background: #24745f; color: white; border: none; }
QPushButton#start:hover { background: #2b8a70; }
QPushButton#start:disabled { background: #2a3436; color: #6f807a; }
QPushButton#quit { background: transparent; color: #a7b6b0; border: 1px solid #2f3b3d; }
QPushButton#quit:hover { color: #e6ecea; border-color: #4a5a5c; }
"""


def logo(size: int = 56) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(36, 116, 95))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(0, 0, size, size, size * 0.24, size * 0.24)
    f = QFont("Microsoft YaHei")
    f.setPixelSize(int(size * 0.6))
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(245, 245, 240))
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "译")
    p.end()
    return pm


class _Check(QObject):
    done = Signal(str, str)  # level, message


class LaunchDialog(QDialog):
    def __init__(self, cfg: Config, running: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Lacyan Translator")
        self.setWindowIcon(QIcon(logo(64)))
        self.setStyleSheet(STYLE)
        pal = self.palette()
        for role in (QPalette.ColorRole.Accent, QPalette.ColorRole.Highlight):
            pal.setColor(role, QColor(43, 138, 112))
        self.setPalette(pal)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setMinimumWidth(460)

        head = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(logo(52))
        titles = QVBoxLayout()
        title = QLabel("Lacyan Translator")
        title.setObjectName("title")
        sub = QLabel(f"Live on-screen translation · v{__version__}")
        sub.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(sub)
        head.addWidget(icon)
        head.addSpacing(10)
        head.addLayout(titles)
        head.addStretch()

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        for col, text in enumerate(("FROM", "", "TO")):
            lab = QLabel(text)
            lab.setObjectName("field")
            grid.addWidget(lab, 0, col)
        self.src = QComboBox()
        self.src.addItem("Chinese")
        self.src.setEnabled(False)  # more source languages are on the roadmap
        arrow = QLabel("→")
        arrow.setStyleSheet("color:#3fa184; font-size:20px;")
        self.dst = QComboBox()
        self.dst.addItems(list(TARGETS))
        self.dst.setCurrentText(cfg.target_language if cfg.target_language in TARGETS else "English")
        grid.addWidget(self.src, 1, 0)
        grid.addWidget(arrow, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.dst, 1, 2)

        cap_label = QLabel("TRANSLATE")
        cap_label.setObjectName("field")
        self.cap_window = QRadioButton("The active window (best for games)")
        self.cap_screen = QRadioButton("The whole screen")
        group = QButtonGroup(self)
        group.addButton(self.cap_window)
        group.addButton(self.cap_screen)
        (self.cap_screen if cfg.capture == "monitor" else self.cap_window).setChecked(True)

        perf_label = QLabel("PERFORMANCE")
        perf_label.setObjectName("field")
        self.perf = QComboBox()
        self.perf.addItems(list(hardware.PROFILES))
        self.perf.setCurrentText(cfg.profile if cfg.profile in hardware.PROFILES else "Balanced")
        self.perf_hint = QLabel(hardware.DESCRIPTIONS[self.perf.currentText()])
        self.perf_hint.setObjectName("hint")
        self.perf_hint.setWordWrap(True)
        self.perf.currentTextChanged.connect(lambda n: self.perf_hint.setText(hardware.DESCRIPTIONS[n]))

        self.status = QLabel("Checking the translation model…")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)

        hint = QLabel("Alt+T  show original   ·   Alt+P  pause   ·   Ctrl+Alt+Q  quit")
        hint.setObjectName("hint")
        self.again = QCheckBox("Show this window when Lacyan Translator starts")
        self.again.setChecked(cfg.show_launcher)

        buttons = QHBoxLayout()
        quit_btn = QPushButton("Close" if running else "Quit")
        quit_btn.setObjectName("quit")
        quit_btn.clicked.connect(self.reject)
        self.start = QPushButton("Apply" if running else "Start translating")
        self.start.setObjectName("start")
        self.start.setDefault(True)
        self.start.clicked.connect(self._accept)
        buttons.addWidget(quit_btn)
        buttons.addStretch()
        buttons.addWidget(self.start)

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 24, 26, 22)
        root.setSpacing(14)
        root.addLayout(head)
        root.addSpacing(4)
        root.addLayout(grid)
        root.addWidget(cap_label)
        root.addWidget(self.cap_window)
        root.addWidget(self.cap_screen)
        root.addWidget(perf_label)
        root.addWidget(self.perf)
        root.addWidget(self.perf_hint)
        root.addWidget(self.status)
        root.addWidget(hint)
        root.addWidget(self.again)
        root.addSpacing(4)
        root.addLayout(buttons)

        self._check = _Check()
        self._check.done.connect(self._show_status)
        if running:
            self._show_status("ok", "Translating now. Changes apply right away.")
        else:
            threading.Thread(target=self._probe, daemon=True).start()

    def _probe(self) -> None:
        if not self._uses_local_ollama():
            self._check.done.emit("ok", f"Using {self.cfg.model} at {self.cfg.endpoint}.")
            return
        if model_setup.ollama_models() is None:
            self._check.done.emit("wait", "Starting Ollama…")
            if not model_setup.start_ollama():
                self._check.done.emit("bad", "Ollama isn't installed or won't start. Install it from ollama.com, then reopen this.")
                return
        models = model_setup.ollama_models() or []
        if any(m.split(":")[0] == self.cfg.model for m in models):
            self._check.done.emit("ok", "Translation model ready. Everything runs on this PC.")
        else:
            self._check.done.emit("ok", "First start: the translation model (about 1.9 GB) downloads once after you press Start.")

    def _uses_local_ollama(self) -> bool:
        return any(h in self.cfg.endpoint for h in ("127.0.0.1:11434", "localhost:11434"))

    def _show_status(self, level: str, message: str) -> None:
        colors = {"ok": "#7fd6b5", "wait": "#e3c26c", "bad": "#f0918a"}
        self.status.setStyleSheet(f"color: {colors.get(level, '#d9e2de')};")
        self.status.setText(message)
        self.start.setEnabled(level != "bad")

    def _accept(self) -> None:
        self.cfg.target_language = self.dst.currentText()
        self.cfg.capture = "monitor" if self.cap_screen.isChecked() else "foreground"
        self.cfg.show_launcher = self.again.isChecked()
        hardware.apply_profile(self.cfg, self.perf.currentText())
        self.cfg.save()
        self.accept()


class PerformanceDialog(QDialog):
    """First launch: show the detected hardware, recommend a profile, and ask the user to confirm it."""

    def __init__(self, cfg: Config, hw: "hardware.Hardware") -> None:
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Lacyan Translator · Performance setup")
        self.setWindowIcon(QIcon(logo(64)))
        self.setStyleSheet(STYLE)
        pal = self.palette()
        for role in (QPalette.ColorRole.Accent, QPalette.ColorRole.Highlight):
            pal.setColor(role, QColor(43, 138, 112))
        self.setPalette(pal)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setMinimumWidth(520)
        rec = hw.recommended()

        title = QLabel("Performance setup")
        title.setObjectName("title")
        found_label = QLabel("THIS PC")
        found_label.setObjectName("field")
        found = QLabel(hw.summary())
        found.setWordWrap(True)
        found.setStyleSheet("color:#eef2f0; font-size:13px;")

        pick_label = QLabel("PROFILE")
        pick_label.setObjectName("field")
        self.group = QButtonGroup(self)
        rows = QVBoxLayout()
        rows.setSpacing(4)
        for name in hardware.PROFILES:
            rb = QRadioButton(f"{name}  (recommended for this PC)" if name == rec else name)
            rb.setProperty("profile", name)
            rb.setChecked(name == rec)
            self.group.addButton(rb)
            desc = QLabel(hardware.DESCRIPTIONS[name])
            desc.setObjectName("hint")
            desc.setWordWrap(True)
            desc.setContentsMargins(26, 0, 0, 6)
            rows.addWidget(rb)
            rows.addWidget(desc)

        warn = QLabel(
            "Heads up: the translator shares your GPU with the game. If a game stutters, pick a lower profile, "
            "or pause translation with Alt+P. You can change this any time in the launch window."
        )
        warn.setObjectName("status")
        warn.setWordWrap(True)
        warn.setStyleSheet("color:#e3c26c;")

        buttons = QHBoxLayout()
        cancel = QPushButton("Quit")
        cancel.setObjectName("quit")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Use this profile")
        ok.setObjectName("start")
        ok.setDefault(True)
        ok.clicked.connect(self._accept)
        buttons.addWidget(cancel)
        buttons.addStretch()
        buttons.addWidget(ok)

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 24, 26, 22)
        root.setSpacing(12)
        root.addWidget(title)
        root.addWidget(found_label)
        root.addWidget(found)
        root.addWidget(pick_label)
        root.addLayout(rows)
        root.addWidget(warn)
        root.addLayout(buttons)

    def _accept(self) -> None:
        chosen = self.group.checkedButton()
        hardware.apply_profile(self.cfg, chosen.property("profile") if chosen else "Balanced")
        self.cfg.profile_confirmed = True
        self.cfg.save()
        self.accept()
