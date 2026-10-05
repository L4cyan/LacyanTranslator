"""User settings, stored as JSON next to the app so the folder stays portable."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "config.json"


@dataclass
class Config:
    # Languages
    source_language: str = "Chinese"
    target_language: str = "English"

    # Translation backend (any OpenAI-compatible server; Ollama by default)
    endpoint: str = "http://127.0.0.1:11434/v1"
    api_key: str = ""
    model: str = "lacyan-mt"
    prompt_style: str = "auto"  # auto | hymt | chat
    glossary_files: list[str] = field(default_factory=lambda: ["glossaries/xianxia.txt"])
    # Exact-match phrasebooks: a block that is exactly one of these is replaced without asking the model.
    phrasebook_files: list[str] = field(default_factory=lambda: ["glossaries/ui.txt", "glossaries/shopping.txt"])
    translation_workers: int = 1  # parallel translation streams (set by the performance profile)
    batch_size: int = 8  # short phrases translated per model call

    # Capture & OCR
    capture: str = "foreground"  # foreground | monitor
    monitor: int = 1  # 1 = primary (mss numbering)
    min_confidence: float = 0.6
    ocr_interval_ms: int = 350
    frame_interval_ms: int = 100
    change_threshold: float = 1.2
    stable_passes: int = 2  # identical OCR passes before translating (handles typewriter text)
    keep_missing_passes: int = 2

    # Look
    font_family: str = "Segoe UI"
    blur_strength: float = 1.0
    backdrop_tint: float = 0.35
    max_grow: float = 1.0  # 1.0 = translations stay exactly inside the original text's area (font shrinks to fit)
    keep_original_color: bool = True
    hide_from_capture: bool = True  # keep overlay out of screenshots (also stops Lacyan Translator reading itself)

    # Performance (chosen at first launch from the detected hardware; see hardware.py)
    profile: str = "Balanced"
    profile_confirmed: bool = False
    max_fps: int = 30

    # Startup
    show_launcher: bool = True  # the window with language choice + Start button

    # Hotkeys
    hotkey_toggle: str = "alt+t"
    hotkey_pause: str = "alt+p"
    hotkey_quit: str = "ctrl+alt+q"

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        if CONFIG_PATH.exists():
            try:
                raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                known = {f.name for f in fields(cls)}
                for k, v in raw.items():
                    if k in known:
                        setattr(cfg, k, v)
            except (OSError, ValueError):
                pass
        cfg.save()  # writes defaults for any new settings
        return cfg


def data_path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name


def open_in_explorer(path: Path) -> None:
    os.startfile(str(path))  # noqa: S606 (Windows only)
