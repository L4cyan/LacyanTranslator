"""Detect what this PC can handle and pick a performance profile."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

# What each profile changes. Higher = smoother and faster to fill a page, but more GPU/CPU while you play.
PROFILES: dict[str, dict] = {
    "Low": {"max_fps": 20, "translation_workers": 1, "batch_size": 6, "ocr_interval_ms": 600, "model_size": "q4"},
    "Balanced": {"max_fps": 30, "translation_workers": 1, "batch_size": 8, "ocr_interval_ms": 350, "model_size": "q6"},
    "High": {"max_fps": 60, "translation_workers": 2, "batch_size": 10, "ocr_interval_ms": 200, "model_size": "q6"},
}

DESCRIPTIONS = {
    "Low": "Light on your PC. Translations follow scrolling at 20 fps and pages fill in more slowly.",
    "Balanced": "Smooth 30 fps following, one translation stream. Good for most gaming PCs.",
    "High": "60 fps following and two translation streams in parallel. Fills busy pages fastest, "
            "but uses noticeably more GPU, so very demanding games may lose some frame rate.",
}


@dataclass
class Hardware:
    gpu: str
    vram_gb: float
    nvidia: bool
    cpu: str
    threads: int
    ram_gb: float

    def summary(self) -> str:
        vram = f" ({self.vram_gb:.0f} GB)" if self.vram_gb else ""
        return f"{self.gpu}{vram} · {self.cpu} ({self.threads} threads) · {self.ram_gb:.0f} GB RAM"

    def recommended(self) -> str:
        if self.nvidia and self.vram_gb >= 7.5 and self.ram_gb >= 14 and self.threads >= 12:
            return "High"
        if (self.vram_gb >= 4 or self.nvidia) and self.ram_gb >= 8:
            return "Balanced"
        return "Low"


def _ram_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 2**30


def _cpu_name() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return " ".join(winreg.QueryValueEx(k, "ProcessorNameString")[0].split())
    except OSError:
        return platform.processor() or "CPU"


def _gpu() -> tuple[str, float, bool]:
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=8, creationflags=0x08000000).stdout.strip()
            name, mem = out.splitlines()[0].rsplit(",", 1)
            return name.strip(), float(mem) / 1024, True
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            pass
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json"],
            capture_output=True, text=True, timeout=15, creationflags=0x08000000).stdout
        cards = json.loads(out)
        cards = cards if isinstance(cards, list) else [cards]
        best = max(cards, key=lambda c: c.get("AdapterRAM") or 0)
        return best["Name"], (best.get("AdapterRAM") or 0) / 2**30, "nvidia" in best["Name"].lower()
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return "Unknown GPU", 0.0, False


def detect() -> Hardware:
    gpu, vram, nvidia = _gpu()
    return Hardware(gpu, vram, nvidia, _cpu_name(), os.cpu_count() or 4, _ram_gb())


def apply_profile(cfg, name: str, model_size: str | None = None) -> None:
    """Apply a profile's settings. The translation model size follows the profile unless one is given."""
    from .model_setup import model_name, variant_of

    cfg.profile = name
    for k, v in PROFILES[name].items():
        if k != "model_size":
            setattr(cfg, k, v)
    if variant_of(cfg.model) is not None:  # only when using the bundled local model, not a custom endpoint
        if model_size:
            cfg.model = model_name(model_size)
        else:
            # Follow the profile's size only if it's installed; someone who installed just the fast model
            # keeps it instead of triggering a surprise download.
            from .model_setup import has_model

            wanted = model_name(PROFILES[name]["model_size"])
            if has_model(wanted):
                cfg.model = wanted
