"""Lacyan Translator setup: a one-file Windows installer.

Bundles the app source and `uv` (a fast Python package manager). At install time it creates a private Python
environment, installs every dependency (picking the GPU runtime for this PC), installs Ollama if missing,
downloads the chosen translation model sizes, and adds shortcuts plus an entry in Windows' installed apps.
Per-user install: no administrator rights needed.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import urllib.request
import winreg
import zipfile
from pathlib import Path
from tkinter import filedialog, ttk

APP = "Lacyan Translator"
VERSION = "0.4.0"
DEFAULT_DIR = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "LacyanTranslator"
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\LacyanTranslator"
OLLAMA_URL = "https://ollama.com/download/OllamaSetup.exe"
NO_WINDOW = 0x08000000

BG, PANEL, FG, MUTED, ACCENT, WARN, BAD = "#15191a", "#1d2426", "#e6ecea", "#8fa39b", "#24745f", "#e3c26c", "#f0918a"


def bundled(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    return base / name


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def ollama_exe() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    p = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(p) if p.exists() else None


def make_shortcut(lnk: Path, target: Path, args: str, workdir: Path, icon: Path) -> None:
    lnk.parent.mkdir(parents=True, exist_ok=True)
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK);"
        "$s.TargetPath=$env:TGT;$s.Arguments=$env:ARGS;$s.WorkingDirectory=$env:WD;"
        "$s.IconLocation=$env:ICO;$s.Description='Live on-screen translator';$s.Save()"
    )
    env = {**os.environ, "LNK": str(lnk), "TGT": str(target), "ARGS": args, "WD": str(workdir), "ICO": str(icon)}
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env, check=True, creationflags=NO_WINDOW)


def shell_folder(name: str) -> Path:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                        r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as k:
        return Path(os.path.expandvars(winreg.QueryValueEx(k, name)[0]))


# --------------------------------------------------------------------------------------------------- install
class Installer:
    def __init__(self, target: Path, models: list[str], cuda: bool, desktop: bool, log) -> None:
        self.dir, self.models, self.cuda, self.desktop, self.log = target, models, cuda, desktop, log

    def run_cmd(self, args: list[str], step: str, env: dict | None = None) -> None:
        self.log("detail", " ".join(str(a) for a in args[:4]) + (" …" if len(args) > 4 else ""))
        proc = subprocess.Popen([str(a) for a in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=self.dir,
                                env=env or self.env, creationflags=NO_WINDOW, text=True, encoding="utf-8", errors="replace")
        tail = []
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            tail = (tail + [line])[-12:]
            if line.startswith("PROGRESS "):
                self.log("progress", line[9:])
            elif line.startswith(("Downloading", "Installed", "Resolved", "Prepared")) or line.endswith("ready"):
                self.log("detail", line)
        if proc.wait() != 0:
            raise RuntimeError(f"{step} failed:\n" + "\n".join(tail))

    @property
    def env(self) -> dict:
        return {**os.environ,
                "UV_PYTHON_INSTALL_DIR": str(self.dir / "python"),
                "UV_CACHE_DIR": str(Path(tempfile.gettempdir()) / "lacyan-uv-cache"),
                "UV_LINK_MODE": "copy", "PYTHONUTF8": "1",
                # Trust Windows' certificate store: antivirus HTTPS scanning (Avast, Kaspersky...) uses its own
                # root certificate, which only the system store knows about.
                "UV_NATIVE_TLS": "1", "UV_SYSTEM_CERTS": "1",
                # Always use a private Python kept inside the install folder, never one already on the PC
                # (uninstalling that one would break the app).
                "UV_PYTHON_PREFERENCE": "only-managed"}

    def install(self) -> None:
        d = self.dir
        d.mkdir(parents=True, exist_ok=True)
        uv = d / "uv.exe"

        self.log("step", "Copying Lacyan Translator files")
        with zipfile.ZipFile(bundled("payload.zip")) as z:
            z.extractall(d)
        shutil.copy2(bundled("uv.exe"), uv)
        shutil.copy2(bundled("icon.ico"), d / "icon.ico")
        if not (d / "config.json").exists():  # first install: start with the installed model size
            json.dump({"model": f"lacyan-mt:{self.models[0]}"}, open(d / "config.json", "w", encoding="utf-8"))

        self.log("step", "Setting up a private Python (only Lacyan Translator uses it)")
        self.run_cmd([uv, "venv", "--python", "3.12", "--allow-existing", d / ".venv"], "Python setup")
        py = d / ".venv" / "Scripts" / "python.exe"

        self.log("step", "Installing app components")
        self.run_cmd([uv, "pip", "install", "--python", py, "-r", d / "requirements.txt"], "Installing components")
        self.run_cmd([uv, "pip", "uninstall", "--python", py, "onnxruntime"], "Removing CPU runtime")
        if self.cuda:
            self.log("step", "Installing NVIDIA GPU acceleration (largest download)")
            self.run_cmd([uv, "pip", "install", "--python", py, "onnxruntime-gpu[cuda,cudnn]"], "GPU runtime")
        else:
            self.log("step", "Installing GPU acceleration (DirectML)")
            self.run_cmd([uv, "pip", "install", "--python", py, "onnxruntime-directml"], "GPU runtime")

        if not ollama_exe():
            self.log("step", "Installing Ollama (runs the translation model)")
            setup = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
            self._download(OLLAMA_URL, setup)
            subprocess.run([str(setup), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], check=True, creationflags=NO_WINDOW)
            for _ in range(60):
                if ollama_exe():
                    break
                time.sleep(1)
            if not ollama_exe():
                raise RuntimeError("Ollama didn't finish installing. Install it from ollama.com, then run this setup again.")

        self.log("step", "Downloading the translation model" + ("s" if len(self.models) > 1 else ""))
        self.run_cmd([py, "-m", "lacyan_translator.model_setup", *self.models], "Model download")

        self.log("step", "Creating shortcuts")
        pyw = d / ".venv" / "Scripts" / "pythonw.exe"
        menu = shell_folder("Programs")
        make_shortcut(menu / f"{APP}.lnk", pyw, "-m lacyan_translator", d, d / "icon.ico")
        if self.desktop:
            make_shortcut(shell_folder("Desktop") / f"{APP}.lnk", pyw, "-m lacyan_translator", d, d / "icon.ico")
        self._register()
        self.log("done", "")

    def _download(self, url: str, dest: Path) -> None:
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    self.log("progress", f"{got * 100 // total}%  {got >> 20} / {total >> 20} MB")

    def _register(self) -> None:
        uninstaller = self.dir / "Uninstall.exe"
        if Path(sys.executable).suffix.lower() == ".exe" and getattr(sys, "frozen", False):
            shutil.copy2(sys.executable, uninstaller)
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as k:
            for name, value in {
                "DisplayName": APP, "DisplayVersion": VERSION, "Publisher": "L4cyan",
                "DisplayIcon": str(self.dir / "icon.ico"), "InstallLocation": str(self.dir),
                "UninstallString": f'"{uninstaller}" --uninstall', "URLInfoAbout": "https://github.com/L4cyan/LacyanTranslator",
            }.items():
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
            winreg.SetValueEx(k, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(k, "NoRepair", 0, winreg.REG_DWORD, 1)


def uninstall(remove_models: bool) -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY) as k:
            target = Path(winreg.QueryValueEx(k, "InstallLocation")[0])
    except OSError:
        target = DEFAULT_DIR
    # Close Lacyan Translator if it's running: only processes started from its own folder.
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($env:TARGET, 'OrdinalIgnoreCase') } "
                    "| Where-Object { $_.Id -ne $env:SELF } | Stop-Process -Force"],
                   env={**os.environ, "TARGET": str(target), "SELF": str(os.getpid())}, creationflags=NO_WINDOW,
                   capture_output=True)
    for lnk in (shell_folder("Programs") / f"{APP}.lnk", shell_folder("Desktop") / f"{APP}.lnk"):
        lnk.unlink(missing_ok=True)
    if remove_models and ollama_exe():
        for tag in ("q4", "q6", "q8", "latest"):
            subprocess.run([ollama_exe(), "rm", f"lacyan-mt:{tag}"], capture_output=True, creationflags=NO_WINDOW)
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    except OSError:
        pass
    # The running Uninstall.exe lives inside the folder: delete the folder after this process exits.
    subprocess.Popen(f'cmd /c ping 127.0.0.1 -n 3 >nul & rmdir /s /q "{target}"', creationflags=NO_WINDOW, shell=True)


# ------------------------------------------------------------------------------------------------------- UI
class Wizard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except OSError:
            pass
        self.title(f"{APP} Setup")
        self.configure(bg=BG)
        self.resizable(False, False)
        try:
            self.iconbitmap(str(bundled("icon.ico")))
        except tk.TclError:
            pass
        self.events: queue.Queue = queue.Queue()
        self._style()
        self.body = tk.Frame(self, bg=BG, padx=28, pady=22)
        self.body.pack(fill="both", expand=True)
        self.page_options()
        self.geometry("640x600")

    def _style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.configure("TRadiobutton", background=BG, foreground=FG, font=("Segoe UI", 11))
        s.map("TRadiobutton", background=[("active", BG)], indicatorcolor=[("selected", ACCENT), ("!selected", PANEL)])
        s.configure("TCheckbutton", background=BG, foreground=FG, font=("Segoe UI", 10))
        s.map("TCheckbutton", background=[("active", BG)], indicatorcolor=[("selected", ACCENT), ("!selected", PANEL)])
        s.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=ACCENT, bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT)

    def clear(self) -> None:
        for w in self.body.winfo_children():
            w.destroy()

    def label(self, text, size=10, color=FG, bold=False, pady=(0, 0), wrap=580):
        l = tk.Label(self.body, text=text, bg=BG, fg=color, font=("Segoe UI", size, "bold" if bold else "normal"),
                     justify="left", anchor="w", wraplength=wrap)
        l.pack(fill="x", pady=pady)
        return l

    def button(self, parent, text, cmd, primary=True):
        b = tk.Button(parent, text=text, command=cmd, bg=ACCENT if primary else BG, fg="white" if primary else MUTED,
                      activebackground="#2b8a70" if primary else PANEL, activeforeground="white", relief="flat",
                      font=("Segoe UI", 11, "bold"), padx=20, pady=8, bd=0, cursor="hand2",
                      highlightthickness=0 if primary else 1, highlightbackground="#2f3b3d")
        return b

    # --- page 1 -----------------------------------------------------------
    def page_options(self) -> None:
        self.clear()
        self.label(APP, 20, bold=True)
        self.label(f"Setup · v{VERSION} · live on-screen translation that runs on your own PC", 10, MUTED, pady=(0, 18))

        self.label("TRANSLATION MODEL", 9, MUTED, bold=True)
        self.models = tk.StringVar(value="fast")
        ttk.Radiobutton(self.body, text="Fastest model only  (recommended · 1.1 GB download)", value="fast",
                        variable=self.models).pack(anchor="w", pady=(6, 0))
        self.label("Quick to download and the fastest at translating. Good quality for games and shops.", 9, MUTED, pady=(0, 6))
        ttk.Radiobutton(self.body, text="Install all model sizes  (4.5 GB download)", value="all",
                        variable=self.models).pack(anchor="w")
        self.label("Fast, Balanced and Best quality, so you can switch any time in the app without waiting.", 9, MUTED, pady=(0, 14))

        self.label("GPU ACCELERATION", 9, MUTED, bold=True)
        nvidia = has_nvidia()
        self.cuda = tk.BooleanVar(value=nvidia)
        cb = ttk.Checkbutton(self.body, text="Use NVIDIA CUDA for the fastest text reading  (about 1.5 GB more)",
                             variable=self.cuda)
        cb.pack(anchor="w", pady=(6, 0))
        if not nvidia:
            cb.state(["disabled"])
        self.label("NVIDIA GPU detected." if nvidia else "No NVIDIA GPU found: DirectML will be used (works on any GPU).",
                   9, MUTED, pady=(0, 14))

        self.label("INSTALL TO", 9, MUTED, bold=True)
        row = tk.Frame(self.body, bg=BG)
        row.pack(fill="x", pady=(6, 4))
        self.dir = tk.StringVar(value=str(DEFAULT_DIR))
        tk.Entry(row, textvariable=self.dir, bg=PANEL, fg=FG, insertbackground=FG, relief="flat",
                 font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True, ipady=6)
        self.button(row, "Browse", self._browse, primary=False).pack(side="left", padx=(8, 0))
        self.desktop = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.body, text="Add a desktop shortcut", variable=self.desktop).pack(anchor="w", pady=(6, 0))

        self.label("Ollama (the free app that runs the translation model) will be installed too if you don't have it. "
                   "Nothing you translate leaves this PC.", 9, MUTED, pady=(16, 0))

        bar = tk.Frame(self.body, bg=BG)
        bar.pack(side="bottom", fill="x")
        self.button(bar, "Cancel", self.destroy, primary=False).pack(side="left")
        self.button(bar, "Install", self.start).pack(side="right")

    def _browse(self) -> None:
        p = filedialog.askdirectory(initialdir=self.dir.get())
        if p:
            self.dir.set(str(Path(p) / "LacyanTranslator") if not p.endswith("LacyanTranslator") else p)

    # --- page 2 -----------------------------------------------------------
    def start(self) -> None:
        models = ["q4"] if self.models.get() == "fast" else ["q4", "q6", "q8"]
        inst = Installer(Path(self.dir.get()), models, self.cuda.get(), self.desktop.get(),
                         lambda kind, text: self.events.put((kind, text)))
        self.clear()
        self.label("Installing…", 18, bold=True, pady=(0, 14))
        self.step = self.label("Starting", 12)
        self.prog = ttk.Progressbar(self.body, mode="indeterminate", length=580)
        self.prog.pack(fill="x", pady=(10, 4))
        self.prog.start(12)
        self.detail = self.label("", 9, MUTED)
        self.logbox = tk.Text(self.body, height=14, bg=PANEL, fg=MUTED, relief="flat", font=("Consolas", 9), wrap="word")
        self.logbox.pack(fill="both", expand=True, pady=(12, 0))
        self.total_steps = 7
        self.done_steps = 0
        threading.Thread(target=self._run, args=(inst,), daemon=True).start()
        self.after(100, self._poll)

    def _run(self, inst: Installer) -> None:
        try:
            inst.install()
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.events.put(("error", str(e)))

    def _poll(self) -> None:
        try:
            while True:
                kind, text = self.events.get_nowait()
                if kind == "step":
                    self.done_steps += 1
                    self.step.config(text=f"{text}  ({self.done_steps}/{self.total_steps})")
                    self.prog.config(mode="indeterminate")
                    self.prog.start(12)
                    self._log("» " + text)
                elif kind == "progress":
                    pct = text.split("%")[0].split()[-1] if "%" in text else ""
                    if pct.isdigit():
                        self.prog.stop()
                        self.prog.config(mode="determinate", value=int(pct))
                    self.detail.config(text=text[-90:])
                elif kind == "detail":
                    self._log("  " + text)
                elif kind == "error":
                    self.page_error(text)
                    return
                elif kind == "done":
                    self.page_done()
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _log(self, text: str) -> None:
        self.logbox.insert("end", text + "\n")
        self.logbox.see("end")

    # --- page 3 -----------------------------------------------------------
    def page_done(self) -> None:
        self.clear()
        self.label("All set", 20, bold=True, pady=(0, 10))
        self.label(f"{APP} is installed. Start it from the desktop or Start menu shortcut.", 11, pady=(0, 6))
        self.label("Alt+T shows the original text · Alt+P pauses · Ctrl+Alt+Q quits. Click the tray icon for options.",
                   10, MUTED, pady=(0, 18))
        self.launch = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.body, text=f"Start {APP} now", variable=self.launch).pack(anchor="w")
        bar = tk.Frame(self.body, bg=BG)
        bar.pack(side="bottom", fill="x")
        self.button(bar, "Finish", self._finish).pack(side="right")

    def _finish(self) -> None:
        if self.launch.get():
            d = Path(self.dir.get())
            subprocess.Popen([str(d / ".venv" / "Scripts" / "pythonw.exe"), "-m", "lacyan_translator"], cwd=d)
        self.destroy()

    def page_error(self, text: str) -> None:
        self.prog.stop()
        self.step.config(text="Setup couldn't finish", fg=BAD)
        self._log("\n" + text)
        bar = tk.Frame(self.body, bg=BG)
        bar.pack(side="bottom", fill="x", pady=(10, 0))
        self.button(bar, "Close", self.destroy, primary=False).pack(side="left")
        self.button(bar, "Try again", self.page_options).pack(side="right")


class UninstallDialog(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"Uninstall {APP}")
        self.configure(bg=BG, padx=26, pady=22)
        self.resizable(False, False)
        tk.Label(self, text=f"Uninstall {APP}?", bg=BG, fg=FG, font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(self, text="Removes the app, its settings and its translation memory.", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 12))
        self.models = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Also delete the translation models from Ollama (frees up to 4.5 GB)", variable=self.models,
                       bg=BG, fg=FG, selectcolor=PANEL, activebackground=BG, activeforeground=FG,
                       font=("Segoe UI", 10)).pack(anchor="w")
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", pady=(18, 0))
        tk.Button(bar, text="Cancel", command=self.destroy, bg=BG, fg=MUTED, relief="flat", font=("Segoe UI", 11),
                  padx=16, pady=6).pack(side="left")
        tk.Button(bar, text="Uninstall", command=self._go, bg="#8a3a35", fg="white", relief="flat",
                  font=("Segoe UI", 11, "bold"), padx=16, pady=6).pack(side="right")

    def _go(self) -> None:
        uninstall(self.models.get())
        self.destroy()


if __name__ == "__main__":
    (UninstallDialog() if "--uninstall" in sys.argv else Wizard()).mainloop()
