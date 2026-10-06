"""Build LacyanTranslator-Setup.exe.

    python installer/build.py

Downloads `uv`, packs the app source, draws the icon, and bundles everything with PyInstaller into
installer/dist/LacyanTranslator-Setup.exe.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / "build"
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
PAYLOAD = ["lacyan_translator", "glossaries", "requirements.txt", "LICENSE", "README.md"]


def venv_python() -> Path:
    venv = HERE / ".venv-build"
    py = venv / "Scripts" / "python.exe"
    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "pyinstaller", "pillow"], check=True)
    return py


def get_uv() -> Path:
    out = BUILD / "uv.exe"
    if not out.exists():
        print("Downloading uv …")
        data = urllib.request.urlopen(UV_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.endswith("uv.exe"))
            out.write_bytes(z.read(name))
    return out


def make_payload() -> Path:
    out = BUILD / "payload.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in PAYLOAD:
            p = ROOT / item
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
            for f in files:
                z.write(f, f.relative_to(ROOT))
    return out


ICON_SCRIPT = r'''
from PIL import Image, ImageDraw, ImageFont
import sys
size = 256
img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([0, 0, size - 1, size - 1], radius=58, fill=(36, 116, 95, 255))
f = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 160)
box = d.textbbox((0, 0), "译", font=f)
d.text(((size - (box[2] - box[0])) / 2 - box[0], (size - (box[3] - box[1])) / 2 - box[1]), "译", font=f, fill=(245, 245, 240, 255))
img.save(sys.argv[1], sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
'''


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    py = venv_python()
    icon = BUILD / "icon.ico"
    subprocess.run([str(py), "-c", ICON_SCRIPT, str(icon)], check=True)
    uv = get_uv()
    payload = make_payload()
    sep = ";"
    subprocess.run([
        str(py), "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
        "--name", "LacyanTranslator-Setup", "--icon", str(icon),
        "--add-data", f"{payload}{sep}.", "--add-data", f"{uv}{sep}.", "--add-data", f"{icon}{sep}.",
        "--distpath", str(HERE / "dist"), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD),
        str(HERE / "setup_app.py"),
    ], check=True)
    exe = HERE / "dist" / "LacyanTranslator-Setup.exe"
    print(f"Built {exe} ({exe.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
