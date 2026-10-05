"""First-run setup: make sure Ollama has the `lacyan-mt` translation model (Tencent Hunyuan-MT 1.5, 1.8B)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("lacyan.setup")

REPO = "hf.co/tencent/HY-MT1.5-1.8B-GGUF"
MODEL_NAME = "lacyan-mt"

# The same translation model at three sizes. Smaller = faster and lighter on the GPU, slightly rougher.
VARIANTS = {
    "q4": {"file": "Q4_K_M", "label": "Fast (Q4 · about 1.1 GB)"},
    "q6": {"file": "Q6_K", "label": "Balanced (Q6 · about 1.5 GB)"},
    "q8": {"file": "Q8_0", "label": "Best quality (Q8 · about 1.9 GB)"},
}


def model_name(variant: str) -> str:
    return f"{MODEL_NAME}:{variant}"


def variant_of(model: str) -> str | None:
    """'lacyan-mt:q4' -> 'q4'; the original untagged 'lacyan-mt' was the Q8 build."""
    if not model.startswith(MODEL_NAME):
        return None
    tag = model.split(":", 1)[1] if ":" in model else "q8"
    return tag if tag in VARIANTS else "q8"

# Ollama's automatic template for this GGUF is broken, so we supply the real Hunyuan chat format.
# Lacyan Translator builds the official translation prompt itself; the template passes the last user message through.
MODELFILE = """FROM {base}
TEMPLATE \"\"\"<｜hy_begin▁of▁sentence｜>{{{{- range $i, $m := .Messages }}}}{{{{- if eq $m.Role "user" }}}}<｜hy_User｜>{{{{ $m.Content }}}}{{{{- else if eq $m.Role "assistant" }}}}<｜hy_Assistant｜>{{{{ $m.Content }}}}<｜hy_place▁holder▁no▁2｜>{{{{- end }}}}{{{{- end }}}}<｜hy_Assistant｜>\"\"\"
PARAMETER stop "<｜hy_place▁holder▁no▁2｜>"
PARAMETER stop "<｜hy_User｜>"
PARAMETER temperature 0.7
PARAMETER top_p 0.6
PARAMETER top_k 20
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 2048
"""


def ollama_models(host: str = "http://127.0.0.1:11434") -> list[str] | None:
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except OSError:
        return None


def ollama_exe() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    default = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    return str(default) if default.exists() else None


def start_ollama(wait_s: float = 20.0) -> bool:
    """Start the Ollama server in the background if it isn't running. Returns True once it answers."""
    if ollama_models() is not None:
        return True
    exe = ollama_exe()
    if not exe:
        return False
    env = dict(os.environ)
    env.setdefault("OLLAMA_NUM_PARALLEL", "2")  # lets the High profile's two translation streams run together
    subprocess.Popen([exe, "serve"], creationflags=0x08000000, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if ollama_models() is not None:
            return True
        time.sleep(0.5)
    return False


def has_model(name: str, models: list[str] | None = None) -> bool:
    models = ollama_models() if models is None else models
    if models is None:
        return False
    want = name if ":" in name else name + ":latest"
    return any(m == want for m in models)


def ensure_model(name: str = MODEL_NAME + ":q6") -> str:
    """Make sure the Ollama model `name` (e.g. 'lacyan-mt:q4') exists, downloading it if needed.
    Returns 'ok', 'created', 'no-ollama' or 'failed'."""
    models = ollama_models()
    if models is None:
        return "no-ollama"
    if has_model(name, models):
        return "ok"
    exe = ollama_exe()
    if not exe:
        return "no-ollama"
    variant = variant_of(name) or "q8"
    base = f"{REPO}:{VARIANTS[variant]['file']}"
    try:
        log.info("Downloading %s ...", base)
        subprocess.run([exe, "pull", base], check=True, capture_output=True, creationflags=0x08000000)
        with tempfile.TemporaryDirectory() as tmp:
            mf = Path(tmp) / "Modelfile"
            mf.write_text(MODELFILE.format(base=base), encoding="utf-8")
            subprocess.run([exe, "create", name, "-f", str(mf)], check=True, capture_output=True, creationflags=0x08000000)
        log.info("Created model %s", name)
        return "created"
    except subprocess.CalledProcessError as e:
        log.error("Model setup failed: %s", (e.stderr or b"")[-400:])
        return "failed"
