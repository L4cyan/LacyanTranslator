"""First-run setup: make sure Ollama has the `lacyan-mt` translation model (Tencent Hunyuan-MT 1.5, 1.8B)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

log = logging.getLogger("lacyan.setup")

BASE_MODEL = "hf.co/tencent/HY-MT1.5-1.8B-GGUF:Q8_0"
MODEL_NAME = "lacyan-mt"

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


def ensure_model(name: str = MODEL_NAME) -> str:
    """Returns 'ok', 'created', 'no-ollama' or 'failed'."""
    models = ollama_models()
    if models is None:
        return "no-ollama"
    if any(m.split(":")[0] == name for m in models):
        return "ok"
    exe = shutil.which("ollama")
    if not exe:
        return "no-ollama"
    try:
        log.info("Downloading %s (about 1.9 GB)...", BASE_MODEL)
        subprocess.run([exe, "pull", BASE_MODEL], check=True, capture_output=True, creationflags=0x08000000)
        with tempfile.TemporaryDirectory() as tmp:
            mf = Path(tmp) / "Modelfile"
            mf.write_text(MODELFILE.format(base=BASE_MODEL), encoding="utf-8")
            subprocess.run([exe, "create", name, "-f", str(mf)], check=True, capture_output=True, creationflags=0x08000000)
        log.info("Created model %s", name)
        return "created"
    except subprocess.CalledProcessError as e:
        log.error("Model setup failed: %s", (e.stderr or b"")[-400:])
        return "failed"
