"""Talks to any OpenAI-compatible server (Ollama, LM Studio, llama.cpp, cloud APIs)."""

from __future__ import annotations

import http.client
import json
import logging
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from .config import ROOT, Config
from .langs import SOURCE_PATTERNS, TARGETS

log = logging.getLogger("veil.translate")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_CJK_TARGETS = {"Japanese", "Korean", "Traditional Chinese"}


def load_glossary(files: list[str]) -> dict[str, str]:
    """Glossary files: one `source = translation` per line, `#` for comments."""
    terms: dict[str, str] = {}
    for name in files:
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if "=" in line:
                src, dst = (p.strip() for p in line.split("=", 1))
                if src and dst:
                    terms[src] = dst
    return terms


class Translator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.glossary = load_glossary(cfg.glossary_files)
        self.phrasebook = load_glossary(cfg.phrasebook_files)
        self._pool = ThreadPoolExecutor(max_workers=max(1, cfg.translation_workers), thread_name_prefix="mt")
        self._local = threading.local()
        url = urlparse(cfg.endpoint.rstrip("/"))
        self._https = url.scheme == "https"
        self._host = url.hostname or "127.0.0.1"
        self._port = url.port or (443 if self._https else 80)
        self._base = url.path or ""
        log.info("Translator: %s model=%s glossary=%d terms", cfg.endpoint, cfg.model, len(self.glossary))

    # --- prompt building -------------------------------------------------
    @property
    def style(self) -> str:
        if self.cfg.prompt_style != "auto":
            return self.cfg.prompt_style
        m = self.cfg.model.lower()
        return "hymt" if ("hy-mt" in m or "veil-mt" in m or "hunyuan-mt" in m) else "chat"

    def _terms_in(self, text: str) -> list[tuple[str, str]]:
        return [(s, d) for s, d in self.glossary.items() if s in text]

    def _messages(self, text: str, fallback: bool = False) -> list[dict]:
        target = self.cfg.target_language
        terms = self._terms_in(text)
        if self.style == "hymt":
            if fallback:  # the model's alternative (English-instruction) prompt format
                return [{"role": "user", "content": f"Translate the following segment into {target}, without additional explanation.\n\n{text}"}]
            tgt = TARGETS.get(target, target)
            prefix = ("参考下面的翻译：" + "；".join(f"{s} 翻译成 {d}" for s, d in terms) + "。") if terms else ""
            # The blank line matters: without it short UI words leak the instruction into the output.
            return [{"role": "user", "content": f"{prefix}将以下文本翻译为{tgt}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"}]
        system = (
            f"You are a professional game localizer. Translate the user's {self.cfg.source_language} game text into "
            f"natural, concise {target} that fits a game UI. Keep names consistent. Output only the translation."
        )
        if terms:
            system += " Use these term translations: " + "; ".join(f"{s} = {d}" for s, d in terms) + "."
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    # --- HTTP ------------------------------------------------------------
    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            conn = cls(self._host, self._port, timeout=60)
            self._local.conn = conn
        return conn

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        payload = json.dumps(body).encode() if body is not None else None
        for attempt in range(2):
            conn = self._conn()
            try:
                conn.request(method, self._base + path, body=payload, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {data[:200]!r}")
                return json.loads(data)
            except (http.client.HTTPException, ConnectionError, OSError):
                conn.close()
                self._local.conn = None
                if attempt:
                    raise
        raise RuntimeError("unreachable")

    def _looks_untranslated(self, src: str, out: str) -> bool:
        if not out or self.cfg.target_language in _CJK_TARGETS:
            return not out
        pat = SOURCE_PATTERNS.get(self.cfg.source_language)
        if pat is None:
            return False
        leftover = len(pat.findall(out))
        leaked = "explanation" in out.lower() or "翻译" in out
        return leaked or leftover > max(1, len(pat.findall(src)) * 0.3)

    def _ask(self, text: str, fallback: bool) -> str:
        body = {
            "model": self.cfg.model,
            "messages": self._messages(text, fallback),
            "stream": False,
            "temperature": 0.3 if (self.style == "chat" or fallback) else 0.7,
            "top_p": 0.6,
            "max_tokens": 512,
        }
        if self.style == "chat":
            body["reasoning_effort"] = "none"  # skip slow "thinking" on reasoning models
        out = self._request("POST", "/chat/completions", body)
        msg = out["choices"][0]["message"].get("content") or ""
        return _THINK.sub("", msg).strip().strip('"“”').strip()

    def translate(self, text: str) -> str:
        exact = self.phrasebook.get(text.strip())
        if exact:
            return exact
        out = self._ask(text, fallback=False)
        if self._looks_untranslated(text, out):
            retry = self._ask(text, fallback=True)
            if not self._looks_untranslated(text, retry):
                return retry
            log.info("Left untranslated: %r -> %r", text, out)
            return ""
        return out

    def warm_up(self) -> None:
        """Load the model into memory now so the first real line isn't slow."""
        try:
            self._ask("你好", fallback=False)
        except Exception as e:  # noqa: BLE001
            log.warning("Warm-up failed: %s", e)

    def submit(self, text: str) -> Future:
        return self._pool.submit(self.translate, text)

    def list_models(self) -> list[str]:
        out = self._request("GET", "/models")
        return [m["id"] for m in out.get("data", [])]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
