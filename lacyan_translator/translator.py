"""Talks to any OpenAI-compatible server (Ollama, LM Studio, llama.cpp, cloud APIs).

Short phrases are translated in numbered batches and streamed back, so a page full of text fills in
line by line instead of one model call per phrase.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import threading
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .cache import normalize
from .config import ROOT, Config
from .langs import SOURCE_PATTERNS, TARGETS

log = logging.getLogger("lacyan.translate")
_THINK = re.compile(r"<think>.*?</think>", re.S)
_NUMBERED = re.compile(r"\s*(\d+)\s*[.、)）:：]\s*(.*)")
_CJK_TARGETS = {"Japanese", "Korean", "Traditional Chinese"}
LONG = 40  # characters; longer texts are translated on their own


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
        self.phrasebook = {normalize(k): v for k, v in load_glossary(cfg.phrasebook_files).items()}
        self._local = threading.local()
        url = urlparse(cfg.endpoint.rstrip("/"))
        self._https = url.scheme == "https"
        self._host = url.hostname or "127.0.0.1"
        self._port = url.port or (443 if self._https else 80)
        self._base = url.path or ""
        log.info("Translator: %s model=%s glossary=%d phrasebook=%d", cfg.endpoint, cfg.model, len(self.glossary), len(self.phrasebook))

    # --- prompts ---------------------------------------------------------
    @property
    def style(self) -> str:
        if self.cfg.prompt_style != "auto":
            return self.cfg.prompt_style
        m = self.cfg.model.lower()
        return "hymt" if ("hy-mt" in m or "lacyan-mt" in m or "hunyuan-mt" in m) else "chat"

    def _terms_in(self, text: str) -> list[tuple[str, str]]:
        return [(s, d) for s, d in self.glossary.items() if s in text]

    def _single(self, text: str, fallback: bool = False) -> list[dict]:
        target = self.cfg.target_language
        terms = self._terms_in(text)
        if self.style == "hymt":
            if fallback:  # the model's alternative (English-instruction) prompt format
                return [{"role": "user", "content": f"Translate the following segment into {target}, without additional explanation.\n\n{text}"}]
            prefix = ("参考下面的翻译：" + "；".join(f"{s} 翻译成 {d}" for s, d in terms) + "。") if terms else ""
            # The blank line matters: without it short UI words leak the instruction into the output.
            return [{"role": "user", "content": f"{prefix}将以下文本翻译为{TARGETS.get(target, target)}，注意只需要输出翻译后的结果，不要额外解释：\n\n{text}"}]
        return self._chat(text, terms)

    def _batch(self, texts: list[str]) -> list[dict]:
        target = self.cfg.target_language
        terms = sorted({t for x in texts for t in self._terms_in(x)})
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
        if self.style == "hymt":
            prefix = ("参考下面的翻译：" + "；".join(f"{s} 翻译成 {d}" for s, d in terms) + "。") if terms else ""
            return [{"role": "user", "content": f"{prefix}将以下编号文本逐行翻译为{TARGETS.get(target, target)}，保留编号，每行对应一行译文，注意只需要输出翻译后的结果，不要额外解释：\n\n{numbered}"}]
        return self._chat(numbered + "\n\n(Translate each numbered line; keep the numbers, one line each.)", terms)

    def _chat(self, text: str, terms) -> list[dict]:
        system = (
            f"You are a professional localizer for games and apps. Translate the user's {self.cfg.source_language} text into "
            f"natural, concise {self.cfg.target_language} that fits a UI. Keep names consistent. Output only the translation."
        )
        if terms:
            system += " Use these term translations: " + "; ".join(f"{s} = {d}" for s, d in terms) + "."
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    # --- HTTP ------------------------------------------------------------
    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            conn = cls(self._host, self._port, timeout=120)
            self._local.conn = conn
        return conn

    def _post(self, body: dict, on_text: Callable[[str], None] | None = None) -> str:
        """POST a chat completion. With `on_text`, stream and call it with each new chunk."""
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        body = {**body, "stream": on_text is not None}
        if self.style == "chat":
            body["reasoning_effort"] = "none"  # skip slow "thinking" on reasoning models
        payload = json.dumps(body).encode()
        for attempt in range(2):
            conn = self._conn()
            try:
                conn.request("POST", self._base + "/chat/completions", body=payload, headers=headers)
                resp = conn.getresponse()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
                if on_text is None:
                    data = json.loads(resp.read())
                    return data["choices"][0]["message"].get("content") or ""
                full = ""
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    chunk = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
                    if chunk:
                        full += chunk
                        on_text(chunk)
                return full
            except (http.client.HTTPException, ConnectionError, OSError):
                conn.close()
                self._local.conn = None
                if attempt:
                    raise
        raise RuntimeError("unreachable")

    # --- translation -----------------------------------------------------
    def phrase(self, text: str) -> str | None:
        """Instant answer from the exact-match phrasebook, if any."""
        return self.phrasebook.get(normalize(text))

    def _looks_untranslated(self, src: str, out: str) -> bool:
        if not out or self.cfg.target_language in _CJK_TARGETS:
            return not out
        pat = SOURCE_PATTERNS.get(self.cfg.source_language)
        if pat is None:
            return False
        leftover = len(pat.findall(out))
        leaked = "explanation" in out.lower() or "翻译" in out
        return leaked or leftover > max(1, len(pat.findall(src)) * 0.3)

    @staticmethod
    def _clean(msg: str) -> str:
        return _THINK.sub("", msg).strip().strip('"“”').strip()

    def translate(self, text: str) -> str:
        hit = self.phrase(text)
        if hit:
            return hit
        body = {"model": self.cfg.model, "messages": self._single(text), "temperature": 0.3 if self.style == "chat" else 0.7,
                "top_p": 0.6, "max_tokens": 512}
        out = self._clean(self._post(body))
        if self._looks_untranslated(text, out):
            body["messages"], body["temperature"] = self._single(text, fallback=True), 0.3
            retry = self._clean(self._post(body))
            if not self._looks_untranslated(text, retry):
                return retry
            log.info("Left untranslated: %r -> %r", text, out)
            return ""
        return out

    def translate_batch(self, texts: list[str], on_result: Callable[[str, str], None]) -> None:
        """Translate several phrases in one streamed call; `on_result(text, translation)` fires per line as it
        arrives. Lines the model skips or leaves untranslated are retried one by one."""
        if len(texts) == 1:
            on_result(texts[0], self.translate(texts[0]))
            return
        done: set[int] = set()
        buf = [""]

        def take(line: str) -> None:
            m = _NUMBERED.match(line)
            if not m:
                return
            i = int(m.group(1)) - 1
            out = self._clean(m.group(2))
            if 0 <= i < len(texts) and i not in done and not self._looks_untranslated(texts[i], out):
                done.add(i)
                on_result(texts[i], out)

        def on_text(chunk: str) -> None:
            buf[0] += chunk
            while "\n" in buf[0]:
                line, buf[0] = buf[0].split("\n", 1)
                take(line)

        body = {"model": self.cfg.model, "messages": self._batch(texts), "temperature": 0.3, "top_p": 0.6,
                "max_tokens": 64 + 48 * len(texts)}
        try:
            self._post(body, on_text)
            take(buf[0])
        except Exception as e:  # noqa: BLE001 - fall back to one-by-one below
            log.warning("Batch failed (%s); translating one by one", e)
        for i, t in enumerate(texts):
            if i not in done:
                on_result(t, self.translate(t))

    def warm_up(self) -> None:
        """Load the model into memory now so the first real line isn't slow."""
        try:
            self.translate("你好")
        except Exception as e:  # noqa: BLE001
            log.warning("Warm-up failed: %s", e)
