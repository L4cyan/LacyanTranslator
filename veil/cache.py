"""Translation memory: every translated phrase is kept on disk so it shows instantly next time."""

from __future__ import annotations

import re
import sqlite3
import threading
from collections import OrderedDict
from difflib import SequenceMatcher

_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACE.sub("", text)


class TranslationCache:
    def __init__(self, path: str, recent: int = 600) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memory (src TEXT, lang TEXT, model TEXT, dst TEXT, hits INTEGER DEFAULT 0,"
            " PRIMARY KEY (src, lang, model))"
        )
        self._lock = threading.Lock()
        self._recent: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._recent_max = recent

    def get(self, text: str, lang: str, model: str, fuzzy: float = 0.92) -> str | None:
        key = (normalize(text), lang, model)
        with self._lock:
            if key in self._recent:
                self._recent.move_to_end(key)
                return self._recent[key]
            row = self._db.execute(
                "SELECT dst FROM memory WHERE src=? AND lang=? AND model=?", key
            ).fetchone()
            if row:
                self._db.execute("UPDATE memory SET hits=hits+1 WHERE src=? AND lang=? AND model=?", key)
                self._remember(key, row[0])
                return row[0]
            # OCR jitter: one misread character shouldn't cost a new translation
            if len(key[0]) >= 6:
                for (src, lg, md), dst in reversed(self._recent.items()):
                    if lg == lang and md == model and abs(len(src) - len(key[0])) <= 2:
                        if SequenceMatcher(None, src, key[0], autojunk=False).ratio() >= fuzzy:
                            return dst
        return None

    def put(self, text: str, lang: str, model: str, translation: str) -> None:
        key = (normalize(text), lang, model)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO memory (src, lang, model, dst) VALUES (?,?,?,?)", (*key, translation))
            self._db.commit()
            self._remember(key, translation)

    def _remember(self, key: tuple[str, str, str], value: str) -> None:
        self._recent[key] = value
        self._recent.move_to_end(key)
        while len(self._recent) > self._recent_max:
            self._recent.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM memory")
            self._db.commit()
            self._recent.clear()

    def size(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM memory").fetchone()[0]
