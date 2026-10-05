"""Translation memory: every translated phrase is kept on disk, so the same phrase shows instantly next time.

Matching is exact on the whole phrase. Only spacing and full-width/half-width forms are normalized
(so "满299减30" and "满 299 减 30" are the same phrase, but "满299减30" and "满298减30" are not).
"""

from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
from collections import OrderedDict

_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACE.sub("", unicodedata.normalize("NFKC", text))


class TranslationCache:
    def __init__(self, path: str, recent: int = 4000) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memory (src TEXT, lang TEXT, model TEXT, dst TEXT, hits INTEGER DEFAULT 0,"
            " PRIMARY KEY (src, lang, model))"
        )
        self._lock = threading.Lock()
        self._recent: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._recent_max = recent

    def get(self, text: str, lang: str, model: str) -> str | None:
        key = (normalize(text), lang, model)
        with self._lock:
            if key in self._recent:
                self._recent.move_to_end(key)
                return self._recent[key]
            row = self._db.execute("SELECT dst FROM memory WHERE src=? AND lang=? AND model=?", key).fetchone()
            if row:
                self._remember(key, row[0])
                return row[0]
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
