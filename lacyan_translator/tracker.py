"""Keep text blocks stable across OCR passes so the overlay doesn't flicker or re-translate needlessly."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .layout import Block

_ids = itertools.count(1)


def similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / float(area(a) + area(b) - inter)


@dataclass
class Track:
    block: Block
    id: int = field(default_factory=lambda: next(_ids))
    stable_text: str = ""  # text that has been seen unchanged for enough passes
    stable_count: int = 0
    translation: str | None = None
    translated_text: str = ""  # which source text `translation` belongs to
    requested_text: str = ""
    misses: int = 0
    shown_at: float = 0.0
    lost_at: float = 0.0
    text_color: tuple[int, int, int] | None = None

    @property
    def visible(self) -> bool:
        return self.translation is not None


class Tracker:
    def __init__(self, stable_passes: int, keep_missing: int) -> None:
        self.tracks: list[Track] = []
        self.stable_passes = stable_passes
        self.keep_missing = keep_missing

    def update(self, blocks: list[Block]) -> list[Track]:
        """Match this pass's blocks to existing tracks. Returns tracks whose text just became stable."""
        now = time.time()
        unmatched = list(self.tracks)
        ready: list[Track] = []
        for b in blocks:
            best, best_score = None, 0.0
            for t in unmatched:
                overlap = iou(b.rect, t.block.rect)
                sim = similar(b.text, t.block.text)
                # a growing typewriter line overlaps a lot and starts with the old text
                grows = b.text.startswith(t.block.text[: max(1, len(t.block.text) - 1)])
                score = overlap * 0.6 + sim * 0.4 + (0.2 if grows and overlap > 0.2 else 0)
                if score > best_score and (overlap > 0.25 or sim > 0.85):
                    best, best_score = t, score
            if best is None:
                best = Track(block=b)
                self.tracks.append(best)
            else:
                unmatched.remove(best)
            t = best
            if t.lost_at:
                t.lost_at = 0.0
            t.misses = 0
            if similar(b.text, t.block.text) >= 0.97:
                t.stable_count += 1
            else:
                t.stable_count = 1
            t.block = b
            if t.stable_count >= self.stable_passes and t.stable_text != b.text:
                t.stable_text = b.text
                ready.append(t)
        for t in unmatched:
            t.misses += 1
            if not t.lost_at:
                t.lost_at = now
        self.tracks = [t for t in self.tracks if t.misses <= self.keep_missing]
        return ready

    def clear(self) -> None:
        self.tracks.clear()
