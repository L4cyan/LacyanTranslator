"""Per-frame tracking of on-screen text by template matching.

Each visible translation keeps a small grayscale picture ("reference") of the text it covers. Every frame
we check that picture is still where we think it is. If it moved (scrolling, a moving window) we find it
again nearby; if it's gone (covered by a popup, scrolled away, replaced) the translation hides at once.
This is a few cheap normalized-correlation checks per frame, far lighter than OCR.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

STAY = 0.80  # correlation needed to say "still here"
MOVE = 0.72  # correlation needed to accept a moved match
GONE = 0.35  # below this the text is clearly not there any more


class Pyramid:
    """Grayscale frame at full, 1/2 and 1/4 resolution, built once per frame."""

    def __init__(self, bgr: np.ndarray) -> None:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = g.shape
        half = cv2.resize(g, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        quarter = cv2.resize(half, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
        self.levels = {1: g, 2: half, 4: quarter}
        self.w, self.h = w, h

    def thumb(self) -> np.ndarray:
        """Small image used to detect which part of the screen changed."""
        q = self.levels[4]
        return cv2.resize(q, (max(1, self.w // 8), max(1, self.h // 8)), interpolation=cv2.INTER_AREA)


@dataclass
class Ref:
    img: np.ndarray
    level: int


def _level_for(h: int) -> int:
    # keep the reference around 10-20 px tall: enough detail to be unique, small enough to be fast
    return 4 if h >= 48 else 2 if h >= 18 else 1


def make_ref(pyr: Pyramid, rect: tuple[int, int, int, int]) -> Ref | None:
    x0, y0, x1, y1 = rect
    lv = _level_for(y1 - y0)
    g = pyr.levels[lv]
    a = g[max(0, y0 // lv):y1 // lv, max(0, x0 // lv):x1 // lv]
    if a.shape[0] < 4 or a.shape[1] < 6 or float(a.std()) < 6.0:
        return None  # too small or too flat to track reliably
    return Ref(a.copy(), lv)


def score_at(pyr: Pyramid, ref: Ref, x: float, y: float) -> float:
    lv = ref.level
    g = pyr.levels[lv]
    hh, ww = ref.img.shape
    xs, ys = int(round(x / lv)), int(round(y / lv))
    if xs < 0 or ys < 0 or xs + ww > g.shape[1] or ys + hh > g.shape[0]:
        return -1.0
    patch = g[ys:ys + hh, xs:xs + ww]
    if float(patch.std()) < 2.0:
        return 0.0
    return float(cv2.matchTemplate(patch, ref.img, cv2.TM_CCOEFF_NORMED)[0, 0])


def search(pyr: Pyramid, ref: Ref, x: float, y: float, mx: int, my: int) -> tuple[float, int, int]:
    """Best match of `ref` within ±(mx, my) full-res pixels of (x, y). Returns (score, x, y)."""
    lv = ref.level
    g = pyr.levels[lv]
    hh, ww = ref.img.shape
    xs, ys = int(round(x / lv)), int(round(y / lv))
    mxs, mys = mx // lv + 1, my // lv + 1
    wx0, wy0 = max(0, xs - mxs), max(0, ys - mys)
    wx1, wy1 = min(g.shape[1], xs + ww + mxs), min(g.shape[0], ys + hh + mys)
    if wx1 - wx0 < ww or wy1 - wy0 < hh:
        return -1.0, int(x), int(y)
    res = cv2.matchTemplate(g[wy0:wy1, wx0:wx1], ref.img, cv2.TM_CCOEFF_NORMED)
    np.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    _, best, _, loc = cv2.minMaxLoc(res)
    return float(best), (wx0 + loc[0]) * lv, (wy0 + loc[1]) * lv


def locate(pyr: Pyramid, ref: Ref, x: int, y: int, h: int, guess: tuple[float, float]) -> tuple[float, int, int]:
    """Where is this text now? Checks in place, then at the predicted position, then searches around it."""
    s = score_at(pyr, ref, x, y)
    if s >= STAY:
        return s, x, y
    gx, gy = x + guess[0], y + guess[1]
    if guess != (0, 0):
        s2 = score_at(pyr, ref, gx, gy)
        if s2 >= STAY:
            return s2, int(round(gx)), int(round(gy))
    my = max(4 * h, 160) + int(abs(guess[1]))
    mx = max(2 * h, 64) + int(abs(guess[0]))
    s3, nx, ny = search(pyr, ref, gx, gy, mx, my)
    if s3 >= MOVE:
        return s3, nx, ny
    return max(s, s3), x, y
