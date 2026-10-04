"""Drawing a translated block: a soft blurred backdrop that hides the original, then fitted text on top."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QTextOption

TEXT_FLAGS = int(Qt.TextFlag.TextWordWrap)


@dataclass
class Item:
    """Everything the overlay needs to draw one block (rect in frame pixels)."""

    id: int
    rect: tuple[int, int, int, int]
    line_h: int
    n_lines: int
    text: str
    color: tuple[int, int, int] | None
    opacity: float


@dataclass
class Layout:
    rect: tuple[int, int, int, int]
    px: float
    align: Qt.AlignmentFlag


def _luma(rgb: tuple[float, float, float]) -> float:
    def ch(c: float) -> float:
        c /= 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    la, lb = sorted((_luma(a), _luma(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def estimate_text_color(frame: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, int, int] | None:
    """Guess the original text colour: the pixels that differ most from the box's background."""
    x0, y0, x1, y1 = rect
    crop = frame[max(0, y0):y1, max(0, x0):x1]
    if crop.size < 30:
        return None
    px = crop.reshape(-1, 3).astype(np.float32)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]]).astype(np.float32)
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(px - bg, axis=1)
    cut = max(60.0, float(np.percentile(dist, 88)))
    fg = px[dist >= cut]
    if len(fg) < 12:
        return None
    b, g, r = np.median(fg, axis=0)
    return int(r), int(g), int(b)


def _hits(r: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], bounds: tuple[int, int]) -> bool:
    if r[0] < 0 or r[1] < 0 or r[2] > bounds[0] or r[3] > bounds[1]:
        return True
    return any(r[0] < o[2] and o[0] < r[2] and r[1] < o[3] and o[1] < r[3] for o in others)


def fit(item: Item, font: QFont, bounds: tuple[int, int], max_grow: float,
        obstacles: list[tuple[int, int, int, int]] = ()) -> Layout:
    """Pick the largest readable font, growing the box if needed, without covering any other text on screen."""
    x0, y0, x1, y1 = item.rect
    w, h = x1 - x0, y1 - y0
    base = max(12.0, item.line_h * 0.86)
    smallest = max(11.0, item.line_h * 0.55)
    left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    centre = Qt.AlignmentFlag.AlignCenter
    # Other text the grown box must not cover (lines inside this block are its own).
    others = [o for o in obstacles if not (o[0] >= x0 - 2 and o[1] >= y0 - 2 and o[2] <= x1 + 2 and o[3] <= y1 + 2)]

    def measure(px: float, width: float) -> tuple[float, float]:
        f = QFont(font)
        f.setPixelSize(int(px))
        r = QFontMetricsF(f).boundingRect(QRectF(0, 0, width, 1e5), TEXT_FLAGS, item.text)
        return r.width(), r.height()

    px = base
    while px >= smallest:
        tw, th = measure(px, w)
        if th <= h * 1.08 and tw <= w + 1:
            return Layout(item.rect, px, left)
        # Single label: widen sideways (e.g. 背包 -> "Inventory"), centred if possible,
        # otherwise toward whichever side has room.
        if item.n_lines == 1:
            lw, lh = measure(px, 1e5)
            if lh <= h * 1.15 and lw <= w * max_grow * 2:
                full, gap = int(lw + px * 0.6), px * 0.3
                cx = (x0 + x1) / 2
                for r in (
                    (int(cx - lw / 2 - gap), y0, int(cx + lw / 2 + gap), y1),
                    (x1 - full, y0, x1, y1),
                    (x0, y0, x0 + full, y1),
                ):
                    if not _hits(r, others, bounds):
                        return Layout(r, px, centre)
        # Paragraph: allow a bit wider and taller, as long as nothing else gets covered.
        for width in (w, int(w * 1.15), int(w * 1.3)):
            tw, th = measure(px, width)
            if th <= h * max_grow and tw <= width + 1:
                r = (x0, y0, x0 + width, int(y0 + max(h, th + px * 0.2)))
                if not _hits(r, others, bounds):
                    return Layout(r, px, left)
        px -= 1
    return Layout(item.rect, smallest, left)


def _side_pads(rect: tuple[int, int, int, int], pad: int, others) -> tuple[int, int, int, int]:
    """Shrink the soft edge on any side where other on-screen text sits close by."""
    l = t = r = b = pad
    for o in others:
        if o[1] < rect[3] and rect[1] < o[3]:  # beside us
            if o[2] <= rect[0]:
                l = min(l, rect[0] - o[2])
            if o[0] >= rect[2]:
                r = min(r, o[0] - rect[2])
        if o[0] < rect[2] and rect[0] < o[2]:  # above / below us
            if o[3] <= rect[1]:
                t = min(t, rect[1] - o[3])
            if o[1] >= rect[3]:
                b = min(b, o[1] - rect[3])
    return max(2, l), max(2, t), max(2, r), max(2, b)


def backdrop(frame: np.ndarray, rect: tuple[int, int, int, int], line_h: int, strength: float, tint: float,
             others=()):
    """Blurred, tinted, edge-feathered patch of what's behind the text. Returns (QImage, x, y, mean_rgb)."""
    H, W = frame.shape[:2]
    pad = int(line_h * 0.45) + 6
    pl, pt, pr, pb = _side_pads(rect, pad, others)
    x0, y0 = max(0, rect[0] - pl), max(0, rect[1] - pt)
    x1, y1 = min(W, rect[2] + pr), min(H, rect[3] + pb)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    # Blur at reduced resolution: much faster and smoother for heavy blur.
    k = 4
    small = cv2.resize(crop, (max(1, cw // k), max(1, ch // k)), interpolation=cv2.INTER_AREA)
    sigma = max(2.0, line_h * 0.32 * strength / k * 2)
    small = cv2.GaussianBlur(small, (0, 0), sigma)
    blur = cv2.resize(small, (cw, ch), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mean = blur.reshape(-1, 3).mean(axis=0)
    blur = blur * (1 - tint) + mean * tint

    # Feathered edges so the patch melts into the scene instead of looking like a sticker.
    # The feather lives entirely in the padding: the original text area stays fully covered.
    mask = np.zeros((ch, cw), np.float32)
    il, it, ir, ib = (max(1, p // 3) for p in (pl, pt, pr, pb))
    cv2.rectangle(mask, (il, it), (cw - ir - 1, ch - ib - 1), 1.0, thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(pl, pt, pr, pb, pad) / 4))
    core = (rect[0] - x0, rect[1] - y0, rect[2] - x0, rect[3] - y0)
    mask[max(0, core[1]):core[3], max(0, core[0]):core[2]] = 1.0
    rgba = np.empty((ch, cw, 4), np.uint8)
    rgba[..., 0] = blur[..., 2]
    rgba[..., 1] = blur[..., 1]
    rgba[..., 2] = blur[..., 0]
    rgba[..., 3] = np.clip(mask * 255, 0, 255)
    img = QImage(rgba.data, cw, ch, cw * 4, QImage.Format.Format_RGBA8888).copy()
    return img, x0, y0, (float(mean[2]), float(mean[1]), float(mean[0]))


def pick_colors(original: tuple[int, int, int] | None, bg: tuple[float, float, float]) -> tuple[QColor, QColor]:
    light, dark = (245, 245, 240), (18, 18, 22)
    if original and contrast(original, bg) >= 3.2:
        fg = original
    else:
        fg = light if contrast(light, bg) >= contrast(dark, bg) else dark
    shadow = (0, 0, 0) if _luma(fg) > 0.35 else (255, 255, 255)
    return QColor(*fg), QColor(*shadow, 150)


def draw_all(painter: QPainter, frame: np.ndarray, items: list[Item], font: QFont, cfg, layout_cache: dict,
             obstacles: list[tuple[int, int, int, int]] = ()) -> None:
    """Lay out every block (each one reserving its space), paint all backdrops, then all text on top."""
    H, W = frame.shape[:2]
    placed: list[tuple[int, int, int, int]] = []
    plans = []
    for item in items:
        key = (item.id, item.text, item.rect)
        lay = layout_cache.get(key)
        if lay is None:
            # avoid other text on screen and the space earlier blocks already took
            lay = layout_cache[key] = fit(item, font, (W, H), cfg.max_grow, list(obstacles) + placed)
        placed.append(lay.rect)
        plans.append((item, lay))

    painted = []
    for i, (item, lay) in enumerate(plans):
        others = [o for o in obstacles if not _inside(o, lay.rect)] + [p.rect for j, (_, p) in enumerate(plans) if j != i]
        bd = backdrop(frame, lay.rect, item.line_h, cfg.blur_strength, cfg.backdrop_tint, others)
        if bd is None:
            continue
        img, bx, by, bg = bd
        painter.setOpacity(item.opacity)
        painter.drawImage(bx, by, img)
        painted.append((item, lay, bg))

    opt_cache = {}
    for item, lay, bg in painted:
        fg, shadow = pick_colors(item.color if cfg.keep_original_color else None, bg)
        f = QFont(font)
        f.setPixelSize(int(lay.px))
        painter.setFont(f)
        painter.setOpacity(item.opacity)
        x0, y0, x1, y1 = lay.rect
        r = QRectF(x0, y0, x1 - x0, y1 - y0)
        opt = opt_cache.get(lay.align)
        if opt is None:
            opt = opt_cache[lay.align] = QTextOption(lay.align)
            opt.setWrapMode(QTextOption.WrapMode.WordWrap)
        off = max(1.0, lay.px / 16)
        painter.setPen(shadow)
        painter.drawText(r.translated(off, off), item.text, opt)
        painter.setPen(fg)
        painter.drawText(r, item.text, opt)
    painter.setOpacity(1.0)


def _inside(o: tuple[int, int, int, int], r: tuple[int, int, int, int]) -> bool:
    return o[0] >= r[0] - 2 and o[1] >= r[1] - 2 and o[2] <= r[2] + 2 and o[3] <= r[3] + 2
