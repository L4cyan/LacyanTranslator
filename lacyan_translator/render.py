"""Drawing a translated block: a soft blurred backdrop that hides the original, then fitted text on top."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QTextLayout, QTextOption

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
    moving: bool = False


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
    # Nothing fits without covering other text: keep the smallest size and grow the box to the text,
    # so text never spills outside the area that gets repainted (that left "residue" behind).
    tw, th = measure(smallest, w)
    return Layout((x0, y0, max(x1, x0 + int(tw) + 2), max(y1, y0 + int(th) + 2)), smallest, left)


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
    """Fill + outline colours for the most readable result: keep the game's text colour when it stands out,
    otherwise near-white; the outline is always the opposite (dark for light text, light for dark text)."""
    light, dark = (250, 250, 246), (16, 16, 20)
    if original and contrast(original, bg) >= 3.0:
        fg = original
    else:
        fg = light if contrast(light, bg) >= contrast(dark, bg) else dark
    outline = (8, 8, 12) if _luma(fg) > 0.3 else (245, 245, 240)
    return QColor(*fg), QColor(*outline)


def text_path(text: str, font: QFont, width: float, height: float, align: Qt.AlignmentFlag) -> QPainterPath:
    """Word-wrapped text as a vector path (so it can be outlined), vertically centred in width×height."""
    opt = QTextOption()
    opt.setWrapMode(QTextOption.WrapMode.WordWrap)
    layout = QTextLayout(text, font)
    layout.setTextOption(opt)
    layout.beginLayout()
    lines, y = [], 0.0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        line.setPosition(QPointF(0, y))
        y += line.height()
        lines.append(line)
    layout.endLayout()
    fm = QFontMetricsF(font)
    top = max(0.0, (height - y) / 2)
    path = QPainterPath()
    centred = bool(align & Qt.AlignmentFlag.AlignHCenter)
    for line in lines:
        seg = text[line.textStart(): line.textStart() + line.textLength()].rstrip()
        x = (width - fm.horizontalAdvance(seg)) / 2 if centred else 0.0
        path.addText(x, top + line.y() + line.ascent(), font, seg)
    return path


@dataclass
class Plan:
    """One block, ready to paint (all coordinates in frame pixels)."""

    item: Item
    rect: tuple[int, int, int, int]
    img: QImage  # blurred backdrop
    bx: int
    by: int
    text: QImage  # pre-rendered text (shadow + outline + fill)
    tx: int
    ty: int
    bounds: tuple[int, int, int, int]

    def shifted(self, dx: int, dy: int) -> "Plan":
        if not dx and not dy:
            return self
        b = self.bounds
        return Plan(self.item, (self.rect[0] + dx, self.rect[1] + dy, self.rect[2] + dx, self.rect[3] + dy), self.img,
                    self.bx + dx, self.by + dy, self.text, self.tx + dx, self.ty + dy, (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy))


class RenderCache:
    """Per-block layouts, text images and backdrops, kept relative to the block so a scrolling block
    just moves. Backdrops are re-blurred a few times a second, spread across frames."""

    BACKDROP_TTL = 0.35
    REFRESH_PER_FRAME = 12

    def __init__(self) -> None:
        self.layouts: dict = {}
        self.texts: dict = {}
        self.backdrops: dict = {}

    def prune(self, live_ids: set[int]) -> None:
        for d in (self.layouts, self.texts):
            for k in [k for k in d if k[0] not in live_ids]:
                del d[k]
        for k in [k for k in self.backdrops if k not in live_ids]:
            del self.backdrops[k]


def _stroke_width(px: float) -> float:
    return max(2.0, px * 0.16)


def render_text(text: str, font: QFont, px: float, w: int, h: int, align, fill: QColor, outline: QColor) -> tuple[QImage, int]:
    """Text with a soft shadow, a crisp outline and the fill, drawn once into an image. Returns (image, margin)."""
    f = QFont(font)
    f.setPixelSize(int(px))
    path = text_path(text, f, w, h, align)
    sw = _stroke_width(px)
    br = path.boundingRect()
    m = int(sw * 2 + 3)
    width = int(max(w, br.right()) + 2 * m) + 1
    height = int(max(h, br.bottom()) + 2 * m) + 1
    img = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.translate(m, m)
    shadow = QColor(outline)
    shadow.setAlpha(90)
    p.save()
    p.translate(sw * 0.6, sw * 0.8)
    p.setPen(QPen(shadow, sw * 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(shadow)
    p.drawPath(path)
    p.restore()
    ol = QColor(outline)
    ol.setAlpha(235)
    p.setPen(QPen(ol, sw, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(path)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(fill)
    p.drawPath(path)
    p.end()
    return img, m


def plan_all(frame: np.ndarray, items: list[Item], font: QFont, cfg, cache: RenderCache,
             obstacles: list[tuple[int, int, int, int]], now: float) -> list[Plan]:
    H, W = frame.shape[:2]
    placed: list[tuple[int, int, int, int]] = []
    laid = []
    for item in items:
        x0, y0, x1, y1 = item.rect
        key = (item.id, item.text, x1 - x0, y1 - y0)
        rel = cache.layouts.get(key)
        if rel is None:
            # avoid other text on screen and the space earlier blocks already took
            lay = fit(item, font, (W, H), cfg.max_grow, list(obstacles) + placed)
            rel = (lay.rect[0] - x0, lay.rect[1] - y0, lay.rect[2] - x0, lay.rect[3] - y0, lay.px, lay.align)
            cache.layouts[key] = rel
        r = (x0 + rel[0], y0 + rel[1], x0 + rel[2], y0 + rel[3])
        placed.append(r)
        laid.append((item, r, rel[4], rel[5]))

    # Re-blur the stalest backdrops first, a limited number per frame, so a busy page never stalls a frame.
    stale = sorted(
        (i for i, (item, r, _, _) in enumerate(laid)
         if (bd := cache.backdrops.get(item.id)) is None or bd[0] != (r[2] - r[0], r[3] - r[1], item.line_h)
         or now - bd[1] >= RenderCache.BACKDROP_TTL),
        key=lambda i: cache.backdrops.get(laid[i][0].id, (None, 0.0))[1],
    )
    must = {i for i in stale if laid[i][0].id not in cache.backdrops
            or cache.backdrops[laid[i][0].id][0] != (laid[i][1][2] - laid[i][1][0], laid[i][1][3] - laid[i][1][1], laid[i][0].line_h)}
    refresh = must | set(stale[: RenderCache.REFRESH_PER_FRAME])

    plans = []
    for i, (item, r, px, align) in enumerate(laid):
        size_key = (r[2] - r[0], r[3] - r[1], item.line_h)
        if i in refresh:
            others = [o for o in obstacles if not _inside(o, r)] + [p[1] for j, p in enumerate(laid) if j != i]
            made = backdrop(frame, r, item.line_h, cfg.blur_strength, cfg.backdrop_tint, others)
            if made is None:
                continue
            img, bx, by, bg = made
            cache.backdrops[item.id] = (size_key, now, img, bx - r[0], by - r[1], bg)
        bd = cache.backdrops.get(item.id)
        if bd is None:
            continue
        _, _, img, rx, ry, bg = bd

        fill, outline = pick_colors(item.color if cfg.keep_original_color else None, bg)
        tkey = (item.id, item.text, r[2] - r[0], r[3] - r[1], px, int(align), fill.rgb(), outline.rgb())
        txt = cache.texts.get(tkey)
        if txt is None:
            txt = cache.texts[tkey] = render_text(item.text, font, px, r[2] - r[0], r[3] - r[1], align, fill, outline)
        timg, m = txt
        bx, by = r[0] + rx, r[1] + ry
        tx, ty = r[0] - m, r[1] - m
        bounds = (min(bx, tx), min(by, ty), max(bx + img.width(), tx + timg.width()), max(by + img.height(), ty + timg.height()))
        plans.append(Plan(item, r, img, bx, by, timg, tx, ty, bounds))
    cache.prune({it.id for it in items})
    return plans


def paint_all(painter: QPainter, plans: list[Plan], cfg) -> None:
    """Backdrops first, then all text on top, so one block's blur never covers another's words."""
    for p in plans:
        painter.setOpacity(p.item.opacity)
        painter.drawImage(p.bx, p.by, p.img)
    for p in plans:
        painter.setOpacity(p.item.opacity)
        painter.drawImage(p.tx, p.ty, p.text)
    painter.setOpacity(1.0)


def _inside(o: tuple[int, int, int, int], r: tuple[int, int, int, int]) -> bool:
    return o[0] >= r[0] - 2 and o[1] >= r[1] - 2 and o[2] <= r[2] + 2 and o[3] <= r[3] + 2
