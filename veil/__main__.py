"""Entry point.

    python -m veil                       run the live translator (tray app)
    python -m veil --snapshot in.png out.png
                                         translate a screenshot and save the result (for testing/demos)
"""

from __future__ import annotations

import argparse
import sys
import time


def snapshot(src: str, dst: str) -> int:
    import cv2
    from PySide6.QtGui import QFont, QImage, QPainter
    from PySide6.QtWidgets import QApplication

    from . import render
    from .cache import TranslationCache
    from .config import Config, data_path
    from .langs import has_source_text
    from .layout import group
    from .ocr import OCR
    from .translator import Translator

    app = QApplication(sys.argv)  # noqa: F841 (fonts need a GUI app)
    cfg = Config.load()
    frame = cv2.imread(src)
    if frame is None:
        print(f"Can't read {src}")
        return 1
    ocr = OCR()
    ocr.read(frame, cfg.min_confidence)  # warm-up
    t = time.perf_counter()
    every = ocr.read(frame, cfg.min_confidence)
    obstacles = [(l.x0, l.y0, l.x1, l.y1) for l in every]
    lines = [l for l in every if has_source_text(l.text, cfg.source_language)]
    ocr_ms = (time.perf_counter() - t) * 1000
    blocks = group(lines)
    tr = Translator(cfg)
    cache = TranslationCache(str(data_path("translations.sqlite")))
    items = []
    for i, b in enumerate(blocks):
        t = time.perf_counter()
        out = cache.get(b.text, cfg.target_language, cfg.model)
        src_tag = "memory"
        if out is None:
            out = tr.translate(b.text)
            cache.put(b.text, cfg.target_language, cfg.model, out)
            src_tag = "model"
        print(f"[{(time.perf_counter() - t) * 1000:5.0f} ms {src_tag:6}] {b.text}  ->  {out}")
        items.append(render.Item(i, b.rect, b.line_h, b.n_lines, out, render.estimate_text_color(frame, b.rect), 1.0))
    print(f"OCR {ocr_ms:.0f} ms on {ocr.device}, {len(lines)} lines -> {len(blocks)} blocks")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888).copy()
    p = QPainter(img)
    p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
    font = QFont(cfg.font_family)
    font.setWeight(QFont.Weight.DemiBold)
    layouts: dict = {}
    render.draw_all(p, frame, items, font, cfg, layouts, obstacles)
    p.end()
    img.save(dst)
    tr.shutdown()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="veil", description="Live on-screen translator")
    ap.add_argument("--snapshot", nargs=2, metavar=("IN", "OUT"), help="translate an image file and save the result")
    args = ap.parse_args()
    if args.snapshot:
        return snapshot(*args.snapshot)
    from .app import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
