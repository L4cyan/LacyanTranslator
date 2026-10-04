"""OCR (PP-OCR via RapidOCR), on the fastest backend available, returning axis-aligned text lines."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("lacyan.ocr")


@dataclass
class Line:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    score: float

    @property
    def h(self) -> int:
        return max(1, self.y1 - self.y0)

    @property
    def w(self) -> int:
        return max(1, self.x1 - self.x0)

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


class OCR:
    """Picks CUDA (NVIDIA) → DirectML (any GPU) → CPU."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from rapidocr_onnxruntime import RapidOCR

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            try:
                ort.preload_dlls()  # CUDA/cuDNN from the pip wheels
            except Exception:  # noqa: BLE001 - older onnxruntime
                pass
            _use_heuristic_cudnn()
            self.engine = RapidOCR(det_use_cuda=True, rec_use_cuda=True)
            self.device = "NVIDIA GPU (CUDA)"
        elif "DmlExecutionProvider" in providers:
            # Detection (fixed-size input) on the GPU; recognition's input shape changes constantly,
            # which DirectML handles badly, so it stays on the CPU.
            self.engine = RapidOCR(det_use_dml=True)
            self.device = "GPU detect (DirectML) + CPU read"
        else:
            self.engine = RapidOCR()
            self.device = "CPU"
        log.info("OCR ready on %s", self.device)

    def read(
        self,
        bgr: np.ndarray,
        min_conf: float,
        dirty: tuple[int, int, int, int] | None = None,
        prev: list[Line] | None = None,
    ) -> list[Line]:
        """Detect text boxes on the whole frame, then read them.

        With `dirty` + `prev`, boxes outside the changed area that match a line from the previous pass
        keep that line's text instead of being read again. Detection is cheap on the GPU; reading
        every line of a text-heavy screen is what costs time.
        """
        e = self.engine
        raw_h, raw_w = bgr.shape[:2]
        img, ratio_h, ratio_w = e.preprocess(bgr)
        record = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}
        img, record = e.maybe_add_letterbox(img, record)
        boxes, _ = e.auto_text_det(img)
        if boxes is None:
            return []
        quads = e._get_origin_points(boxes, record, raw_h, raw_w)
        rects = [(int(q[:, 0].min()), int(q[:, 1].min()), int(q[:, 0].max()), int(q[:, 1].max())) for q in quads]

        out: list[Line | None] = [None] * len(rects)
        todo: list[int] = []
        for i, r in enumerate(rects):
            if dirty is not None and prev and not _touches(r, dirty):
                match = max(prev, key=lambda l: _iou(r, l.rect), default=None)
                if match is not None and _iou(r, match.rect) > 0.7:
                    out[i] = Line(*r, match.text, match.score)
                    continue
            todo.append(i)
        if todo:
            crops = e.get_crop_img_list(img, [boxes[i] for i in todo])
            results, _ = e.text_rec(crops)
            for i, res in zip(todo, results):
                out[i] = Line(*rects[i], str(res[0]).strip(), float(res[1]))
        return [l for l in out if l is not None and l.score >= min_conf and l.text]


def _use_heuristic_cudnn() -> None:
    """RapidOCR hardcodes cuDNN's EXHAUSTIVE algorithm search, which re-benchmarks every new input
    shape (seconds per new text-line width). Heuristic search is near-identical in speed without the stalls."""
    import rapidocr_onnxruntime.utils.infer_engine as ie

    original = ie.OrtInferSession._get_ep_list
    if getattr(original, "_lacyan", False):
        return

    def patched(self):
        return [
            (name, {**opts, "cudnn_conv_algo_search": "HEURISTIC"}) if name == "CUDAExecutionProvider" else (name, opts)
            for name, opts in original(self)
        ]

    patched._lacyan = True
    ie.OrtInferSession._get_ep_list = patched


def _touches(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if not inter:
        return 0.0
    return inter / float((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
