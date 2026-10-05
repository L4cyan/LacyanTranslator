"""Group OCR lines into text blocks: fragments on one line join up, stacked lines become paragraphs."""

from __future__ import annotations

from dataclasses import dataclass

from .langs import join_lines
from .ocr import Line


@dataclass
class Block:
    x0: int
    y0: int
    x1: int
    y1: int
    text: str
    line_h: int
    n_lines: int
    vertical: bool = False

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def _belongs_together(a: Line, b: Line) -> bool:
    if a.vertical or b.vertical:  # vertical columns stay on their own
        return False
    h = min(a.h, b.h)
    if max(a.h, b.h) > 1.7 * h:  # very different font sizes: title vs body
        return False
    v_overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
    h_gap = max(a.x0, b.x0) - min(a.x1, b.x1)
    # Same visual line, split by OCR (e.g. "师姐：" + "道友留步")
    if v_overlap > 0.5 * h and h_gap < 1.2 * h:
        return True
    # Stacked lines of one paragraph: same font size, close together, aligned. (Different sizes stacked
    # closely are usually separate UI labels: a product title over its sales count, a name over a price.)
    if max(a.h, b.h) > 1.2 * h:
        return False
    v_gap = max(a.y0, b.y0) - min(a.y1, b.y1)
    h_overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
    left_aligned = abs(a.x0 - b.x0) < 0.8 * h
    return v_gap < 0.75 * h and (h_overlap > 0.5 * min(a.w, b.w) or left_aligned)


def group(lines: list[Line]) -> list[Block]:
    parent = list(range(len(lines)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _belongs_together(lines[i], lines[j]):
                parent[find(i)] = find(j)

    clusters: dict[int, list[Line]] = {}
    for i, ln in enumerate(lines):
        clusters.setdefault(find(i), []).append(ln)

    blocks: list[Block] = []
    for members in clusters.values():
        # reading order: rows top-to-bottom (by vertical centre), then left-to-right within a row
        members.sort(key=lambda l: (l.y0 + l.y1) / 2)
        rows: list[list[Line]] = []
        for ln in members:
            if rows and min(ln.y1, rows[-1][0].y1) - max(ln.y0, rows[-1][0].y0) > 0.5 * min(ln.h, rows[-1][0].h):
                rows[-1].append(ln)
            else:
                rows.append([ln])
        for row in rows:
            row.sort(key=lambda l: l.x0)
        text = join_lines([join_lines([l.text for l in row]) for row in rows])
        blocks.append(
            Block(
                min(l.x0 for l in members),
                min(l.y0 for l in members),
                max(l.x1 for l in members),
                max(l.y1 for l in members),
                text,
                # for a vertical column the character size is the column's width
                members[0].w if members[0].vertical else int(sorted(l.h for l in members)[len(members) // 2]),
                len(rows),
                len(members) == 1 and members[0].vertical,
            )
        )
    return blocks
