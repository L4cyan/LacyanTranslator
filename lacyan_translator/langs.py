"""Language names, script detection and the prompt wording Hunyuan-MT expects."""

from __future__ import annotations

import re

# Target languages Hunyuan-MT 1.5 supports, with the Chinese name its official prompt uses.
TARGETS: dict[str, str] = {
    "English": "英语",
    "Filipino": "菲律宾语",
    "Indonesian": "印尼语",
    "Japanese": "日语",
    "Korean": "韩语",
    "Vietnamese": "越南语",
    "Thai": "泰语",
    "Malay": "马来语",
    "Spanish": "西班牙语",
    "Portuguese": "葡萄牙语",
    "French": "法语",
    "German": "德语",
    "Italian": "意大利语",
    "Russian": "俄语",
    "Arabic": "阿拉伯语",
    "Turkish": "土耳其语",
    "Hindi": "印地语",
    "Traditional Chinese": "繁体中文",
}

# What counts as "foreign text worth translating" for each source language.
SOURCE_PATTERNS: dict[str, re.Pattern[str]] = {
    "Chinese": re.compile(r"[㐀-䶿一-鿿豈-﫿]"),
}

_CJK = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")


def has_source_text(text: str, source: str, min_chars: int = 1) -> bool:
    pat = SOURCE_PATTERNS.get(source, SOURCE_PATTERNS["Chinese"])
    return len(pat.findall(text)) >= min_chars


def join_lines(lines: list[str]) -> str:
    """Join OCR lines: no space between CJK characters, a space between Latin words."""
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if out and not (_CJK.search(out[-1]) or _CJK.search(line[0])):
            out += " "
        out += line
    return out
