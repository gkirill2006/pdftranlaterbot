from dataclasses import dataclass
from typing import Optional

from reportlab.pdfbase import pdfmetrics


@dataclass
class FitResult:
    font_size: float
    line_height: float
    lines: list[str]
    align: str


def wrap_text(text: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    if max_width <= 1:
        return [text]

    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue

        words = paragraph.split()
        current = ""
        for word in words:
            if not current:
                candidate = word
            else:
                candidate = f"{current} {word}"

            if _line_width(candidate, font_name, font_size) <= max_width:
                current = candidate
                continue

            if current:
                lines.append(current)
                current = ""

            if _line_width(word, font_name, font_size) <= max_width:
                current = word
            else:
                split_parts = _split_long_token(word, font_name, font_size, max_width)
                if split_parts:
                    lines.extend(split_parts[:-1])
                    current = split_parts[-1]
                else:
                    current = word
        if current:
            lines.append(current)

    return lines or [text]


def fit_text_to_box(
    text: str,
    font_name: str,
    max_font_size: float,
    min_font_size: float,
    box_w: float,
    box_h: float,
    align: str = "left",
    line_spacing: float = 1.15,
) -> Optional[FitResult]:
    if not text.strip():
        return None
    if box_w <= 1 or box_h <= 1:
        return None

    max_size = max(min_font_size, max_font_size)
    size = max_size
    while size >= min_font_size - 1e-6:
        lines = wrap_text(text, font_name, size, box_w)
        line_height = size * line_spacing
        total_h = len(lines) * line_height
        if total_h <= box_h:
            return FitResult(font_size=size, line_height=line_height, lines=lines, align=align)
        size -= 0.5

    fallback_size = min_font_size
    line_height = fallback_size * line_spacing
    max_lines = max(1, int(box_h // line_height))
    lines = wrap_text(text, font_name, fallback_size, box_w)
    lines = lines[:max_lines]
    if lines:
        lines[-1] = _ellipsize(lines[-1], font_name, fallback_size, box_w)
    return FitResult(font_size=fallback_size, line_height=line_height, lines=lines, align=align)


def _line_width(text: str, font_name: str, font_size: float) -> float:
    return pdfmetrics.stringWidth(text, font_name, font_size)


def _split_long_token(token: str, font_name: str, font_size: float, max_width: float) -> list[str]:
    parts: list[str] = []
    chunk = ""
    for char in token:
        candidate = f"{chunk}{char}"
        if _line_width(candidate, font_name, font_size) <= max_width:
            chunk = candidate
        else:
            if chunk:
                parts.append(chunk)
            chunk = char
    if chunk:
        parts.append(chunk)
    return parts


def _ellipsize(text: str, font_name: str, font_size: float, max_width: float) -> str:
    if _line_width(text, font_name, font_size) <= max_width:
        return text
    ellipsis = "..."
    if _line_width(ellipsis, font_name, font_size) > max_width:
        return ""
    trimmed = text
    while trimmed and _line_width(trimmed + ellipsis, font_name, font_size) > max_width:
        trimmed = trimmed[:-1]
    return trimmed + ellipsis
