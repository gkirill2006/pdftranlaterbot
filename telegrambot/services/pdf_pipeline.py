import asyncio
from collections import defaultdict
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Awaitable, Callable, Optional
from uuid import uuid4

import fitz

from config import Settings
from services.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

LATIN_PATTERN = re.compile(r"[A-Za-z]")


@dataclass
class TextBlock:
    block_id: str
    x: float
    y: float
    w: float
    h: float
    text_en: str
    text_ru: str = ""
    bold: bool = False
    size_hint: Optional[float] = None
    align: str = "left"
    single_line: bool = True
    render_y: Optional[float] = None
    render_h: Optional[float] = None


@dataclass
class PageData:
    index: int
    width_pt: float
    height_pt: float
    blocks: list[TextBlock] = field(default_factory=list)
    used_vision_fallback: bool = False


@dataclass
class TableGeometry:
    x_guides: list[float]
    y_guides: list[float]
    bounds: fitz.Rect


class PDFPipeline:
    FONT_NAME_REGULAR = "F0"
    FONT_NAME_BOLD = "F1"

    def __init__(self, settings: Settings, openai_client: OpenAIClient) -> None:
        self.settings = settings
        self.openai_client = openai_client
        self._translation_cache: dict[str, str] = {}
        self._translation_lock = asyncio.Lock()
        self._translate_sem = asyncio.Semaphore(settings.translate_parallelism)
        self._font_regular_path, self._font_bold_path = self._resolve_font_paths()
        self._font_regular = fitz.Font(fontfile=str(self._font_regular_path))
        self._font_bold = fitz.Font(fontfile=str(self._font_bold_path))

    async def process_pdf(
        self,
        input_pdf_path: Path,
        progress_cb: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> Path:
        doc = fitz.open(str(input_pdf_path))
        try:
            pages = await self._extract_pages_with_blocks(doc, input_pdf_path)
            if not pages:
                raise RuntimeError("PDF has no pages")

            total = len(pages)
            await self._translate_pages(pages, progress_cb, total)

            for page_data in pages:
                page = doc[page_data.index]
                self._apply_translations_to_page(page, page_data.blocks)

            output_path = self.settings.output_dir / f"translated_{uuid4().hex}.pdf"
            doc.save(
                str(output_path),
                garbage=4,
                deflate=True,
                clean=True,
            )
            return output_path
        finally:
            doc.close()

    async def _extract_pages_with_blocks(self, doc: fitz.Document, input_pdf_path: Path) -> list[PageData]:
        pages: list[PageData] = []
        for page_index in range(doc.page_count):
            page = doc[page_index]
            page_data = PageData(
                index=page_index,
                width_pt=float(page.rect.width),
                height_pt=float(page.rect.height),
            )

            native_blocks = self._extract_native_blocks_from_pdf_page(page)
            if native_blocks:
                page_data.blocks = native_blocks
            else:
                page_data.blocks = await self._extract_vision_blocks_for_page(
                    page=page,
                    page_index=page_index,
                    input_pdf_path=input_pdf_path,
                )
                page_data.used_vision_fallback = bool(page_data.blocks)

            pages.append(page_data)
        return pages

    def _extract_native_blocks_from_pdf_page(self, page: fitz.Page) -> list[TextBlock]:
        raw = page.get_text("dict")
        blocks: list[TextBlock] = []
        table = self._detect_table_geometry(page)
        table_cells: dict[tuple[int, int], list[tuple[float, float, str, float, bool]]] = defaultdict(list)

        for idx, raw_block in enumerate(raw.get("blocks", []), start=1):
            if raw_block.get("type") != 0:
                continue

            block_bbox = raw_block.get("bbox")
            if not block_bbox or len(block_bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in block_bbox]
            w = x1 - x0
            h = y1 - y0
            if w <= 1 or h <= 1:
                continue

            line_records: list[tuple[str, tuple[float, float, float, float], float, bool]] = []
            for line in raw_block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join((str(span.get("text", "")) for span in spans)).strip()
                line_bbox = line.get("bbox")
                if (
                    not line_text
                    or not LATIN_PATTERN.search(line_text)
                    or not line_bbox
                    or len(line_bbox) != 4
                ):
                    continue

                span_sizes: list[float] = []
                bold_hits = 0
                span_total = 0
                for span in spans:
                    span_text = str(span.get("text", ""))
                    if not LATIN_PATTERN.search(span_text):
                        continue
                    size = span.get("size")
                    if isinstance(size, (int, float)):
                        span_sizes.append(float(size))
                    font_name = str(span.get("font", "")).lower()
                    flags = int(span.get("flags", 0) or 0)
                    if "bold" in font_name or "demi" in font_name or (flags & 16):
                        bold_hits += 1
                    span_total += 1

                size_hint = median(span_sizes) if span_sizes else max(7.0, (line_bbox[3] - line_bbox[1]) * 0.7)
                is_bold = span_total > 0 and (bold_hits / span_total) >= 0.45
                line_records.append((line_text, tuple(float(v) for v in line_bbox), size_hint, is_bold))

            if not line_records:
                continue

            if table and self._rect_intersects_table(x0, y0, x1, y1, table):
                self._append_table_line_records(table_cells, table, line_records)
                continue

            text_en = "\n".join(line for line, _, _, _ in line_records)
            size_hint = median([sz for _, _, sz, _ in line_records])
            is_bold = sum(1 for _, _, _, b in line_records if b) >= (len(line_records) / 2)

            blocks.append(
                TextBlock(
                    block_id=f"native_{page.number + 1}_{idx}",
                    x=x0,
                    y=y0,
                    w=w,
                    h=h,
                    text_en=text_en,
                    bold=is_bold,
                    size_hint=size_hint,
                    align="left",
                    single_line=len(line_records) == 1,
                )
            )

        if table_cells:
            for (row_idx, col_idx), lines_data in table_cells.items():
                if row_idx < 0 or col_idx < 0:
                    continue
                if row_idx >= len(table.y_guides) - 1 or col_idx >= len(table.x_guides) - 1:
                    continue
                lines_data.sort(key=lambda item: (round(item[0], 2), round(item[1], 2)))
                text_lines = [item[2] for item in lines_data if item[2].strip()]
                if not text_lines:
                    continue
                sizes = [item[3] for item in lines_data if item[3] > 0]
                bold_ratio = sum(1 for item in lines_data if item[4]) / len(lines_data)

                x_left = table.x_guides[col_idx]
                x_right = table.x_guides[col_idx + 1]
                y_top = table.y_guides[row_idx]
                y_bottom = table.y_guides[row_idx + 1]
                pad_x = min(4.0, max(1.2, (x_right - x_left) * 0.015))
                pad_y = min(3.0, max(0.8, (y_bottom - y_top) * 0.06))

                blocks.append(
                    TextBlock(
                        block_id=f"tbl_{page.number + 1}_{row_idx}_{col_idx}",
                        x=x_left + pad_x,
                        y=y_top + pad_y,
                        w=max(1.0, (x_right - x_left) - 2 * pad_x),
                        h=max(1.0, (y_bottom - y_top) - 2 * pad_y),
                        text_en="\n".join(text_lines),
                        bold=bold_ratio >= 0.5,
                        size_hint=median(sizes) if sizes else None,
                        align="left",
                        single_line=len(text_lines) == 1,
                    )
                )

        blocks.sort(key=lambda block: (round(block.y, 2), round(block.x, 2)))
        return blocks

    @staticmethod
    def _cluster_positions(values: list[float], tolerance: float = 1.1) -> list[float]:
        if not values:
            return []
        values = sorted(values)
        groups: list[list[float]] = [[values[0]]]
        for value in values[1:]:
            if abs(value - groups[-1][-1]) <= tolerance:
                groups[-1].append(value)
            else:
                groups.append([value])
        return [sum(group) / len(group) for group in groups]

    def _detect_table_geometry(self, page: fitz.Page) -> Optional[TableGeometry]:
        vertical_raw: list[tuple[float, float, float]] = []
        horizontal_raw: list[tuple[float, float, float]] = []
        for drawing in page.get_drawings():
            for item in drawing.get("items", []):
                if not item or item[0] != "l":
                    continue
                p1, p2 = item[1], item[2]
                if abs(p1.x - p2.x) <= 0.7 and abs(p1.y - p2.y) >= 60:
                    vertical_raw.append((float(p1.x), float(min(p1.y, p2.y)), float(max(p1.y, p2.y))))
                elif abs(p1.y - p2.y) <= 0.7 and abs(p1.x - p2.x) >= 80:
                    horizontal_raw.append((float(p1.y), float(min(p1.x, p2.x)), float(max(p1.x, p2.x))))

        if not vertical_raw or not horizontal_raw:
            return None

        x_guides = self._cluster_positions([v[0] for v in vertical_raw], tolerance=1.2)
        if len(x_guides) < 3:
            return None
        x_min, x_max = min(x_guides), max(x_guides)
        table_width = x_max - x_min
        if table_width < 140:
            return None

        y_candidates: list[float] = []
        for y, hx0, hx1 in horizontal_raw:
            overlap = max(0.0, min(hx1, x_max) - max(hx0, x_min))
            if overlap >= table_width * 0.75:
                y_candidates.append(y)
        y_guides = self._cluster_positions(y_candidates, tolerance=1.2)
        if len(y_guides) < 3:
            return None

        y_min, y_max = min(y_guides), max(y_guides)
        if (y_max - y_min) < 80:
            return None

        return TableGeometry(
            x_guides=sorted(x_guides),
            y_guides=sorted(y_guides),
            bounds=fitz.Rect(x_min, y_min, x_max, y_max),
        )

    @staticmethod
    def _rect_intersects_table(x0: float, y0: float, x1: float, y1: float, table: TableGeometry) -> bool:
        rect = fitz.Rect(x0, y0, x1, y1)
        return rect.intersects(table.bounds)

    @staticmethod
    def _interval_index(guides: list[float], value: float) -> Optional[int]:
        if len(guides) < 2:
            return None
        for i in range(len(guides) - 1):
            if guides[i] - 0.8 <= value <= guides[i + 1] + 0.8:
                return i
        return None

    def _append_table_line_records(
        self,
        table_cells: dict[tuple[int, int], list[tuple[float, float, str, float, bool]]],
        table: TableGeometry,
        line_records: list[tuple[str, tuple[float, float, float, float], float, bool]],
    ) -> None:
        if len(table.x_guides) < 3 or len(table.y_guides) < 2:
            return

        col_count = len(table.x_guides) - 1
        two_col_table = col_count == 2
        divider_x = table.x_guides[1] if two_col_table else None

        for text, bbox, size_hint, is_bold in line_records:
            x0, y0, x1, y1 = bbox
            y_center = (y0 + y1) / 2.0
            row_idx = self._interval_index(table.y_guides, y_center)
            if row_idx is None:
                continue

            if two_col_table and divider_x and "|" in text and (x0 < divider_x < x1):
                left_text, right_text = text.split("|", 1)
                left_text = left_text.strip()
                right_text = right_text.strip()
                if left_text:
                    table_cells[(row_idx, 0)].append((y0, x0, left_text, size_hint, is_bold))
                if right_text:
                    table_cells[(row_idx, 1)].append((y0, divider_x, right_text, size_hint, is_bold))
                continue

            x_center = (x0 + x1) / 2.0
            col_idx = self._interval_index(table.x_guides, x_center)
            if col_idx is None:
                continue
            table_cells[(row_idx, col_idx)].append((y0, x0, text.strip(), size_hint, is_bold))

    async def _extract_vision_blocks_for_page(
        self,
        page: fitz.Page,
        page_index: int,
        input_pdf_path: Path,
    ) -> list[TextBlock]:
        scale = self.settings.render_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = self.settings.tmp_dir / f"{input_pdf_path.stem}_p{page_index}_vision.png"
        pix.save(str(image_path))

        try:
            layout = await self.openai_client.extract_layout(image_path)
            return self._normalize_vision_blocks(
                layout=layout,
                image_width_px=pix.width,
                image_height_px=pix.height,
                page_width_pt=float(page.rect.width),
                page_height_pt=float(page.rect.height),
                page_number=page_index + 1,
            )
        except Exception as exc:
            logger.warning("Vision layout failed on page %s: %s", page_index + 1, exc)
            return []
        finally:
            image_path.unlink(missing_ok=True)

    @staticmethod
    def _normalize_vision_blocks(
        layout: dict,
        image_width_px: int,
        image_height_px: int,
        page_width_pt: float,
        page_height_pt: float,
        page_number: int,
    ) -> list[TextBlock]:
        if image_width_px <= 0 or image_height_px <= 0:
            return []

        sx = page_width_pt / image_width_px
        sy = page_height_pt / image_height_px
        blocks: list[TextBlock] = []

        for idx, block in enumerate(layout.get("blocks", []), start=1):
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "text")).lower() != "text":
                continue

            bbox = block.get("bbox", {}) or {}
            try:
                x = float(bbox.get("x", 0))
                y = float(bbox.get("y", 0))
                w = float(bbox.get("w", 0))
                h = float(bbox.get("h", 0))
            except (TypeError, ValueError):
                continue

            if w <= 1 or h <= 1:
                continue

            text_en = str(block.get("text_en", "")).strip()
            if not text_en:
                continue

            style = block.get("style", {}) or {}
            raw_size = style.get("size_hint")
            if isinstance(raw_size, (int, float)):
                size_hint = float(raw_size) * sy
            else:
                size_hint = None

            align = str(style.get("align", "left")).lower()
            if align not in {"left", "center", "right"}:
                align = "left"

            blocks.append(
                TextBlock(
                    block_id=f"vision_{page_number}_{idx}",
                    x=max(0.0, x * sx),
                    y=max(0.0, y * sy),
                    w=max(1.0, w * sx),
                    h=max(1.0, h * sy),
                    text_en=text_en,
                    bold=bool(style.get("bold", False)),
                    size_hint=size_hint,
                    align=align,
                    single_line="\n" not in text_en,
                )
            )

        blocks.sort(key=lambda block: (round(block.y, 2), round(block.x, 2)))
        return blocks

    async def _translate_pages(
        self,
        pages: list[PageData],
        progress_cb: Optional[Callable[[int, int], Awaitable[None]]],
        total: int,
    ) -> None:
        done = 0
        page_sem = asyncio.Semaphore(self.settings.page_parallelism)

        async def translate_page(page_data: PageData) -> int:
            if not page_data.blocks:
                return page_data.index
            async with page_sem:
                await self.translate_blocks(page_data.blocks)
            return page_data.index

        tasks = [asyncio.create_task(translate_page(page_data)) for page_data in pages]
        for task in asyncio.as_completed(tasks):
            await task
            done += 1
            if progress_cb:
                await progress_cb(done, total)

    async def translate_blocks(self, blocks: list[TextBlock]) -> None:
        unique_texts = {b.text_en for b in blocks if b.text_en.strip()}

        async def translate_one(source: str) -> tuple[str, str]:
            translated = await self._translate_text_cached(source)
            return source, translated

        text_to_ru: dict[str, str] = {}
        tasks = [asyncio.create_task(translate_one(text)) for text in unique_texts]
        for task in asyncio.as_completed(tasks):
            source, translated = await task
            text_to_ru[source] = translated

        for block in blocks:
            block.text_ru = text_to_ru.get(block.text_en, block.text_en)

    async def _translate_text_cached(self, text_en: str) -> str:
        if text_en in self._translation_cache:
            return self._translation_cache[text_en]

        if not LATIN_PATTERN.search(text_en):
            return text_en

        async with self._translate_sem:
            translated = await self.openai_client.translate_text(text_en)

        async with self._translation_lock:
            self._translation_cache[text_en] = translated
        return translated

    def _apply_translations_to_page(self, page: fitz.Page, blocks: list[TextBlock]) -> None:
        prepared: list[tuple[TextBlock, str]] = []

        for block in blocks:
            text = self._sanitize_text_for_pdf((block.text_ru or "").strip())
            if not text:
                continue

            rect = fitz.Rect(block.x, block.y, block.x + block.w, block.y + block.h)
            if rect.width <= 0.8 or rect.height <= 0.8:
                continue

            prepared.append((block, text))

        if not prepared:
            return

        page_scale = self._compute_page_font_scale(prepared)

        draw_specs: list[tuple[fitz.Rect, str, bool, str, Optional[float]]] = []
        for block, text in prepared:
            original_rect = fitz.Rect(block.x, block.y, block.x + block.w, block.y + block.h)
            draw_rect = original_rect
            if draw_rect.width <= 0.8 or draw_rect.height <= 0.8:
                continue

            redaction_inset = min(0.35, original_rect.width / 18.0, original_rect.height / 18.0)
            redaction_rect = fitz.Rect(
                original_rect.x0 + redaction_inset,
                original_rect.y0 + redaction_inset,
                original_rect.x1 - redaction_inset,
                original_rect.y1 - redaction_inset,
            )
            if redaction_rect.width <= 0.2 or redaction_rect.height <= 0.2:
                redaction_rect = original_rect

            page.add_redact_annot(redaction_rect, fill=(1, 1, 1))
            draw_specs.append((draw_rect, text, block.bold, block.align, block.size_hint))

        if not draw_specs:
            return

        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        for fill_rect, text, bold, align, size_hint in draw_specs:
            self._draw_text_preserve_size(
                page=page,
                rect=fill_rect,
                text=text,
                bold=bold,
                align=align,
                size_hint=size_hint,
                page_scale=page_scale,
            )

    def _compute_page_font_scale(self, prepared: list[tuple[TextBlock, str]]) -> float:
        ratios: list[float] = []
        line_spacing = 1.15
        for block, text in prepared:
            base_size = self._source_font_size(block.size_hint, block.h)
            font = self._font_bold if block.bold else self._font_regular
            lines = self._wrap_text(text, font, base_size, max(1.0, block.w))
            needed_h = max(base_size, len(lines) * base_size * line_spacing)
            if needed_h > 0:
                ratios.append(min(1.0, block.h / needed_h))

        if not ratios:
            return 1.0

        ratios.sort()
        p20_index = max(0, int(len(ratios) * 0.2) - 1)
        p20 = ratios[p20_index]
        # One global factor for whole page to avoid local font-size jumps.
        return max(0.72, min(1.0, p20 * 0.98))

    def _apply_vertical_flow_layout(
        self,
        prepared: list[tuple[TextBlock, str]],
        page_scale: float,
    ) -> None:
        row_tolerance = 2.2
        ordered = sorted(prepared, key=lambda item: (round(item[0].y, 2), round(item[0].x, 2)))
        rows: list[list[tuple[TextBlock, str]]] = []

        current_row: list[tuple[TextBlock, str]] = []
        anchor_y: Optional[float] = None
        for item in ordered:
            block = item[0]
            if anchor_y is None or abs(block.y - anchor_y) <= row_tolerance:
                current_row.append(item)
                n = len(current_row)
                anchor_y = block.y if n == 1 else ((anchor_y * (n - 1)) + block.y) / n
                continue
            rows.append(current_row)
            current_row = [item]
            anchor_y = block.y

        if current_row:
            rows.append(current_row)

        offset_y = 0.0
        for row in rows:
            row_top = min(block.y for block, _ in row)
            row_bottom = max(block.y + block.h for block, _ in row)
            row_height = max(1.0, row_bottom - row_top)
            row_needed = row_height

            for block, text in row:
                needed = self._estimate_needed_height(block, text, page_scale)
                row_needed = max(row_needed, needed)

            for block, _ in row:
                block.render_y = block.y + offset_y
                block.render_h = max(block.h, row_needed)

            offset_y += max(0.0, row_needed - row_height)

    def _estimate_needed_height(self, block: TextBlock, text: str, page_scale: float) -> float:
        font = self._font_bold if block.bold else self._font_regular
        size = self._source_font_size(block.size_hint, block.h) * page_scale
        lines = self._wrap_text(text, font, size, max(1.0, block.w))
        return max(size, len(lines) * size * 1.15)

    @staticmethod
    def _source_font_size(size_hint: Optional[float], fallback_h: float) -> float:
        if isinstance(size_hint, (int, float)) and size_hint > 0:
            return max(3.2, float(size_hint))
        return max(4.0, min(11.0, fallback_h * 0.78))

    def _draw_text_preserve_size(
        self,
        page: fitz.Page,
        rect: fitz.Rect,
        text: str,
        bold: bool,
        align: str,
        size_hint: Optional[float],
        page_scale: float,
    ) -> None:
        font_path = self._font_bold_path if bold else self._font_regular_path
        font_name = self.FONT_NAME_BOLD if bold else self.FONT_NAME_REGULAR
        font = self._font_bold if bold else self._font_regular
        font_size = self._source_font_size(size_hint, rect.height) * page_scale
        line_spacing = 1.15

        lines = self._wrap_text(text, font, font_size, rect.width)
        line_height = font_size * line_spacing
        max_lines = max(1, int(rect.height // line_height))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = self._ellipsize(lines[-1], font, font_size, rect.width)

        y_baseline = rect.y0 + font_size
        for line in lines:
            if y_baseline > rect.y1 + 0.1:
                break
            line_width = font.text_length(line, fontsize=font_size)
            if align == "center":
                x = rect.x0 + max(0.0, (rect.width - line_width) / 2.0)
            elif align == "right":
                x = rect.x0 + max(0.0, rect.width - line_width)
            else:
                x = rect.x0
            page.insert_text(
                point=fitz.Point(x, y_baseline),
                text=line,
                fontsize=font_size,
                fontname=font_name,
                fontfile=str(font_path),
                color=(0, 0, 0),
                overlay=True,
            )
            y_baseline += line_height

    @staticmethod
    def _wrap_text(text: str, font: fitz.Font, font_size: float, max_width: float) -> list[str]:
        if max_width <= 1:
            return [text]

        lines: list[str] = []
        for paragraph in text.splitlines() or [""]:
            paragraph = paragraph.strip()
            if not paragraph:
                lines.append("")
                continue

            words = paragraph.split()
            current = ""
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if font.text_length(candidate, fontsize=font_size) <= max_width:
                    current = candidate
                    continue

                if current:
                    lines.append(current)
                else:
                    # Keep long token as-is; fitting loop will reduce font first.
                    current = word
                    continue

                current = word
            if current:
                lines.append(current)
        return lines or [text]

    @staticmethod
    def _ellipsize(text: str, font: fitz.Font, font_size: float, max_width: float) -> str:
        if font.text_length(text, fontsize=font_size) <= max_width:
            return text

        dots = "..."
        if font.text_length(dots, fontsize=font_size) > max_width:
            return ""

        out = text
        while out and font.text_length(out + dots, fontsize=font_size) > max_width:
            out = out[:-1]
        return out + dots

    @staticmethod
    def _sanitize_text_for_pdf(text: str) -> str:
        replacements = {
            "\u3000": " ",
            "\u00a0": " ",
            "\uff1a": ":",
            "\uff1b": ";",
            "\uff0c": ",",
            "\uff0e": ".",
            "\uff08": "(",
            "\uff09": ")",
        }
        cleaned = "".join(replacements.get(ch, ch) for ch in text)
        cleaned = "".join(ch for ch in cleaned if ch == "\n" or ch == "\t" or ord(ch) >= 32)
        return cleaned.strip()

    @staticmethod
    def _resolve_font_paths() -> tuple[Path, Path]:
        candidate_pairs = [
            (
                Path(__file__).resolve().parent.parent / "assets" / "fonts" / "OpenSans-Regular.ttf",
                Path(__file__).resolve().parent.parent / "assets" / "fonts" / "OpenSans-Bold.ttf",
            ),
            (
                Path(__file__).resolve().parent.parent / "assets" / "fonts" / "OpenSans-Regular.ttf",
                Path(__file__).resolve().parent.parent / "assets" / "fonts" / "OpenSans-SemiBold.ttf",
            ),
            (
                Path("/usr/share/fonts/truetype/open-sans/OpenSans-Regular.ttf"),
                Path("/usr/share/fonts/truetype/open-sans/OpenSans-Bold.ttf"),
            ),
            (
                Path("/usr/share/fonts/truetype/opensans/OpenSans-Regular.ttf"),
                Path("/usr/share/fonts/truetype/opensans/OpenSans-Bold.ttf"),
            ),
            (
                Path(__file__).resolve().parent.parent / "assets" / "fonts" / "DejaVuSans.ttf",
                Path(__file__).resolve().parent.parent
                / "assets"
                / "fonts"
                / "DejaVuSans-Bold.ttf",
            ),
            (
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ),
            (
                Path("/usr/local/share/fonts/DejaVuSans.ttf"),
                Path("/usr/local/share/fonts/DejaVuSans-Bold.ttf"),
            ),
        ]

        for regular, bold in candidate_pairs:
            if regular.exists() and bold.exists():
                logger.info("Using font pair: %s | %s", regular.name, bold.name)
                return regular, bold

        raise FileNotFoundError(
            "TTF fonts not found. Place OpenSans-Regular.ttf and OpenSans-Bold.ttf "
            "in telegrambot/assets/fonts/ (or DejaVu fallback fonts)."
        )
