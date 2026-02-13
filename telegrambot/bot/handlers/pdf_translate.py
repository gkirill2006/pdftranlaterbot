import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from aiogram import Bot, F, Router
from aiogram.types import FSInputFile, Message

from config import Settings
from services.openai_client import OpenAIClient
from services.pdf_pipeline import PDFPipeline

logger = logging.getLogger(__name__)


def _safe_name(name: Optional[str]) -> str:
    if not name:
        return "input.pdf"
    return Path(name).name.replace("/", "_")


def create_pdf_router(settings: Settings) -> Router:
    router = Router(name="pdf_translate")
    openai_client = OpenAIClient(settings)
    pipeline = PDFPipeline(settings, openai_client)

    @router.message(F.document)
    async def handle_pdf_document(message: Message, bot: Bot) -> None:
        document = message.document
        if not document:
            return

        file_name = _safe_name(document.file_name)
        is_pdf = (
            file_name.lower().endswith(".pdf")
            or (document.mime_type or "").lower() == "application/pdf"
        )
        if not is_pdf:
            await message.answer("Нужен PDF-файл. Отправьте документ с расширением .pdf.")
            return

        input_name = f"{message.chat.id}_{message.message_id}_{uuid4().hex}_{file_name}"
        input_path = settings.tmp_dir / input_name
        output_path: Optional[Path] = None

        status_message = await message.answer("Принял. Перевожу…")

        try:
            tg_file = await bot.get_file(document.file_id)
            await bot.download(tg_file, destination=input_path)
            logger.info("Downloaded PDF: %s", input_path)

            async def progress_cb(done: int, total: int) -> None:
                try:
                    await status_message.edit_text(f"Перевожу… Страница {done}/{total}")
                except Exception:
                    logger.debug("Progress update skipped", exc_info=True)

            output_path = await pipeline.process_pdf(
                input_pdf_path=input_path,
                progress_cb=progress_cb,
            )

            await message.answer_document(
                FSInputFile(str(output_path)),
                caption="Готово. Переведённый PDF во вложении.",
            )
            await status_message.edit_text("Перевод завершён.")
        except Exception as exc:
            logger.exception("PDF translation failed: %s", exc)
            await status_message.edit_text(
                "Не удалось обработать PDF. Попробуйте другой файл или повторите позже."
            )
        finally:
            if input_path.exists():
                input_path.unlink(missing_ok=True)
            if output_path and output_path.exists():
                output_path.unlink(missing_ok=True)

    return router
