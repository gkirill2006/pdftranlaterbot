from dataclasses import dataclass
from pathlib import Path
import os


def _read_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip("\"'")


def _read_int(name: str, default: int) -> int:
    raw = _read_env(name, "")
    if not raw:
        return default
    return int(raw)


def _read_float(name: str, default: float) -> float:
    raw = _read_env(name, "")
    if not raw:
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    openai_api_key: str
    tmp_dir: Path
    output_dir: Path
    render_dpi: int
    openai_vision_model: str
    openai_text_model: str
    openai_timeout_sec: float
    openai_retries: int
    page_parallelism: int
    translate_parallelism: int


def load_settings() -> Settings:
    base_dir = Path(__file__).resolve().parent
    tmp_dir = Path(_read_env("PDF_TMP_DIR", str(base_dir / "data" / "tmp"))).resolve()
    output_dir = Path(_read_env("PDF_OUTPUT_DIR", str(base_dir / "data" / "out"))).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        telegram_bot_token=_read_env("TELEGRAM_BOT_TOKEN"),
        openai_api_key=_read_env("OPENAI_API_KEY"),
        tmp_dir=tmp_dir,
        output_dir=output_dir,
        render_dpi=_read_int("PDF_RENDER_DPI", 200),
        openai_vision_model=_read_env("OPENAI_VISION_MODEL", "gpt-4.1-mini"),
        openai_text_model=_read_env("OPENAI_TEXT_MODEL", "gpt-4.1-mini"),
        openai_timeout_sec=_read_float("OPENAI_TIMEOUT_SEC", 90.0),
        openai_retries=_read_int("OPENAI_RETRIES", 3),
        page_parallelism=_read_int("PDF_PAGE_PARALLELISM", 2),
        translate_parallelism=_read_int("PDF_TRANSLATE_PARALLELISM", 5),
    )

    if not settings.telegram_bot_token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment")
    if not settings.openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment")

    return settings
