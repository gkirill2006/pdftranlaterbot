import asyncio
import base64
import json
import logging
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from config import Settings

logger = logging.getLogger(__name__)

LAYOUT_SYSTEM_PROMPT = "You are a document layout analyzer. Return valid JSON only."
LAYOUT_USER_PROMPT = (
    "Analyze this page image. Extract text blocks with bounding boxes in pixels relative "
    "to the image (origin top-left). Return blocks with id, bbox(x,y,w,h), style hints "
    "(bold/regular, approximate font size), and exact English text. Ignore non-text graphics.\n"
    "Return strict JSON with this shape:\n"
    "{"
    "\"blocks\": ["
    "{\"id\": \"b1\", \"type\": \"text\", \"bbox\": {\"x\": 0, \"y\": 0, \"w\": 0, \"h\": 0}, "
    "\"style\": {\"bold\": false, \"size_hint\": 11, \"align\": \"left\"}, \"text_en\": \"...\"}"
    "]"
    "}"
)

TRANSLATE_SYSTEM_PROMPT = "You are a professional technical translator EN→RU."
TRANSLATE_USER_PROMPT = (
    "Translate English text to Russian preserving technical meaning and formatting. "
    "Do not change numbers, units, SKUs, model codes, URLs, emails, and punctuation structure. "
    "Keep non-English fragments unchanged. Keep capitalization where sensible. "
    "Prefer concise wording close to source length; avoid unnecessary expansion. "
    "Return only translated text."
)


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def extract_layout(self, image_path: Path) -> dict[str, Any]:
        image_bytes = image_path.read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{image_base64}"

        response_text = await self._request_text(
            model=self.settings.openai_vision_model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": LAYOUT_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": LAYOUT_USER_PROMPT},
                        {"type": "input_image", "image_url": image_url},
                    ],
                },
            ],
        )
        raw = self._extract_json_payload(response_text)
        parsed = json.loads(raw)

        blocks = parsed.get("blocks", [])
        if not isinstance(blocks, list):
            return {"blocks": []}
        return {"blocks": blocks}

    async def translate_text(self, text_en: str) -> str:
        source = text_en.strip()
        if not source:
            return source

        response_text = await self._request_text(
            model=self.settings.openai_text_model,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": TRANSLATE_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{TRANSLATE_USER_PROMPT}\n\nText:\n{source}",
                        }
                    ],
                },
            ],
        )
        return response_text.strip()

    async def _request_text(self, model: str, messages: list[dict[str, Any]]) -> str:
        last_error: Exception | None = None

        for attempt in range(1, self.settings.openai_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.responses.create(
                        model=model,
                        temperature=0,
                        input=messages,
                    ),
                    timeout=self.settings.openai_timeout_sec,
                )
                text = self._response_to_text(response)
                if not text.strip():
                    raise RuntimeError("OpenAI returned empty response text")
                return text
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "OpenAI request failed (attempt %s/%s): %s",
                    attempt,
                    self.settings.openai_retries,
                    exc,
                )
                if attempt < self.settings.openai_retries:
                    await asyncio.sleep(1.5 * attempt)

        raise RuntimeError(f"OpenAI request failed after retries: {last_error}") from last_error

    @staticmethod
    def _response_to_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text:
            return text

        output = getattr(response, "output", None)
        if not output:
            return ""
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None) or []
            for c in content:
                c_type = getattr(c, "type", "")
                if c_type in {"output_text", "text"}:
                    value = getattr(c, "text", "")
                    if value:
                        chunks.append(value)
        return "\n".join(chunks)

    @staticmethod
    def _extract_json_payload(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not find JSON object in layout response")
        return cleaned[start : end + 1]
