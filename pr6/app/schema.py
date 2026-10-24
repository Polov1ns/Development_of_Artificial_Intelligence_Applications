"""Контракт відповіді моделі: що саме вона має повернути і як це перевіряється.

Схема — домовленість між моделлю та рештою застосунку:
* answer — текст відповіді для клієнта;
* sources — перелік номерів джерел (наприклад [1, 2]), на які спирається відповідь;
* found — чи знайдено відповідь у наданих фрагментах.
"""

import json
from pydantic import BaseModel, Field


class AssistantResponse(BaseModel):
    """Схема структурованої відповіді моделі."""

    answer: str = Field(
        description="Текст відповіді для клієнта, побудований суворо на основі наданих фрагментів, або повідомлення про відсутність інформації"
    )
    sources: list[int] = Field(
        default_factory=list,
        description="Перелік номерів використаних джерел (наприклад [1, 2]), на які спирається відповідь"
    )
    found: bool = Field(
        description="true, якщо відповідь є в наданих фрагментах; false, якщо в базі знань немає інформації для відповіді"
    )


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі."""
    return AssistantResponse.model_json_schema()


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути валідовані дані.

    Очищає markdown-розмітку (```json ... ```) та перевіряє відповідність схемі.
    Підіймає ValueError у разі невалідності.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Відповідь моделі не є валідним JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Очікувався JSON-обʼєкт (словник), отримано інший тип")

    try:
        validated = AssistantResponse(**data)
        return validated.model_dump()
    except Exception as exc:
        raise ValueError(f"Відповідь не відповідає схемі: {exc}") from exc
