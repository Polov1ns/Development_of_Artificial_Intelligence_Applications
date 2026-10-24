"""Відбір фрагментів для моделі та збирання контексту.

Між пошуком із ПР5 і мовною моделлю стоїть шар, якого в ПР5 не було: він
вирішує, *що саме* модель побачить. Пошук повертає влучення з оцінками —
стільки, скільки попросили; модель потребує тексту — обмеженого за
розміром, упорядкованого, з позначками джерел, на які вона зможе
послатися. Цей шар — тут.
"""

import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

from . import embeddings
from .documents import Chunk
from .index import Hit, SearchIndex, SIMILARITY_THRESHOLD, search
from .keyword import KeywordIndex

load_dotenv()

# Скільки фрагментів іде в контекст моделі й скільки токенів на них є.
CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "8"))
MAX_CHUNKS_PER_DOC = int(os.getenv("MAX_CHUNKS_PER_DOC", "2"))


@dataclass
class Source:
    """Джерело, показане моделі й користувачеві.

    `ref` — номер, під яким фрагмент стоїть у контексті й на який модель
    посилається у відповіді. `chunk` — сам фрагмент із метаданими;
    `score` — оцінка пошуку, за якою його відібрано.
    """

    ref: int
    chunk: Chunk
    score: float


def estimate_tokens(text: str) -> int:
    """Приблизна оцінка кількості токенів для тексту (українська/англійська мови)."""
    # Для кирилиці зазвичай 1 токен відповідає приблизно 2.5-3.5 символам
    return max(1, int(len(text) / 3.2))


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти, які варто показати моделі.

    Забезпечує:
    1. Жорсткий запобіжник: внутрішні документи (audience == 'персонал')
       ніколи не потрапляють до моделі клієнта, незалежно від параметрів запиту.
    2. Архівні документи: відсікаються за замовчуванням, окрім явного запиту
       клієнта на 2025 рік або старі редакції.
    3. Семантичний пошук із порогом схожості (threshold).
    4. Згортання/обмеження кількості фрагментів з одного документа (до MAX_CHUNKS_PER_DOC),
       щоб у контекст вмістилися різні релевантні документи.
    5. Обмеження результуючого списку до CONTEXT_CHUNKS найкращих влучень.
    """
    if not query.strip() or len(index.chunks) == 0:
        return []

    # Копіюємо фільтри користувача
    user_filters = dict(filters) if filters else {}

    # Захист: користувач не може зняти або підмінити обмеження на audience
    # Внутрішні документи суворо заблоковані на рівні коду
    user_filters.pop("audience", None)

    # Чи є в запиті явне посилання на архівні редакції / минулі роки?
    q_lower = query.lower()
    explicit_archive = any(kw in q_lower for kw in ["2025", "архів", "минул", "старих правил", "старій редакці"])

    # Отримуємо вектор запиту через embeddings.embed_query
    query_vector = embeddings.embed_query(query)

    # Шукаємо ширше (SEARCH_TOP_K), щоб після фільтрації мати якісні кандидати
    search_k = max(SEARCH_TOP_K, CONTEXT_CHUNKS * 2)
    raw_hits = search(
        index=index,
        query_vector=query_vector,
        top_k=search_k,
        filters=user_filters,
        threshold=SIMILARITY_THRESHOLD,
    )

    filtered_hits: list[Hit] = []
    doc_chunk_count: dict[str, int] = {}

    for hit in raw_hits:
        meta = hit.chunk.metadata or {}

        # 1. Запобіжник для внутрішніх документів
        if meta.get("audience", "").strip().lower() == "персонал":
            continue

        # 2. Обробка архівних документів
        if not explicit_archive and meta.get("status", "").strip().lower() == "архів":
            continue

        # 3. Обмеження кількості фрагментів з одного документа
        source_doc = hit.chunk.source
        current_count = doc_chunk_count.get(source_doc, 0)
        if current_count >= MAX_CHUNKS_PER_DOC:
            continue

        doc_chunk_count[source_doc] = current_count + 1
        filtered_hits.append(hit)

        if len(filtered_hits) >= CONTEXT_CHUNKS:
            break

    # Сортуємо за спаданням оцінки: найкраще влучення завжди на початку
    filtered_hits.sort(key=lambda h: h.score, reverse=True)
    return filtered_hits


def build_context(hits: list[Hit], budget: int = CONTEXT_BUDGET) -> tuple[str, list[Source]]:
    """Зібрати з влучень текст контексту для моделі та перелік джерел.

    Кожен фрагмент містить чітку позначку з номером [ref], назвою документа,
    розділом та датою редакції:
        [1] Доставка › Вартість (2026-08-01):
        «Текст фрагмента...»

    Найкраще влучення ставиться на початок контексту.
    Контекст обмежується бюджетом токенів.
    """
    if not hits:
        return "", []

    sources: list[Source] = []
    context_blocks: list[str] = []
    used_tokens = 0

    for i, hit in enumerate(hits, start=1):
        meta = hit.chunk.metadata or {}
        title = meta.get("title") or hit.chunk.source
        heading = meta.get("heading", "")
        updated = meta.get("updated", "")

        header_parts = [title]
        if heading:
            header_parts.append(f"› {heading}")
        header_line = " ".join(header_parts)
        if updated:
            header_line += f" ({updated})"

        block = f"[{i}] {header_line}:\n«{hit.chunk.text.strip()}»"
        block_tokens = estimate_tokens(block)

        # Перевірка бюджету: якщо блок не вміщується, не додаємо його
        if used_tokens + block_tokens > budget and sources:
            break

        used_tokens += block_tokens
        sources.append(Source(ref=i, chunk=hit.chunk, score=hit.score))
        context_blocks.append(block)

    full_context = "\n\n".join(context_blocks)
    return full_context, sources
