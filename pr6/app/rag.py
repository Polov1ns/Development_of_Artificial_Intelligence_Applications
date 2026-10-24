"""Конвеєр відповіді за базою знань: від питання до відповіді з джерелами.

Поєднує:
1. Пошук і відбір фрагментів (`retrieval.retrieve`).
2. Збирання контексту в межах бюджету (`retrieval.build_context`).
3. Контрольовану відповідь без виклику моделі, якщо пошук не знайшов нічого вище порогу.
4. Звернення до моделі (`llm.ask`).
5. Програмну верифікацію посилань та статусу «знайдено»:
   - фільтрація неіснуючих посилань;
   - захист від галюцинацій («знайдено: так» без дійсних джерел перетворюється на «знайдено: ні»);
6. Зведення результату `Answer` для веб-рівня.
"""

import time
from dataclasses import dataclass, field

from . import llm, retrieval
from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source


@dataclass
class Answer:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `text` — відповідь для клієнта.
    `found` — чи відповідь спирається на базу знань (після програмної перевірки).
    `sources` — перевірені джерела, на які модель послалася.
    `retrieved` — усі фрагменти, передані в контекст моделі (для діагностики).
    `model` — назва використаної моделі.
    `elapsed` — час окремо за етапами (retrieval, generation).
    `usage` — кількість використаних токенів.
    """

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    """Відповісти на питання за базою знань через RAG-конвеєр."""
    # Етап 1: Пошук та відбір фрагментів
    t_ret_start = time.perf_counter()
    hits = retrieval.retrieve(
        query=question,
        index=index,
        keyword_index=keyword_index,
        filters=filters,
    )
    context, sources = retrieval.build_context(hits)
    elapsed_retrieval = time.perf_counter() - t_ret_start

    # Етап 2: Якщо пошук не повернув жодного релевантного фрагмента (нижче порогу) —
    # контрольована миттєва відмова без витрати токенів і без виклику LLM
    if not hits or not context.strip():
        return Answer(
            text="У базі знань магазину немає інформації для відповіді на це запитання.",
            found=False,
            sources=[],
            retrieved=[],
            model=None,
            elapsed={"retrieval": round(elapsed_retrieval, 4), "generation": 0.0},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # Етап 3: Виклик мовної моделі
    llm_output = llm.ask(question=question, context=context)
    result_data = llm_output["result"]
    elapsed_generation = llm_output["elapsed"]

    raw_text = result_data.get("answer", "").strip()
    raw_sources = result_data.get("sources", [])
    raw_found = bool(result_data.get("found", False))

    # Етап 4: Програмна перевірка посилань і поля «знайдено»
    sources_by_ref = {s.ref: s for s in sources}
    valid_cited_sources: list[Source] = []

    # Перевіряємо чи кожен номер джерела існує серед показаних
    for ref in raw_sources:
        if isinstance(ref, int) and ref in sources_by_ref:
            if sources_by_ref[ref] not in valid_cited_sources:
                valid_cited_sources.append(sources_by_ref[ref])

    final_found = raw_found
    final_text = raw_text

    # Перевірка надійності:
    # 1. Якщо модель стверджує "found: true", але не вказала жодного валідного джерела —
    # це підозріла відповідь (галюцинація або відповідь «з голови»). Не довіряємо їй.
    if raw_found and not valid_cited_sources:
        final_found = False
        valid_cited_sources = []
        final_text = "У наданих документах бази знань не знайдено підтвердження для відповіді на це питання."

    # 2. Якщо модель вказала "found: false", джерела скидаються
    if not raw_found:
        final_found = False
        valid_cited_sources = []

    return Answer(
        text=final_text,
        found=final_found,
        sources=valid_cited_sources,
        retrieved=hits,
        model=llm_output.get("model"),
        elapsed={"retrieval": round(elapsed_retrieval, 4), "generation": round(elapsed_generation, 3)},
        usage=llm_output.get("usage"),
    )
