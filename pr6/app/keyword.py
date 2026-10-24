"""Пошук за ключовими словами — точка порівняння для семантичного.

Той самий набір фрагментів, той самий формат влучень (`Hit`), ті самі
фільтри — інший спосіб ранжувати. Без цього модуля не буде з чим
порівнювати семантичний пошук, а без порівняння — не буде відповіді на
питання, чи він узагалі потрібен для цієї колекції.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення:

* як розбивати текст на слова: що робити з регістром, розділовими
  знаками, апострофом у «звʼязок», числами й артикулами на кшталт
  `OR-X2-BLK`;
* чи зводити слова до основи — і чим, якщо так; без цього «повернути»
  й «повернення» для пошуку різні слова;
* чим ранжувати: кількістю збігів, TF-IDF, BM25 (пакет `rank_bm25`
  вже в залежностях);
* у яких одиницях оцінка й чи можна її порівнювати з оцінкою
  семантичного пошуку.
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit, matches_filter


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами на основі BM25Okapi."""

    chunks: list[Chunk]
    bm25: BM25Okapi | None = None
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Враховує українські літери, цифри, апострофи та дефіси в кодах/артикулах.
    """
    clean = text.replace("’", "'").replace("ʼ", "'").lower()
    base_tokens = re.findall(r"[a-za-яіїєґ0-9]+(?:['\-][a-za-яіїєґ0-9]+)*", clean)
    tokens: list[str] = []
    for token in base_tokens:
        tokens.append(token)
        if "-" in token:
            for part in token.split("-"):
                if part and part != token:
                    tokens.append(part)
    return tokens


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів за алгоритмом BM25."""
    if not chunks:
        return KeywordIndex(chunks=[], bm25=None)

    tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    return KeywordIndex(
        chunks=chunks,
        bm25=bm25,
        extra={"num_chunks": len(chunks)},
    )


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти top_k фрагментів за словами запиту за алгоритмом BM25."""
    if not index.chunks or index.bm25 is None:
        return []

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    if filters:
        candidate_indices = [i for i, chunk in enumerate(index.chunks) if matches_filter(chunk, filters)]
    else:
        candidate_indices = list(range(len(index.chunks)))

    if not candidate_indices:
        return []

    all_scores = index.bm25.get_scores(q_tokens)

    hits: list[Hit] = []
    for idx in candidate_indices:
        score = float(all_scores[idx])
        if score > 0.0:
            hits.append(Hit(chunk=index.chunks[idx], score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
