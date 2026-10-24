"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих.

Індекс — це вектори всіх фрагментів, самі фрагменти з метаданими й назва
моделі, якою вектори отримано. Він будується окремою командою
(`ingest.py`) і зберігається на диску, а застосунок при старті лише
читає його: перераховувати ембедінги колекції на кожен запуск — марна
трата часу, а на кожен запит — тим паче.

Веб-рівень (`app/main.py`) звертається сюди з вектором запиту й отримує
список влучень із оцінкою схожості. Звідки береться вектор — не справа
індексу; що показувати клієнтові — не його справа теж.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яку міру схожості взяти й що зробити з векторами перед порівнянням;
* як шукати: перебір усіх векторів (для сотень фрагментів — цілком
  доречно) чи бібліотека наближеного пошуку;
* у якому вигляді зберігати індекс на диску і що обовʼязково покласти
  поруч із векторами, щоб потім не переплутати, чиї вони;
* де застосовувати фільтри за метаданими — до ранжування чи після — і що
  станеться з top-k у кожному з варіантів;
* що робити з кількома фрагментами одного документа у видачі;
* чи потрібен поріг схожості й звідки взяти його значення.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

# Скільки результатів повертати за замовчуванням і чи відсікати слабкі.
# Порожній поріг означає «без порога».
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None

KNOWN_FILTER_FIELDS = {
    "category", "product", "audience", "status",
    "title", "heading", "updated", "source", "chunk_index"
}


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті.

    `vectors` — масив (кількість фрагментів × розмірність); рядок i
    відповідає `chunks[i]`. `model_name` — модель, якою отримано вектори:
    без неї індекс, збудований однією моделлю, мовчки шукатиме векторами
    іншої.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів.

    Перевіряє відповідність розмірностей та забезпечує нормалізацію L2.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Кількість фрагментів ({len(chunks)}) не збігається з кількістю векторів ({len(vectors)})"
        )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    norm_vectors = (vectors / norms).astype(np.float32)

    return SearchIndex(
        chunks=chunks,
        vectors=norm_vectors,
        model_name=model_name,
        extra={"num_chunks": len(chunks), "embedding_dim": int(vectors.shape[1]) if len(vectors) > 0 else 0},
    )


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск: вектори в .npy, метадані та фрагменти в .json."""
    path.mkdir(parents=True, exist_ok=True)
    vectors_file = path / "vectors.npy"
    meta_file = path / "index.json"

    np.save(vectors_file, index.vectors)

    data = {
        "model_name": index.model_name,
        "extra": index.extra,
        "chunks": [
            {
                "text": chunk.text,
                "source": chunk.source,
                "metadata": chunk.metadata,
            }
            for chunk in index.chunks
        ],
    }
    meta_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска.

    Підіймає зрозумілий FileNotFoundError, якщо індекс ще не збережено.
    """
    vectors_file = path / "vectors.npy"
    meta_file = path / "index.json"

    if not vectors_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Індекс не знайдено у {path}. Спочатку виконайте 'python ingest.py'.")

    vectors = np.load(vectors_file)
    data = json.loads(meta_file.read_text(encoding="utf-8"))

    chunks = [
        Chunk(
            text=c["text"],
            source=c["source"],
            metadata=c.get("metadata", {}),
        )
        for c in data["chunks"]
    ]

    return SearchIndex(
        chunks=chunks,
        vectors=vectors,
        model_name=data["model_name"],
        extra=data.get("extra", {}),
    )


def matches_filter(chunk: Chunk, filters: dict) -> bool:
    """Перевірити відповідність фрагмента фільтрам."""
    for key, expected in filters.items():
        if key not in KNOWN_FILTER_FIELDS:
            raise ValueError(f"Невідоме поле фільтра: '{key}'. Дозволені поля: {sorted(KNOWN_FILTER_FIELDS)}")
        if expected is None or expected == "":
            continue
        actual = chunk.metadata.get(key, "")
        if str(actual).strip().lower() != str(expected).strip().lower():
            return False
    return True


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти top_k фрагментів, найближчих до вектора запиту."""
    if len(index.chunks) == 0:
        return []

    if query_vector.shape[-1] != index.vectors.shape[-1]:
        raise ValueError(
            f"Розмірність вектора запиту ({query_vector.shape[-1]}) не відповідає індексу ({index.vectors.shape[-1]})."
        )

    q_norm = np.linalg.norm(query_vector)
    q_vec = (query_vector / q_norm).astype(np.float32) if q_norm != 0 else query_vector.astype(np.float32)

    if filters:
        candidate_indices = [i for i, chunk in enumerate(index.chunks) if matches_filter(chunk, filters)]
    else:
        candidate_indices = list(range(len(index.chunks)))

    if not candidate_indices:
        return []

    cand_vectors = index.vectors[candidate_indices]
    scores = cand_vectors @ q_vec

    hits: list[Hit] = []
    for idx_pos, score_val in enumerate(scores):
        score = float(score_val)
        if threshold is not None and score < threshold:
            continue
        original_idx = candidate_indices[idx_pos]
        hits.append(Hit(chunk=index.chunks[original_idx], score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
