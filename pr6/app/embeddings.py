"""Модуль ембедінгів: єдине місце застосунку, яке знає, яка модель
перетворює текст на вектор і як її викликати.

Індекс (`app/index.py`) і веб-рівень (`app/main.py`) отримують звідси
готові вектори й не знають, локальна це модель чи API. Замінивши модель
тут, ви не змінюєте решту коду — але маєте перебудувати індекс: вектори
різних моделей непорівнянні.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яку модель узяти й чому саме її: мови, розмірність, довжина входу,
  розмір, швидкість на CPU;
* чи потрібні моделі префікси для запиту й для тексту (у деяких моделей
  запит і документ кодуються по-різному — дивіться картку моделі);
* чи нормалізувати вектори і де — тут чи в індексі;
* як кодувати кілька сотень фрагментів: по одному чи пакетами.

Назва моделі читається з `.env`; значення за замовчуванням — відправна
точка, а не рекомендація.
"""

import os

import numpy as np
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

load_dotenv()

# Назва моделі ембедінгів. Для локальної моделі — ідентифікатор на
# Hugging Face Hub; для моделі через API — назва в провайдера.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Повернути готову до роботи модель.

    Локальна модель завантажується один раз (singleton) та кешується у памʼяті.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Повертає масив розміру (кількість текстів × розмірність моделі).
    Для моделей родини e5 додається обовʼязковий префікс 'passage: '.
    Вектори нормалізуються (L2) для швидкого скалярного добутку.
    """
    model = get_model()
    is_e5 = "e5" in MODEL_NAME.lower()
    prefixed_texts = [f"passage: {t}" if is_e5 and not t.startswith("passage: ") else t for t in texts]

    embeddings = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на вектор тієї самої розмірності.

    Для моделей родини e5 додається обовʼязковий префікс 'query: '.
    Вектор нормалізується (L2).
    """
    model = get_model()
    is_e5 = "e5" in MODEL_NAME.lower()
    prefixed_text = f"query: {text}" if is_e5 and not text.startswith("query: ") else text

    embedding = model.encode(
        prefixed_text,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embedding, dtype=np.float32)
