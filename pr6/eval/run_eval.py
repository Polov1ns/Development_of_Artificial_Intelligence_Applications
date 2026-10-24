"""Скрипт для прогону набору тестових запитань через RAG-конвеєр та аналізу результатів.

Зберігає повні дані про кожен запит:
- знайдені та передані в контекст фрагменти;
- перевірені джерела та відповідь моделі;
- токени, час пошуку й генерації;
- класифікацію помилок (помилки пошуку vs помилки генерації).
"""

import json
import os
import sys
import time
from pathlib import Path

# Додаємо корінь pr6 до sys.path
PR6_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PR6_ROOT))

from app import index, keyword, rag, retrieval


def evaluate(output_file: str = "eval/results_run1.json", context_chunks: int | None = None) -> dict:
    if context_chunks is not None:
        retrieval.CONTEXT_CHUNKS = context_chunks

    print(f"\n--- Запуск оцінювання (RAG_CONTEXT_CHUNKS={retrieval.CONTEXT_CHUNKS}) ---")

    idx = index.load()
    kw_idx = keyword.build(idx.chunks)

    questions_file = PR6_ROOT / "eval" / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["питання"]
    results = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_retrieval_time = 0.0
    total_generation_time = 0.0

    count_expected_in_context = 0
    count_expected_docs_total = 0
    count_correct_answers = 0
    count_correct_refusals = 0
    count_unwanted_leaks = 0
    count_search_errors = 0
    count_generation_errors = 0

    for q in questions:
        qid = q["id"]
        q_text = q["питання"]
        q_type = q["вид"]
        filters = q.get("filters", {})
        expected_docs = q.get("очікувані_документи", [])
        expected_ans = q.get("очікувана_відповідь", "")
        expect_answer = q.get("очікується_відповідь", True)

        print(f"[{qid:02d}/14] {q_type}: «{q_text[:45]}...»")

        # Виклик конвеєра
        ans: rag.Answer = rag.answer(
            question=q_text,
            index=idx,
            keyword_index=kw_idx,
            filters=filters,
        )

        retrieved_sources = [h.chunk.source for h in ans.retrieved]
        cited_sources = [s.chunk.source for s in ans.sources]
        cited_refs = [s.ref for s in ans.sources]

        # Перевірка очікуваних документів у контексті
        docs_found = []
        if expected_docs:
            count_expected_docs_total += 1
            all_found = all(doc in retrieved_sources for doc in expected_docs)
            if all_found:
                count_expected_in_context += 1
            docs_found = [doc for doc in expected_docs if doc in retrieved_sources]

        # Перевірка на витік внутрішніх або непотрібних архівних документів
        has_leak = False
        for h in ans.retrieved:
            if h.chunk.metadata.get("audience") == "персонал":
                has_leak = True
            if not any(k in q_text.lower() for k in ["2025", "архів", "старих"]):
                if h.chunk.metadata.get("status") == "архів":
                    has_leak = True
        if has_leak:
            count_unwanted_leaks += 1

        # Перевірка коректності відмови / відповіді
        error_type = None
        error_stage = None

        if not expect_answer:
            # Має бути відмова (found == False)
            if not ans.found:
                count_correct_refusals += 1
            else:
                # Помилка: відповіла там, де мала бути відмова
                if any(h.chunk.metadata.get("audience") == "персонал" for h in ans.retrieved):
                    error_stage = "пошук"
                    error_type = "витік внутрішнього документа"
                    count_search_errors += 1
                else:
                    error_stage = "модель"
                    error_type = "відповідь «з голови» або вигадка без підстави"
                    count_generation_errors += 1
        else:
            # Має бути відповідь (found == True)
            if not ans.found:
                # Відмова там, де мала бути відповідь
                if not docs_found:
                    error_stage = "пошук"
                    error_type = "не знайдено потрібний документ"
                    count_search_errors += 1
                else:
                    error_stage = "модель"
                    error_type = "безпідставна відмова при наявному контексті"
                    count_generation_errors += 1
            else:
                # Перевірка, чи всі очікувані документи були в контексті
                missing_docs = [d for d in expected_docs if d not in retrieved_sources]
                if missing_docs:
                    error_stage = "пошук"
                    error_type = f"неповний контекст (відсутній {missing_docs})"
                    count_search_errors += 1
                else:
                    count_correct_answers += 1

        u = ans.usage or {}
        p_tok = u.get("prompt_tokens", 0)
        c_tok = u.get("completion_tokens", 0)
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok

        r_time = ans.elapsed.get("retrieval", 0.0)
        g_time = ans.elapsed.get("generation", 0.0)
        total_retrieval_time += r_time
        total_generation_time += g_time

        item_res = {
            "id": qid,
            "question": q_text,
            "type": q_type,
            "filters": filters,
            "expected_docs": expected_docs,
            "expected_answer_fact": expected_ans,
            "expect_answer": expect_answer,
            "found": ans.found,
            "answer": ans.text,
            "cited_sources": cited_sources,
            "cited_refs": cited_refs,
            "retrieved_sources": retrieved_sources,
            "docs_in_context": docs_found,
            "error_stage": error_stage,
            "error_type": error_type,
            "elapsed": ans.elapsed,
            "usage": ans.usage,
            "context_seen": [
                {
                    "ref": i + 1,
                    "source": h.chunk.source,
                    "title": h.chunk.metadata.get("title", ""),
                    "heading": h.chunk.metadata.get("heading", ""),
                    "score": round(h.score, 4),
                    "text": h.chunk.text,
                }
                for i, h in enumerate(ans.retrieved)
            ],
        }
        results.append(item_res)

    summary = {
        "context_chunks_setting": retrieval.CONTEXT_CHUNKS,
        "total_questions": len(questions),
        "questions_expecting_answer": count_expected_docs_total,
        "expected_doc_in_context": count_expected_in_context,
        "correct_answers": count_correct_answers,
        "correct_refusals": count_correct_refusals,
        "unwanted_leaks": count_unwanted_leaks,
        "search_errors": count_search_errors,
        "generation_errors": count_generation_errors,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_prompt_tokens_per_q": round(total_prompt_tokens / len(questions), 1),
        "avg_completion_tokens_per_q": round(total_completion_tokens / len(questions), 1),
        "avg_retrieval_time": round(total_retrieval_time / len(questions), 3),
        "avg_generation_time": round(total_generation_time / len(questions), 3),
    }

    output_path = PR6_ROOT / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print("\nПідсумки прогону:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Збережено у {output_path}")
    return summary


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "eval/results_run1.json"
    chunks = int(sys.argv[2]) if len(sys.argv) > 2 else None
    evaluate(out, chunks)
