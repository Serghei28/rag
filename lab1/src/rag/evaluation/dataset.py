"""Набор вопросов для оценки качества (data/eval/questions.yaml).

Каждый вопрос размечен вручную: тип, эталонный ответ и статьи, в которых он находится.
По `relevant_titles` считаются метрики поиска (Recall@K, MRR), по `expected_answer` —
качество ответа LLM.
"""

from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

QuestionType = Literal["factoid", "multi_doc", "specific", "context", "no_answer"]

TYPE_LABELS = {
    "factoid": "простой фактологический",
    "multi_doc": "по нескольким документам",
    "specific": "поиск конкретной информации",
    "context": "требует понимания контекста",
    "no_answer": "ответа нет в базе",
}


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    type: QuestionType
    expected_answer: str  # для no_answer: что именно должен сказать RAG
    relevant_titles: list[str] = []  # статьи, где есть ответ (пусто для no_answer)
    # Необязательно: разделы, например "Государственное устройство > Президент"
    relevant_sections: list[str] = []
    notes: str = ""

    @field_validator("question", "expected_answer")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("не может быть пустым")
        return v.strip()

    @model_validator(mode="after")
    def titles_match_type(self) -> "EvalQuestion":
        if self.type == "no_answer" and self.relevant_titles:
            raise ValueError("у вопроса no_answer не должно быть relevant_titles")
        if self.type != "no_answer" and not self.relevant_titles:
            raise ValueError("укажите relevant_titles — статьи, где есть ответ")
        if self.type == "multi_doc" and len(self.relevant_titles) < 2:
            raise ValueError("у вопроса multi_doc должно быть ≥ 2 relevant_titles")
        return self


class EvalDataset(BaseModel):
    questions: list[EvalQuestion]

    @model_validator(mode="after")
    def unique_ids(self) -> "EvalDataset":
        dup = [i for i, n in Counter(q.id for q in self.questions).items() if n > 1]
        if dup:
            raise ValueError(f"повторяются id: {dup}")
        return self

    def by_type(self) -> Counter:
        return Counter(q.type for q in self.questions)


def load_dataset(path: Path) -> EvalDataset:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return EvalDataset.model_validate(raw)


def unknown_titles(dataset: EvalDataset, known: set[str]) -> dict[str, list[str]]:
    """id вопроса → статьи из relevant_titles, которых нет в корпусе (опечатки)."""
    result = {}
    for q in dataset.questions:
        missing = [t for t in q.relevant_titles if t not in known]
        if missing:
            result[q.id] = missing
    return result
