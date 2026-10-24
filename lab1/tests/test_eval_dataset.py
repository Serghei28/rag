import pytest
from pydantic import ValidationError

from rag.config import PROJECT_ROOT
from rag.evaluation.dataset import EvalDataset, load_dataset, unknown_titles


def q(**kw):
    base = {
        "id": "q1",
        "question": "Вопрос?",
        "type": "factoid",
        "expected_answer": "Ответ",
        "relevant_titles": ["Молдова"],
    }
    return {**base, **kw}


def test_repository_dataset_is_valid():
    dataset = load_dataset(PROJECT_ROOT / "data" / "eval" / "questions.yaml")
    assert dataset.questions


@pytest.mark.parametrize(
    "bad",
    [
        q(type="no_answer"),  # у no_answer не должно быть статей
        q(relevant_titles=[]),  # у остальных — должны быть
        q(type="multi_doc"),  # multi_doc — минимум две статьи
        q(type="unknown"),
        q(question="  "),
    ],
)
def test_invalid_questions_are_rejected(bad):
    with pytest.raises(ValidationError):
        EvalDataset.model_validate({"questions": [bad]})


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError):
        EvalDataset.model_validate({"questions": [q(), q()]})


def test_unknown_titles():
    ds = EvalDataset.model_validate({"questions": [q(relevant_titles=["Молдова", "Молдовa"])]})
    assert unknown_titles(ds, {"Молдова"}) == {"q1": ["Молдовa"]}  # латинская «a» в конце
