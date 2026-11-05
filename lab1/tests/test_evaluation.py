import json

import pytest

from rag.evaluation import metrics as m
from rag.evaluation.dataset import EvalDataset
from rag.evaluation.judge import Verdict, parse_verdict
from rag.evaluation.runner import run_evaluation
from rag.generation import Generator
from tests.test_generation import FakeLLM
from tests.test_retrieval import retriever  # noqa: F401 — фикстура

# ---------------------------------------------------------------- metrics ---

RANKED = ["Молдова", "Парламент", "Молдова", "Гимн", "Кишинёв"]


def test_hit_and_mrr():
    assert m.hit_at_k(RANKED, {"Парламент"}, 1) == 0.0
    assert m.hit_at_k(RANKED, {"Парламент"}, 2) == 1.0
    assert m.reciprocal_rank(RANKED, {"Гимн"}) == 0.25
    assert m.reciprocal_rank(RANKED, {"Нет такой"}) == 0.0


def test_recall_counts_distinct_articles():
    relevant = {"Молдова", "Кишинёв", "Гимн"}
    assert m.recall_at_k(RANKED, relevant, 3) == pytest.approx(1 / 3)  # «Молдова» дважды
    assert m.recall_at_k(RANKED, relevant, 5) == 1.0


def test_precision_divides_by_k():
    assert m.precision_at_k(RANKED, {"Молдова"}, 5) == pytest.approx(2 / 5)
    assert m.precision_at_k(["Молдова"], {"Молдова"}, 5) == pytest.approx(1 / 5)  # недобор


def test_retrieval_metrics_keys_and_mean():
    out = m.retrieval_metrics(RANKED, {"Парламент"}, [1, 5])
    assert set(out) == {
        "mrr",
        "hit@1",
        "recall@1",
        "precision@1",
        "hit@5",
        "recall@5",
        "precision@5",
    }
    assert m.mean([1.0, 0.0, None]) == 0.5 and m.mean([None]) is None


# ------------------------------------------------------------------ judge ---


@pytest.mark.parametrize(
    ("raw", "score"),
    [
        ('{"score": 2, "reason": "ok"}', 1.0),
        ('{"score": 1}', 0.5),
        ('{"score": "0"}', 0.0),
        ('{"score": 5}', None),
        ("не json", None),
        ('{"reason": "нет оценки"}', None),
    ],
)
def test_parse_verdict(raw, score):
    assert parse_verdict(raw).score == score


# ----------------------------------------------------------------- runner ---


class FakeJudge:
    model = "fake-judge"
    prompt_version = "fake"

    def __init__(self):
        self.calls: list[str] = []

    def correctness(self, question, expected, answer):
        self.calls.append("correctness")
        return Verdict(1.0 if "Кишинёв" in answer else 0.0, "сравнение с эталоном")

    def faithfulness(self, question, answer, sources):
        self.calls.append("faithfulness")
        return Verdict(0.5, "есть неподтверждённая деталь")


DATASET = EvalDataset.model_validate(
    {
        "questions": [
            {
                "id": "q1",
                "question": "столица молдовы",
                "type": "factoid",
                "expected_answer": "Кишинёв",
                "relevant_titles": ["Кишинёв"],
            },
            {
                "id": "q2",
                "question": "рецепт борща со сметаной",
                "type": "no_answer",
                "expected_answer": "нет информации",
            },
        ]
    }
)


def run(tmp_path, retriever, llm_text):  # noqa: F811
    retriever.config.paths.experiments_dir = tmp_path
    judge = FakeJudge()
    generator = Generator(retriever.config, retriever=retriever, llm=FakeLLM(llm_text))
    result = run_evaluation(
        retriever.config,
        name="test",
        dataset=DATASET,
        retriever=retriever,
        generator=generator,
        judge=judge,
    )
    return result, judge


def test_runner_scores_answer_and_refusal(tmp_path, retriever):  # noqa: F811
    result, judge = run(tmp_path, retriever, "Столица — Кишинёв [1].")
    q1, q2 = result.results

    assert q1.retrieval_metrics["hit@1"] == 1.0 and q1.first_relevant_rank == 1
    assert q1.status == "answered" and q1.correctness == 1.0 and q1.faithfulness == 0.5
    assert q2.retrieval_empty and q2.status == "no_context" and q2.correctness == 1.0
    assert judge.calls == ["correctness", "faithfulness"]  # судья не нужен для отказа

    s = result.summary
    assert s["overall"]["retrieval"]["mrr"] == 1.0
    assert s["overall"]["no_answer_rejected_by_retrieval"] == 1.0
    assert s["by_type"]["factoid"]["generation"]["correctness"] == 1.0


def test_runner_false_refusal_without_judge_call(tmp_path, retriever):  # noqa: F811
    result, judge = run(tmp_path, retriever, "Президент — Санду.")  # нет ссылок → unsupported
    q1 = result.results[0]
    assert q1.status == "unsupported" and q1.correctness == 0.0 and q1.faithfulness is None
    assert judge.calls == []
    assert result.summary["overall"]["generation"]["false_refusal_rate"] == 1.0


def test_runner_writes_files(tmp_path, retriever):  # noqa: F811
    result, _ = run(tmp_path, retriever, "Столица — Кишинёв [1].")
    out = tmp_path / "runs" / "test"
    assert result.path == out
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["q1", "q2"]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["params"]["judge"] == "fake-judge (fake)"
    assert "## Поиск" in (out / "report.md").read_text(encoding="utf-8")


def test_retrieval_only_mode(tmp_path, retriever):  # noqa: F811
    retriever.config.paths.experiments_dir = tmp_path
    result = run_evaluation(
        retriever.config, name="r", dataset=DATASET, retriever=retriever, generate=False
    )
    assert result.results[0].status is None and "generation" not in result.summary["overall"]
    assert result.params["llm"] is None


# ------------------------------------------------------------ judge check ---


def test_check_judge_agreement_and_disagreements():
    from rag.evaluation.judge_check import check_judge

    class Judge:
        prompt_version = "vX"

        def correctness(self, question, expected, answer):
            return Verdict(1.0 if "Кишинёв" in answer else 0.5, "причина")

        def faithfulness(self, question, answer, sources):
            return Verdict(1.0, "")

    results = {
        "a": {
            "status": "answered",
            "question": "?",
            "expected_answer": "Кишинёв",
            "answer": "Кишинёв [1]",
            "sources_text": "[1] ...",
        },
        "b": {
            "status": "answered",
            "question": "?",
            "expected_answer": "Парламент",
            "answer": "Однопалатный парламент [1]",
            "sources_text": "[1] ...",
        },
        "c": {
            "status": "no_context",
            "question": "?",
            "expected_answer": "x",
            "answer": "отказ",
            "sources_text": "",
        },
    }
    labels = {
        "a": {"correctness": 1.0, "faithfulness": 1.0},
        "b": {"correctness": 1.0, "faithfulness": 1.0},
        "c": {"correctness": 0.0},
    }  # отказ судьёй не оценивается

    check = check_judge(Judge(), results, labels)
    assert check.agreement == {"correctness": 0.5, "faithfulness": 1.0}
    assert check.mae["correctness"] == 0.25
    assert [(d.id, d.metric) for d in check.disagreements] == [("b", "correctness")]


def test_check_judge_detects_changed_answer():
    from rag.evaluation.judge_check import check_judge

    results = {
        "a": {
            "status": "answered",
            "question": "?",
            "expected_answer": "x",
            "answer": "новый ответ",
            "sources_text": "",
        }
    }
    with pytest.raises(ValueError):
        check_judge(FakeJudge(), results, {"a": {"answer": "старый ответ", "correctness": 1.0}})
