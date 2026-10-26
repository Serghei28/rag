"""Прогон набора вопросов через RAG и подсчёт метрик.

Для каждого вопроса:
1. Retrieval с top_n = max(ks) → метрики поиска (Hit@K, Recall@K, Precision@K, MRR).
2. Генерация ответа (как в боевом режиме, с настроенным top_n).
3. Оценка ответа:
   - no_answer: правильно, если система отказалась (правило, без судьи);
   - вопрос с ответом, система отказалась: «ложный отказ», correctness = 0;
   - вопрос с ответом, ответ дан: LLM-судья, correctness и faithfulness.
   Если система ответила на no_answer, судья проверяет faithfulness этого ответа.

Результаты: experiments/runs/<name>/
├── results.jsonl   — всё по каждому вопросу: найденные статьи, ответ, оценки судьи
├── summary.json    — итоговые метрики (общие и по типам вопросов)
└── report.md       — те же метрики таблицами для README
Оценки судьи дополнительно прикрепляются к трейсам Langfuse, все трейсы прогона — в
одну сессию с именем прогона.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from rag.config import AppConfig
from rag.evaluation.dataset import TYPE_LABELS, EvalDataset, EvalQuestion, load_dataset
from rag.evaluation.judge import LLMJudge, make_judge
from rag.evaluation.metrics import first_relevant_rank, mean, retrieval_metrics
from rag.generation.prompts import format_sources
from rag.tracing import flush, score, session


@dataclass
class QuestionResult:
    id: str
    type: str
    question: str
    expected_answer: str
    relevant_titles: list[str]
    # retrieval
    retrieved: list[dict] = field(default_factory=list)  # title, section, scores
    first_relevant_rank: int | None = None
    retrieval_metrics: dict = field(default_factory=dict)
    retrieval_empty: bool = False
    # generation
    status: str | None = None
    answer: str | None = None
    llm_answer: str | None = None  # ответ LLM до проверок
    cited_titles: list[str] = field(default_factory=list)
    sources_text: str = ""  # источники, которые видела LLM, — для повторной оценки судьёй
    trace_id: str | None = None
    # оценка
    correctness: float | None = None
    correctness_reason: str = ""
    faithfulness: float | None = None
    faithfulness_reason: str = ""
    timings_ms: dict = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return self.type != "no_answer"

    @property
    def refused(self) -> bool:
        return self.status is not None and self.status != "answered"


@dataclass
class EvalRun:
    name: str
    params: dict
    results: list[QuestionResult]
    summary: dict
    path: Path


def default_run_name(params: dict) -> str:
    combo = "__".join(
        str(params[k]) for k in ("chunking", "embedding", "reranker") if params.get(k)
    )
    return f"{datetime.now():%Y%m%d-%H%M}_{combo}"


def run_evaluation(
    config: AppConfig,
    *,
    name: str | None = None,
    dataset: EvalDataset | None = None,
    retriever=None,
    generator=None,
    judge: LLMJudge | None = None,
    generate: bool = True,
    use_judge: bool = True,
    limit: int | None = None,
    overrides: dict | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> EvalRun:
    """`overrides` — параметры retrieval (top_k, top_n, rerank, …), как у Retriever.retrieve."""
    from rag.generation import Generator
    from rag.retrieval import Retriever

    overrides = overrides or {}
    dataset = dataset or load_dataset(config.evaluation.dataset)
    questions = dataset.questions[:limit] if limit else dataset.questions
    ks = sorted(config.evaluation.ks)

    retriever = retriever or Retriever(config)
    if generate:
        generator = generator or Generator(config, retriever=retriever)
        if use_judge and judge is None:
            judge = make_judge(config)

    retrieval_params = {**asdict(retriever.defaults), **overrides}
    params = {
        "embedding": retriever.embedding_spec.name,
        "chunking": retriever.chunking,
        "reranker": retriever.reranker_name if retrieval_params["rerank"] else None,
        "retrieval": retrieval_params,
        "with_header": config.embeddings.with_header,
        "llm": config.generation.model if generate else None,
        "judge": f"{judge.model} ({getattr(judge, 'prompt_version', '?')})"
        if (generate and judge)
        else None,
        "dataset": str(config.evaluation.dataset),
        "questions": len(questions),
    }
    name = name or default_run_name(params)
    out_dir = config.paths.experiments_dir / "runs" / name

    results = []
    with session(name, tags=["eval"], metadata={k: str(v) for k, v in params.items()}):
        for q in questions:
            if on_progress:
                on_progress(q.id)
            results.append(_evaluate_question(q, retriever, generator, judge, ks, overrides))
    flush()

    summary = summarize(results, ks)
    run = EvalRun(name=name, params=params, results=results, summary=summary, path=out_dir)
    save_run(run)
    return run


def _evaluate_question(
    q: EvalQuestion, retriever, generator, judge, ks: list[int], overrides: dict
) -> QuestionResult:
    r = QuestionResult(
        id=q.id,
        type=q.type,
        question=q.question,
        expected_answer=q.expected_answer,
        relevant_titles=q.relevant_titles,
    )
    relevant = set(q.relevant_titles)

    # 1. Retrieval: берём max(K) chunks, чтобы посчитать метрики для всех K за один запрос
    started = time.perf_counter()
    retrieval = retriever.retrieve(q.question, **{**overrides, "top_n": max(ks)})
    r.timings_ms["retrieval_total"] = round((time.perf_counter() - started) * 1000, 1)
    ranked = [c.chunk.title for c in retrieval.chunks]
    r.retrieved = [
        {
            "title": c.chunk.title,
            "section": c.chunk.section,
            "vector_score": round(c.vector_score, 3),
            "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 3),
            "relevant": c.chunk.title in relevant,
        }
        for c in retrieval.chunks
    ]
    r.retrieval_empty = retrieval.is_empty
    if r.answerable:
        r.first_relevant_rank = first_relevant_rank(ranked, relevant)
        r.retrieval_metrics = retrieval_metrics(ranked, relevant, ks)

    if generator is None:
        return r

    # 2. Генерация в боевом режиме (свой top_n, проверки ссылок)
    answer = generator.answer(q.question, **overrides)
    r.status, r.answer, r.llm_answer = answer.status, answer.text, answer.llm_text
    r.cited_titles = [c.chunk.title for _, c in answer.cited_sources]
    r.sources_text = format_sources(answer.sources)
    r.trace_id = answer.trace_id
    r.timings_ms.update(answer.timings_ms)

    # 3. Оценка
    if not r.answerable:
        r.correctness = 0.0 if answer.status == "answered" else 1.0
        r.correctness_reason = (
            "система ответила на вопрос без ответа в базе"
            if answer.status == "answered"
            else f"корректный отказ ({answer.status})"
        )
    elif answer.status != "answered":
        r.correctness, r.correctness_reason = 0.0, f"ложный отказ ({answer.status})"
    elif judge is not None:
        verdict = judge.correctness(q.question, q.expected_answer, answer.text)
        r.correctness, r.correctness_reason = verdict.score, verdict.reason

    if answer.status == "answered" and judge is not None:
        verdict = judge.faithfulness(q.question, answer.text, r.sources_text)
        r.faithfulness, r.faithfulness_reason = verdict.score, verdict.reason

    for metric, value, reason in [
        ("correctness", r.correctness, r.correctness_reason),
        ("faithfulness", r.faithfulness, r.faithfulness_reason),
        ("retrieval_mrr", r.retrieval_metrics.get("mrr"), ""),
    ]:
        if value is not None:
            score(r.trace_id, metric, value, reason)
    return r


def summarize(results: list[QuestionResult], ks: list[int]) -> dict:
    def block(rs: list[QuestionResult]) -> dict:
        answerable = [r for r in rs if r.answerable]
        no_answer = [r for r in rs if not r.answerable]
        generated = [r for r in rs if r.status is not None]
        out: dict = {"questions": len(rs)}
        if answerable:
            keys = answerable[0].retrieval_metrics.keys()
            out["retrieval"] = {k: mean([r.retrieval_metrics[k] for r in answerable]) for k in keys}
        if no_answer:
            out["no_answer_rejected_by_retrieval"] = mean(
                [float(r.retrieval_empty) for r in no_answer]
            )
        if generated:
            ans_gen = [r for r in answerable if r.status is not None]
            out["generation"] = {
                "correctness": mean([r.correctness for r in generated]),
                "faithfulness": mean([r.faithfulness for r in generated]),
                "answered_rate": mean([float(not r.refused) for r in ans_gen]) if ans_gen else None,
                "false_refusal_rate": mean([float(r.refused) for r in ans_gen])
                if ans_gen
                else None,
                "no_answer_refusal_rate": (
                    mean([float(r.refused) for r in no_answer if r.status is not None])
                    if no_answer
                    else None
                ),
            }
        return out

    latency = {}
    for key in ("retrieval_total", "embed", "search", "filters", "rerank", "llm"):
        values = [r.timings_ms[key] for r in results if key in r.timings_ms]
        if values:
            latency[key] = round(sum(values) / len(values), 1)

    return {
        "overall": block(results),
        "by_type": {
            t: block([r for r in results if r.type == t])
            for t in TYPE_LABELS
            if any(r.type == t for r in results)
        },
        "latency_ms": latency,
    }


def save_run(run: EvalRun) -> None:
    run.path.mkdir(parents=True, exist_ok=True)
    with (run.path / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in run.results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    meta = {"name": run.name, "params": run.params, "summary": run.summary}
    (run.path / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run.path / "report.md").write_text(render_report(run), encoding="utf-8")


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def render_report(run: EvalRun) -> str:
    s, p = run.summary, run.params
    lines = [
        f"# Прогон `{run.name}`",
        "",
        f"embedding `{p['embedding']}` · chunking `{p['chunking']}` · reranker "
        f"`{p['reranker'] or 'нет'}` · LLM `{p['llm'] or '—'}` · судья `{p['judge'] or '—'}` · "
        f"вопросов: {p['questions']}",
        "",
        "## Поиск (вопросы с ответом)",
        "",
    ]
    types = list(s["by_type"])
    retrieval_keys = list(s["overall"].get("retrieval", {}))
    if retrieval_keys:
        header = ["Метрика", "Все", *[t for t in types if t != "no_answer"]]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for key in retrieval_keys:
            row = [key, _fmt(s["overall"]["retrieval"][key])]
            row += [_fmt(s["by_type"][t].get("retrieval", {}).get(key)) for t in header[2:]]
            lines.append("| " + " | ".join(row) + " |")
    if "generation" in s["overall"]:
        lines += [
            "",
            "## Ответы",
            "",
            "| Метрика | Все | " + " | ".join(types) + " |",
            "|" + "---|" * (len(types) + 2),
        ]
        for key in s["overall"]["generation"]:
            row = [key, _fmt(s["overall"]["generation"][key])]
            row += [_fmt(s["by_type"][t].get("generation", {}).get(key)) for t in types]
            lines.append("| " + " | ".join(row) + " |")
    if "no_answer_rejected_by_retrieval" in s["overall"]:
        lines += [
            "",
            "no_answer отсечены ещё поиском: "
            f"{_fmt(s['overall']['no_answer_rejected_by_retrieval'])}",
        ]
    lines += [
        "",
        "## Время, мс (среднее)",
        "",
        ", ".join(f"{k}: {v}" for k, v in s["latency_ms"].items()),
        "",
    ]

    failures = [r for r in run.results if r.correctness is not None and r.correctness < 1]
    if failures:
        lines += [
            "## Ошибки",
            "",
            "| id | тип | correctness | faithfulness | ответ | причина |",
            "|---|---|---|---|---|---|",
        ]
        for r in failures:
            answer = (r.answer or "").replace("\n", " ").replace("|", "/")[:160]
            reason = r.correctness_reason.replace("\n", " ").replace("|", "/")[:160]
            lines.append(
                f"| {r.id} | {r.type} | {_fmt(r.correctness)} | "
                f"{_fmt(r.faithfulness)} | {answer} | {reason} |"
            )
    return "\n".join(lines) + "\n"
