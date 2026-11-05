"""Эксперименты: серии прогонов оценки, в которых меняется один компонент RAG.

Каждая серия (`Suite`) отвечает на один исследовательский вопрос, например «какая
стратегия chunking лучше». Вариант серии (`Variant`) — изменения относительно базовой
конфигурации: патч YAML-конфига, параметры retrieval, embedding / chunking / reranker.

    pixi run rag experiment chunking            # одна серия
    pixi run rag experiment --retrieval         # все серии, которые не требуют LLM

Результаты:
    experiments/runs/<серия>__<вариант>/   — полный прогон каждого варианта (как rag eval)
    experiments/<серия>.md / .csv / .svg   — сводная таблица и график для README
"""

import copy
import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rag.config import AppConfig, ChunkingStrategy
from rag.evaluation.charts import bar_chart, line_chart
from rag.evaluation.runner import EvalRun, run_evaluation

log = logging.getLogger(__name__)


@dataclass
class Variant:
    name: str
    label: str  # подпись в таблице и на графике
    embedding: str | None = None
    chunking: str | None = None
    reranker: str | None = None
    config: dict = field(default_factory=dict)  # патч конфига, например {"generation": {...}}
    overrides: dict = field(default_factory=dict)  # параметры Retriever.retrieve
    x: float | None = None  # значение для графика-линии (размер, порог, K)
    # Та же конфигурация, что у готового прогона (например, «qwen 7b» = «повтор 1»):
    # результат берётся оттуда, без повторного запуска на 6 минут
    same_as: str | None = None


@dataclass
class Suite:
    name: str
    title: str
    question: str
    variants: list[Variant]
    generate: bool = False  # True — с ответами LLM и судьёй (~6 мин на вариант)
    extra_chunking: list[ChunkingStrategy] = field(default_factory=list)
    plot_metrics: list[str] = field(default_factory=lambda: ["hit@1", "hit@5", "mrr"])
    x_label: str | None = None  # если задано — график-линия по Variant.x


# ------------------------------------------------------------------ серии ---

THRESHOLDS = [None, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]


def _suites() -> dict[str, Suite]:
    def strat(name: str, method: str, size: int, overlap: int = 0) -> ChunkingStrategy:
        return ChunkingStrategy(name=name, method=method, size=size, overlap=overlap)

    suites = [
        Suite(
            "chunking",
            "Стратегия chunking × embedding-модель",
            "Какой способ разбиения и какая модель дают лучший поиск?",
            [
                Variant(f"{s}__{m}", f"{s} · {m}", embedding=m, chunking=s)
                for m in ("e5-base", "bge-m3")
                for s in ("fixed_800", "fixed_800_ov150", "paragraph_800", "section_800")
            ],
            plot_metrics=["hit@1", "mrr", "recall@5"],
        ),
        Suite(
            "embeddings",
            "Embedding-модели без reranker'а",
            "Какая embedding-модель лучше сама по себе, без пересортировки reranker'ом?",
            [
                Variant(
                    f"{s}__{m}", f"{s} · {m}", embedding=m, chunking=s, overrides={"rerank": False}
                )
                for m in ("e5-base", "bge-m3")
                for s in ("fixed_800", "section_800")
            ],
            plot_metrics=["hit@1", "mrr", "recall@5"],
        ),
        Suite(
            "chunk_size",
            "Размер chunk",
            "Как размер chunk влияет на поиск (fixed и section, bge-m3)?",
            [
                Variant(f"{m}_{n}", f"{m} {n}", chunking=f"{m}_{n}", x=n)
                for m in ("fixed", "section")
                for n in (400, 800, 1200)
            ],
            extra_chunking=[
                strat(f"{m}_{n}", m, n) for m in ("fixed", "section") for n in (400, 1200)
            ],
            plot_metrics=["mrr", "recall@5"],
            x_label="размер chunk, символов",
        ),
        Suite(
            "overlap",
            "Overlap для fixed-разбиения",
            "Помогает ли перекрытие соседних chunks (fixed, 800 символов)?",
            [
                Variant("ov0", "overlap 0", chunking="fixed_800", x=0),
                Variant("ov150", "overlap 150", chunking="fixed_800_ov150", x=150),
                Variant("ov300", "overlap 300", chunking="fixed_800_ov300", x=300),
            ],
            extra_chunking=[strat("fixed_800_ov300", "fixed", 800, 300)],
            plot_metrics=["hit@1", "mrr", "recall@5"],
            x_label="overlap, символов",
        ),
        Suite(
            "header",
            "Заголовок «Статья. Раздел» перед текстом chunk",
            "Нужен ли контекст статьи в тексте, который векторизуется?",
            [
                Variant("with_header", "с заголовком"),
                Variant(
                    "no_header", "без заголовка", config={"embeddings": {"with_header": False}}
                ),
            ],
        ),
        Suite(
            "top_k",
            "Число кандидатов из векторного поиска (top_k)",
            "Сколько кандидатов отдавать reranker'у?",
            [Variant(f"k{k}", f"top_k={k}", overrides={"top_k": k}, x=k) for k in (5, 10, 20, 50)],
            plot_metrics=["hit@1", "hit@5", "mrr"],
            x_label="top_k",
        ),
        Suite(
            "reranker",
            "Reranker",
            "Даёт ли reranker прирост и какой лучше?",
            [
                Variant("none", "без reranker'а", overrides={"rerank": False}),
                Variant("mminilm", "mminilm", reranker="mminilm"),
                Variant("bge", "bge-reranker", reranker="bge-reranker"),
            ],
        ),
        Suite(
            "thresholds",
            "Порог reranker'а",
            "Где граница между отсечением вопросов без ответа и потерей правильных?",
            [
                Variant(
                    f"thr_{t}",
                    f"порог {t if t is not None else 'нет'}",
                    overrides={"rerank_threshold": t},
                    x=t or 0.0,
                )
                for t in THRESHOLDS
            ],
            plot_metrics=["no_answer_rejected", "answerable_rejected", "hit@5"],
            x_label="порог rerank score",
        ),
        Suite(
            "variance",
            "Разброс между одинаковыми прогонами",
            "Насколько отличаются метрики при повторе той же конфигурации?",
            [Variant(f"rep{i}", f"повтор {i}") for i in (1, 2, 3)],
            generate=True,
            plot_metrics=["correctness", "faithfulness"],
        ),
        Suite(
            "top_n",
            "Число chunks в контексте LLM (top_n)",
            "Сколько фрагментов давать LLM?",
            [
                Variant(
                    f"n{n}",
                    f"top_n={n}",
                    overrides={"top_n": n},
                    x=n,
                    same_as="variance__rep1" if n == 5 else None,
                )
                for n in (3, 5, 10)
            ],
            generate=True,
            plot_metrics=["correctness", "faithfulness"],
            x_label="top_n",
        ),
        Suite(
            "llm",
            "LLM",
            "Как размер модели влияет на ответы (qwen2.5 7B и 3B)?",
            [
                Variant("qwen7b", "qwen2.5:7b", same_as="variance__rep1"),
                Variant("qwen3b", "qwen2.5:3b", config={"generation": {"model": "qwen2.5:3b"}}),
            ],
            generate=True,
            plot_metrics=["correctness", "faithfulness"],
        ),
        Suite(
            "prompt",
            "Системный промпт",
            "Помогают ли правила против найденных ошибок (v2) по сравнению с базовым (v1)?",
            [
                Variant("v1", "промпт v1", same_as="variance__rep1"),
                Variant("v2", "промпт v2", config={"generation": {"prompt": "v2"}}),
            ],
            generate=True,
            plot_metrics=["correctness", "faithfulness"],
        ),
    ]
    return {s.name: s for s in suites}


SUITES = _suites()


# ------------------------------------------------------------- подготовка ---


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def patched_config(base: AppConfig, patch: dict, extra: list[ChunkingStrategy]) -> AppConfig:
    data = deep_merge(base.model_dump(), patch)
    known = {s["name"] for s in data["chunking"]}
    data["chunking"] += [s.model_dump() for s in extra if s.name not in known]
    return AppConfig.model_validate(data)


def ensure_collection(config: AppConfig, chunking: str, embedding: str) -> None:
    """Достроить chunks и коллекцию Qdrant, если их ещё нет (инкрементально)."""
    from rag.embeddings import collection_name, get_store, index_chunks
    from rag.embeddings.encoder import get_encoder
    from rag.preprocessing.pipeline import build_chunks, chunks_path, read_chunks, read_documents

    strategy = config.chunking_strategy(chunking)
    if not chunks_path(config, chunking).exists():
        log.info("строю chunks %s", chunking)
        build_chunks(config, read_documents(config), strategy)

    store = get_store(config.paths.index_dir / "qdrant")
    emb = config.embeddings
    name = collection_name(chunking, embedding, emb.with_header)
    spec = config.embedding_model(embedding)
    encoder = get_encoder(spec, emb.device, emb.batch_size)
    result = index_chunks(
        store, encoder, name, read_chunks(config, chunking), with_header=emb.with_header
    )
    if result.added or result.updated:
        log.info("индекс %s: +%d, изменено %d", name, result.added, result.updated)


# ------------------------------------------------------------------ прогон ---


def run_suite(
    base: AppConfig,
    suite: Suite,
    *,
    only: list[str] | None = None,
    on_variant: Callable[[Variant], None] | None = None,
) -> list[tuple[Variant, dict]]:
    from rag.retrieval import Retriever

    rows = []
    for v in suite.variants:
        run_name = f"{suite.name}__{v.name}"
        run_dir = base.paths.experiments_dir / "runs" / run_name
        if only and v.name not in only:
            summary = _load_summary(run_dir)  # остальные варианты — из прошлых прогонов
            if summary:
                rows.append((v, summary))
            continue
        if v.same_as:
            summary = _load_summary(base.paths.experiments_dir / "runs" / v.same_as)
            if summary is None:
                raise FileNotFoundError(f"{v.label}: сначала нужен прогон {v.same_as}")
            rows.append((v, summary))
            continue
        if on_variant:
            on_variant(v)

        cfg = patched_config(base, v.config, suite.extra_chunking)
        embedding = v.embedding or cfg.active.embedding
        chunking = v.chunking or cfg.active.chunking
        ensure_collection(cfg, chunking, embedding)

        retriever = Retriever(cfg, embedding=embedding, chunking=chunking, reranker=v.reranker)
        run: EvalRun = run_evaluation(
            cfg,
            name=run_name,
            retriever=retriever,
            generate=suite.generate,
            overrides=v.overrides,
        )
        rows.append((v, {"params": run.params, "summary": run.summary}))

    write_suite_report(base.paths.experiments_dir, suite, rows)
    return rows


def _load_summary(run_dir: Path) -> dict | None:
    import json

    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ------------------------------------------------------------------- отчёт ---

RETRIEVAL_COLUMNS = [
    ("hit@1", "Hit@1"),
    ("hit@5", "Hit@5"),
    ("mrr", "MRR"),
    ("recall@5", "Recall@5"),
    ("precision@5", "Prec@5"),
    ("multi_doc_recall@5", "Recall@5 multi_doc"),
    ("answerable_rejected", "Ложн. отсев"),
    ("no_answer_rejected", "Отсев no_answer"),
    ("retrieval_ms", "Поиск, мс"),
]
GENERATION_COLUMNS = [
    ("correctness", "Correctness"),
    ("faithfulness", "Faithfulness"),
    ("false_refusal_rate", "Ложн. отказы"),
    ("no_answer_refusal_rate", "Отказ no_answer"),
    ("llm_ms", "LLM, мс"),
]


def flatten(summary: dict) -> dict[str, float | None]:
    o = summary["overall"]
    flat = dict(o.get("retrieval", {}))
    flat["multi_doc_recall@5"] = (
        summary["by_type"].get("multi_doc", {}).get("retrieval", {}).get("recall@5")
    )
    flat["answerable_rejected"] = o.get("answerable_rejected_by_retrieval")
    flat["no_answer_rejected"] = o.get("no_answer_rejected_by_retrieval")
    flat["retrieval_ms"] = summary["latency_ms"].get("retrieval_total")
    flat.update(o.get("generation", {}))
    flat["llm_ms"] = summary["latency_ms"].get("llm")
    return flat


def _fmt(value: float | None, key: str) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}" if key.endswith("_ms") else f"{value:.3f}"


def write_suite_report(out_dir: Path, suite: Suite, rows: list[tuple[Variant, dict]]) -> None:
    columns = RETRIEVAL_COLUMNS + (GENERATION_COLUMNS if suite.generate else [])
    flats = [(v, flatten(data["summary"])) for v, data in rows]

    lines = [f"# {suite.title}", "", f"**Вопрос:** {suite.question}", ""]
    lines += [
        "| Вариант | " + " | ".join(t for _, t in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for v, flat in flats:
        cells = " | ".join(_fmt(flat.get(k), k) for k, _ in columns)
        lines.append(f"| {v.label} | {cells} |")
    lines += ["", f"Прогоны: `experiments/runs/{suite.name}__<вариант>/`", ""]
    (out_dir / f"{suite.name}.md").write_text("\n".join(lines), encoding="utf-8")

    with (out_dir / f"{suite.name}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", *[k for k, _ in columns]])
        for v, flat in flats:
            writer.writerow([v.name, *[flat.get(k) for k, _ in columns]])

    _plot(out_dir / f"{suite.name}.svg", suite, flats)


def _plot(path: Path, suite: Suite, flats: list[tuple[Variant, dict]]) -> None:
    metrics = [m for m in suite.plot_metrics if any(f.get(m) is not None for _, f in flats)]
    if suite.x_label and all(v.x is not None for v, _ in flats):
        series: dict[str, list[tuple[float, float | None]]] = {}
        for v, f in flats:  # в chunk_size две линии на метрику: fixed и section
            group = v.label.rsplit(" ", 1)[0] if suite.name == "chunk_size" else ""
            for m in metrics:
                series.setdefault(f"{group} {m}".strip(), []).append((v.x, f.get(m)))
        line_chart(path, suite.title, suite.x_label, series)
    else:
        labels = [v.label for v, _ in flats]
        bar_chart(path, suite.title, labels, {m: [f.get(m) for _, f in flats] for m in metrics})
