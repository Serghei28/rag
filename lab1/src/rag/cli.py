"""Точка входа командной строки: `pixi run rag <команда>`."""

import logging
import platform
import shutil
import subprocess
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from rag import __version__
from rag.config import get_settings, load_config
from rag.tracing import get_langfuse

app = typer.Typer(help="RAG по государству и праву Республики Молдова", no_args_is_help=True)
console = Console()

ConfigOption = typer.Option(
    None, "--config", "-c", help="YAML-конфиг (по умолчанию configs/default.yaml)"
)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный лог")) -> None:
    """RAG по государству и праву Республики Молдова."""
    # Сторонние библиотеки (httpx, huggingface_hub, qdrant) — только предупреждения,
    # наш код (логгеры rag.*) — INFO или DEBUG с флагом -v
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    logging.getLogger("rag").setLevel(logging.DEBUG if verbose else logging.INFO)


@app.command()
def grab(
    config: Path | None = ConfigOption,
    force: bool = typer.Option(False, help="Скачать всё заново, даже неизменённые статьи"),
    dry_run: bool = typer.Option(False, help="Только показать, что будет скачано"),
) -> None:
    """Скачать новые и изменённые статьи Википедии в data/raw."""
    from rag.grabber import Grabber

    grabber = Grabber(load_config(config))
    with Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[title]}"),
        console=console,
    ) as progress:
        task = progress.add_task("Выборка статей…", total=None, title="")
        result = grabber.run(
            force=force,
            dry_run=dry_run,
            on_selected=lambda n: progress.update(task, total=n, description="Загрузка"),
            on_progress=lambda title: progress.update(task, advance=1, title=title[:50]),
        )

    table = Table(title="Итог" + (" (dry run)" if dry_run else ""), show_header=False)
    labels = {
        "selected": "В выборке",
        "new": "Новые",
        "updated": "Изменённые",
        "unchanged": "Без изменений",
        "removed": "Выпали из выборки",
        "duplicates": "Дубликаты",
        "failed": "Ошибки",
    }
    for key, value in result.summary().items():
        table.add_row(labels[key], str(value))
    console.print(table)
    for doc_id, error in result.failed.items():
        console.print(f"[red]{doc_id}[/]: {error}")


@app.command()
def preprocess(config: Path | None = ConfigOption) -> None:
    """Очистить документы из data/raw и построить chunks всеми стратегиями из конфига."""
    from rag.preprocessing.pipeline import build_chunks, clean_documents, length_stats

    cfg = load_config(config)
    with Progress(
        TextColumn("[bold]Очистка"), BarColumn(), MofNCompleteColumn(), console=console
    ) as progress:
        task = progress.add_task("", total=None)
        docs = clean_documents(cfg, on_progress=lambda _: progress.advance(task))

    doc_stats = length_stats([d.char_count for d in docs])
    console.print(
        f"Документов: {len(docs)}, символов: {doc_stats['total_chars']:,}, "
        f"секций: {sum(len(d.sections) for d in docs)}".replace(",", " ")
    )

    table = Table(title="Chunks по стратегиям (длина в символах)")
    for col in ["Стратегия", "Chunks", "Средняя", "Медиана", "p95", "Макс."]:
        table.add_column(col, justify="right" if col != "Стратегия" else "left")
    for strategy in cfg.chunking:
        st = length_stats([len(c.text) for c in build_chunks(cfg, docs, strategy)])
        table.add_row(
            strategy.name, *(str(st[k]) for k in ["count", "mean", "median", "p95", "max"])
        )
    console.print(table)
    console.print(f"Результат: {cfg.paths.processed_dir}")


@app.command()
def index(
    model: list[str] = typer.Option(None, "--model", "-m", help="Модель (по умолчанию все)"),
    strategy: list[str] = typer.Option(
        None, "--strategy", "-s", help="Стратегия (по умолчанию все)"
    ),
    config: Path | None = ConfigOption,
) -> None:
    """Векторизовать chunks и записать в Qdrant: коллекция на каждую пару модель × стратегия."""
    import torch

    from rag.embeddings import collection_name, get_store, index_chunks
    from rag.embeddings.encoder import Encoder
    from rag.preprocessing.pipeline import read_chunks

    cfg = load_config(config)
    models_ = [cfg.embedding_model(m) for m in model] if model else cfg.embeddings.models
    strategies = [cfg.chunking_strategy(s) for s in strategy] if strategy else cfg.chunking
    store = get_store(cfg.paths.index_dir / "qdrant")

    table = Table(title="Индексация")
    for col in ["Коллекция", "Chunks", "Новые", "Изм.", "Удал.", "Без изм.", "Сек"]:
        table.add_column(col, justify="left" if col == "Коллекция" else "right", no_wrap=True)

    for spec in models_:
        encoder = Encoder(spec, cfg.embeddings.device, cfg.embeddings.batch_size)
        for s in strategies:
            name = collection_name(s.name, spec.name)
            console.print(f"[bold]{name}[/] (dim={encoder.dim})")
            r = index_chunks(
                store,
                encoder,
                name,
                read_chunks(cfg, s.name),
                with_header=cfg.embeddings.with_header,
                show_progress=True,
            )
            table.add_row(
                name, *map(str, [r.total, r.added, r.updated, r.deleted, r.unchanged, r.seconds])
            )
        del encoder  # освобождаем видеопамять перед следующей моделью
        torch.cuda.empty_cache()
    console.print(table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Вопрос на русском"),
    model: str | None = typer.Option(None, "--model", "-m"),
    strategy: str | None = typer.Option(None, "--strategy", "-s"),
    k: int = typer.Option(5, "--k", "-k", help="Сколько результатов показать"),
    config: Path | None = ConfigOption,
) -> None:
    """Векторный поиск без LLM: какие chunks ближе всего к вопросу."""
    from rag.embeddings import collection_name, get_store
    from rag.embeddings.encoder import get_encoder

    cfg = load_config(config)
    spec = cfg.embedding_model(model or cfg.active.embedding)
    name = collection_name(strategy or cfg.active.chunking, spec.name)
    store = get_store(cfg.paths.index_dir / "qdrant")
    if not store.exists(name):
        console.print(f"[red]Коллекции {name} нет — запустите `pixi run index`[/]")
        raise typer.Exit(1)

    encoder = get_encoder(spec, cfg.embeddings.device, cfg.embeddings.batch_size)
    hits = store.search(name, encoder.encode_query(query), limit=k)
    console.print(f"[dim]{name}[/]")
    for rank, hit in enumerate(hits, start=1):
        c = hit.chunk
        console.rule(f"[bold]#{rank}[/] score={hit.score:.3f} · {c.title} · {c.section}")
        console.print(
            c.text[:400] + ("…" if len(c.text) > 400 else ""), markup=False, highlight=False
        )


DROP_REASONS = {
    "score_threshold": "ниже порога similarity",
    "too_short": "слишком короткие",
    "duplicate": "дубликаты",
    "per_document_limit": "лимит chunks на статью",
    "rerank_threshold": "ниже порога reranker'а",
    "beyond_top_n": "не вошли в top_n",
}


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Вопрос на русском"),
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding-модель"),
    strategy: str | None = typer.Option(None, "--strategy", "-s"),
    reranker: str | None = typer.Option(None, "--reranker", "-r"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Без reranker'а"),
    top_k: int | None = typer.Option(None, "--k", "-k", help="Кандидатов из векторного поиска"),
    top_n: int | None = typer.Option(None, "--n", "-n", help="Сколько chunks в итоге"),
    title: list[str] = typer.Option(None, "--title", "-t", help="Искать только в статье"),
    show_text: bool = typer.Option(False, "--text", help="Показать текст chunks"),
    config: Path | None = ConfigOption,
) -> None:
    """Полный retrieval: векторный поиск → фильтры → reranker. Показывает каждый шаг."""
    from rag.retrieval import MetadataFilter, Retriever

    cfg = load_config(config)
    retriever = Retriever(cfg, embedding=model, chunking=strategy, reranker=reranker)
    overrides = {
        k: v
        for k, v in {"top_k": top_k, "top_n": top_n, "rerank": False if no_rerank else None}.items()
        if v is not None
    }
    result = retriever.retrieve(
        query, metadata_filter=MetadataFilter(titles=title) if title else None, **overrides
    )

    console.print(
        f"[dim]{result.collection} · reranker: {result.reranker or 'нет'} · "
        f"кандидатов: {len(result.candidates)}[/]"
    )
    table = Table(title=query, show_lines=show_text)
    for col in ["#", "Rerank", "Vector", "Место до", "Статья · раздел"]:
        table.add_column(col, justify="right" if col != "Статья · раздел" else "left")
    for rank, c in enumerate(result.chunks, start=1):
        moved = c.vector_rank - rank
        arrow = f"[green]↑{moved}[/]" if moved > 0 else f"[red]↓{-moved}[/]" if moved < 0 else "="
        where = f"{c.chunk.title} · {c.chunk.section}"
        if show_text:
            where += "\n[dim]" + c.chunk.text[:300].replace("[", r"\[") + "…[/]"
        rerank = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "—"
        table.add_row(str(rank), rerank, f"{c.vector_score:.3f}", f"{c.vector_rank} {arrow}", where)
    console.print(table)

    if result.is_empty:
        console.print(
            "[yellow]Ничего релевантного не найдено — LLM ответит «недостаточно информации»[/]"
        )
    for reason, items in result.dropped.items():
        names = ", ".join(f"{c.chunk.title} ({c.vector_score:.2f})" for c in items[:3])
        more = f" и ещё {len(items) - 3}" if len(items) > 3 else ""
        console.print(f"[dim]Отброшено, {DROP_REASONS[reason]}: {len(items)} — {names}{more}[/]")
    console.print(
        "[dim]Время, мс: " + ", ".join(f"{k} {v}" for k, v in result.timings_ms.items()) + "[/]"
    )


@app.command()
def ask(
    query: str = typer.Argument(..., help="Вопрос на русском"),
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding-модель"),
    strategy: str | None = typer.Option(None, "--strategy", "-s"),
    reranker: str | None = typer.Option(None, "--reranker", "-r"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Без reranker'а"),
    top_n: int | None = typer.Option(None, "--n", "-n", help="Сколько chunks отдать LLM"),
    show_raw: bool = typer.Option(False, "--raw", help="Показать ответ LLM до проверок"),
    config: Path | None = ConfigOption,
) -> None:
    """Ответ на вопрос: retrieval → LLM → проверка ссылок на источники."""
    from rich.panel import Panel

    from rag.generation import STATUS_LABELS, Generator
    from rag.tracing import flush

    cfg = load_config(config)
    generator = Generator(cfg, embedding=model, chunking=strategy, reranker=reranker)
    overrides = {
        key: value
        for key, value in {"top_n": top_n, "rerank": False if no_rerank else None}.items()
        if value is not None
    }
    try:
        answer = generator.answer(query, **overrides)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from None
    finally:
        flush()

    color = "green" if answer.status == "answered" else "yellow"
    console.print(
        Panel(
            answer.text,
            title=query,
            subtitle=f"{answer.status}: {STATUS_LABELS[answer.status]}",
            border_style=color,
        )
    )
    for number, candidate in answer.cited_sources:
        chunk = candidate.chunk
        console.print(f"[bold][{number}][/] {chunk.title} · {chunk.section}  [dim]{chunk.url}[/]")
    if show_raw and answer.llm_text:
        console.print(Panel(answer.llm_text, title="Ответ LLM до проверок", border_style="dim"))
    console.print(
        f"[dim]{answer.retrieval.collection} · LLM: {cfg.generation.model} · "
        f"источников в промпте: {len(answer.sources)}[/]"
    )
    console.print(
        "[dim]Время, мс: "
        + ", ".join(f"{key} {value}" for key, value in answer.timings_ms.items())
        + "[/]"
    )


@app.command("eval")
def evaluate(
    name: str | None = typer.Option(None, "--name", help="Имя прогона (папка в experiments/runs)"),
    model: str | None = typer.Option(None, "--model", "-m", help="Embedding-модель"),
    strategy: str | None = typer.Option(None, "--strategy", "-s"),
    reranker: str | None = typer.Option(None, "--reranker", "-r"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Без reranker'а"),
    top_k: int | None = typer.Option(None, "--k", "-k", help="Кандидатов из векторного поиска"),
    top_n: int | None = typer.Option(None, "--n", "-n", help="Сколько chunks отдать LLM"),
    retrieval_only: bool = typer.Option(False, "--retrieval-only", help="Только метрики поиска"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Без LLM-судьи"),
    limit: int | None = typer.Option(None, "--limit", help="Только первые N вопросов"),
    config: Path | None = ConfigOption,
) -> None:
    """Прогнать набор вопросов: метрики поиска, ответы LLM и оценки LLM-судьи."""
    from rag.evaluation.dataset import load_dataset
    from rag.evaluation.runner import run_evaluation
    from rag.retrieval import Retriever

    cfg = load_config(config)
    total = len(load_dataset(cfg.evaluation.dataset).questions)
    total = min(total, limit) if limit else total
    retriever = Retriever(cfg, embedding=model, chunking=strategy, reranker=reranker)
    overrides = {
        k: v
        for k, v in {"top_k": top_k, "top_n": top_n, "rerank": False if no_rerank else None}.items()
        if v is not None
    }
    with Progress(
        TextColumn("[bold]Оценка"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[qid]}"),
        console=console,
    ) as progress:
        task = progress.add_task("", total=total, qid="")
        run = run_evaluation(
            cfg,
            name=name,
            retriever=retriever,
            generate=not retrieval_only,
            use_judge=not no_judge,
            limit=limit,
            overrides=overrides,
            on_progress=lambda qid: progress.update(task, advance=1, qid=qid),
        )

    s = run.summary["overall"]
    table = Table(title=run.name, show_header=False)
    for key, value in s.get("retrieval", {}).items():
        if key == "mrr" or key.endswith(("@1", "@5")):
            table.add_row(f"retrieval {key}", f"{value:.3f}")
    if "no_answer_rejected_by_retrieval" in s:
        table.add_row("no_answer отсечены поиском", f"{s['no_answer_rejected_by_retrieval']:.3f}")
    for key, value in s.get("generation", {}).items():
        table.add_row(key, "—" if value is None else f"{value:.3f}")
    console.print(table)
    console.print(f"Отчёт: {run.path / 'report.md'}")


@app.command("judge-check")
def judge_check(
    version: list[str] = typer.Option(
        ["v1", "v2", "v3"], "--prompt", "-p", help="Версии промпта судьи"
    ),
    config: Path | None = ConfigOption,
) -> None:
    """Сравнить оценки LLM-судьи с ручной разметкой (experiments/judge_validation)."""
    from rag.evaluation.judge import make_judge
    from rag.evaluation.judge_check import check_judge, load_labels, load_results, render_report

    cfg = load_config(config)
    out_dir = cfg.paths.experiments_dir / "judge_validation"
    run, labels = load_labels(out_dir / "labels.yaml")
    results = load_results(cfg.paths.experiments_dir / "runs" / run)

    checks = []
    for v in version:
        with console.status(f"Судья, промпт {v}…"):
            checks.append(check_judge(make_judge(cfg, v), results, labels))
    report = render_report(run, checks, len(labels))
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    table = Table(title=f"Судья vs ручная разметка ({len(labels)} ответов прогона {run})")
    for col in ["Промпт", "Согласие C", "MAE C", "Согласие F", "MAE F", "Расхождений"]:
        table.add_column(col)
    for c in checks:
        table.add_row(
            c.version,
            f"{c.agreement['correctness']:.0%}",
            f"{c.mae['correctness']:.2f}",
            f"{c.agreement['faithfulness']:.0%}",
            f"{c.mae['faithfulness']:.2f}",
            str(len(c.disagreements)),
        )
    console.print(table)
    console.print(f"Отчёт: {out_dir / 'report.md'}")


@app.command()
def titles(
    contains: str | None = typer.Argument(None, help="Часть названия, например «выборы»"),
    config: Path | None = ConfigOption,
) -> None:
    """Список статей в корпусе (точные названия — для relevant_titles в вопросах)."""
    from rag.grabber.store import RawStore

    cfg = load_config(config)
    docs = [
        r
        for r in RawStore(cfg.paths.raw_dir).load_manifest().documents.values()
        if r.status == "active" and (not contains or contains.lower() in r.title.lower())
    ]
    for r in sorted(docs, key=lambda r: r.title):
        console.print(f"{r.title}  [dim]{r.url}[/]", highlight=False)
    console.print(f"[dim]Всего: {len(docs)}[/]")


@app.command("eval-check")
def eval_check(
    path: Path | None = typer.Option(
        None, "--path", "-p", help="По умолчанию data/eval/questions.yaml"
    ),
    config: Path | None = ConfigOption,
) -> None:
    """Проверить файл с вопросами: формат, типы, названия статей."""
    from pydantic import ValidationError

    from rag.evaluation.dataset import TYPE_LABELS, load_dataset, unknown_titles
    from rag.grabber.store import RawStore

    cfg = load_config(config)
    path = path or cfg.paths.data_dir / "eval" / "questions.yaml"
    try:
        dataset = load_dataset(path)
    except ValidationError as exc:
        console.print(f"[red]Ошибки в {path}:[/]")
        for err in exc.errors():
            where = ".".join(str(x) for x in err["loc"])
            console.print(f"  [red]•[/] {where}: {err['msg']}")
        raise typer.Exit(1) from None

    known = {r.title for r in RawStore(cfg.paths.raw_dir).load_manifest().documents.values()}
    missing = unknown_titles(dataset, known)
    for qid, titles in missing.items():
        console.print(f"[red]{qid}[/]: нет статей {titles} — проверьте точное название")

    table = Table(title=f"{path} — {len(dataset.questions)} вопросов")
    table.add_column("Тип")
    table.add_column("Описание")
    table.add_column("Кол-во", justify="right")
    counts = dataset.by_type()
    for t, label in TYPE_LABELS.items():
        table.add_row(t, label, str(counts.get(t, 0)))
    console.print(table)
    if missing:
        raise typer.Exit(1)
    console.print("[green]Файл корректен[/]")


@app.command()
def chunks(
    title: str = typer.Argument(..., help="Название статьи, например «Конституция Молдовы»"),
    strategy: str = typer.Option("section_800", "--strategy", "-s"),
    config: Path | None = ConfigOption,
) -> None:
    """Показать chunks одной статьи — чтобы глазами проверить качество разбиения."""
    from rag.preprocessing.pipeline import read_chunks

    found = [c for c in read_chunks(load_config(config), strategy) if c.title == title]
    if not found:
        console.print(f"[red]Нет chunks для «{title}» в стратегии {strategy}[/]")
        raise typer.Exit(1)
    for c in found:
        console.rule(f"[bold]{c.chunk_id}[/] · {c.section} · {len(c.text)} симв.")
        console.print(c.text, markup=False, highlight=False)


@app.command()
def check() -> None:
    """Проверить окружение: конфиг, GPU, Langfuse, Ollama."""
    settings = get_settings()
    config = load_config()

    table = Table(title=f"rag-md {__version__}", show_header=False)
    table.add_row("Python", platform.python_version())
    table.add_row("Data dir", str(config.paths.data_dir))
    table.add_row("GPU", _gpu_status())
    table.add_row("Langfuse", _langfuse_status())
    table.add_row("Ollama", _ollama_status(settings.ollama_host, config.generation.model))
    console.print(table)


def _gpu_status() -> str:
    if not shutil.which("nvidia-smi"):
        return "[yellow]nvidia-smi не найден[/]"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return f"[green]{out.stdout.strip()}[/]" if out.returncode == 0 else "[red]ошибка[/]"


def _langfuse_status() -> str:
    client = get_langfuse()
    if client is None:
        return "[yellow]отключён (нет ключей в .env)[/]"
    try:
        ok = client.auth_check()
    except Exception as exc:  # сеть, неверный host и т.п.
        return f"[red]ошибка: {exc}[/]"
    return f"[green]OK ({get_settings().langfuse_host})[/]" if ok else "[red]неверные ключи[/]"


def _ollama_status(host: str, model: str) -> str:
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=3)
        resp.raise_for_status()
    except httpx.HTTPError:
        return f"[yellow]недоступна ({host})[/]"
    models = [m["name"] for m in resp.json().get("models", [])]
    if model not in models:
        return f"[yellow]нет модели {model}: `ollama pull {model}`[/]"
    return f"[green]OK, модели: {', '.join(models)}[/]"


if __name__ == "__main__":
    app()
