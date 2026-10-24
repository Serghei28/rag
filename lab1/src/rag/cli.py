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
    table.add_row("Ollama", _ollama_status(settings.ollama_host))
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


def _ollama_status(host: str) -> str:
    try:
        resp = httpx.get(f"{host}/api/tags", timeout=3)
        resp.raise_for_status()
    except httpx.HTTPError:
        return f"[yellow]недоступна ({host})[/]"
    models = [m["name"] for m in resp.json().get("models", [])]
    return f"[green]OK, модели: {', '.join(models) or 'нет'}[/]"


if __name__ == "__main__":
    app()
