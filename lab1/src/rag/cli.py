"""Точка входа командной строки: `pixi run rag <команда>`."""

import platform
import shutil
import subprocess

import httpx
import typer
from rich.console import Console
from rich.table import Table

from rag import __version__
from rag.config import get_settings, load_config
from rag.tracing import get_langfuse

app = typer.Typer(help="RAG по законодательству Республики Молдова", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """RAG по законодательству Республики Молдова."""


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
