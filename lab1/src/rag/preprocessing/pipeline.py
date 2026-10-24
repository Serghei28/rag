"""Pipeline preprocessing: data/raw → data/processed.

data/processed/
├── documents.jsonl          # очищенные документы с секциями (один JSON на строку)
└── chunks/<strategy>.jsonl  # chunks каждой стратегии из конфига
"""

import logging
import statistics
from collections.abc import Callable
from pathlib import Path

from rag.config import AppConfig, ChunkingStrategy
from rag.grabber.store import RawStore
from rag.preprocessing.chunking import chunk_document
from rag.preprocessing.html_cleaner import clean_html
from rag.preprocessing.models import Chunk, CleanDocument

log = logging.getLogger(__name__)


def documents_path(config: AppConfig) -> Path:
    return config.paths.processed_dir / "documents.jsonl"


def chunks_path(config: AppConfig, strategy: str) -> Path:
    return config.paths.processed_dir / "chunks" / f"{strategy}.jsonl"


def write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    tmp.replace(path)


def read_documents(config: AppConfig) -> list[CleanDocument]:
    with documents_path(config).open(encoding="utf-8") as f:
        return [CleanDocument.model_validate_json(line) for line in f]


def read_chunks(config: AppConfig, strategy: str) -> list[Chunk]:
    with chunks_path(config, strategy).open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f]


def clean_documents(
    config: AppConfig, on_progress: Callable[[str], None] | None = None
) -> list[CleanDocument]:
    store = RawStore(config.paths.raw_dir)
    records = [r for r in store.load_manifest().documents.values() if r.status == "active"]
    if not records:
        raise FileNotFoundError("data/raw пуст — сначала запустите `pixi run grab`")

    docs = []
    for rec in records:
        if on_progress:
            on_progress(rec.title)
        sections = clean_html(store.read_document(rec), config.preprocessing)
        if not sections:
            log.warning("после очистки пусто: %s", rec.title)
            continue
        docs.append(
            CleanDocument(
                document_id=rec.document_id,
                source=rec.source,
                title=rec.title,
                url=rec.url,
                updated_at=rec.updated_at,
                categories=rec.categories,
                sections=sections,
            )
        )
    write_jsonl(documents_path(config), docs)
    return docs


def build_chunks(
    config: AppConfig, docs: list[CleanDocument], strategy: ChunkingStrategy
) -> list[Chunk]:
    chunks = [c for doc in docs for c in chunk_document(doc, strategy)]
    write_jsonl(chunks_path(config, strategy.name), chunks)
    return chunks


def length_stats(lengths: list[int]) -> dict[str, float]:
    lengths = sorted(lengths)
    return {
        "count": len(lengths),
        "mean": round(statistics.mean(lengths)),
        "median": statistics.median(lengths),
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
        "total_chars": sum(lengths),
    }
