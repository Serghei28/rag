"""Хранилище сырых документов и манифест.

data/raw/
├── manifest.json        # метаданные всех документов: revid, sha256, путь, статус
├── runs.jsonl           # журнал запусков grabber'а (сколько новых/изменённых/ошибок)
└── wikipedia/<pageid>.html

Манифест — «память» grabber'а между запусками: по нему определяется, какие документы
новые, какие изменились, а какие можно не скачивать.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class DocumentRecord(BaseModel):
    document_id: str  # стабильный id: "ruwiki:<pageid>"
    source: str
    title: str
    url: str
    pageid: int
    revid: int
    updated_at: datetime  # дата последней правки статьи
    fetched_at: datetime  # когда grabber скачал эту версию
    sha256: str
    path: str  # путь к файлу относительно raw_dir
    seed: str  # через какую категорию/заголовок документ попал в выборку
    categories: list[str] = []
    status: Literal["active", "removed"] = "active"


class Manifest(BaseModel):
    documents: dict[str, DocumentRecord] = {}

    def by_content_hash(self, sha256: str) -> DocumentRecord | None:
        return next(
            (r for r in self.documents.values() if r.sha256 == sha256 and r.status == "active"),
            None,
        )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    """Запись через временный файл: прерванный запуск не оставит битый файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class RawStore:
    def __init__(self, raw_dir: Path):
        self.root = raw_dir
        self.manifest_path = raw_dir / "manifest.json"
        self.runs_path = raw_dir / "runs.jsonl"

    def load_manifest(self) -> Manifest:
        if not self.manifest_path.exists():
            return Manifest()
        return Manifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: Manifest) -> None:
        manifest.documents = dict(sorted(manifest.documents.items()))
        atomic_write(self.manifest_path, manifest.model_dump_json(indent=2))

    def write_document(self, rel_path: str, content: str) -> None:
        atomic_write(self.root / rel_path, content)

    def read_document(self, rec: DocumentRecord) -> str:
        return (self.root / rec.path).read_text(encoding="utf-8")

    def has_file(self, rec: DocumentRecord) -> bool:
        return (self.root / rec.path).exists()

    def log_run(self, summary: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {"finished_at": datetime.now(UTC).isoformat(), **summary}
        with self.runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
