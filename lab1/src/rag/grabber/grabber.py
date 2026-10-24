"""Инкрементальный сбор статей.

Алгоритм одного запуска:
1. Собрать выборку статей: явные заголовки + обход категорий, без дубликатов по pageid.
2. Одним пакетным запросом узнать текущий revid каждой статьи.
3. Скачать только новые статьи и статьи, у которых изменился revid.
4. Статьи, выпавшие из выборки, пометить в манифесте как `removed`.
5. Ошибка на одной статье не останавливает запуск — она попадает в отчёт.
"""

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

from rag.config import AppConfig
from rag.grabber.store import DocumentRecord, RawStore, sha256_text
from rag.grabber.wikipedia import PageRef, RetryableError, WikiApiError, WikipediaClient

log = logging.getLogger(__name__)

SOURCE = "wikipedia-ru"


@dataclass
class GrabResult:
    selected: int = 0
    new: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, int]:
        return {
            "selected": self.selected,
            "new": len(self.new),
            "updated": len(self.updated),
            "unchanged": len(self.unchanged),
            "removed": len(self.removed),
            "duplicates": len(self.duplicates),
            "failed": len(self.failed),
        }


def document_id(pageid: int) -> str:
    return f"ruwiki:{pageid}"


class Grabber:
    def __init__(
        self,
        config: AppConfig,
        client: WikipediaClient | None = None,
        store: RawStore | None = None,
    ):
        self.cfg = config.grabber
        self.client = client or WikipediaClient(self.cfg)
        self.store = store or RawStore(config.paths.raw_dir)
        self._wiki_base = self.cfg.api_url.replace("/w/api.php", "/wiki/")
        self._exclude_titles = [re.compile(p, re.I) for p in self.cfg.exclude_titles]

    def select_pages(self) -> list[PageRef]:
        """Явные заголовки идут первыми: при ограничении max_pages они не должны выпасть."""
        refs: dict[int, PageRef] = {}

        def add(ref: PageRef) -> None:
            if ref.pageid in refs:
                return
            if any(p.search(ref.title) for p in self._exclude_titles):
                log.debug("пропуск статьи %s", ref.title)
                return
            refs[ref.pageid] = ref

        for ref in self.client.resolve_titles(self.cfg.titles):
            add(ref)
        for seed in self.cfg.categories:
            for ref in self.client.iter_category(seed.name, seed.depth):
                add(ref)

        pages = list(refs.values())
        if self.cfg.max_pages is not None and len(pages) > self.cfg.max_pages:
            log.warning("выборка обрезана: %d → %d статей", len(pages), self.cfg.max_pages)
            pages = pages[: self.cfg.max_pages]
        return pages

    def run(
        self,
        *,
        force: bool = False,
        dry_run: bool = False,
        on_progress: Callable[[str], None] | None = None,
        on_selected: Callable[[int], None] | None = None,
    ) -> GrabResult:
        result = GrabResult()
        manifest = self.store.load_manifest()

        pages = self.select_pages()
        result.selected = len(pages)
        if on_selected:
            on_selected(len(pages))
        revisions = self.client.revisions(p.pageid for p in pages)

        selected_ids: set[str] = set()
        for i, ref in enumerate(pages, start=1):
            doc_id = document_id(ref.pageid)
            selected_ids.add(doc_id)
            if on_progress:
                on_progress(ref.title)

            rev = revisions.get(ref.pageid)
            if rev is None:
                result.failed[doc_id] = "нет данных о правках (статья удалена?)"
                continue

            old = manifest.documents.get(doc_id)
            is_fresh = (
                old is not None
                and old.status == "active"
                and old.revid == rev.revid
                and self.store.has_file(old)
            )
            if is_fresh and not force:
                result.unchanged.append(doc_id)
                continue
            if dry_run:
                (result.updated if old else result.new).append(doc_id)
                continue

            try:
                page = self.client.parse(ref.pageid)
            except (WikiApiError, RetryableError, httpx.HTTPError) as exc:
                log.error("не удалось скачать %s: %s", ref.title, exc)
                result.failed[doc_id] = str(exc)
                continue

            sha = sha256_text(page.html)
            dup = manifest.by_content_hash(sha)
            if dup is not None and dup.document_id != doc_id:
                log.warning("%s дублирует %s, пропуск", ref.title, dup.title)
                result.duplicates.append(doc_id)
                continue

            rel_path = f"wikipedia/{page.pageid}.html"
            self.store.write_document(rel_path, page.html)
            manifest.documents[doc_id] = DocumentRecord(
                document_id=doc_id,
                source=SOURCE,
                title=page.title,
                url=self._wiki_base + page.title.replace(" ", "_"),
                pageid=page.pageid,
                revid=page.revid,
                updated_at=rev.timestamp,
                fetched_at=datetime.now(UTC),
                sha256=sha,
                path=rel_path,
                seed=ref.seed,
                categories=page.categories,
            )
            (result.updated if old else result.new).append(doc_id)

            if i % 25 == 0:  # промежуточное сохранение: прерванный запуск не теряет работу
                self.store.save_manifest(manifest)

        for doc_id, rec in manifest.documents.items():
            if doc_id not in selected_ids and rec.status == "active":
                rec.status = "removed"
                result.removed.append(doc_id)

        if not dry_run:
            self.store.save_manifest(manifest)
            self.store.log_run(result.summary())
        return result
