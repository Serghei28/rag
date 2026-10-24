"""Клиент MediaWiki API (https://www.mediawiki.org/wiki/API:Main_page).

Используем только официальный API, а не парсинг страниц сайта:
- стабильный формат ответа (JSON);
- редиректы разрешаются на стороне API, поэтому одна статья не скачивается дважды;
- `revid` (номер правки) позволяет дёшево проверить, изменилась ли статья.
"""

import logging
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.config import GrabberConfig

log = logging.getLogger(__name__)

BATCH = 50  # максимум titles/pageids в одном запросе для обычного клиента API


class RetryableError(Exception):
    """Временная ошибка: сеть, 429, 5xx, maxlag. Запрос будет повторён."""


class WikiApiError(Exception):
    """Ошибка API, повтор которой не поможет (например, страница не найдена)."""


@dataclass(frozen=True)
class PageRef:
    pageid: int
    title: str
    seed: str  # категория или явный заголовок, через который статья попала в выборку


@dataclass(frozen=True)
class Revision:
    pageid: int
    title: str
    revid: int
    timestamp: datetime


@dataclass
class ParsedPage:
    pageid: int
    title: str
    revid: int
    html: str
    categories: list[str] = field(default_factory=list)


class WikipediaClient:
    def __init__(self, cfg: GrabberConfig, http: httpx.Client | None = None):
        self.cfg = cfg
        self.http = http or httpx.Client(
            timeout=cfg.timeout, headers={"User-Agent": cfg.user_agent}, follow_redirects=True
        )
        self._exclude_cats = [re.compile(p, re.I) for p in cfg.exclude_categories]
        # повторяем только временные ошибки, с экспоненциальной паузой 1, 2, 4... сек
        self._get = retry(
            retry=retry_if_exception_type(RetryableError),
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential(multiplier=1, max=30),
            reraise=True,
        )(self._get_once)

    def _get_once(self, **params) -> dict:
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        try:
            resp = self.http.get(self.cfg.api_url, params=params)
        except httpx.TransportError as exc:
            raise RetryableError(f"сеть: {exc}") from exc
        finally:
            time.sleep(self.cfg.request_delay)

        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableError(f"HTTP {resp.status_code}")
        resp.raise_for_status()

        data = resp.json()
        if error := data.get("error"):
            if error.get("code") == "maxlag":
                raise RetryableError("maxlag: серверы Википедии перегружены")
            raise WikiApiError(f"{error.get('code')}: {error.get('info')}")
        return data

    def _query_all(self, **params) -> Iterator[dict]:
        """Запрос с постраничной выдачей через `continue`."""
        cont: dict = {}
        while True:
            data = self._get(action="query", **params, **cont)
            yield data.get("query", {})
            if "continue" not in data:
                return
            cont = data["continue"]

    # ------------------------------------------------------------ выборка ---

    def iter_category(self, name: str, depth: int, seed: str | None = None) -> Iterator[PageRef]:
        """Статьи категории и её подкатегорий до глубины `depth` (обход в ширину)."""
        seed = seed or f"Категория:{name}"
        queue, seen = [(name, depth)], set()
        while queue:
            cat, d = queue.pop(0)
            if cat in seen:
                continue
            seen.add(cat)
            for chunk in self._query_all(
                list="categorymembers",
                cmtitle=f"Категория:{cat}",
                cmtype="page|subcat",
                cmlimit="500",
            ):
                for m in chunk.get("categorymembers", []):
                    if m["ns"] == 0:
                        yield PageRef(m["pageid"], m["title"], seed)
                    elif m["ns"] == 14 and d > 0:
                        sub = m["title"].split(":", 1)[1]
                        if any(p.search(sub) for p in self._exclude_cats):
                            log.debug("пропуск подкатегории %s", sub)
                            continue
                        queue.append((sub, d - 1))

    def resolve_titles(self, titles: Iterable[str]) -> Iterator[PageRef]:
        """Заголовки → pageid с учётом редиректов. Несуществующие статьи пропускаются."""
        titles = list(titles)
        for i in range(0, len(titles), BATCH):
            batch = titles[i : i + BATCH]
            data = self._get(action="query", titles="|".join(batch), redirects="1")
            for page in data.get("query", {}).get("pages", []):
                if page.get("missing") or page.get("invalid"):
                    log.warning("статья не найдена: %s", page.get("title"))
                    continue
                yield PageRef(page["pageid"], page["title"], page["title"])

    # ------------------------------------------------------------ контент ---

    def revisions(self, pageids: Iterable[int]) -> dict[int, Revision]:
        """Номер и дата последней правки для каждой статьи — пакетами по 50."""
        pageids = list(pageids)
        result: dict[int, Revision] = {}
        for i in range(0, len(pageids), BATCH):
            batch = pageids[i : i + BATCH]
            data = self._get(
                action="query",
                pageids="|".join(map(str, batch)),
                prop="revisions",
                rvprop="ids|timestamp",
            )
            for page in data.get("query", {}).get("pages", []):
                if page.get("missing") or not page.get("revisions"):
                    continue
                rev = page["revisions"][0]
                result[page["pageid"]] = Revision(
                    pageid=page["pageid"],
                    title=page["title"],
                    revid=rev["revid"],
                    timestamp=datetime.fromisoformat(rev["timestamp"].replace("Z", "+00:00")),
                )
        return result

    def parse(self, pageid: int) -> ParsedPage:
        """Отрендеренный HTML статьи (как на сайте, без ссылок «править» и оглавления)."""
        data = self._get(
            action="parse",
            pageid=str(pageid),
            prop="text|revid|categories",
            disableeditsection="1",
            disabletoc="1",
        )["parse"]
        return ParsedPage(
            pageid=data["pageid"],
            title=data["title"],
            revid=data["revid"],
            html=data["text"],
            categories=[
                c["category"].replace("_", " ")
                for c in data.get("categories", [])
                if not c.get("hidden")
            ],
        )
