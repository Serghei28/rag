from datetime import UTC, datetime

import pytest

from rag.config import AppConfig, CategorySeed, GrabberConfig, PathsConfig
from rag.grabber import Grabber, RawStore
from rag.grabber.wikipedia import PageRef, ParsedPage, Revision, WikiApiError

TS = datetime(2026, 1, 1, tzinfo=UTC)


class FakeClient:
    """Подменяет WikipediaClient: статьи живут в словаре pageid → (title, revid, html)."""

    def __init__(self, pages: dict[int, tuple[str, int, str]]):
        self.pages = pages
        self.parsed: list[int] = []
        self.broken: set[int] = set()

    def resolve_titles(self, titles):
        return []

    def iter_category(self, name, depth, seed=None):
        for pid, (title, _, _) in self.pages.items():
            yield PageRef(pid, title, f"Категория:{name}")

    def revisions(self, pageids):
        return {
            pid: Revision(pid, self.pages[pid][0], self.pages[pid][1], TS)
            for pid in pageids
            if pid in self.pages
        }

    def parse(self, pageid):
        if pageid in self.broken:
            raise WikiApiError("boom")
        self.parsed.append(pageid)
        title, revid, html = self.pages[pageid]
        return ParsedPage(pageid, title, revid, html, ["Право Молдовы"])


@pytest.fixture
def make_grabber(tmp_path):
    def make(client, **grabber_kwargs):
        cfg = AppConfig(
            paths=PathsConfig().resolve(tmp_path),
            grabber=GrabberConfig(categories=[CategorySeed(name="Тест")], **grabber_kwargs),
        )
        return Grabber(cfg, client=client, store=RawStore(cfg.paths.raw_dir))

    return make


def test_first_run_downloads_everything(make_grabber):
    client = FakeClient({1: ("А", 10, "<p>a</p>"), 2: ("Б", 20, "<p>b</p>")})
    result = make_grabber(client).run()

    assert sorted(result.new) == ["ruwiki:1", "ruwiki:2"]
    manifest = RawStore(make_grabber(client).store.root).load_manifest()
    rec = manifest.documents["ruwiki:1"]
    assert rec.title == "А" and rec.revid == 10 and rec.url.endswith("/wiki/А")
    assert rec.categories == ["Право Молдовы"]


def test_second_run_skips_unchanged_and_fetches_changed(make_grabber):
    client = FakeClient({1: ("А", 10, "<p>a</p>"), 2: ("Б", 20, "<p>b</p>")})
    make_grabber(client).run()
    client.parsed.clear()

    client.pages[2] = ("Б", 21, "<p>b v2</p>")  # статью Б отредактировали
    result = make_grabber(client).run()

    assert result.unchanged == ["ruwiki:1"]
    assert result.updated == ["ruwiki:2"]
    assert client.parsed == [2]  # скачана только изменённая


def test_page_dropped_from_selection_is_marked_removed(make_grabber):
    client = FakeClient({1: ("А", 10, "<p>a</p>"), 2: ("Б", 20, "<p>b</p>")})
    grabber = make_grabber(client)
    grabber.run()

    del client.pages[2]
    result = make_grabber(client).run()

    assert result.removed == ["ruwiki:2"]
    assert grabber.store.load_manifest().documents["ruwiki:2"].status == "removed"


def test_error_on_one_page_does_not_stop_run(make_grabber):
    client = FakeClient({1: ("А", 10, "<p>a</p>"), 2: ("Б", 20, "<p>b</p>")})
    client.broken.add(1)
    result = make_grabber(client).run()

    assert result.new == ["ruwiki:2"]
    assert "ruwiki:1" in result.failed


def test_identical_content_is_not_stored_twice(make_grabber):
    client = FakeClient({1: ("А", 10, "<p>same</p>"), 2: ("А (копия)", 20, "<p>same</p>")})
    result = make_grabber(client).run()

    assert result.new == ["ruwiki:1"]
    assert result.duplicates == ["ruwiki:2"]


def test_missing_file_is_redownloaded(make_grabber):
    client = FakeClient({1: ("А", 10, "<p>a</p>")})
    grabber = make_grabber(client)
    grabber.run()
    (grabber.store.root / "wikipedia" / "1.html").unlink()

    result = make_grabber(client).run()
    assert result.updated == ["ruwiki:1"]


def test_exclude_titles_and_max_pages(make_grabber):
    client = FakeClient({1: ("А", 10, "a"), 2: ("Чемпионат", 20, "b"), 3: ("В", 30, "c")})
    result = make_grabber(client, exclude_titles=["чемпионат"], max_pages=1).run()

    assert result.selected == 1
    assert result.new == ["ruwiki:1"]


def test_dry_run_writes_nothing(make_grabber):
    client = FakeClient({1: ("А", 10, "<p>a</p>")})
    grabber = make_grabber(client)
    result = grabber.run(dry_run=True)

    assert result.new == ["ruwiki:1"]
    assert client.parsed == []
    assert not grabber.store.manifest_path.exists()
