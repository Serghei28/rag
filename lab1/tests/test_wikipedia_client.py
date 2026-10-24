import httpx
import pytest

from rag.config import GrabberConfig
from rag.grabber.wikipedia import RetryableError, WikiApiError, WikipediaClient

CFG = GrabberConfig(request_delay=0, max_retries=3, exclude_categories=["Персоналии"])


def make_client(handler) -> WikipediaClient:
    client = WikipediaClient(CFG, http=httpx.Client(transport=httpx.MockTransport(handler)))
    client._get.retry.wait = lambda _: 0  # без пауз между повторами в тестах
    return client


def test_retries_on_server_error_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"query": {"pages": []}})

    make_client(handler).revisions([1])
    assert len(calls) == 3


def test_gives_up_after_max_retries():
    client = make_client(lambda request: httpx.Response(429))
    with pytest.raises(RetryableError):
        client.revisions([1])


def test_api_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"error": {"code": "nosuchpageid", "info": "нет"}})

    with pytest.raises(WikiApiError):
        make_client(handler).parse(1)
    assert len(calls) == 1


def test_category_walk_follows_continue_and_skips_excluded_subcats():
    def handler(request):
        p = request.url.params
        cat, cont = p["cmtitle"], p.get("cmcontinue")
        if cat == "Категория:Корень" and cont is None:
            return httpx.Response(
                200,
                json={
                    "continue": {"cmcontinue": "next", "continue": "-||"},
                    "query": {
                        "categorymembers": [
                            {"ns": 0, "pageid": 1, "title": "Статья 1"},
                            {"ns": 14, "pageid": 100, "title": "Категория:Под"},
                        ]
                    },
                },
            )
        if cat == "Категория:Корень":
            return httpx.Response(
                200,
                json={
                    "query": {
                        "categorymembers": [
                            {"ns": 0, "pageid": 2, "title": "Статья 2"},
                            {"ns": 14, "pageid": 101, "title": "Категория:Персоналии:Молдова"},
                        ]
                    }
                },
            )
        if cat == "Категория:Под":
            return httpx.Response(
                200,
                json={
                    "query": {
                        "categorymembers": [
                            {"ns": 0, "pageid": 3, "title": "Статья 3"},
                        ]
                    }
                },
            )
        raise AssertionError(f"неожиданный запрос {cat}")

    refs = list(make_client(handler).iter_category("Корень", depth=1))
    assert [r.pageid for r in refs] == [1, 2, 3]
    assert {r.seed for r in refs} == {"Категория:Корень"}


def test_resolve_titles_skips_missing():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": [
                        {"pageid": 7, "title": "Молдова"},
                        {"title": "Нет такой", "missing": True},
                    ]
                }
            },
        )

    refs = list(make_client(handler).resolve_titles(["Молдавия", "Нет такой"]))
    assert [(r.pageid, r.title) for r in refs] == [(7, "Молдова")]
