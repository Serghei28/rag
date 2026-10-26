from rag.config import AppConfig
from rag.generation import Generator, LLMResponse
from rag.generation.prompts import build_messages, is_refusal, parse_citations
from rag.retrieval import RetrievalResult
from tests.test_retrieval import cand

REFUSAL = AppConfig().generation.refusal


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, metadata_filter=None, **overrides):
        return RetrievalResult(query=query, collection="fake", chunks=self.chunks, candidates=[])


class FakeLLM:
    model = "fake"
    parameters: dict = {}

    def __init__(self, text: str):
        self.text = text
        self.calls: list[list[dict]] = []

    def chat(self, messages):
        self.calls.append(messages)
        return LLMResponse(text=self.text)


def generator(llm_text: str, n_chunks: int = 2, **generation):
    config = AppConfig()
    config.generation = config.generation.model_copy(update=generation)
    llm = FakeLLM(llm_text)
    chunks = [cand(i, 0.9) for i in range(1, n_chunks + 1)]
    return Generator(config, retriever=FakeRetriever(chunks), llm=llm), llm


def test_parse_citations_order_and_groups():
    assert parse_citations("Майя Санду [2]. Избрана в 2020 [1, 2], [3]") == [2, 1, 3]
    assert parse_citations("без ссылок") == []


def test_build_messages_numbers_sources():
    messages = build_messages("Кто президент?", [cand(1, 0.9), cand(2, 0.8)], REFUSAL)
    assert REFUSAL in messages[0]["content"]
    user = messages[1]["content"]
    assert "[1] Статья 1 — Введение" in user and "[2] Статья 2" in user
    assert user.endswith("Вопрос: Кто президент?")


def test_is_refusal():
    assert is_refusal(REFUSAL)
    assert not is_refusal("Президент Молдовы — Майя Санду [1].")


def test_no_context_skips_llm():
    gen, llm = generator("не должен вызываться", n_chunks=0)
    answer = gen.answer("Как приготовить мамалыгу?")
    assert answer.status == "no_context" and answer.text == REFUSAL
    assert llm.calls == []


def test_answer_with_citations():
    gen, llm = generator("Президент — Майя Санду [2].")
    answer = gen.answer("Кто президент?")
    assert answer.status == "answered" and answer.citations == [2]
    assert [number for number, _ in answer.cited_sources] == [2]
    assert answer.cited_sources[0][1].chunk.title == "Статья 2"
    assert len(llm.calls) == 1 and "llm" in answer.timings_ms


def test_llm_refusal():
    answer = generator(REFUSAL)[0].answer("Сколько стоит паспорт?")
    assert answer.status == "refused" and answer.text == REFUSAL and answer.citations == []


def test_answer_without_citations_is_replaced():
    answer = generator("Президент — Майя Санду.")[0].answer("Кто президент?")
    assert answer.status == "unsupported" and answer.text == REFUSAL
    assert answer.llm_text == "Президент — Майя Санду."


def test_answer_without_citations_allowed():
    answer = generator("Майя Санду.", require_citations=False)[0].answer("Кто президент?")
    assert answer.status == "answered" and answer.text == "Майя Санду."


def test_out_of_range_citation_ignored():
    answer = generator("Майя Санду [7].")[0].answer("Кто президент?")
    assert answer.status == "unsupported"
