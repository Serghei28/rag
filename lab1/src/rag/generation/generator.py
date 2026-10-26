"""Generator: retrieval → промпт → LLM → проверка ответа.

Защита от галлюцинаций:
1. пустой результат retrieval не вызывает LLM;
2. промпт ограничивает ответ найденными источниками;
3. ответ без ссылок на реальные источники заменяется отказом.
"""

import time
from dataclasses import dataclass, field
from typing import Literal

from rag.config import AppConfig, get_settings
from rag.generation.llm import OllamaClient
from rag.generation.prompts import build_messages, is_refusal, parse_citations
from rag.retrieval import Candidate, MetadataFilter, RetrievalResult, Retriever
from rag.tracing import current_trace_id, observe

Status = Literal["answered", "no_context", "refused", "unsupported"]

STATUS_LABELS: dict[str, str] = {
    "answered": "ответ со ссылками на источники",
    "no_context": "retriever ничего не нашёл, LLM не вызывалась",
    "refused": "LLM ответила, что информации недостаточно",
    "unsupported": "ответ LLM без ссылок на источники заменён отказом",
}


@dataclass
class Answer:
    query: str
    text: str
    status: Status
    sources: list[Candidate]
    citations: list[int]
    retrieval: RetrievalResult
    llm_text: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    trace_id: str | None = None  # id трейса в Langfuse: к нему прикрепляются оценки

    @property
    def cited_sources(self) -> list[tuple[int, Candidate]]:
        return [(number, self.sources[number - 1]) for number in self.citations]


class Generator:
    def __init__(
        self,
        config: AppConfig,
        *,
        retriever: Retriever | None = None,
        llm: OllamaClient | None = None,
        **retriever_kwargs,
    ):
        self.config = config
        self.retriever = retriever or Retriever(config, **retriever_kwargs)
        self.llm = llm or OllamaClient(get_settings().ollama_host, config.generation)

    def answer(
        self, query: str, metadata_filter: MetadataFilter | None = None, **overrides
    ) -> Answer:
        generation = self.config.generation
        with observe("rag-answer", as_type="chain", input=query) as trace:
            with observe("retrieval", as_type="retriever", input=query) as span:
                result = self.retriever.retrieve(query, metadata_filter, **overrides)
                if span:
                    span.update(
                        output=[_source_info(candidate) for candidate in result.chunks],
                        metadata={
                            "collection": result.collection,
                            "reranker": result.reranker,
                            "timings_ms": result.timings_ms,
                            "dropped": {
                                reason: len(items) for reason, items in result.dropped.items()
                            },
                        },
                    )

            if result.is_empty:
                answer = Answer(query, generation.refusal, "no_context", [], [], result)
            else:
                answer = self._generate(query, result)

            answer.timings_ms = {**result.timings_ms, **answer.timings_ms}
            answer.trace_id = current_trace_id()
            if trace:
                trace.update(
                    output=answer.text,
                    metadata={"status": answer.status, "citations": answer.citations},
                )
        return answer

    def _generate(self, query: str, result: RetrievalResult) -> Answer:
        generation = self.config.generation
        messages = build_messages(query, result.chunks, generation.refusal, generation.prompt)

        started = time.perf_counter()
        with observe(
            "llm",
            as_type="generation",
            input=messages,
            model=self.llm.model,
            model_parameters=self.llm.parameters,
        ) as span:
            response = self.llm.chat(messages)
            if span:
                span.update(
                    output=response.text,
                    usage_details={
                        "input": response.prompt_tokens,
                        "output": response.completion_tokens,
                    },
                )
        timings = {"llm": round((time.perf_counter() - started) * 1000, 1)}

        citations = [
            number for number in parse_citations(response.text) if 1 <= number <= len(result.chunks)
        ]
        if is_refusal(response.text) and not citations:
            status, text = "refused", generation.refusal
        elif not citations and generation.require_citations:
            status, text = "unsupported", generation.refusal
        else:
            status, text = "answered", response.text

        return Answer(
            query=query,
            text=text,
            status=status,
            sources=result.chunks,
            citations=citations if status == "answered" else [],
            retrieval=result,
            llm_text=response.text,
            timings_ms=timings,
        )


def _source_info(candidate: Candidate) -> dict:
    return {
        "title": candidate.chunk.title,
        "section": candidate.chunk.section,
        "vector_score": round(candidate.vector_score, 3),
        "rerank_score": (
            None if candidate.rerank_score is None else round(candidate.rerank_score, 3)
        ),
    }
