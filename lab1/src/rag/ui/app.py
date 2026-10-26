"""Streamlit-интерфейс для проверки retrieval в RAG."""

import streamlit as st

from rag.config import AppConfig, load_config
from rag.generation import STATUS_LABELS, Generator
from rag.retrieval import Retriever


@st.cache_resource
def get_config() -> AppConfig:
    return load_config()


@st.cache_resource
def get_retriever(
    embedding: str | None,
    chunking: str | None,
    reranker: str | None,
) -> Retriever:
    return Retriever(
        get_config(),
        embedding=embedding or None,
        chunking=chunking or None,
        reranker=reranker or None,
    )


@st.cache_resource
def get_generator(
    embedding: str | None,
    chunking: str | None,
    reranker: str | None,
) -> Generator:
    return Generator(
        get_config(),
        retriever=get_retriever(embedding, chunking, reranker),
    )


def _option_names(config: AppConfig, kind: str) -> list[str]:
    if kind == "embedding":
        return [model.name for model in config.embeddings.models]
    if kind == "chunking":
        return [strategy.name for strategy in config.chunking]
    return [model.name for model in config.reranking.models]


def _default_option(options: list[str], active: str) -> str:
    return active if active in options else (options[0] if options else "")


def _render_result(result) -> None:
    if result.is_empty:
        st.warning("Ничего релевантного не найдено")

    if result.chunks:
        st.subheader(f"Найдено фрагментов: {len(result.chunks)}")
        for rank, candidate in enumerate(result.chunks, start=1):
            chunk = candidate.chunk
            with st.expander(
                f"{rank}. {chunk.title}" + (f" — {chunk.section}" if chunk.section else ""),
                expanded=rank == 1,
            ):
                st.markdown(f"[{chunk.title}]({chunk.url})")
                rerank_score = (
                    f"{candidate.rerank_score:.3f}" if candidate.rerank_score is not None else "—"
                )
                st.caption(
                    f"rerank score: {rerank_score} · vector score: {candidate.vector_score:.3f} "
                    f"· место: {candidate.vector_rank} → {rank}"
                )
                st.write(chunk.text)

    with st.expander("Отброшено фильтрами"):
        if result.dropped:
            for reason, dropped in result.dropped.items():
                st.write(f"{reason}: {len(dropped)}")
        else:
            st.write("Нет отброшенных chunks")

    with st.expander("Время выполнения"):
        for step, duration in result.timings_ms.items():
            st.write(f"{step}: {duration} мс")


def main() -> None:
    st.set_page_config(page_title="RAG Молдова", page_icon="📚", layout="wide")
    st.title("RAG по государству и праву Молдовы")
    st.write("Задайте вопрос, чтобы увидеть найденные фрагменты корпуса и оценки retrieval.")

    config = get_config()
    embedding_options = _option_names(config, "embedding")
    chunking_options = _option_names(config, "chunking")
    reranker_options = _option_names(config, "reranker")

    with st.sidebar:
        st.header("Параметры поиска")
        embedding = st.selectbox(
            "Embedding-модель",
            embedding_options,
            index=embedding_options.index(
                _default_option(embedding_options, config.active.embedding)
            )
            if embedding_options
            else None,
        )
        chunking = st.selectbox(
            "Стратегия chunking",
            chunking_options,
            index=chunking_options.index(_default_option(chunking_options, config.active.chunking))
            if chunking_options
            else None,
        )
        rerank_enabled = st.checkbox("Использовать reranker", value=config.reranking.enabled)
        reranker = st.selectbox(
            "Reranker-модель",
            reranker_options,
            index=(
                reranker_options.index(config.active.reranker)
                if config.active.reranker in reranker_options
                else 0
            ),
            disabled=not rerank_enabled,
        )
        top_k = st.slider("top_k", 5, 50, 20)
        top_n = st.slider("top_n", 1, 10, 5)

    query = st.text_area(
        "Вопрос",
        placeholder="Например: Сколько депутатов в Парламенте Молдовы?",
        height=90,
    )
    submitted = st.button("Найти", type="primary", use_container_width=True)

    if not submitted:
        return
    if not query.strip():
        st.error("Введите вопрос.")
        return

    try:
        selected_reranker = reranker if rerank_enabled else None
        generator = get_generator(embedding, chunking, selected_reranker)
        with st.spinner("Ищу релевантные фрагменты…"):
            answer = generator.answer(
                query.strip(),
                top_k=top_k,
                top_n=top_n,
                rerank=rerank_enabled,
            )
    except LookupError as exc:
        st.error(str(exc))
        return
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Не удалось выполнить поиск: {exc}")
        return

    st.subheader("Ответ")
    if answer.status == "answered":
        st.success(answer.text)
    else:
        st.warning(answer.text)
    st.caption(f"{answer.status}: {STATUS_LABELS[answer.status]}")
    for number, candidate in answer.cited_sources:
        chunk = candidate.chunk
        st.markdown(f"**[{number}]** [{chunk.title}]({chunk.url}) · {chunk.section}")

    _render_result(answer.retrieval)


if __name__ == "__main__":
    main()
