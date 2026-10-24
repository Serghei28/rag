"""Обёртка над sentence-transformers: одинаковый интерфейс для разных embedding-моделей.

Различия моделей (префиксы, длина контекста) задаются в конфиге, а не в коде,
поэтому новая модель для эксперимента добавляется одной записью в YAML.
"""

import logging
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging

from rag.config import EmbeddingModel
from rag.preprocessing.models import Chunk

log = logging.getLogger(__name__)
hf_logging.disable_progress_bar()  # прогресс-бар «Loading weights» при каждой загрузке модели


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def passage_text(chunk: Chunk, with_header: bool) -> str:
    """Текст chunk для векторизации. Заголовок даёт контекст коротким фрагментам:
    «Президент: Майя Санду» без названия статьи неясно, о какой стране речь."""
    if not with_header:
        return chunk.text
    return f"{chunk.title}. {chunk.section}\n{chunk.text}"


class Encoder:
    def __init__(self, spec: EmbeddingModel, device: str = "auto", batch_size: int = 32):
        self.spec = spec
        self.batch_size = batch_size
        self.device = resolve_device(device)
        log.info("загрузка %s на %s", spec.model_id, self.device)
        dtype = torch.float16 if spec.fp16 and self.device == "cuda" else torch.float32
        self.model = SentenceTransformer(
            spec.model_id, device=self.device, model_kwargs={"torch_dtype": dtype}
        )
        self.model.max_seq_length = spec.max_seq_length
        self._encode(["прогрев"])  # инициализация CUDA-ядер, чтобы не попала в замеры времени

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def _encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        # Векторы единичной длины: cosine similarity = скалярное произведение
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vectors.astype(np.float32)

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self.spec.passage_prefix + t for t in texts], show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([self.spec.query_prefix + query])[0]

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])


@lru_cache(maxsize=2)
def get_encoder(spec: EmbeddingModel, device: str = "auto", batch_size: int = 32) -> Encoder:
    """Модель грузится секунды и занимает видеопамять — держим загруженной."""
    return Encoder(spec, device, batch_size)
