"""Preprocessing: очистка HTML, нормализация текста и chunking документов."""

from rag.preprocessing.chunking import chunk_document
from rag.preprocessing.html_cleaner import clean_html
from rag.preprocessing.models import Block, Chunk, CleanDocument, Section
from rag.preprocessing.normalize import normalize_text

__all__ = [
    "Block",
    "Chunk",
    "CleanDocument",
    "Section",
    "chunk_document",
    "clean_html",
    "normalize_text",
]
