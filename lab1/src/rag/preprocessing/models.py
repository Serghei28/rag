"""Структуры данных этапа preprocessing (сериализуются в JSONL)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

BlockKind = Literal["paragraph", "list", "table", "infobox", "quote"]


class Block(BaseModel):
    kind: BlockKind
    text: str


class Section(BaseModel):
    # Путь заголовков от h2 вглубь: ["История", "Принятие конституции"]. [] — вводная часть.
    heading_path: list[str] = []
    blocks: list[Block] = []

    @property
    def heading(self) -> str:
        return " > ".join(self.heading_path) or "Введение"

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


class CleanDocument(BaseModel):
    document_id: str
    source: str
    title: str
    url: str
    updated_at: datetime
    categories: list[str] = []
    sections: list[Section] = []

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


class Chunk(BaseModel):
    chunk_id: str  # "<document_id>#<strategy>:<номер>"
    document_id: str
    title: str
    url: str
    section: str  # путь заголовков, в котором начинается chunk
    updated_at: datetime
    strategy: str
    position: int  # порядковый номер chunk внутри документа
    text: str
