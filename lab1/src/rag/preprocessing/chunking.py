"""Стратегии разбиения документов на chunks.

| method    | Единица разбиения                        | Границы секций | Структура      |
|-----------|------------------------------------------|----------------|----------------|
| fixed     | окно `size` символов (+ `overlap`)       | игнорирует     | режет по пробелу |
| paragraph | целые абзацы/списки/таблицы, до `size`   | пересекает     | абзацы целые   |
| section   | целая секция, если влезает; иначе абзацы | не пересекает  | секции целые   |

Во всех стратегиях блок длиннее `size` делится по предложениям, предложение длиннее
`size` — по пробелам. Chunks короче `min_chars` отбрасываются (обрывки, пустые заголовки).
"""

import re
from collections.abc import Iterator

from rag.config import ChunkingStrategy
from rag.preprocessing.models import Chunk, CleanDocument

SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\dA-ZА-ЯЁ])")


def split_on_space(text: str, size: int) -> list[str]:
    """Жёсткое деление по ближайшему пробелу перед границей `size`."""
    parts = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def split_long(text: str, size: int) -> list[str]:
    """Текст длиннее `size` → куски по строкам и предложениям, каждый ≤ size."""
    if len(text) <= size:
        return [text]
    if "\n" in text:  # списки, таблицы, карточка: делим по строкам, строки не дробим
        units = [u for line in text.split("\n") for u in split_long(line, size)]
        return list(pack(units, size, sep="\n"))
    units = []
    for sentence in SENTENCE_END.split(text):
        units.extend(split_on_space(sentence, size) if len(sentence) > size else [sentence])
    return list(pack(units, size, sep=" "))


def pack(units: list[str], size: int, sep: str = "\n\n") -> Iterator[str]:
    """Жадно склеивает подряд идущие куски, пока сумма не превышает `size`."""
    buf: list[str] = []
    length = 0
    for unit in units:
        extra = len(unit) + (len(sep) if buf else 0)
        if buf and length + extra > size:
            yield sep.join(buf)
            buf, length = [], 0
            extra = len(unit)
        buf.append(unit)
        length += extra
    if buf:
        yield sep.join(buf)


def _fixed(doc: CleanDocument, s: ChunkingStrategy) -> list[tuple[str, str]]:
    # Документ как сплошной текст; заголовки остаются строками внутри текста.
    pieces, offsets = [], []  # offsets: (начало секции в тексте, заголовок)
    pos = 0
    for section in doc.sections:
        part = (f"{section.heading}\n" if section.heading_path else "") + section.text
        offsets.append((pos, section.heading))
        pieces.append(part)
        pos += len(part) + 2
    text = "\n\n".join(pieces)

    def section_at(i: int) -> str:
        return next(h for start, h in reversed(offsets) if start <= i) if offsets else ""

    step = s.size - s.overlap
    if step <= 0:
        raise ValueError(f"{s.name}: overlap должен быть меньше size")

    out, start = [], 0
    while start < len(text):
        end = min(start + s.size, len(text))
        if end < len(text):  # не режем слово пополам: ищем пробел во второй половине окна
            space = text.rfind(" ", start + max(s.size // 2, s.overlap + 1), end)
            end = space if space != -1 else end
        out.append((section_at(start), text[start:end].strip()))
        if end >= len(text):
            break
        nxt = max(end - s.overlap, start + 1)
        if s.overlap:  # начало следующего окна тоже сдвигаем к границе слова
            space = text.find(" ", nxt, end)
            nxt = space + 1 if space != -1 else nxt
        start = nxt
    return out


def _paragraph(doc: CleanDocument, s: ChunkingStrategy) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = [
        (section.heading, piece)
        for section in doc.sections
        for block in section.blocks
        for piece in split_long(block.text, s.size)
    ]
    out, buf, length = [], [], 0
    for heading, piece in units:
        if buf and length + len(piece) + 2 > s.size:
            out.append((buf[0][0], "\n\n".join(p for _, p in buf)))
            buf, length = [], 0
        buf.append((heading, piece))
        length += len(piece) + 2
    if buf:
        out.append((buf[0][0], "\n\n".join(p for _, p in buf)))
    return out


def _section(doc: CleanDocument, s: ChunkingStrategy) -> list[tuple[str, str]]:
    out = []
    for section in doc.sections:
        if len(section.text) <= s.size:
            out.append((section.heading, section.text))
            continue
        units = [piece for b in section.blocks for piece in split_long(b.text, s.size)]
        out.extend((section.heading, part) for part in pack(units, s.size))
    return out


METHODS = {"fixed": _fixed, "paragraph": _paragraph, "section": _section}


def chunk_document(doc: CleanDocument, strategy: ChunkingStrategy) -> list[Chunk]:
    raw = METHODS[strategy.method](doc, strategy)
    chunks = []
    for heading, text in raw:
        if len(text) < strategy.min_chars:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}#{strategy.name}:{len(chunks)}",
                document_id=doc.document_id,
                title=doc.title,
                url=doc.url,
                section=heading,
                updated_at=doc.updated_at,
                strategy=strategy.name,
                position=len(chunks),
                text=text,
            )
        )
    return chunks
