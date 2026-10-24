"""Нормализация текста.

Каждое правило убирает конкретную проблему, найденную в корпусе:
- знаки ударения (`Молдо́ва`): слово с ударением и без — разные строки для поиска;
- маркеры сносок (`[18]`, `[a]`, `[источник не указан 123 дня]`) — шум в embeddings;
- неразрывные и «тонкие» пробелы, мягкие переносы, zero-width символы;
- пробелы перед знаками препинания, которые появляются при склейке HTML-элементов.
"""

import re
import unicodedata

STRESS = "́"  # combining acute accent
INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))
SPACES = re.compile(r"[  -   　\t]")
FOOTNOTE = re.compile(
    r"\[\s*(?:\d{1,3}|[a-zа-я]|прим\.?\s*\d*|источник не указан[^\]]*|нет в источнике|"
    r"уточнить|неавторитетный источник|en|ro|англ\.|рум\.)\s*\]",
    re.I,
)
TREND_ARROWS = re.compile(r"[↘↗↑↓▲▼➘➚]\s*")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?)\]»%])")
SPACE_AFTER_OPEN = re.compile(r"([(\[«])\s+")
MULTI_SPACE = re.compile(r" {2,}")
MULTI_NEWLINE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace(STRESS, "").translate(INVISIBLE)
    text = SPACES.sub(" ", text)
    text = FOOTNOTE.sub("", text)
    text = TREND_ARROWS.sub("", text)
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = SPACE_AFTER_OPEN.sub(r"\1", text)
    text = MULTI_SPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()
