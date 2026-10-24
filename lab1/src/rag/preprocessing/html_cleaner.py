"""HTML статьи Википедии → список секций с текстовыми блоками.

Что удаляется (выбрано по частоте элементов в корпусе, см. README):
- навигация: navbox (520 шт.), hatnote «Не следует путать с…» (283), служебные плашки mbox;
- сноски `<sup class="reference">` и списки литературы reflist (271);
- изображения, галереи, подписи, координаты, стили;
- служебные секции: «Примечания», «Ссылки», «Литература» и т.д. (настраивается).

Что сохраняется в преобразованном виде:
- карточка (infobox, 273 шт.) → строки «Ключ: значение» — это плотные факты
  (столица, президент, население), на которые хорошо отвечать;
- таблицы (wikitable, 240 шт.) → строки «ячейка | ячейка» — результаты выборов, составы.
"""

import re

from bs4 import BeautifulSoup, Tag

from rag.config import PreprocessingConfig
from rag.preprocessing.models import Block, Section
from rag.preprocessing.normalize import normalize_text

REMOVE_SELECTORS = [
    "style",
    "script",
    "link",
    "sup.reference",
    "sup.noprint",
    ".mw-editsection",
    ".navbox",
    ".vertical-navbox",
    ".hatnote",
    ".metadata",
    ".ambox",
    "[class*='mbox-']",
    ".reflist",
    "ol.references",
    ".noprint",
    "figure",
    ".gallery",
    ".thumb",
    ".mw-empty-elt",
    "#coordinates",
    "[class*='Родственные_проекты']",
    ".sistersitebox",
]
HEADING = re.compile(r"^h([2-6])$")
INFOBOX_HEADING = "Основные сведения"


def _text(el: Tag, sep: str = "") -> str:
    return normalize_text(el.get_text(sep))


def _cell_text(el: Tag) -> str:
    # Внутри ячеек много <br> и вложенных тегов: разделяем пробелом, а не склеиваем
    return normalize_text(el.get_text(" ")).replace("\n", " ")


def infobox_to_text(table: Tag) -> str:
    """Строки «Ключ: значение». Заголовки групп («Население») становятся префиксом
    для следующих строк с маркером «•»: «Население — Оценка (2025): 2 381 300 чел.»"""
    lines, group = [], None
    for tr in table.select("tr"):
        th, td = tr.find("th", recursive=False), tr.find("td", recursive=False)
        cells = tr.find_all(["th", "td"], recursive=False)
        if th and td:
            key, value = _cell_text(th), _cell_text(td)
            if not value:
                continue
            if key.startswith("•") and group:
                key = f"{group} — {key.lstrip('• ').strip()}"
            lines.append(f"{key}: {value}")
        elif len(cells) == 1:
            text = _cell_text(cells[0])
            if text and len(text) < 60:  # короткая строка без пары — заголовок группы
                group = text
    return "\n".join(lines)


def table_to_text(table: Tag) -> str:
    rows = []
    for tr in table.select("tr"):
        cells = [_cell_text(c) for c in tr.find_all(["th", "td"], recursive=False)]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def list_to_text(lst: Tag) -> str:
    items = []
    for li in lst.find_all(["li", "dd", "dt"], recursive=False):
        text = _text(li, " ").replace("\n", " ")
        if text:
            items.append(f"- {text}")
    return "\n".join(items)


def clean_html(html: str, cfg: PreprocessingConfig) -> list[Section]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.select_one("div.mw-parser-output") or soup

    infobox_blocks = []
    for table in root.select("table.infobox"):
        if cfg.keep_infobox and (text := infobox_to_text(table)):
            infobox_blocks.append(Block(kind="infobox", text=text))
        table.decompose()
    for el in root.select(", ".join(REMOVE_SELECTORS)):
        el.decompose()
    for br in root.find_all("br"):
        br.replace_with("\n")

    drop = {s.lower() for s in cfg.drop_sections}
    sections = [Section(heading_path=[], blocks=[])]
    skip_level: int | None = None  # внутри удаляемой секции этого уровня

    def emit(block: Block) -> None:
        if skip_level is None and block.text:
            sections[-1].blocks.append(block)

    def walk(el: Tag) -> None:
        nonlocal skip_level
        name = el.name
        if name == "div" and "mw-heading" in (el.get("class") or []):
            h = el.find(HEADING)
            if h is None:
                return
            level, title = int(h.name[1]), _text(h, " ")
            if skip_level is not None and level > skip_level:
                return  # подзаголовок внутри удаляемой секции
            skip_level = level if title.lower() in drop else None
            parent = sections[-1].heading_path[: level - 2]
            sections.append(Section(heading_path=[*parent, title]))
        elif name == "p":
            emit(Block(kind="paragraph", text=_text(el)))
        elif name in ("ul", "ol", "dl"):
            emit(Block(kind="list", text=list_to_text(el)))
        elif name == "table":
            if cfg.keep_tables:
                emit(Block(kind="table", text=table_to_text(el)))
        elif name == "blockquote" or (
            name == "div" and "Цитата" in " ".join(el.get("class") or [])
        ):
            emit(Block(kind="quote", text=_text(el, " ")))
        elif name == "div":
            for child in el.find_all(recursive=False):
                walk(child)

    for child in root.find_all(recursive=False):
        walk(child)

    if infobox_blocks:
        sections.insert(0, Section(heading_path=[INFOBOX_HEADING], blocks=infobox_blocks))
    return [s for s in sections if s.blocks]
