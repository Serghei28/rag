from datetime import UTC, datetime

import pytest

from rag.config import ChunkingStrategy, PreprocessingConfig
from rag.preprocessing import Block, CleanDocument, Section, chunk_document, clean_html
from rag.preprocessing.chunking import split_long
from rag.preprocessing.normalize import normalize_text

# ------------------------------------------------------------ normalize ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Молдо́ва", "Молдова"),  # ударение
        ("Кишинёв[12], Бельцы[a]", "Кишинёв, Бельцы"),  # сноски
        ("факт[источник не указан 123 дня].", "факт."),
        ("33 846 км²", "33 846 км²"),  # неразрывные пробелы
        ("Бельцы , Тирасполь ( город )", "Бельцы, Тирасполь (город)"),
        ("↘ 2 381 300 чел.", "2 381 300 чел."),
        ("а\n\n\n\nб", "а\n\nб"),
        ("пере­нос", "перенос"),  # мягкий перенос
    ],
)
def test_normalize(raw, expected):
    assert normalize_text(raw) == expected


# ---------------------------------------------------------- html_cleaner ---

HTML = """
<div class="mw-parser-output">
  <table class="infobox">
    <tr><th>Столица</th><td>Кишинёв<sup class="reference">[1]</sup></td></tr>
    <tr><td colspan="2">Население</td></tr>
    <tr><th>• Перепись (2014)</th><td>2 913 281 чел.</td></tr>
  </table>
  <div class="hatnote">Не следует путать с Молдавией (регион).</div>
  <p>Вводный <b>абзац</b> о стране.<sup class="reference">[2]</sup></p>
  <div class="mw-heading mw-heading2"><h2>История</h2><span class="mw-editsection">править</span></div>
  <p>Абзац истории.</p>
  <div class="mw-heading mw-heading3"><h3>Независимость</h3></div>
  <ul><li>пункт один</li><li>пункт два</li></ul>
  <table class="wikitable"><tr><th>Партия</th><th>%</th></tr><tr><td>ПДС</td><td>52,8</td></tr></table>
  <figure><figcaption>Подпись картинки</figcaption></figure>
  <div class="navbox">Навигация по странам</div>
  <div class="mw-heading mw-heading2"><h2>Примечания</h2></div>
  <div class="reflist"><ol class="references"><li>Источник 1</li></ol></div>
  <div class="mw-heading mw-heading3"><h3>Подсекция примечаний</h3></div>
  <p>Тоже служебный текст.</p>
  <div class="mw-heading mw-heading2"><h2>География</h2></div>
  <p>Абзац географии.</p>
</div>
"""


def test_clean_html_structure():
    sections = clean_html(HTML, PreprocessingConfig())
    assert [s.heading for s in sections] == [
        "Основные сведения",
        "Введение",
        "История",
        "История > Независимость",
        "География",
    ]


def test_clean_html_removes_junk_and_keeps_content():
    sections = clean_html(HTML, PreprocessingConfig())
    full = "\n".join(s.text for s in sections)
    for junk in [
        "[1]",
        "[2]",
        "править",
        "Не следует путать",
        "Подпись",
        "Навигация",
        "Источник 1",
        "служебный",
    ]:
        assert junk not in full
    assert "Вводный абзац о стране." in full


def test_infobox_and_table_to_text():
    sections = clean_html(HTML, PreprocessingConfig())
    infobox = sections[0].blocks[0]
    assert infobox.kind == "infobox"
    assert infobox.text == "Столица: Кишинёв\nНаселение — Перепись (2014): 2 913 281 чел."

    kinds = [b.kind for b in sections[3].blocks]
    assert kinds == ["list", "table"]
    assert sections[3].blocks[0].text == "- пункт один\n- пункт два"
    assert sections[3].blocks[1].text == "Партия | %\nПДС | 52,8"


def test_infobox_and_tables_can_be_disabled():
    cfg = PreprocessingConfig(keep_infobox=False, keep_tables=False)
    kinds = {b.kind for s in clean_html(HTML, cfg) for b in s.blocks}
    assert kinds == {"paragraph", "list"}


# -------------------------------------------------------------- chunking ---

WORDS = " ".join(f"слово{i}" for i in range(400))  # ~3000 символов


def make_doc(*sections: tuple[list[str], list[str]]) -> CleanDocument:
    return CleanDocument(
        document_id="ruwiki:1",
        source="test",
        title="Тест",
        url="https://example.org",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        sections=[
            Section(heading_path=path, blocks=[Block(kind="paragraph", text=t) for t in texts])
            for path, texts in sections
        ],
    )


def test_fixed_respects_size_and_word_boundaries():
    doc = make_doc(([], [WORDS]))
    chunks = chunk_document(doc, ChunkingStrategy(name="f", method="fixed", size=300))
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.text.startswith("слово") and c.text[-1].isdigit() for c in chunks)
    # без overlap текст восстанавливается полностью
    assert " ".join(c.text for c in chunks) == WORDS


def test_fixed_overlap_repeats_tail_of_previous_chunk():
    doc = make_doc(([], [WORDS]))
    s = ChunkingStrategy(name="f", method="fixed", size=300, overlap=100)
    chunks = chunk_document(doc, s)
    for a, b in zip(chunks, chunks[1:], strict=False):
        first_word = b.text.split()[0]
        assert first_word in a.text.split()[-20:]  # следующий начинается внутри хвоста
    no_overlap = chunk_document(doc, ChunkingStrategy(name="f", method="fixed", size=300))
    assert len(chunks) > len(no_overlap)


def test_overlap_must_be_smaller_than_size():
    doc = make_doc(([], [WORDS]))
    with pytest.raises(ValueError):
        chunk_document(doc, ChunkingStrategy(name="f", method="fixed", size=100, overlap=100))


def test_paragraph_keeps_paragraphs_whole_and_may_cross_sections():
    p1, p2, p3 = "А" * 100, "Б" * 100, "В" * 100
    doc = make_doc((["Один"], [p1, p2]), (["Два"], [p3]))
    chunks = chunk_document(doc, ChunkingStrategy(name="p", method="paragraph", size=350))
    assert [c.text for c in chunks] == [f"{p1}\n\n{p2}\n\n{p3}"]
    assert chunks[0].section == "Один"


def test_section_never_crosses_sections():
    p1, p2, p3 = "А" * 100, "Б" * 100, "В" * 100
    doc = make_doc((["Один"], [p1, p2]), (["Два"], [p3]))
    chunks = chunk_document(doc, ChunkingStrategy(name="s", method="section", size=350))
    assert [(c.section, c.text) for c in chunks] == [("Один", f"{p1}\n\n{p2}"), ("Два", p3)]


def test_long_paragraph_is_split_by_sentences():
    text = " ".join(f"Предложение номер {i} о государстве." for i in range(40))
    parts = split_long(text, 200)
    assert all(len(p) <= 200 for p in parts)
    assert all(p.endswith(".") for p in parts)
    assert " ".join(parts) == text


def test_multiline_block_is_split_by_lines_only():
    text = "\n".join(f"Население — Оценка ({y}): 2 381 300 чел. (142-е)" for y in range(30))
    parts = split_long(text, 300)
    lines = [line for p in parts for line in p.split("\n")]
    assert lines == text.split("\n")  # ни одна строка не разрезана


def test_short_chunks_are_dropped_and_ids_are_unique():
    doc = make_doc((["Один"], ["коротко"]), (["Два"], ["Д" * 100]))
    chunks = chunk_document(doc, ChunkingStrategy(name="s", method="section", min_chars=40))
    assert [c.section for c in chunks] == ["Два"]
    assert chunks[0].chunk_id == "ruwiki:1#s:0"
