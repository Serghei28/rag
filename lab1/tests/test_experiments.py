import csv

from rag.config import AppConfig, ChunkingStrategy
from rag.evaluation.experiments import (
    SUITES,
    Suite,
    Variant,
    deep_merge,
    flatten,
    patched_config,
    write_suite_report,
)
from rag.generation.prompts import PROMPTS, build_messages


def test_deep_merge_keeps_untouched_keys():
    base = {"generation": {"model": "a", "temperature": 0.0}, "x": 1}
    out = deep_merge(base, {"generation": {"model": "b"}})
    assert out == {"generation": {"model": "b", "temperature": 0.0}, "x": 1}
    assert base["generation"]["model"] == "a"  # исходный словарь не изменён


def test_patched_config_applies_patch_and_extra_chunking():
    base = AppConfig(chunking=[ChunkingStrategy(name="s", method="section")])
    extra = [
        ChunkingStrategy(name="fixed_400", method="fixed", size=400),
        ChunkingStrategy(name="s", method="section"),
    ]  # дубликат не добавляется
    cfg = patched_config(base, {"generation": {"prompt": "v2"}}, extra)
    assert cfg.generation.prompt == "v2"
    assert [s.name for s in cfg.chunking] == ["s", "fixed_400"]
    assert base.generation.prompt == "v1"


def test_all_suites_are_consistent():
    for suite in SUITES.values():
        names = [v.name for v in suite.variants]
        assert len(names) == len(set(names)), suite.name
        if suite.x_label:
            assert all(v.x is not None for v in suite.variants), suite.name
        if suite.generate:
            assert all(m in ("correctness", "faithfulness") for m in suite.plot_metrics)


def test_prompt_versions():
    assert set(PROMPTS) == {"v1", "v2"}
    v2 = build_messages("Вопрос?", [], "ОТКАЗ", "v2")[0]["content"]
    assert "ОТКАЗ" in v2 and "действующее" in v2


SUMMARY = {
    "overall": {
        "retrieval": {
            "hit@1": 0.7,
            "hit@5": 0.97,
            "mrr": 0.81,
            "recall@5": 0.8,
            "precision@5": 0.5,
        },
        "answerable_rejected_by_retrieval": 0.03,
        "no_answer_rejected_by_retrieval": 1.0,
        "generation": {"correctness": 0.83, "faithfulness": 0.95},
    },
    "by_type": {"multi_doc": {"retrieval": {"recall@5": 0.61}}},
    "latency_ms": {"retrieval_total": 350.0, "llm": 3700.0},
}


def test_flatten():
    flat = flatten(SUMMARY)
    assert flat["mrr"] == 0.81 and flat["multi_doc_recall@5"] == 0.61
    assert flat["no_answer_rejected"] == 1.0 and flat["correctness"] == 0.83
    assert flat["llm_ms"] == 3700.0


def test_write_suite_report(tmp_path):
    suite = Suite(
        "demo",
        "Демо",
        "Работает ли отчёт?",
        [Variant("a", "вариант A", x=1), Variant("b", "вариант B", x=2)],
        generate=True,
        plot_metrics=["correctness"],
        x_label="x",
    )
    rows = [(v, {"summary": SUMMARY}) for v in suite.variants]
    write_suite_report(tmp_path, suite, rows)

    md = (tmp_path / "demo.md").read_text(encoding="utf-8")
    assert "| вариант A |" in md and "Correctness" in md and "0.830" in md
    with (tmp_path / "demo.csv").open(encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    assert [r["variant"] for r in table] == ["a", "b"] and table[0]["mrr"] == "0.81"
    svg = (tmp_path / "demo.svg").read_text(encoding="utf-8")
    assert svg.startswith("<svg") and "polyline" in svg and "Демо" in svg
