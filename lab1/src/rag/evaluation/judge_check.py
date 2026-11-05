"""Проверка LLM-судьи на ручной разметке.

LLM-as-a-Judge — тоже модель, и она ошибается. Прежде чем доверять её оценкам в
экспериментах, сравниваем их с ручной разметкой тех же ответов
(experiments/judge_validation/labels.yaml):
- agreement — доля ответов, где оценка судьи совпала с ручной;
- MAE — средняя абсолютная разница оценок (0 — идеально, 1 — максимум).

Ответы берутся из готового прогона (results.jsonl), поэтому разные версии промпта
судьи сравниваются на одних и тех же ответах, без повторной генерации.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.evaluation.judge import LLMJudge
from rag.evaluation.metrics import mean


@dataclass
class Disagreement:
    id: str
    metric: str
    manual: float
    judge: float | None
    reason: str


@dataclass
class JudgeCheck:
    version: str
    agreement: dict[str, float | None] = field(default_factory=dict)
    mae: dict[str, float | None] = field(default_factory=dict)
    disagreements: list[Disagreement] = field(default_factory=list)


def load_labels(path: Path) -> tuple[str, dict[str, dict]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw["run"], {item["id"]: item for item in raw["labels"]}


def load_results(run_dir: Path) -> dict[str, dict]:
    with (run_dir / "results.jsonl").open(encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def check_judge(judge: LLMJudge, results: dict[str, dict], labels: dict[str, dict]) -> JudgeCheck:
    out = JudgeCheck(version=judge.prompt_version)
    pairs: dict[str, list[tuple[float, float | None]]] = {"correctness": [], "faithfulness": []}

    for qid, label in labels.items():
        r = results.get(qid)
        if r is None or r["status"] != "answered":
            continue  # отказы оцениваются правилом, а не судьёй
        if label.get("answer") and label["answer"] != r["answer"]:
            raise ValueError(f"{qid}: ответ в прогоне отличается от размеченного")
        verdicts = {
            "correctness": judge.correctness(r["question"], r["expected_answer"], r["answer"]),
            "faithfulness": judge.faithfulness(r["question"], r["answer"], r["sources_text"]),
        }
        for metric, verdict in verdicts.items():
            manual = label.get(metric)
            if manual is None:
                continue
            pairs[metric].append((manual, verdict.score))
            if verdict.score != manual:
                out.disagreements.append(
                    Disagreement(qid, metric, manual, verdict.score, verdict.reason)
                )

    for metric, items in pairs.items():
        out.agreement[metric] = mean([float(m == j) for m, j in items])
        out.mae[metric] = mean([abs(m - j) for m, j in items if j is not None])
    return out


def render_report(run: str, checks: list[JudgeCheck], n_labels: int) -> str:
    lines = [
        "# Проверка LLM-судьи на ручной разметке",
        "",
        f"Ответы прогона `{run}`, размечено вручную: {n_labels} ответов.",
        "",
        "| Промпт судьи | Согласие correctness | MAE correctness | Согласие faithfulness "
        "| MAE faithfulness |",
        "|---|---|---|---|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c.version} | {_pct(c.agreement['correctness'])} | {_num(c.mae['correctness'])} "
            f"| {_pct(c.agreement['faithfulness'])} | {_num(c.mae['faithfulness'])} |"
        )
    for c in checks:
        lines += ["", f"## Расхождения, промпт {c.version}", ""]
        if not c.disagreements:
            lines.append("Нет.")
            continue
        lines += ["| id | метрика | вручную | судья | причина судьи |", "|---|---|---|---|---|"]
        for d in c.disagreements:
            reason = d.reason.replace("\n", " ").replace("|", "/")[:150]
            lines.append(f"| {d.id} | {d.metric} | {d.manual} | {_num(d.judge)} | {reason} |")
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"
