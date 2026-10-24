"""Сгенерировать requirements.txt из pyproject.toml.

Основной способ установки — pixi (`pixi install`). requirements.txt нужен для тех,
кто ставит проект обычным pip: `pip install -r requirements.txt && pip install -e .`
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
deps = project["project"]["dependencies"] + project["dependency-groups"]["dev"]

header = "# Сгенерировано scripts/export_requirements.py из pyproject.toml — не редактировать вручную.\n"
(ROOT / "requirements.txt").write_text(header + "\n".join(deps) + "\n", encoding="utf-8")
print(f"requirements.txt: {len(deps)} зависимостей")
