"""Скачать все модели из конфига (embeddings + reranker'ы) в кэш Hugging Face.

Зачем отдельный скрипт: модели весят ~6 ГБ, и при медленной сети удобнее скачать их
заранее, с повторами и докачкой, чем ждать первого `pixi run index`.

    pixi run download-models            # обычная загрузка
    pixi run download-models --no-xet   # если загрузка зависает (протокол Xet)
"""

import os
import sys
import time

if "--no-xet" in sys.argv:
    os.environ["HF_HUB_DISABLE_XET"] = "1"  # до импорта huggingface_hub

from huggingface_hub import snapshot_download

from rag.config import load_config

# Только веса и токенизатор: без ONNX/OpenVINO-копий и картинок из README
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "1_Pooling/*", "2_Normalize/*"]

cfg = load_config()
repos = [m.model_id for m in cfg.embeddings.models] + [m.model_id for m in cfg.reranking.models]

for repo in repos:
    for attempt in range(1, 6):
        try:
            started = time.time()
            path = snapshot_download(repo, allow_patterns=PATTERNS)
            print(f"ok   {repo}  ({time.time() - started:.0f} с) → {path}", flush=True)
            break
        except Exception as exc:  # сеть: повторяем, уже скачанное докачивается
            print(f"fail {repo}, попытка {attempt}: {exc}", flush=True)
            time.sleep(10 * attempt)
    else:
        sys.exit(f"не удалось скачать {repo}")
