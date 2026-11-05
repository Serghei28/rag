"""LLM-as-a-Judge: оценка ответов RAG другой (или той же) LLM.

Две независимые оценки по шкале 0–2:
- correctness — совпадает ли ответ по смыслу с эталоном из набора вопросов;
- faithfulness — подтверждается ли каждое утверждение ответа найденными источниками.

Они ловят разные ошибки. Ответ может быть верным, но взятым «из головы» модели
(correctness 2, faithfulness 0), или честно пересказывать источник, который сам не
отвечает на вопрос (correctness 0, faithfulness 2).

Шкала 0–2 вместо 1–10: малые модели плохо различают тонкие градации, а три уровня
(«неверно / частично / верно») размечаются стабильно. Судья отвечает строгим JSON
(режим `format: json` в Ollama).
"""

import json
from dataclasses import dataclass

from rag.generation.llm import OllamaClient
from rag.generation.prompts import format_sources
from rag.retrieval import Candidate

# v1 — первая версия. На ручной разметке базового прогона была слишком придирчива к верным
# ответам с лишними деталями и слишком мягка к ответам, называющим другой факт (README).
CORRECTNESS_V1 = """\
Ты — строгий экзаменатор. Сравни ответ системы с эталонным ответом на вопрос.

Оценка:
2 — ответ содержит все ключевые факты эталона (имена, числа, даты) и не противоречит ему;
    формулировка может отличаться, лишние верные детали допустимы;
1 — ответ частично верен: есть часть ключевых фактов, но чего-то не хватает или есть неточность;
0 — ответ неверен, противоречит эталону, не отвечает на вопрос или система отказалась отвечать.

Верни только JSON: {"score": 0|1|2, "reason": "кратко, почему"}"""

FAITHFULNESS_V1 = """\
Ты проверяешь ответ системы на галлюцинации. Источники — единственное, что знала система.
Проверь каждое утверждение ответа: следует ли оно из источников?
Номера в квадратных скобках [1], [2] — ссылки на источники; проверь, что источник
действительно говорит то, что ему приписано.

Оценка:
2 — все утверждения подтверждаются источниками;
1 — главное утверждение подтверждено, но есть неподтверждённые детали (число, дата, имя);
0 — главное утверждение не подтверждается источниками или противоречит им.

Верни только JSON: {"score": 0|1|2, "reason": "кратко: какое утверждение не подтверждено"}"""

# v2 — явные правила для двух типичных ошибок v1 и короткие примеры. Примеры придуманы
# заново и не взяты из набора вопросов, чтобы судья не «знал» тестовые ответы.
CORRECTNESS_V2 = """\
Ты сравниваешь ответ системы с эталонным ответом. Оцени ТОЛЬКО одно: содержит ли ответ
те же ключевые факты, что и эталон (имена, названия, числа, даты).

Правила:
- Другая формулировка, синонимы и дополнительные верные подробности НЕ снижают оценку.
- Если ответ называет ДРУГОЙ факт, чем эталон (другое имя, число, язык, год), — это 0,
  даже если ответ звучит уверенно и правдоподобно.
- Если в эталоне несколько фактов, а в ответе верна только часть — это 1.

Оценка: 2 — все ключевые факты эталона есть; 1 — только часть; 0 — нет или противоречат.

Примеры:
Эталон: «Бухарест.» Ответ: «Столица страны — город Бухарест, крупнейший город Румынии.» → 2
Эталон: «Шесть лет.» Ответ: «Срок полномочий составляет 6 лет [1].» → 2
Эталон: «Мирча Снегур.» Ответ: «Первым президентом был Пётр Лучинский [2].» → 0
Эталон: «В 1998 — 40 мест, в 2001 — 71 место.» Ответ: «В 1998 году получила 40 мест [1].» → 1

Верни только JSON: {"score": 0|1|2, "reason": "кратко, почему"}"""

FAITHFULNESS_V2 = """\
Ты проверяешь ответ системы на выдуманные факты. Источники — единственное, что знала система.

Правила:
- Проверяй только то, что ЕСТЬ в ответе. Если ответ чего-то не упоминает — это не ошибка.
- Утверждение подтверждено, если оно есть хотя бы в одном источнике, пусть и другими словами.
- Утверждение не подтверждено, если его нет ни в одном источнике или источник говорит
  другое (например, число относится к другой партии или другому году).

Оценка:
2 — все утверждения ответа есть в источниках;
1 — главное утверждение есть в источниках, но какая-то деталь (число, дата, имя) — нет;
0 — главное утверждение отсутствует в источниках или противоречит им.

Верни только JSON: {"score": 0|1|2, "reason": "кратко: какое утверждение не подтверждено"}"""

# v3 — лучшее из двух по проверке на ручной разметке: правила v2 исправили correctness, но
# сделали faithfulness подозрительной ко всему (ложные «не подтверждено»), поэтому
# faithfulness остаётся из v1. Результаты сравнения — experiments/judge_validation/report.md.
PROMPTS = {
    "v1": (CORRECTNESS_V1, FAITHFULNESS_V1),
    "v2": (CORRECTNESS_V2, FAITHFULNESS_V2),
    "v3": (CORRECTNESS_V2, FAITHFULNESS_V1),
}


@dataclass
class Verdict:
    score: float | None  # нормировано в 0..1; None — судья вернул непонятный ответ
    reason: str
    raw: str = ""


class LLMJudge:
    def __init__(self, llm: OllamaClient, prompt_version: str = "v2"):
        self.llm = llm
        self.prompt_version = prompt_version
        self.correctness_prompt, self.faithfulness_prompt = PROMPTS[prompt_version]

    @property
    def model(self) -> str:
        return self.llm.model

    def _ask(self, system: str, user: str) -> Verdict:
        response = self.llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=True,
        )
        return parse_verdict(response.text)

    def correctness(self, question: str, expected: str, answer: str) -> Verdict:
        return self._ask(
            self.correctness_prompt,
            f"Вопрос: {question}\n\nЭталонный ответ: {expected}\n\nОтвет системы: {answer}",
        )

    def faithfulness(self, question: str, answer: str, sources: list[Candidate] | str) -> Verdict:
        """`sources` — найденные chunks или уже отформатированный текст источников."""
        text = sources if isinstance(sources, str) else format_sources(sources)
        return self._ask(
            self.faithfulness_prompt,
            f"Источники:\n\n{text}\n\nВопрос: {question}\n\nОтвет системы: {answer}",
        )


def make_judge(config, prompt_version: str | None = None) -> LLMJudge:
    from rag.config import get_settings

    judge_cfg = config.generation.model_copy(
        update={"model": config.evaluation.judge_model, "max_tokens": 256}
    )
    return LLMJudge(
        OllamaClient(get_settings().ollama_host, judge_cfg),
        prompt_version or config.evaluation.judge_prompt,
    )


def parse_verdict(text: str) -> Verdict:
    try:
        data = json.loads(text)
        score = int(data["score"])
        if score not in (0, 1, 2):
            raise ValueError(score)
    except (ValueError, KeyError, TypeError):
        return Verdict(score=None, reason="судья вернул некорректный ответ", raw=text)
    return Verdict(score=score / 2, reason=str(data.get("reason", "")), raw=text)
