"""Cascade: the local student answers first; the teacher is asked only when the student is unsure.

    student risk < 0.35                  -> 🟢 green   (teacher not asked)
    0.35 <= risk < student threshold     -> 🟡 yellow  (grey zone: ask the teacher)
    risk >= student threshold (0.73)     -> 🔴 red     (teacher not asked)

The grey zone is decided by the STUDENT's risk (docs/PLAN.md, Phase 7). The red border is
the student's own decision threshold, chosen by cross-validation on the training ads.

The teacher is a bonus, never a requirement (owner's decision 2026-11-25):
- one attempt, no retries, and we wait at most TEACHER_TIMEOUT_SECONDS;
- at most TEACHER_DAILY_LIMIT requests a day — the free Gemini key allows ~20 a day and
  the background labelling needs the rest;
- any problem (no key, quota, busy server, timeout, no internet) -> the answer stays the
  student's, with a short note. The cascade never raises because of the teacher.
"""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import date
from functools import partial
from pathlib import Path

from aldau import llm
from aldau.schema import Ad, TeacherLabel
from aldau.student import predict, predict_ad_from_text
from aldau.teacher import (
    TeacherBatch,
    append_cache,
    build_prompt,
    cache_key,
    parse_batch,
    read_cache,
)
from aldau.train_baseline import BaselineModel

# Rule floor (owner's decision 2026-11-26): known scam schemes from the teacher's prompt
# (teacher.INSTRUCTIONS, "Типичные схемы") raise the level to at least this, whatever the
# student's risk. The student learned from 74 ads and knows only what was frequent there
# (e.g. "оплата по ссылке" is in 2 training ads), so expert rules cover rare schemes.
# Not in the list on purpose — the prompt says they are NOT a scam by themselves:
# deposit, urgency, messenger. Prepayment is left to the model: honest bookings use it too.
SCHEME_FLOORS = {
    "fake_delivery": "red",  # card details on a link from the seller: always a scheme
    "from_abroad": "yellow",  # "из Кореи", "на таможне" — honest importers exist too
    "remote_seller": "yellow",  # "хозяин в другом городе", "отправлю почтой"
    "sends_documents": "yellow",  # "пришлю фото удостоверения"
    "pressure": "yellow",  # "желающих много, кто первый оплатит"
}
LEVEL_ORDER = ["green", "yellow", "red"]

GREEN_BELOW = 0.35  # student risk below this is "low risk"
TEACHER_GREEN_BELOW = 0.4  # the teacher's own scale (prompt v2): 0.15–0.4 = "one weak sign"
TEACHER_RED_FROM = 0.7  # prompt v2: 0.7–1.0 = "a recognisable scam scheme"
TEACHER_TIMEOUT_SECONDS = 20.0
TEACHER_DAILY_LIMIT = 10
# Teacher answers from the bot / demo go to their own cache file, so they never mix with
# the labelling run's cache (data/cache/teacher.jsonl).
LIVE_CACHE_PATH = Path("data/cache/live_teacher.jsonl")
BUDGET_PATH = Path("data/cache/teacher_budget.json")

# Notes shown under the answer when the teacher was needed but did not answer.
NOTE_OFFLINE = "Офлайн-режим: ответ локальной модели, углублённая проверка выключена."
NOTE_LIMIT = "Углублённая проверка на сегодня исчерпана — ответ локальной модели."
NOTE_TIMEOUT = "Углублённая проверка не успела ответить — ответ локальной модели."
NOTE_ERROR = "Углублённая проверка сейчас недоступна — ответ локальной модели."

# One background thread for teacher requests. On a timeout we stop WAITING, but the
# request finishes in the background and its answer still lands in the cache.
_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="teacher")


@dataclass
class CheckResult:
    ad: Ad  # the checked ad, personal data already masked
    student_risk: float
    student_level: str  # green / yellow / red, from the student alone
    signals: list[str] = field(default_factory=list)  # regex signals found in the text
    teacher: TeacherLabel | None = None  # the teacher's answer, if it was asked and answered
    note: str | None = None  # why the teacher's answer is missing, if it was needed
    rule: str | None = None  # the scheme signal that raised the level (SCHEME_FLOORS), if any

    @property
    def level(self) -> str:
        """The final level: the teacher decides the grey zone when it answered."""
        return teacher_level(self.teacher.risk_score) if self.teacher else self.student_level

    @property
    def needs_teacher(self) -> bool:
        return self.student_level == "yellow"


def student_level(risk: float, threshold: float) -> str:
    if risk >= max(threshold, GREEN_BELOW):
        return "red"
    return "green" if risk < GREEN_BELOW else "yellow"


def apply_floor(level: str, signals: list[str]) -> tuple[str, str | None]:
    """Raise the level to the strongest floor among the found scheme signals.

    Returns (level, the signal that raised it or None if the level did not change).
    """
    best, rule = level, None
    for signal in signals:
        floor = SCHEME_FLOORS.get(signal)
        if floor and LEVEL_ORDER.index(floor) > LEVEL_ORDER.index(best):
            best, rule = floor, signal
    return best, rule


def level_for(risk: float, threshold: float, signals: list[str], rules: bool = True) -> str:
    """The student's level for an ad, with or without the rule floor (used in reports)."""
    level = student_level(risk, threshold)
    return apply_floor(level, signals)[0] if rules else level


def teacher_level(risk: float) -> str:
    if risk >= TEACHER_RED_FROM:
        return "red"
    return "green" if risk < TEACHER_GREEN_BELOW else "yellow"


def check_student(
    model: BaselineModel,
    text: str,
    price_kzt: int | None = None,
    category: str = "другое",
    city: str | None = None,
    rules: bool = True,
) -> CheckResult:
    """Step 1, always local and instant (~1 ms): student risk -> level -> rule floor."""
    ad = predict_ad_from_text(text, price_kzt=price_kzt, category=category, city=city)
    result = predict(model, ad)
    level, rule = student_level(result.risk, model.threshold), None
    if rules:
        level, rule = apply_floor(level, result.signals)
    return CheckResult(ad, result.risk, level, result.signals, rule=rule)


# --- the daily teacher budget ----------------------------------------------------------


def calls_today(path: Path = BUDGET_PATH, today: date | None = None) -> int:
    today = today or date.today()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return data.get("count", 0) if data.get("date") == today.isoformat() else 0


def count_call(path: Path = BUDGET_PATH, today: date | None = None) -> None:
    today = today or date.today()
    count = calls_today(path, today) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": today.isoformat(), "count": count}), encoding="utf-8")


# --- step 2: the teacher -----------------------------------------------------------------


def ask_teacher_once(
    ad: Ad,
    model: BaselineModel,
    teacher_model: str,
    ask: Callable[..., dict],
    cache_path: Path,
) -> TeacherLabel:
    """One request for one ad; the answer is cached. Raises llm.LLMError on any problem."""
    answer = ask(build_prompt([ad], model.prices), TeacherBatch)
    labels, problems = parse_batch(answer, [ad])
    if ad.id not in labels:
        raise llm.LLMError("; ".join(problems) or "пустой ответ учителя")
    append_cache(cache_path, cache_key(ad, teacher_model), ad, teacher_model, labels[ad.id])
    return labels[ad.id]


def add_teacher(
    result: CheckResult,
    model: BaselineModel,
    timeout: float = TEACHER_TIMEOUT_SECONDS,
    daily_limit: int = TEACHER_DAILY_LIMIT,
    ask: Callable[..., dict] | None = None,
    cache_path: Path = LIVE_CACHE_PATH,
    budget_path: Path = BUDGET_PATH,
) -> CheckResult:
    """Step 2, only for the grey zone. Fills result.teacher or result.note; never raises."""
    if not result.needs_teacher:
        return result
    try:
        teacher_model = llm.model_id()
    except llm.LLMError:
        result.note = NOTE_ERROR
        return result
    cached = read_cache(cache_path).get(cache_key(result.ad, teacher_model))
    if cached:  # the same ad was checked before: free and instant
        result.teacher = TeacherLabel.model_validate(cached["label"])
        return result
    if calls_today(budget_path) >= daily_limit:
        result.note = NOTE_LIMIT
        return result
    count_call(budget_path)
    ask = ask or partial(llm.complete_json, retries=0, log=lambda message: None)
    future = _POOL.submit(ask_teacher_once, result.ad, model, teacher_model, ask, cache_path)
    try:
        result.teacher = future.result(timeout=timeout)
    except FutureTimeout:
        result.note = NOTE_TIMEOUT
    except llm.QuotaError:
        result.note = NOTE_LIMIT
    except Exception:  # LLMError, network, a bug in the answer: the student's answer stands
        result.note = NOTE_ERROR
    return result


def check(
    model: BaselineModel,
    text: str,
    price_kzt: int | None = None,
    category: str = "другое",
    city: str | None = None,
    use_teacher: bool = True,
    **teacher_options,
) -> CheckResult:
    """Both steps in one call (the CLI and the web page use this; the bot does them apart)."""
    result = check_student(model, text, price_kzt, category, city)
    if not result.needs_teacher:
        return result
    if not use_teacher:
        result.note = NOTE_OFFLINE
        return result
    return add_teacher(result, model, **teacher_options)
