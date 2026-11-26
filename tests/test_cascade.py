import time
from datetime import date

import pytest
from test_baseline import toy_data

from aldau import cascade, llm
from aldau.cascade import CheckResult, add_teacher, calls_today, check, count_call
from aldau.collect import build_ad
from aldau.explain import DISCLAIMER, PENDING, format_answer
from aldau.schema import TeacherLabel
from aldau.train_baseline import BaselineModel


@pytest.fixture(scope="module")
def model():
    return BaselineModel(mode="both", c=1.0, threshold=0.73).fit(toy_data())


def grey(text="Квартира посуточно, бронь по предоплате, пишите в ватсап", **extra):
    ad = build_ad("olx", "другое", text[:30], text)
    return CheckResult(ad, 0.5, "yellow", ["prepayment", "messenger"], **extra)


def fake_teacher(risk=0.9, calls=None, delay=0.0, error=None):
    """Stand-in for llm.complete_json: answers for whatever ad is in the prompt."""

    def ask(prompt, schema):
        if calls is not None:
            calls.append(prompt)
        if delay:
            time.sleep(delay)
        if error:
            raise error
        ad_id = prompt.split('"id": "', 1)[1].split('"', 1)[0]
        label = "scam" if risk >= 0.5 else "legit"
        return {"results": [{"id": ad_id, "label": label, "risk_score": risk,
                             "signals": ["prepayment"], "explanation_ru": "Бронь до просмотра."}]}

    return ask


def paths(tmp_path):
    return {"cache_path": tmp_path / "live.jsonl", "budget_path": tmp_path / "budget.json"}


# --- levels -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "risk, level", [(0.1, "green"), (0.34, "green"), (0.35, "yellow"), (0.72, "yellow"),
                    (0.73, "red"), (0.99, "red")],
)  # fmt: skip
def test_student_level(risk, level):
    assert cascade.student_level(risk, threshold=0.73) == level


def test_low_threshold_never_makes_the_grey_zone_negative():
    assert cascade.student_level(0.2, threshold=0.1) == "green"
    assert cascade.student_level(0.4, threshold=0.1) == "red"


@pytest.mark.parametrize("risk, level", [(0.1, "green"), (0.5, "yellow"), (0.7, "red")])
def test_teacher_level_uses_the_prompt_scale(risk, level):
    assert cascade.teacher_level(risk) == level


# --- the rule floor ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "level, signals, expected",
    [("green", ["fake_delivery"], ("red", "fake_delivery")),
     ("green", ["from_abroad", "messenger"], ("yellow", "from_abroad")),
     ("green", ["remote_seller", "fake_delivery"], ("red", "fake_delivery")),
     ("red", ["remote_seller"], ("red", None)),  # a floor never lowers the level
     ("green", ["messenger", "urgency", "deposit", "prepayment"], ("green", None))],
)  # fmt: skip
def test_apply_floor(level, signals, expected):
    assert cascade.apply_floor(level, signals) == expected


def test_floor_signals_are_real_signal_names():
    from aldau.features import SIGNAL_NAMES

    assert set(cascade.SCHEME_FLOORS) <= set(SIGNAL_NAMES)
    assert not {"deposit", "urgency", "messenger"} & set(cascade.SCHEME_FLOORS)


def test_check_student_applies_the_floor_and_can_skip_it(model):
    text = "Продаю телефон, оплата через OLX доставку, введите данные карты по ссылке"
    with_rules = cascade.check_student(model, text)
    assert with_rules.student_level == "red" and with_rules.rule == "fake_delivery"
    assert "Уровень поднят правилом" in format_answer(with_rules)
    without = cascade.check_student(model, text, rules=False)
    assert without.rule is None
    assert without.student_level == cascade.student_level(without.student_risk, 0.73)


def test_level_for():
    assert cascade.level_for(0.1, 0.73, ["fake_delivery"]) == "red"
    assert cascade.level_for(0.1, 0.73, ["fake_delivery"], rules=False) == "green"


# --- the cascade --------------------------------------------------------------------------


def test_green_and_red_never_ask_the_teacher(model, tmp_path):
    calls = []
    for level in ("green", "red"):
        result = CheckResult(grey().ad, 0.1, level)
        add_teacher(result, model, ask=fake_teacher(calls=calls), **paths(tmp_path))
        assert result.teacher is None and result.note is None
    assert calls == []


def test_grey_zone_asks_the_teacher_once_then_uses_the_cache(model, tmp_path):
    calls = []
    first = add_teacher(grey(), model, ask=fake_teacher(0.9, calls), **paths(tmp_path))
    assert first.teacher.risk_score == 0.9 and first.level == "red"
    again = add_teacher(grey(), model, ask=fake_teacher(0.9, calls), **paths(tmp_path))
    assert again.teacher.risk_score == 0.9
    assert len(calls) == 1  # the second check came from the cache
    assert calls_today(tmp_path / "budget.json") == 1


def test_teacher_can_clear_a_grey_ad(model, tmp_path):
    result = add_teacher(grey(), model, ask=fake_teacher(0.15), **paths(tmp_path))
    assert result.level == "green" and result.student_level == "yellow"


def test_daily_limit_stops_asking(model, tmp_path):
    calls = []
    for _ in range(3):
        count_call(tmp_path / "budget.json")
    result = add_teacher(grey(), model, daily_limit=3, ask=fake_teacher(calls=calls),
                         **paths(tmp_path))  # fmt: skip
    assert calls == [] and result.note == cascade.NOTE_LIMIT and result.level == "yellow"


def test_budget_resets_on_a_new_day(tmp_path):
    path = tmp_path / "budget.json"
    count_call(path, date(2026, 12, 13))
    assert calls_today(path, date(2026, 12, 13)) == 1
    assert calls_today(path, date(2026, 12, 14)) == 0


def test_slow_teacher_times_out_and_the_student_answer_stands(model, tmp_path):
    result = add_teacher(grey(), model, timeout=0.05, ask=fake_teacher(delay=0.5),
                         **paths(tmp_path))  # fmt: skip
    assert result.teacher is None and result.note == cascade.NOTE_TIMEOUT
    assert result.level == "yellow"


@pytest.mark.parametrize(
    "error, note",
    [(llm.QuotaError("429"), cascade.NOTE_LIMIT), (llm.OverloadedError("503"), cascade.NOTE_ERROR),
     (ValueError("broken answer"), cascade.NOTE_ERROR)],
)  # fmt: skip
def test_teacher_errors_never_break_the_check(model, tmp_path, error, note):
    result = add_teacher(grey(), model, ask=fake_teacher(error=error), **paths(tmp_path))
    assert result.teacher is None and result.note == note


def test_missing_teacher_settings_give_a_note(model, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "")
    result = add_teacher(grey(), model, ask=fake_teacher(), **paths(tmp_path))
    assert result.note == cascade.NOTE_ERROR


def test_check_offline_marks_the_grey_zone(model, tmp_path, monkeypatch):
    monkeypatch.setattr(cascade, "student_level", lambda risk, threshold: "yellow")
    result = check(model, "Сдаю квартиру посуточно", use_teacher=False)
    assert result.note == cascade.NOTE_OFFLINE


def test_check_masks_personal_data_and_reads_the_price(model):
    result = check(model, "Продам диван за 25 000 тг, звоните 87011234567", use_teacher=False)
    assert "87011234567" not in result.ad.text and "[PHONE]" in result.ad.text
    assert result.ad.price_kzt == 25_000


# --- the answer text ------------------------------------------------------------------------


def test_answer_always_ends_with_the_disclaimer(model):
    for level in ("green", "yellow", "red"):
        text = format_answer(CheckResult(grey().ad, 0.5, level, ["prepayment"]))
        assert text.endswith(DISCLAIMER)


def test_answer_lists_reasons_and_soft_prepayment_advice():
    text = format_answer(CheckResult(grey().ad, 0.95, "red", ["prepayment", "messenger"]))
    assert text.startswith("🔴")
    assert "Оценка локальной модели: 95 %" in text
    assert "просят предоплату" in text
    assert "бывает и у честных" in text


def test_answer_shows_the_teacher_and_pending_line():
    label = TeacherLabel(label="legit", risk_score=0.2, signals=[], explanation_ru="Обычная бронь.")
    text = format_answer(grey(teacher=label))
    assert text.startswith("🟢") and "углублённая проверка: Обычная бронь." in text
    assert "(локальная модель: 50 %)" in text
    assert PENDING in format_answer(grey(), pending_teacher=True)


def test_signal_advice_is_shown_even_when_the_level_is_green():
    text = format_answer(CheckResult(grey().ad, 0.06, "green", ["fake_delivery"]))
    assert text.startswith("🟢") and "Никогда не вводите данные карты" in text
    assert "Всё равно встречайтесь лично" in format_answer(CheckResult(grey().ad, 0.05, "green"))


def test_answer_without_signals_still_gives_a_reason():
    assert "явных признаков" in format_answer(CheckResult(grey().ad, 0.05, "green"))
    assert "похож на мошеннические" in format_answer(CheckResult(grey().ad, 0.9, "red"))


def test_at_most_four_reasons():
    signals = ["prepayment", "messenger", "urgency", "pressure", "remote_seller", "says_cheap"]
    text = format_answer(CheckResult(grey().ad, 0.9, "red", signals))
    reasons_part = text.split("Почему:")[1].split("Что проверить:")[0]
    assert reasons_part.count("•") == 4
