"""Turn a CheckResult into the answer a person reads (bot, CLI and web page use the same text).

Answer = level (🟢/🟡/🔴) + risk, 2–4 reasons, what to check, and ALWAYS the disclaimer:
the bot is advisory, the result is a probability, not a verdict (CLAUDE.md, hard rules).
"""

from aldau.cascade import CheckResult
from aldau.features import SIGNAL_DESCRIPTIONS_RU

HEADERS = {
    "green": "🟢 Низкий риск",
    "yellow": "🟡 Есть сомнения",
    "red": "🔴 Высокий риск мошенничества",
}
DISCLAIMER = (
    "ℹ️ Это оценка вероятности, а не вердикт: модель может ошибаться. "
    "Решение принимаете вы."
)
PENDING = "⏳ Идёт углублённая проверка, ответ обновится…"
MAX_REASONS = 4
MAX_ADVICE = 3

# What to check, per signal. "prepayment" is worded softly on purpose: booking by
# prepayment is also normal for honest daily rents and dealers (all 8 false alarms of the
# demo model on gold were such ads), so the advice says how to check, not "it is a scam".
ADVICE_RU = {
    "fake_delivery": "Никогда не вводите данные карты по ссылке от продавца — оплачивайте только "
    "в официальном приложении площадки.",
    "prepayment": "Бронь по предоплате бывает и у честных продавцов. Перед оплатой проверьте "
    "договор, оплату на ИП или компанию (а не на личную карту), отзывы; лучше не платить до "
    "осмотра.",
    "remote_seller": "Не платите за то, что не видели: настаивайте на личной встрече и осмотре.",
    "from_abroad": "Не платите вперёд за «пригон» или «растаможку» без договора с компанией.",
    "deposit": "Депозит обычно платят при заселении или осмотре, а не заранее переводом.",
    "pressure": "Не поддавайтесь спешке: «желающих много» — частый приём давления.",
    "urgency": "Срочность — повод проверить всё особенно внимательно.",
    "no_calls": "Попросите созвониться: мошенники часто избегают звонков.",
    "messenger": "Попросите созвониться: мошенники часто избегают звонков.",
    "sends_documents": "Фото удостоверения легко подделать — оно не доказывает честность.",
    "says_cheap": "Сравните цену с похожими объявлениями: слишком дёшево — повод насторожиться.",
    "has_link": "Не переходите по ссылкам из объявления, особенно для оплаты.",
}
ADVICE_DEFAULT = "Не переводите деньги до встречи и проверки товара."
ADVICE_GREEN = "Всё равно встречайтесь лично и проверяйте товар до оплаты."
REASON_TEXT_ONLY = "текст похож на мошеннические объявления из обучающих данных"
REASON_NOTHING = "явных признаков мошенничества в тексте не найдено"


def percent(risk: float) -> str:
    return f"{round(risk * 100)} %"


def reasons(result: CheckResult) -> list[str]:
    """2–4 short reasons: the teacher's explanation first (if any), then the regex signals."""
    items = []
    if result.teacher and result.teacher.explanation_ru:
        items.append(f"углублённая проверка: {result.teacher.explanation_ru.strip()}")
    items += [SIGNAL_DESCRIPTIONS_RU[s] for s in result.signals if s in SIGNAL_DESCRIPTIONS_RU]
    if not items:
        items.append(REASON_NOTHING if result.level == "green" else REASON_TEXT_ONLY)
    return items[:MAX_REASONS]


def advice(result: CheckResult) -> list[str]:
    """Tips for the signals found — at every level: even when the model says 🟢, a found
    "оплата по ссылке" deserves its warning (the model is small and can miss a scheme)."""
    tips = []
    for signal in ADVICE_RU:  # ADVICE_RU is ordered from the most to the least important
        if signal in result.signals and ADVICE_RU[signal] not in tips:
            tips.append(ADVICE_RU[signal])
    if tips:
        return tips[:MAX_ADVICE]
    return [ADVICE_GREEN if result.level == "green" else ADVICE_DEFAULT]


def score_line(result: CheckResult) -> str:
    if result.teacher:
        return (
            f"Оценка углублённой проверки: {percent(result.teacher.risk_score)} "
            f"(локальная модель: {percent(result.student_risk)})"
        )
    return f"Оценка локальной модели: {percent(result.student_risk)}"


def format_answer(result: CheckResult, pending_teacher: bool = False) -> str:
    """Plain text (no HTML/Markdown), so nothing in the ad can break the formatting."""
    lines = [HEADERS[result.level], score_line(result)]
    if result.rule and not result.teacher:
        scheme = SIGNAL_DESCRIPTIONS_RU[result.rule]
        lines.append(f"Уровень поднят правилом: известная схема — {scheme}")
    lines += ["", "Почему:"]
    lines += [f"• {reason}" for reason in reasons(result)]
    lines += ["", "Что проверить:"]
    lines += [f"• {tip}" for tip in advice(result)]
    if pending_teacher:
        lines += ["", PENDING]
    elif result.note:
        lines += ["", result.note]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)
