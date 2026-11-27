"""Bot tests: handlers are called directly with fake messages — no Telegram, no API."""

import asyncio

import pytest
from test_baseline import toy_data

from aldau import cascade
from aldau.bot import main as bot
from aldau.bot import storage
from aldau.cascade import CheckResult
from aldau.collect import build_ad
from aldau.explain import DISCLAIMER, PENDING
from aldau.schema import TeacherLabel
from aldau.train_baseline import BaselineModel


class FakeReply:
    def __init__(self, text, reply_markup=None):
        self.text, self.reply_markup, self.edits = text, reply_markup, []

    async def edit_text(self, text, reply_markup=None):
        self.edits.append(text)

    async def edit_reply_markup(self, reply_markup=None):
        self.reply_markup = reply_markup


class FakeMessage:
    def __init__(self, text=None):
        self.text, self.replies = text, []

    async def answer(self, text, reply_markup=None):
        reply = FakeReply(text, reply_markup)
        self.replies.append(reply)
        return reply


class FakeCallback:
    def __init__(self, data):
        self.data, self.message, self.answers = data, FakeReply("старый ответ", "kb"), []

    async def answer(self, text=None):
        self.answers.append(text)


@pytest.fixture
def app():
    model = BaselineModel(mode="both", threshold=0.73).fit(toy_data())
    return bot.App(model=model, conn=storage.connect(":memory:"), use_teacher=True)


def force_level(monkeypatch, level):
    """Make the student answer a fixed level, whatever the toy model thinks."""
    real = cascade.check_student

    def fake(model, text, *args, **kwargs):
        result = real(model, text, *args, **kwargs)
        result.student_level = level
        return result

    monkeypatch.setattr(bot, "check_student", fake)


TEXT = "Сдаю квартиру посуточно, бронь по предоплате на карту, пишите в ватсап 87011234567"


# --- storage ----------------------------------------------------------------------------


def result(level="red", teacher=None):
    ad = build_ad("olx", "другое", "Квартира", "Предоплата, звоните 87011234567")
    return CheckResult(ad, 0.9, level, ["prepayment"], teacher=teacher)


def test_storage_logs_masked_text_feedback_and_stats():
    conn = storage.connect(":memory:")
    first = storage.log_check(conn, result("red"))
    storage.log_check(conn, result("green"))
    assert storage.set_feedback(conn, first, "right")
    assert not storage.set_feedback(conn, 999, "wrong")
    text = conn.execute("SELECT ad_text FROM checks WHERE id = ?", (first,)).fetchone()[0]
    assert "87011234567" not in text and "[PHONE]" in text
    s = storage.stats(conn)
    assert (s["total"], s["red"], s["green"], s["right"], s["wrong"]) == (2, 1, 1, 1, 0)


def test_storage_keeps_the_text_as_sent_without_repeating_the_title(app):
    sent = ["Продаю iPhone 14\nСостояние хорошее, 135 000 тг", "Продаю диван, 60 000 тг"]
    for text in sent:
        storage.log_check(app.conn, cascade.check_student(app.model, text))
    stored = [row[0] for row in app.conn.execute("SELECT ad_text FROM checks ORDER BY id")]
    assert stored == sent


def test_storage_has_no_user_columns():
    conn = storage.connect(":memory:")
    columns = [row[1] for row in conn.execute("PRAGMA table_info(checks)")]
    assert not any("user" in c or "chat" in c or "name" in c for c in columns)


def test_update_teacher_changes_the_final_level():
    conn = storage.connect(":memory:")
    r = result("yellow")
    check_id = storage.log_check(conn, r)
    r.teacher = TeacherLabel(label="legit", risk_score=0.1, signals=[], explanation_ru="ок")
    storage.update_teacher(conn, check_id, r)
    row = conn.execute("SELECT teacher_risk, level FROM checks").fetchone()
    assert row == (0.1, "green")


# --- handlers ------------------------------------------------------------------------------


def test_start_and_help_end_with_the_disclaimer():
    for handler in (bot.on_start, bot.on_help):
        message = FakeMessage("/start")
        asyncio.run(handler(message))
        assert message.replies[0].text.endswith(DISCLAIMER)


def test_short_text_and_photos_get_a_hint(app):
    message = FakeMessage("привет")
    asyncio.run(bot.on_text(message, app))
    assert message.replies[0].text == bot.TOO_SHORT
    photo = FakeMessage()
    asyncio.run(bot.on_photo(photo))
    assert photo.replies[0].text == bot.NO_PHOTOS


def test_red_ad_is_answered_at_once_without_the_teacher(app, monkeypatch):
    force_level(monkeypatch, "red")
    monkeypatch.setattr(bot, "add_teacher", lambda *a, **k: pytest.fail("teacher asked"))
    message = FakeMessage(TEXT)
    asyncio.run(bot.on_text(message, app))
    reply = message.replies[0]
    assert reply.text.startswith("🔴") and reply.text.endswith(DISCLAIMER)
    assert reply.edits == [] and reply.reply_markup is not None
    assert storage.stats(app.conn)["red"] == 1


def test_grey_ad_shows_pending_then_the_teacher_answer(app, monkeypatch):
    force_level(monkeypatch, "yellow")

    def fake_teacher(result, model):
        result.teacher = TeacherLabel(label="scam", risk_score=0.9, signals=["prepayment"],
                                      explanation_ru="Бронь до просмотра.")  # fmt: skip
        return result

    monkeypatch.setattr(bot, "add_teacher", fake_teacher)
    message = FakeMessage(TEXT)
    asyncio.run(bot.on_text(message, app))
    reply = message.replies[0]
    assert PENDING in reply.text
    assert reply.edits[0].startswith("🔴") and "Бронь до просмотра." in reply.edits[0]
    assert storage.stats(app.conn)["teacher"] == 1


def test_grey_ad_without_teacher_says_offline(app, monkeypatch):
    force_level(monkeypatch, "yellow")
    app.use_teacher = False
    message = FakeMessage(TEXT)
    asyncio.run(bot.on_text(message, app))
    assert cascade.NOTE_OFFLINE in message.replies[0].text
    assert message.replies[0].edits == []


def test_feedback_button_is_saved_and_removed(app):
    check_id = storage.log_check(app.conn, result())
    callback = FakeCallback(f"fb:{check_id}:wrong")
    asyncio.run(bot.on_feedback(callback, app))
    assert callback.answers == [bot.THANKS] and callback.message.reply_markup is None
    assert storage.stats(app.conn)["wrong"] == 1


@pytest.mark.parametrize(
    "data, parsed",
    [("fb:12:right", (12, "right")), ("fb:3:wrong", (3, "wrong")), ("fb:x:right", None),
     ("fb:1:maybe", None), ("other", None), (None, None)],
)  # fmt: skip
def test_parse_feedback(data, parsed):
    assert bot.parse_feedback(data) == parsed


def test_stats_text():
    s = {"total": 5, "green": 2, "yellow": 1, "red": 2, "teacher": 1, "right": 3, "wrong": 1}
    text = bot.stats_text(s)
    assert "Проверено объявлений: 5" in text and "75 %" in text
