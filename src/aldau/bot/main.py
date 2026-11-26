"""Telegram bot (aiogram 3): send an ad's text, get the risk, reasons and advice.

    python -m aldau.bot                # student + teacher in the grey zone
    python -m aldau.bot --no-teacher   # student + regex only (no Gemini requests at all)

Long polling: the laptop itself asks Telegram for new messages, so no server, domain or
open port is needed — good for a demo stand. The token is read from .env
(TELEGRAM_BOT_TOKEN) and never printed.

Flow for one message:
1. the student answers at once (~1 ms) and the answer is sent;
2. if the ad is in the grey zone 🟡, the message says "⏳ углублённая проверка…", the
   teacher is asked in a background thread (max 20 s, max 10 requests a day), and the
   same message is edited with the final answer.
"""

import argparse
import asyncio
import logging
import os
import sqlite3
from dataclasses import dataclass

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv

from aldau.bot import storage
from aldau.cascade import NOTE_OFFLINE, TEACHER_DAILY_LIMIT, add_teacher, check_student
from aldau.explain import DISCLAIMER, format_answer
from aldau.student import load_student
from aldau.train_baseline import BaselineModel

MIN_LENGTH = 30  # characters; shorter messages are not ads
MAX_LENGTH = 5000

START_TEXT = (
    "Здравствуйте! Я Aldau — помогаю оценить, не мошенническое ли объявление "
    "(OLX, Kolesa, Krisha).\n\n"
    "Пришлите текст объявления целиком — можно с ценой. Понимаю русский и казахский.\n"
    "Я отвечу: 🟢 низкий риск, 🟡 есть сомнения или 🔴 высокий риск, — объясню почему "
    "и подскажу, что проверить.\n\n"
    "/help — как это работает, /stats — статистика.\n\n" + DISCLAIMER
)
HELP_TEXT = (
    "Как это работает:\n"
    "1. Локальная модель (TF-IDF + логистическая регрессия) и правила ищут признаки схем "
    "обмана: предоплата до осмотра, «хозяин в другом городе», оплата по ссылке и т. п.\n"
    "2. Если модель не уверена (🟡), объявление дополнительно проверяет большая языковая "
    "модель — если она доступна. Это не больше 20 секунд.\n"
    "3. Кнопки «Верно / Неверно» помогают улучшить модель.\n\n"
    "Ограничения: модель — учебный прототип на нескольких сотнях объявлений; фото пока не "
    "проверяю; личные данные (телефоны, ссылки, имена) убираются из текста до проверки, "
    "данные о вас не сохраняются.\n\n" + DISCLAIMER
)
TOO_SHORT = "Пришлите, пожалуйста, полный текст объявления — хотя бы пару предложений."
NO_PHOTOS = "Фото пока не проверяю. Пришлите текст объявления — его можно скопировать с сайта."
THANKS = "Спасибо! Это поможет улучшить модель."

router = Router()


@dataclass
class App:
    """Everything the handlers need; passed to them by aiogram as the `app` argument."""

    model: BaselineModel
    conn: sqlite3.Connection
    use_teacher: bool = True


def feedback_keyboard(check_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="👍 Верно", callback_data=f"fb:{check_id}:right"),
            InlineKeyboardButton(text="👎 Неверно", callback_data=f"fb:{check_id}:wrong"),
        ]]
    )  # fmt: skip


def parse_feedback(data: str) -> tuple[int, str] | None:
    """'fb:12:right' -> (12, 'right'); anything else -> None."""
    parts = (data or "").split(":")
    if len(parts) != 3 or parts[0] != "fb" or not parts[1].isdigit():
        return None
    if parts[2] not in ("right", "wrong"):
        return None
    return int(parts[1]), parts[2]


def stats_text(s: dict) -> str:
    rated = s["right"] + s["wrong"]
    accuracy = f"{round(100 * s['right'] / rated)} %" if rated else "—"
    return (
        f"Проверено объявлений: {s['total']}\n"
        f"🟢 {s['green']}   🟡 {s['yellow']}   🔴 {s['red']}\n"
        f"Углублённых проверок: {s['teacher']} (лимит {TEACHER_DAILY_LIMIT} в день)\n"
        f"Отзывы: 👍 {s['right']}   👎 {s['wrong']}   (верно по отзывам: {accuracy})"
    )


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(START_TEXT)


@router.message(Command("help"))
async def on_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


@router.message(Command("stats"))
async def on_stats(message: Message, app: App) -> None:
    await message.answer(stats_text(storage.stats(app.conn)))


@router.message(F.photo)
async def on_photo(message: Message) -> None:
    await message.answer(NO_PHOTOS)


@router.message(F.text)
async def on_text(message: Message, app: App) -> None:
    text = message.text.strip()
    if text.startswith("/"):
        await message.answer(HELP_TEXT)
        return
    if len(text) < MIN_LENGTH:
        await message.answer(TOO_SHORT)
        return
    result = check_student(app.model, text[:MAX_LENGTH])
    ask_teacher = result.needs_teacher and app.use_teacher
    if result.needs_teacher and not app.use_teacher:
        result.note = NOTE_OFFLINE
    check_id = storage.log_check(app.conn, result)
    keyboard = feedback_keyboard(check_id)
    reply = await message.answer(format_answer(result, pending_teacher=ask_teacher),
                                 reply_markup=keyboard)  # fmt: skip
    if not ask_teacher:
        return
    # The teacher call blocks (network), so it runs in a thread: the bot keeps answering others.
    result = await asyncio.to_thread(add_teacher, result, app.model)
    storage.update_teacher(app.conn, check_id, result)
    try:
        await reply.edit_text(format_answer(result), reply_markup=keyboard)
    except TelegramBadRequest:
        pass  # e.g. the message was deleted meanwhile; nothing to fix


@router.callback_query(F.data.startswith("fb:"))
async def on_feedback(callback: CallbackQuery, app: App) -> None:
    parsed = parse_feedback(callback.data)
    if parsed:
        storage.set_feedback(app.conn, *parsed)
    await callback.answer(THANKS)
    if callback.message:
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except TelegramBadRequest:
            pass


async def run(app: App, token: str) -> None:
    bot = Bot(token=token)
    dispatcher = Dispatcher(app=app)  # `app` is passed to every handler that asks for it
    dispatcher.include_router(router)
    await dispatcher.start_polling(bot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aldau Telegram bot.")
    parser.add_argument("--no-teacher", action="store_true", help="never ask the teacher")
    args = parser.parse_args()
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN в .env не задан")
    try:
        model = load_student()
    except FileNotFoundError:
        raise SystemExit("Нет модели: сначала python -m aldau.train_baseline") from None
    # INFO shows that polling started; message texts and the token are never logged.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = App(model=model, conn=storage.connect(), use_teacher=not args.no_teacher)
    mode = "без учителя"
    if not args.no_teacher:
        mode = f"учитель в серой зоне, до {TEACHER_DAILY_LIMIT} запросов в день"
    print(f"Бот запущен ({mode}). Ctrl+C — остановить.")
    try:
        asyncio.run(run(app, token))
    except KeyboardInterrupt:
        print("\nОстановлено.")


if __name__ == "__main__":
    main()
