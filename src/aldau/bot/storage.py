"""SQLite log of the bot's checks (standard sqlite3 module, one file: data/bot.db).

What is stored: the ad text with personal data already masked, the price, the risks and
the level, and the user's "верно / неверно" feedback. What is NOT stored: anything about
the Telegram user (no user id, no name, no username) — the log is about ads, not people.
The masked texts can later become new training data (active learning, after the EXPO).
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from aldau.cascade import CheckResult

DB_PATH = Path("data/bot.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created       TEXT NOT NULL,      -- local time, ISO format
    ad_id         TEXT NOT NULL,      -- hash of the masked text (same as in data/raw)
    ad_text       TEXT NOT NULL,      -- masked: [PHONE], [LINK], [NAME] ...
    price_kzt     INTEGER,
    student_risk  REAL NOT NULL,
    teacher_risk  REAL,               -- NULL if the teacher was not asked or did not answer
    level         TEXT NOT NULL,      -- final level: green / yellow / red
    signals       TEXT NOT NULL,      -- JSON list of regex signal names
    feedback      TEXT                -- NULL, 'right' or 'wrong'
)
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: aiogram may run handlers in different threads of one process
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def full_text(result: CheckResult) -> str:
    """The ad as the person sent it (masked): title line + text, without repeating the
    title of a one-line ad (there the title is also used as the text)."""
    ad = result.ad
    return ad.text if ad.text == ad.title else ad.title + "\n" + ad.text


def log_check(conn: sqlite3.Connection, result: CheckResult) -> int:
    """Save one check, return its row id (used in the feedback buttons)."""
    cursor = conn.execute(
        "INSERT INTO checks (created, ad_id, ad_text, price_kzt, student_risk, teacher_risk, "
        "level, signals) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now().isoformat(timespec="seconds"),
            result.ad.id,
            full_text(result),
            result.ad.price_kzt,
            result.student_risk,
            result.teacher.risk_score if result.teacher else None,
            result.level,
            json.dumps(result.signals),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def update_teacher(conn: sqlite3.Connection, check_id: int, result: CheckResult) -> None:
    """After the teacher answered (or not): store its risk and the final level."""
    conn.execute(
        "UPDATE checks SET teacher_risk = ?, level = ? WHERE id = ?",
        (result.teacher.risk_score if result.teacher else None, result.level, check_id),
    )
    conn.commit()


def set_feedback(conn: sqlite3.Connection, check_id: int, feedback: str) -> bool:
    """Save 'right' / 'wrong'. Returns False if there is no such check."""
    if feedback not in ("right", "wrong"):
        raise ValueError(f"unknown feedback: {feedback}")
    cursor = conn.execute("UPDATE checks SET feedback = ? WHERE id = ?", (feedback, check_id))
    conn.commit()
    return cursor.rowcount == 1


def stats(conn: sqlite3.Connection) -> dict:
    """Counts for /stats."""
    total = conn.execute("SELECT COUNT(*) FROM checks").fetchone()[0]
    levels = dict(conn.execute("SELECT level, COUNT(*) FROM checks GROUP BY level").fetchall())
    feedback = dict(
        conn.execute(
            "SELECT feedback, COUNT(*) FROM checks WHERE feedback IS NOT NULL GROUP BY feedback"
        ).fetchall()
    )
    teacher = conn.execute("SELECT COUNT(*) FROM checks WHERE teacher_risk IS NOT NULL").fetchone()
    return {
        "total": total,
        "green": levels.get("green", 0),
        "yellow": levels.get("yellow", 0),
        "red": levels.get("red", 0),
        "teacher": teacher[0],
        "right": feedback.get("right", 0),
        "wrong": feedback.get("wrong", 0),
    }
