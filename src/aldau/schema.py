"""Data models shared by the whole project.

Two models live here:
- Ad            — one classified ad (what we collect and classify)
- TeacherLabel  — what the teacher LLM returns for one ad

Pydantic checks types and allowed values when an object is created, so a broken
record (unknown source, risk score of 1.7, ...) fails loudly here instead of
silently spoiling the training data later.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Literal = "only these exact strings are allowed"
Source = Literal["olx", "kolesa", "krisha"]
Lang = Literal["ru", "kk"]
AdLabel = Literal["scam", "legit", "unknown"]
# The teacher must always decide, so it has no "unknown" option.
TeacherVerdict = Literal["scam", "legit"]


class Ad(BaseModel):
    """One classified ad."""

    id: str
    source: Source
    category: str
    title: str
    text: str
    price_kzt: int | None = Field(default=None, ge=0)  # None = price not given
    city: str | None = None
    photos: list[str] = Field(default_factory=list)  # paths to image files
    seller_info: str | None = None  # optional, must not contain personal data
    lang: Lang = "ru"
    label: AdLabel = "unknown"  # human label; "unknown" until someone labels it


class TeacherLabel(BaseModel):
    """The teacher LLM's answer for one ad."""

    label: TeacherVerdict
    risk_score: float = Field(ge=0.0, le=1.0)  # 0 = surely legit, 1 = surely scam
    signals: list[str] = Field(default_factory=list)  # short fraud-signal names
    explanation_ru: str  # human-readable explanation in Russian
