"""Use the trained student: one ad in, risk and reasons out.

This is what the cascade, the bot and the offline demo call. It never touches the
internet: the model file and the regex signals are enough.
"""

from dataclasses import dataclass
from pathlib import Path

from aldau.collect import build_ad
from aldau.features import find_signals
from aldau.schema import Ad
from aldau.train_baseline import MODEL_PATH, BaselineModel, load


@dataclass
class StudentResult:
    risk: float  # probability of scam, 0..1
    label: str  # "scam" if risk >= the model's threshold, else "legit"
    signals: list[str]  # regex signals found in the text (names from features.SIGNAL_PATTERNS)


def load_student(path: Path = MODEL_PATH) -> BaselineModel:
    return load(path)


def predict(model: BaselineModel, ad: Ad) -> StudentResult:
    risk = float(model.risk([ad])[0])
    signals = [s for s in find_signals(ad.title + "\n" + ad.text) if s != "has_phone"]
    return StudentResult(risk, "scam" if risk >= model.threshold else "legit", signals)


def predict_text(
    model: BaselineModel,
    text: str,
    title: str = "",
    price_kzt: int | None = None,
    category: str = "другое",
    city: str | None = None,
) -> StudentResult:
    """Score a pasted ad. Personal data is masked first, exactly as for training data."""
    title = title or text.strip().split("\n", 1)[0][:100]
    ad = build_ad("olx", category, title, text, price_kzt=price_kzt, city=city)
    return predict(model, ad)
