"""Use the trained student: one ad in, risk and reasons out.

This is what the cascade, the bot and the offline demo call. It never touches the
internet: the model file and the regex signals are enough.
"""

from dataclasses import dataclass
from pathlib import Path

from aldau.collect import build_ad
from aldau.features import extract_prices, find_signals
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


def predict_ad_from_text(
    text: str,
    title: str = "",
    price_kzt: int | None = None,
    category: str = "другое",
    city: str | None = None,
) -> Ad:
    """Turn a pasted ad into an Ad. Personal data is masked, exactly as for training data.

    If no price is given, the first price written in the text is used ("25 000 тг").
    """
    title = title or text.strip().split("\n", 1)[0][:100]
    if price_kzt is None:
        prices = extract_prices(text)
        price_kzt = prices[0] if prices else None
    return build_ad("olx", category, title, text, price_kzt=price_kzt, city=city)


def predict_text(
    model: BaselineModel,
    text: str,
    title: str = "",
    price_kzt: int | None = None,
    category: str = "другое",
    city: str | None = None,
) -> StudentResult:
    """Score a pasted ad."""
    return predict(model, predict_ad_from_text(text, title, price_kzt, category, city))
