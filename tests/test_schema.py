import pytest
from pydantic import ValidationError

from aldau.schema import Ad, TeacherLabel


def make_ad(**overrides) -> Ad:
    """A valid ad; tests override single fields to check one rule at a time."""
    fields = {
        "id": "ad-1",
        "source": "olx",
        "category": "phones",
        "title": "iPhone 13",
        "text": "Продам iPhone 13, состояние отличное.",
        "price_kzt": 250_000,
        "city": "Алматы",
    }
    fields.update(overrides)
    return Ad(**fields)


def test_valid_ad_gets_defaults():
    ad = make_ad()
    assert ad.lang == "ru"
    assert ad.label == "unknown"
    assert ad.photos == []
    assert ad.seller_info is None


def test_ad_price_is_optional():
    assert make_ad(price_kzt=None).price_kzt is None


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("source", "avito"),  # not a Kazakhstan marketplace we support
        ("lang", "en"),
        ("label", "maybe"),
        ("price_kzt", -1),
    ],
)
def test_ad_rejects_bad_values(field, bad_value):
    with pytest.raises(ValidationError):
        make_ad(**{field: bad_value})


def test_ad_requires_text():
    with pytest.raises(ValidationError):
        Ad(id="ad-1", source="olx", category="phones", title="iPhone 13")


def test_ad_json_roundtrip_keeps_kazakh_text():
    ad = make_ad(text="Пәтер жалға беріледі, алдын ала төлем", lang="kk")
    restored = Ad.model_validate_json(ad.model_dump_json())
    assert restored == ad


def test_valid_teacher_label():
    label = TeacherLabel(
        label="scam",
        risk_score=0.9,
        signals=["prepayment", "price_too_low"],
        explanation_ru="Просят предоплату, цена вдвое ниже рынка.",
    )
    assert label.label == "scam"
    assert label.signals == ["prepayment", "price_too_low"]


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_teacher_label_rejects_score_outside_0_1(score):
    with pytest.raises(ValidationError):
        TeacherLabel(label="scam", risk_score=score, explanation_ru="...")


def test_teacher_label_must_decide():
    # The teacher is not allowed to answer "unknown".
    with pytest.raises(ValidationError):
        TeacherLabel(label="unknown", risk_score=0.5, explanation_ru="...")
