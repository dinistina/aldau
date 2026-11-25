import math

import numpy as np
import pytest

from aldau.collect import build_ad
from aldau.evaluate import recall_with_and_without_t, slice_metrics, worst_misses
from aldau.student import predict, predict_text
from aldau.train_baseline import (
    BaselineModel,
    assert_no_overlap,
    best_threshold,
    jaccard,
    load,
    save,
    split_train_gold,
    train,
)

SCAM_TEXTS = [
    "Сдаю квартиру, хозяин в другом городе, ключи после оплаты на карту, желающих много",
    "Предоплата 50% на Kaspi, отправлю почтой, сам в командировке",
    "Бронь 30 000 переводом, иначе отдам другим, звонки не отвечаю, пишите в ватсап",
    "Оплата по ссылке OLX доставка, введите данные карты, лично встретиться не могу",
    "Авто из Кореи под заказ, 100% предоплата, мест ограничено",
    "Алдын ала төлем Kaspi-ге аударыңыз, кілтті көршім береді",
]
LEGIT_TEXTS = [
    "Продам диван, самовывоз, торг при встрече",
    "Стиральная машина LG рабочая, звоните, можно посмотреть",
    "Ноутбук Lenovo, батарея держит, проверка на месте",
    "Сдам квартиру на долгий срок семейным, депозит при заселении",
    "Телефон Samsung, коробка, документы, встреча в центре",
    "Пәтер сатылады, жағдайы жақсы, келіп көріңіз",
]


def ad(n, label, text, origin=None, category="разное", price=100_000):
    origin = origin or ("case_based" if label == "scam" else "real")
    return build_ad("olx", category, f"Объявление {n}", text, price, "Алматы", label, origin)


def toy_data(copies=3):
    """A small, clearly separable set: scam texts vs ordinary texts, a few variants each."""
    ads = []
    for k in range(copies):
        ads += [ad(f"s{k}{i}", "scam", t + f" вариант {k}") for i, t in enumerate(SCAM_TEXTS)]
        ads += [ad(f"l{k}{i}", "legit", t + f" вариант {k}") for i, t in enumerate(LEGIT_TEXTS)]
    return ads


# --- data split ------------------------------------------------------------------------


def test_split_excludes_gold_unlabelled_and_near_copies():
    gold = [ad("g1", "scam", "Предоплата на карту, хозяин в другом городе, ключи соседка передаст")]
    copy = ad("c1", "scam", "Предоплата на карту, хозяин в другом городе, ключи соседка передаст!")
    other = ad("o1", "legit", "Продам диван, самовывоз")
    unknown = ad("u1", "unknown", "Что-то непонятное")
    train_ads, dropped = split_train_gold([gold[0], copy, other, unknown], gold)
    assert [a.id for a in train_ads] == [other.id]
    assert [a.id for a in dropped] == [copy.id]


def test_assert_no_overlap_catches_gold_in_training():
    gold = [ad("g1", "scam", "Предоплата")]
    with pytest.raises(ValueError, match="gold ads in training"):
        assert_no_overlap(gold, gold)
    assert_no_overlap([ad("t1", "legit", "Диван")], gold)  # no error


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0


# --- model -----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["text", "features", "both"])
def test_model_learns_a_separable_toy_set(mode):
    data = toy_data()
    model = BaselineModel(mode=mode, c=1.0).fit(data)
    risk = model.risk(data)
    assert risk.min() >= 0 and risk.max() <= 1
    scam = np.array([a.label == "scam" for a in data])
    assert risk[scam].mean() > risk[~scam].mean()


def test_price_medians_come_from_training_ads_only():
    data = toy_data(copies=1)
    model = BaselineModel(mode="features").fit(data)
    assert model.prices.by_category["разное"] == 100_000  # every training ad costs 100 000


def test_best_threshold_maximises_f1():
    risk = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    y = np.array([0, 0, 1, 0, 1])
    t = best_threshold(risk, y)
    assert t == pytest.approx(0.6)  # catches both scam ads with one false alarm (F1 0.8)


def test_train_picks_settings_by_cross_validation_and_records_them():
    model, summary = train(toy_data(copies=4), modes=["text", "both"])
    assert {row["mode"] for row in summary} == {"text", "both"}
    assert model.mode in ("text", "both")
    assert model.info["train_size"] == 48
    assert 0 <= model.threshold <= 1


def test_forced_mode_is_saved_and_the_cv_choice_is_kept():
    model, _ = train(toy_data(copies=4), modes=["text", "features"], force_mode="features")
    assert model.mode == "features"
    assert model.info["chosen_by"] == "owner_after_gold"
    assert model.info["cv_choice"]["mode"] in ("text", "features")
    plain, _ = train(toy_data(copies=4), modes=["text", "features"])
    assert plain.info["chosen_by"] == "cv"
    assert plain.mode == plain.info["cv_choice"]["mode"]


def test_saved_model_loads_in_a_fresh_python_process(tmp_path):
    """The bot loads the model in another process: the class must be pickled by its real
    module name, not as __main__.BaselineModel."""
    import subprocess
    import sys

    path = tmp_path / "m.joblib"
    save(BaselineModel(mode="features").fit(toy_data(copies=1)), path)
    code = f"from aldau.train_baseline import load; print(load(r'{path}').mode)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "features"


def test_save_and_load_give_the_same_risk(tmp_path):
    data = toy_data()
    model = BaselineModel(mode="both").fit(data)
    save(model, tmp_path / "m.joblib")
    again = load(tmp_path / "m.joblib")
    assert np.allclose(model.risk(data), again.risk(data))


# --- evaluation ------------------------------------------------------------------------


def test_slice_metrics():
    risk = np.array([0.9, 0.2, 0.8, 0.1])
    y = np.array([1, 1, 0, 0])
    m = slice_metrics(risk, y, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    assert m["roc_auc"] == 0.75


def test_slice_metrics_with_one_class_has_no_auc():
    m = slice_metrics(np.array([0.9, 0.3]), np.array([1, 1]), 0.5)
    assert math.isnan(m["roc_auc"]) and math.isnan(m["pr_auc"])
    assert m["recall"] == 0.5


def test_recall_with_and_without_t():
    hidden = ad("h", "scam", "Обычный текст", origin="real").model_copy(
        update={"not_from_text": True}
    )
    seen = ad("s", "scam", "Предоплата на карту")
    lines = recall_with_and_without_t(np.array([0.1, 0.9]), [hidden, seen], 0.5)
    assert "recall по всем scam: 0.50 (1 из 2)" in lines[0]
    assert "1.00 (1 из 1)" in lines[1]
    assert "0.00 (0 из 1); без t: — (0 из 0)" in lines[2]


def test_worst_misses_most_confident_first():
    model = BaselineModel(threshold=0.5)
    gold = [ad("a", "scam", "x"), ad("b", "legit", "y"), ad("c", "scam", "z")]
    misses = worst_misses(model, gold, np.array([0.4, 0.95, 0.05]))
    assert [a.id for a, _ in misses] == [gold[1].id, gold[2].id, gold[0].id]


# --- student ---------------------------------------------------------------------------


def test_predict_text_masks_pii_and_returns_signals():
    model = BaselineModel(mode="both").fit(toy_data())
    result = predict_text(model, "Предоплата на карту, хозяин в другом городе. Тел 87011234567")
    assert result.label in ("scam", "legit")
    assert "prepayment" in result.signals and "remote_seller" in result.signals
    assert "has_phone" not in result.signals
    assert predict(model, toy_data()[0]).risk == pytest.approx(model.risk([toy_data()[0]])[0])
