import math

from aldau.collect import append_ad, build_ad
from aldau.eda import build_report, cross_val_rate, roc_auc, stratified_folds


def test_roc_auc_perfect_random_and_reversed():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5  # ties = half


def test_roc_auc_counts_pairs():
    # scam scores 0.9 and 0.4; legit 0.5 and 0.1 -> 3 of 4 pairs ordered correctly
    assert roc_auc([0.9, 0.4, 0.5, 0.1], [True, True, False, False]) == 0.75


def test_roc_auc_needs_both_classes():
    assert math.isnan(roc_auc([0.1, 0.2], [True, True]))


def test_stratified_folds_keep_the_class_balance():
    is_scam = [True] * 10 + [False] * 40
    fold_of = stratified_folds(is_scam, folds=5)
    for fold in range(5):
        members = [y for y, f in zip(is_scam, fold_of, strict=True) if f == fold]
        assert members.count(True) == 2 and members.count(False) == 8


def test_field_that_matches_the_label_gets_auc_near_one():
    values = ["запчасти"] * 10 + ["одежда"] * 10
    is_scam = [True] * 10 + [False] * 10
    assert roc_auc(cross_val_rate(values, is_scam), is_scam) == 1.0


def test_useless_field_gets_auc_near_half():
    # the same scam share (50 %) for both values: the field says nothing
    values = (["ru"] * 5 + ["kk"] * 5) * 4
    is_scam = [True, False] * 20
    auc = roc_auc(cross_val_rate(values, is_scam), is_scam)
    assert 0.35 <= auc <= 0.65


def test_build_report_runs_on_a_small_dataset(tmp_path):
    raw = tmp_path / "ads.jsonl"
    examples = [
        ("Сдам квартиру, предоплата на карту, желающих много", "scam", 50_000),
        ("Сдам квартиру, бронь 30 000, хозяин в другом городе", "scam", 60_000),
        ("Сдам квартиру на долгий срок, без животных", "legit", 150_000),
        ("Сдам квартиру, депозит 50 000", "legit", 160_000),
        ("Сдам квартиру рядом с метро", "legit", 140_000),
        ("Сдам квартиру, торг", "unknown", 145_000),
    ]
    for n, (text, label, price) in enumerate(examples):
        ad = build_ad("krisha", "аренда квартир", f"Квартира {n}", text, price, "Алматы", label)
        append_ad(raw, ad)

    report = build_report(raw)

    assert "scam 2, legit 3" in report
    assert "| `prepayment` |" in report
    assert "| категория |" in report
    assert "Фото пока не собраны" in report
