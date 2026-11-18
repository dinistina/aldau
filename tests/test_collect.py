import pytest

from aldau.collect import (
    append_ad,
    build_ad,
    crosstab,
    label_gold,
    make_ad_id,
    parse_gold_answer,
    pick_unlabelled,
    read_ads,
    save_new_ad,
)


def make_ad(n: int, **overrides):
    fields = {
        "source": "olx",
        "category": "Телефоны",
        "title": f"Телефон номер {n}",
        "text": f"Продам телефон, вариант {n}.",
        "price_kzt": 100_000,
        "city": "Алматы",
    }
    fields.update(overrides)
    return build_ad(**fields)


def fake_input(answers):
    """Stand-in for input(): returns the prepared answers one by one."""
    it = iter(answers)
    return lambda prompt: next(it)


def test_build_ad_masks_pii_and_fills_fields():
    ad = make_ad(1, text="Звоните 87011234567, меня зовут Арман")
    assert "87011234567" not in ad.text
    assert "[PHONE]" in ad.text and "[NAME]" in ad.text
    assert ad.category == "телефоны"  # normalised to lower case
    assert ad.lang == "ru"
    assert ad.label == "unknown"


def test_build_ad_detects_kazakh():
    assert make_ad(1, text="Телефон сатылады, жағдайы өте жақсы").lang == "kk"


def test_ad_id_ignores_case_and_spaces():
    assert make_ad_id("iPhone 13", "Продам  телефон") == make_ad_id("iphone 13", "продам телефон")
    assert make_ad_id("iPhone 13", "Продам") != make_ad_id("iPhone 14", "Продам")


def test_save_and_read_roundtrip(tmp_path):
    path = tmp_path / "raw" / "ads.jsonl"  # the folder does not exist yet
    ad = make_ad(1, text="Пәтер жалға беріледі")
    assert save_new_ad(ad, path) is True
    assert read_ads(path) == [ad]


def test_duplicate_ad_is_not_saved_twice(tmp_path):
    path = tmp_path / "ads.jsonl"
    assert save_new_ad(make_ad(1), path) is True
    assert save_new_ad(make_ad(1), path) is False
    assert len(read_ads(path)) == 1


def test_read_ads_from_missing_file_is_empty(tmp_path):
    assert read_ads(tmp_path / "nothing.jsonl") == []


def test_pick_unlabelled_skips_gold_and_is_stable():
    ads = [make_ad(n) for n in range(10)]
    gold = ads[:3]
    todo = pick_unlabelled(ads, gold)
    assert {ad.id for ad in todo} == {ad.id for ad in ads[3:]}
    assert todo == pick_unlabelled(ads, gold)  # same order on every run


def test_pick_unlabelled_never_offers_synthetic_ads():
    real = make_ad(1)
    case = make_ad(2, origin="case_based")
    synthetic = make_ad(3, origin="synthetic")
    todo = pick_unlabelled([real, case, synthetic], [])
    assert {ad.id for ad in todo} == {real.id, case.id}


def test_crosstab_counts_labels_per_category():
    ads = [make_ad(1, label="scam"), make_ad(2, label="legit"), make_ad(3, category="авто")]
    table = crosstab(ads, "category", "категория").splitlines()
    assert table[1].split() == ["авто", "0", "0", "1", "1", "-"]
    assert table[2].split() == ["телефоны", "1", "1", "0", "2", "50%"]


def test_label_gold_saves_answers_and_stops_on_q(tmp_path):
    raw, gold = tmp_path / "ads.jsonl", tmp_path / "gold.jsonl"
    for n in range(5):
        append_ad(raw, make_ad(n))

    # scam, legit, skip, quit -> two ads end up in gold
    count = label_gold(raw, gold, ask=fake_input(["s", "l", "n", "q"]))

    assert count == 2
    assert [ad.label for ad in read_ads(gold)] == ["scam", "legit"]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("s", ("scam", False)),
        ("l", ("legit", False)),
        ("st", ("scam", True)),
        ("ts", ("scam", True)),
        ("lt", ("legit", True)),
        ("t", None),  # t alone: no label
        ("sl", None),  # both labels
        ("n", None),
        ("x", None),
    ],
)
def test_parse_gold_answer(answer, expected):
    assert parse_gold_answer(answer) == expected


def test_label_gold_saves_not_from_text_mark(tmp_path):
    raw, gold = tmp_path / "ads.jsonl", tmp_path / "gold.jsonl"
    for n in range(2):
        append_ad(raw, make_ad(n))
    label_gold(raw, gold, ask=fake_input(["st", "l"]))
    assert [(ad.label, ad.not_from_text) for ad in read_ads(gold)] == [
        ("scam", True),
        ("legit", False),
    ]


def test_label_gold_continues_where_it_stopped(tmp_path):
    raw, gold = tmp_path / "ads.jsonl", tmp_path / "gold.jsonl"
    for n in range(4):
        append_ad(raw, make_ad(n))

    label_gold(raw, gold, ask=fake_input(["s", "q"]))
    label_gold(raw, gold, ask=fake_input(["l", "l", "l"]))

    gold_ids = [ad.id for ad in read_ads(gold)]
    assert len(gold_ids) == 4
    assert len(set(gold_ids)) == 4  # no ad was labelled twice


def test_label_gold_stops_at_target(tmp_path):
    raw, gold = tmp_path / "ads.jsonl", tmp_path / "gold.jsonl"
    for n in range(5):
        append_ad(raw, make_ad(n))

    count = label_gold(raw, gold, target=2, ask=fake_input(["s", "l", "s", "s", "s"]))

    assert count == 2
    assert len(read_ads(gold)) == 2
