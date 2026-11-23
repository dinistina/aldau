import json
import math

import pytest

from aldau.agreement import (
    agreement_report,
    cohen_kappa,
    draw_sample,
    label_second,
    read_second,
)
from aldau.collect import append_ad, build_ad, label_gold, read_ads
from aldau.gold import change_label, check_unchanged, freeze, gold_vs_import


def make_ad(n, label="legit", origin="real"):
    return build_ad("olx", "телефоны", f"Объявление {n}", f"Текст номер {n}", 1000, "Алматы",
                    label, origin)  # fmt: skip


@pytest.fixture
def gold_files(tmp_path):
    gold = tmp_path / "gold" / "gold.jsonl"
    for n in range(4):
        append_ad(gold, make_ad(n, "scam" if n == 0 else "legit"))
    return {
        "gold_path": gold,
        "changes_path": tmp_path / "gold_changes.jsonl",
        "freeze_path": tmp_path / "gold" / "FROZEN.json",
    }


# --- review --------------------------------------------------------------------------


def test_gold_vs_import_lists_only_changed_labels():
    raw = [make_ad(1, "scam"), make_ad(2, "legit"), make_ad(3, "unknown")]
    gold = [make_ad(2, "legit"), make_ad(1, "legit"), make_ad(3, "legit")]
    rows = gold_vs_import(gold, raw)
    assert [(number, ad.title) for number, ad, _ in rows] == [
        (2, "Объявление 1"),
        (3, "Объявление 3"),
    ]
    assert rows[0][2].label == "scam"  # what it was at import


# --- logged changes -------------------------------------------------------------------


def test_change_label_updates_gold_and_writes_log(gold_files):
    ad_id = read_ads(gold_files["gold_path"])[1].id
    record = change_label(ad_id, "scam", True, "перечитал: схема с предоплатой", **gold_files)

    changed = read_ads(gold_files["gold_path"])[1]
    assert (changed.label, changed.not_from_text) == ("scam", True)
    logged = json.loads(gold_files["changes_path"].read_text(encoding="utf-8"))
    assert logged == record
    assert logged["was"] == {"label": "legit", "not_from_text": False}
    assert logged["now"] == {"label": "scam", "not_from_text": True}
    assert logged["reason"] == "перечитал: схема с предоплатой"
    assert len(read_ads(gold_files["gold_path"])) == 4  # nothing lost, order kept


@pytest.mark.parametrize(
    "ad_index, label, t, reason, message",
    [
        (1, "scam", False, "  ", "причина"),
        (1, "legit", False, "x", "уже такая"),
        (1, "unknown", False, "x", "только scam или legit"),
        (None, "scam", False, "x", "нет объявления"),
    ],
)
def test_change_label_refuses_bad_requests(gold_files, ad_index, label, t, reason, message):
    ad_id = "nope" if ad_index is None else read_ads(gold_files["gold_path"])[ad_index].id
    with pytest.raises(ValueError, match=message):
        change_label(ad_id, label, t, reason, **gold_files)
    assert not gold_files["changes_path"].exists()


# --- freezing -------------------------------------------------------------------------


def test_freeze_then_check_and_no_more_changes(gold_files):
    paths = {k: gold_files[k] for k in ("gold_path", "freeze_path")}
    info = freeze(**paths)
    assert (info["ads"], info["scam"], info["legit"]) == (4, 1, 3)
    assert check_unchanged(**paths)

    ad_id = read_ads(gold_files["gold_path"])[1].id
    with pytest.raises(ValueError, match="заморожен"):
        change_label(ad_id, "scam", False, "x", **gold_files)
    with pytest.raises(ValueError, match="уже заморожен"):
        freeze(**paths)


def test_check_notices_an_edit_after_freezing(gold_files):
    paths = {k: gold_files[k] for k in ("gold_path", "freeze_path")}
    freeze(**paths)
    text = gold_files["gold_path"].read_text(encoding="utf-8")
    gold_files["gold_path"].write_text(text.replace('"legit"', '"scam"', 1), encoding="utf-8")
    assert not check_unchanged(**paths)


def test_gold_labelling_refuses_when_frozen(gold_files, tmp_path):
    freeze(gold_path=gold_files["gold_path"], freeze_path=gold_files["freeze_path"])
    raw = tmp_path / "raw.jsonl"
    append_ad(raw, make_ad(9))
    with pytest.raises(SystemExit, match="заморожен"):
        label_gold(raw, gold_files["gold_path"], ask=lambda prompt: "s")


# --- second annotator and kappa -------------------------------------------------------


def test_cohen_kappa_textbook_values():
    assert cohen_kappa(["scam", "legit"] * 5, ["scam", "legit"] * 5)[0] == 1.0
    # p_o = 0.5, p_e = 0.5 -> kappa 0 (exactly what chance gives)
    kappa, p_o, p_e = cohen_kappa(
        ["scam", "scam", "legit", "legit"], ["scam", "legit", "scam", "legit"]
    )
    assert (kappa, p_o, p_e) == (0.0, 0.5, 0.5)


def test_kappa_is_zero_when_both_always_say_legit_except_rarely():
    # 9 of 10 agree, but only because both mostly say legit: kappa is low
    first = ["legit"] * 9 + ["scam"]
    second = ["legit"] * 10
    kappa, p_o, _ = cohen_kappa(first, second)
    assert p_o == 0.9
    assert kappa == 0.0


def test_kappa_undefined_without_data():
    assert math.isnan(cohen_kappa([], [])[0])


def test_draw_sample_is_reproducible_and_unique():
    gold = [make_ad(n) for n in range(50)]
    sample = draw_sample(gold, 30)
    assert sample == draw_sample(gold, 30)
    assert len(set(sample)) == 30


def test_second_annotator_does_not_see_first_label(tmp_path, capsys):
    gold = [make_ad(1, "scam", "case_based"), make_ad(2, "legit")]
    answers = iter(["l", "st"])
    out = tmp_path / "second.jsonl"
    count = label_second(gold, [ad.id for ad in gold], out, ask=lambda prompt: next(answers))

    shown = capsys.readouterr().out
    assert "case_based" not in shown and "scam" not in shown.split("> ")[0]
    assert count == 2
    second = read_second(out)
    assert second[gold[0].id]["label"] == "legit"
    assert second[gold[1].id] == {"id": gold[1].id, "label": "scam", "not_from_text": True}


def test_second_annotator_resumes(tmp_path):
    gold = [make_ad(n) for n in range(3)]
    ids = [ad.id for ad in gold]
    out = tmp_path / "second.jsonl"
    label_second(gold, ids, out, ask=lambda prompt: "q")
    answers = iter(["s", "q"])
    label_second(gold, ids, out, ask=lambda prompt: next(answers))
    answers = iter(["l", "l"])
    label_second(gold, ids, out, ask=lambda prompt: next(answers))
    assert list(read_second(out)) == ids


def test_agreement_report():
    gold = [make_ad(1, "scam"), make_ad(2, "legit"), make_ad(3, "legit")]
    second = {
        gold[0].id: {"id": gold[0].id, "label": "scam", "not_from_text": False},
        gold[1].id: {"id": gold[1].id, "label": "scam", "not_from_text": False},
        gold[2].id: {"id": gold[2].id, "label": "legit", "not_from_text": False},
    }
    report = "\n".join(agreement_report(gold, second))
    assert "размеченных обоими: 3" in report
    assert "Расхождения (1)" in report and gold[1].id in report
    assert "Cohen's kappa: 0.40" in report
