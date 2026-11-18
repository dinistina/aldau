import json

import pytest

from aldau import llm
from aldau.collect import build_ad
from aldau.features import PriceReference
from aldau.schema import TeacherLabel
from aldau.teacher import (
    TeacherBatch,
    build_prompt,
    cache_key,
    cached_labels,
    classification_metrics,
    compare_report,
    compare_versions,
    export_labels,
    label_ads,
    parse_batch,
    read_cache,
)

MODEL = "gemini:test-model"


def make_ad(n, label="legit", origin="real", text=None):
    text = text or f"Продам телефон номер {n}, состояние хорошее"
    return build_ad("olx", "телефоны", f"Телефон {n}", text, 100_000, "Алматы", label, origin)


def answer_for(batch, label="legit", score=0.1):
    """What a well-behaved teacher returns for a batch."""
    return {
        "results": [
            {
                "id": ad.id,
                "label": label,
                "risk_score": score,
                "signals": [],
                "explanation_ru": "Обычное объявление.",
            }
            for ad in batch
        ]
    }


def ads_in(prompt):
    """The ads part of a prompt: the JSON list after the line 'Объявления:'."""
    return json.loads(prompt.rsplit("Объявления:\n", 1)[1])


class FakeTeacher:
    """Stand-in for llm.complete_json: answers every batch, can fail on a chosen request."""

    def __init__(self, fail_on=None):
        self.prompts = []
        self.fail_on = fail_on

    def __call__(self, prompt, schema):
        assert schema is TeacherBatch
        self.prompts.append(prompt)
        if len(self.prompts) == self.fail_on:
            raise llm.LLMError("ошибка 429: дневной лимит исчерпан")
        items = ads_in(prompt)
        return {
            "results": [
                {
                    "id": item["id"],
                    "label": "legit",
                    "risk_score": 0.1,
                    "signals": [],
                    "explanation_ru": "ok",
                }
                for item in items
            ]
        }


def run(ads, tmp_path, teacher, batch_size=2):
    prices = PriceReference(min_count=1).fit(ads)
    return label_ads(
        ads,
        prices,
        MODEL,
        batch_size=batch_size,
        pause=0,
        cache_path=tmp_path / "cache.jsonl",
        ask=teacher,
        sleep=lambda s: None,
        log=lambda m: None,
    )


# --- prompt ------------------------------------------------------------------------


def test_prompt_never_shows_human_label_or_origin():
    ad = make_ad(1, label="scam", origin="case_based", text="Предоплата 20 000 на карту")
    prompt = build_prompt([ad], PriceReference(min_count=1).fit([ad]))
    items = ads_in(prompt)
    assert set(items[0]) == {
        "id", "category", "city", "price_kzt", "price_ratio", "title", "text", "regex_hints",
    }  # fmt: skip
    assert "case_based" not in prompt
    assert "просят предоплату, задаток или бронь" in items[0]["regex_hints"]


def test_prompt_keeps_kazakh_text_readable():
    ad = make_ad(1, text="Пәтер жалға беріледі")
    assert "Пәтер жалға беріледі" in build_prompt([ad], PriceReference())


# --- cache key ---------------------------------------------------------------------


def test_cache_key_depends_on_model_and_prompt_version(monkeypatch):
    ad = make_ad(1)
    assert cache_key(ad, "gemini:a") != cache_key(ad, "gemini:b")
    before = cache_key(ad, MODEL)
    monkeypatch.setattr("aldau.teacher.PROMPT_VERSION", "v999")
    assert cache_key(ad, MODEL) != before


def test_cache_key_changes_when_ad_changes():
    ad = make_ad(1)
    cheaper = ad.model_copy(update={"price_kzt": 50_000})
    assert cache_key(ad, MODEL) != cache_key(cheaper, MODEL)


# --- labelling ---------------------------------------------------------------------


def test_ads_are_sent_in_batches_and_cached(tmp_path):
    ads = [make_ad(n) for n in range(5)]
    teacher = FakeTeacher()
    report = run(ads, tmp_path, teacher, batch_size=2)

    assert len(teacher.prompts) == 3  # 2 + 2 + 1
    assert report.requests == 3
    assert sorted(report.labelled) == sorted(ad.id for ad in ads)
    assert len(read_cache(tmp_path / "cache.jsonl")) == 5


def test_second_run_uses_cache_and_makes_no_requests(tmp_path):
    ads = [make_ad(n) for n in range(4)]
    run(ads, tmp_path, FakeTeacher())
    teacher = FakeTeacher()
    report = run(ads, tmp_path, teacher)
    assert teacher.prompts == []
    assert report.requests == 0


def test_run_stops_on_api_error_and_resumes_later(tmp_path):
    ads = [make_ad(n) for n in range(6)]
    report = run(ads, tmp_path, FakeTeacher(fail_on=2), batch_size=2)
    assert report.stopped_early
    assert len(report.labelled) == 2  # only the first batch got through
    assert "429" in report.problems[0]

    teacher = FakeTeacher()
    report = run(ads, tmp_path, teacher, batch_size=2)
    assert len(teacher.prompts) == 2  # the remaining 4 ads, not all 6
    assert len(read_cache(tmp_path / "cache.jsonl")) == 6


def test_labels_of_another_model_are_not_reused(tmp_path):
    ads = [make_ad(1)]
    run(ads, tmp_path, FakeTeacher())
    assert cached_labels(ads, MODEL, tmp_path / "cache.jsonl")
    assert cached_labels(ads, "anthropic:claude", tmp_path / "cache.jsonl") == {}


def test_broken_cache_line_is_skipped(tmp_path):
    ads = [make_ad(1), make_ad(2)]
    run(ads, tmp_path, FakeTeacher())
    with (tmp_path / "cache.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"key": "half-written')  # the PC died mid-write
    assert len(read_cache(tmp_path / "cache.jsonl")) == 2


# --- checking the answer -----------------------------------------------------------


def test_parse_batch_accepts_good_answer():
    batch = [make_ad(1), make_ad(2)]
    labels, problems = parse_batch(answer_for(batch), batch)
    assert set(labels) == {batch[0].id, batch[1].id}
    assert problems == []


def test_parse_batch_drops_bad_items_and_reports_them():
    batch = [make_ad(1), make_ad(2), make_ad(3)]
    answer = answer_for(batch)
    answer["results"][0]["risk_score"] = 1.7  # out of range
    answer["results"][1]["id"] = "someone-else"  # wrong id
    labels, problems = parse_batch(answer, batch)

    assert list(labels) == [batch[2].id]
    assert any("схемы" in p for p in problems)
    assert any("чужой id" in p for p in problems)
    assert any("нет ответа для 2" in p for p in problems)


def test_parse_batch_without_results_list():
    labels, problems = parse_batch({"answer": "oops"}, [make_ad(1)])
    assert labels == {} and problems == ["в ответе нет списка results"]


def test_bad_items_are_not_cached_and_are_asked_again(tmp_path):
    ads = [make_ad(1), make_ad(2)]

    def half_broken(prompt, schema):
        answer = answer_for(ads)
        answer["results"][1]["label"] = "unknown"  # the teacher must decide
        return answer

    report = run(ads, tmp_path, half_broken)
    assert report.labelled == [ads[0].id]
    teacher = FakeTeacher()
    run(ads, tmp_path, teacher)
    assert len(teacher.prompts) == 1 and ads[1].id in teacher.prompts[0]


def test_export_labels(tmp_path):
    ads = [make_ad(1), make_ad(2)]
    run(ads[:1], tmp_path, FakeTeacher())
    out = tmp_path / "labels.jsonl"
    assert export_labels(ads, MODEL, tmp_path / "cache.jsonl", out) == 1
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["id"] == ads[0].id and row["model"] == MODEL and row["label"] == "legit"


# --- metrics -----------------------------------------------------------------------


def label(value, score):
    return TeacherLabel(label=value, risk_score=score, explanation_ru="...")


def test_classification_metrics():
    pairs = [
        ("scam", label("scam", 0.9)),  # TP
        ("scam", label("legit", 0.3)),  # FN
        ("legit", label("scam", 0.6)),  # FP
        ("legit", label("legit", 0.1)),  # TN
        ("legit", label("legit", 0.2)),  # TN
    ]
    m = classification_metrics(pairs)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["accuracy"] == pytest.approx(0.6)


def test_recall_shown_with_and_without_not_from_text_ads():
    hidden = make_ad(1, "scam", "real").model_copy(update={"not_from_text": True})
    visible = make_ad(2, "scam", "case_based")
    labels = {hidden.id: label("legit", 0.05), visible.id: label("scam", 0.9)}
    report = compare_report([hidden, visible], labels, MODEL, "test")
    assert "recall по всем scam: 0.50 (1 из 2)" in report
    assert "recall без помеченных «по тексту не распознать»: 1.00 (1 из 1)" in report
    assert "помечено таких scam: 1" in report


def test_compare_versions_reads_both_prompt_versions(tmp_path, monkeypatch):
    ads = [make_ad(1), make_ad(2)]
    monkeypatch.setattr("aldau.teacher.PROMPT_VERSION", "v1")
    run(ads, tmp_path, FakeTeacher())
    monkeypatch.setattr("aldau.teacher.PROMPT_VERSION", "v2")
    teacher = FakeTeacher()
    run(ads, tmp_path, teacher)
    assert len(teacher.prompts) == 1  # v1 answers were not reused for v2

    table = compare_versions(ads, MODEL, "v1", "v2", tmp_path / "cache.jsonl")
    assert sum(line.startswith(f"| {ad.id}") for ad in ads for line in table) == 2


def test_prompt_contains_the_risk_scale():
    prompt = build_prompt([make_ad(1)], PriceReference())
    for band in ("0.00–0.15", "0.15–0.40", "0.40–0.70", "0.70–1.00"):
        assert band in prompt


def test_compare_report_splits_by_origin():
    ads = [
        make_ad(1, "scam", "case_based"),
        make_ad(2, "scam", "real"),
        make_ad(3, "legit", "real"),
        make_ad(4, "unknown", "real"),  # no human label: left out
    ]
    labels = {
        ads[0].id: label("scam", 0.9),
        ads[1].id: label("legit", 0.4),
        ads[2].id: label("legit", 0.1),
        ads[3].id: label("legit", 0.1),
    }
    report = compare_report(ads, labels, MODEL, "test")
    table = [line for line in report.splitlines() if line.startswith("| ")]
    rows = {line.split("|")[1].strip(): line for line in table}
    assert "| 3 |" in rows["все"]
    assert "| 2 |" in rows["real"]
    assert "| 1 |" in rows["case_based"]
