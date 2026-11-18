import pytest

from aldau.collect import import_inbox, parse_block, read_ads, split_inbox

GOOD_INBOX = """\
olx | Телефоны | 220 000 | Алматы | legit | real
iPhone 13 128 гб
Продам айфон, состояние хорошее.
Звоните 87011234567
---
krisha | аренда квартир | 130000 | Павлодар | scam | case_based
Сдам квартиру
Задаток 30 000 тг на карту, ключи после оплаты.
---

olx | ноутбуки |  | Актобе |  |
Ноутбук сатылады
Жағдайы өте жақсы
---
"""


def write_inbox(tmp_path, text):
    path = tmp_path / "inbox.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_split_inbox_reports_header_line_numbers():
    blocks = split_inbox(GOOD_INBOX)
    assert [line_no for line_no, _ in blocks] == [1, 6, 11]
    assert blocks[2][1][0].startswith("olx | ноутбуки")


def test_parse_block_reads_header_title_and_text():
    ad = parse_block(
        ["olx | Телефоны | 220 000 | Алматы | legit | real", "iPhone 13", "Строка 1", "", "Строка 2"]  # noqa: E501
    )
    assert (ad.source, ad.category, ad.price_kzt, ad.city) == ("olx", "телефоны", 220000, "Алматы")
    assert (ad.label, ad.origin) == ("legit", "real")
    assert ad.title == "iPhone 13"
    assert ad.text == "Строка 1\n\nСтрока 2"


def test_parse_block_empty_fields_get_defaults():
    ad = parse_block(["olx | ноутбуки |  |  |  | ", "Ноутбук сатылады", "Жағдайы өте жақсы"])
    assert ad.price_kzt is None
    assert ad.city is None
    assert (ad.label, ad.origin, ad.lang) == ("unknown", "real", "kk")


def test_parse_block_optional_seventh_field_is_source_note():
    ad = parse_block(
        ["olx | аренда квартир | 90000 | Алматы | scam | real | tg-канал, скриншот", "Сдам", "..."]
    )
    assert ad.source_note == "tg-канал, скриншот"


def test_source_note_keeps_news_link_but_masks_phone():
    note = "https://tengrinews.kz/news/123/ тел 87011234567"
    header = f"olx | мебель | 135000 |  | scam | real | {note}"
    ad = parse_block([header, "Продам гарнитур"])
    assert ad.source_note == "https://tengrinews.kz/news/123/ тел [PHONE]"
    assert parse_block(["olx | мебель | 5000 | Тараз | legit | real", "Стул"]).source_note is None


def test_parse_block_one_line_ad_uses_title_as_text():
    ad = parse_block(["olx | мебель | 5000 | Тараз | legit | real", "Продам стул"])
    assert ad.title == ad.text == "Продам стул"


@pytest.mark.parametrize(
    "lines, message",
    [
        (["olx | телефоны | 100 | Алматы | legit", "Заголовок"], "6 полей"),
        (["avito | телефоны | 100 | Алматы | legit | real", "Заголовок"], "источник"),
        (["olx |  | 100 | Алматы | legit | real", "Заголовок"], "категория"),
        (["olx | телефоны | дёшево | Алматы | legit | real", "Заголовок"], "цена"),
        (["olx | телефоны | 100 | Алматы | maybe | real", "Заголовок"], "метка"),
        (["olx | телефоны | 100 | Алматы | legit | news", "Заголовок"], "origin"),
        (["olx | телефоны | 100 | Алматы | legit | real", "  "], "нет текста"),
    ],
)
def test_parse_block_rejects_bad_ads(lines, message):
    with pytest.raises(ValueError, match=message):
        parse_block(lines)


def test_import_adds_all_ads_and_masks_pii(tmp_path):
    raw = tmp_path / "ads.jsonl"
    report = import_inbox(write_inbox(tmp_path, GOOD_INBOX), raw)

    saved = read_ads(raw)
    assert len(report.added) == len(saved) == 3
    assert report.errors == [] and report.duplicates == []
    assert "87011234567" not in saved[0].text
    assert "[PHONE]" in saved[0].text
    assert len(report.masked) == 1  # only the first ad had personal data
    assert [ad.origin for ad in saved] == ["real", "case_based", "real"]


def test_import_skips_bad_ads_but_keeps_good_ones(tmp_path):
    inbox = GOOD_INBOX + "avito | телефоны | 100 | Алматы | legit | real\nПлохой источник\n---\n"
    raw = tmp_path / "ads.jsonl"
    report = import_inbox(write_inbox(tmp_path, inbox), raw)

    assert len(report.added) == 3
    assert len(report.errors) == 1
    assert "№4" in report.errors[0] and "строка 15" in report.errors[0]


def test_import_twice_adds_nothing_new(tmp_path):
    inbox, raw = write_inbox(tmp_path, GOOD_INBOX), tmp_path / "ads.jsonl"
    import_inbox(inbox, raw)
    report = import_inbox(inbox, raw)

    assert report.added == []
    assert len(report.duplicates) == 3
    assert len(read_ads(raw)) == 3


def test_import_detects_duplicate_inside_one_file(tmp_path):
    block = "olx | мебель | 5000 | Тараз | legit | real\nПродам стул\n---\n"
    report = import_inbox(write_inbox(tmp_path, block + block), tmp_path / "ads.jsonl")
    assert len(report.added) == 1
    assert len(report.duplicates) == 1


def test_dry_run_writes_nothing(tmp_path):
    raw = tmp_path / "ads.jsonl"
    report = import_inbox(write_inbox(tmp_path, GOOD_INBOX), raw, dry_run=True)
    assert len(report.added) == 3
    assert not raw.exists()


def test_import_accepts_file_with_bom(tmp_path):
    path = tmp_path / "inbox.txt"
    path.write_text(GOOD_INBOX, encoding="utf-8-sig")  # how Windows Notepad may save it
    assert len(import_inbox(path, tmp_path / "ads.jsonl").added) == 3
