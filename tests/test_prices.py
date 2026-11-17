import pytest

from aldau.collect import (
    build_ad,
    find_price_problems,
    import_inbox,
    make_ad_id,
    price_mismatch,
    read_ads,
    read_price_ok,
)
from aldau.features import extract_prices


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Цена: 350 000 ₸", [350_000]),
        ("отдам за 25.000 окончательно", [25_000]),
        ("окончатлно 340к", [340_000]),
        ("Бағасы: 120 мың тг", [120_000]),
        ("180 000 мынга сатылады", [180_000]),  # already in full, not x1000
        ("3 бөлмелі пәтер. 9.5млн", [9_500_000]),
        ("Цена 20 млн тенге", [20_000_000]),
        ("Продам за 150000тг", [150_000]),
        ("Продам телефон, цена 60000", [60_000]),  # bare number after a price word
    ],
)
def test_extract_prices_finds_asking_price(text, expected):
    assert extract_prices(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Пробег 87 000 км, 2015 год",  # no currency, no price word
        "Acer Extensa 215-51K",  # model name
        "Сдам 2к квартиру",  # 2-room flat
        "Депозит 35 000 теңге",  # deposit, not the price
        "задаток 30 000 тг на Kaspi",
        "Покупали за 100 тысяч",  # old price
        "Новый стоит 36 тыс",
        "базардағы бағадан 2 миллионға арзан",  # "2 million cheaper"
    ],
)
def test_extract_prices_ignores_other_numbers(text):
    assert extract_prices(text) == []


def make_ad(text, price):
    return build_ad("olx", "техника", "Заголовок", text, price_kzt=price)


def test_no_warning_when_header_matches_text():
    assert price_mismatch(make_ad("Цена: 350 000 ₸", 350_000)) is None
    assert price_mismatch(make_ad("Цена 340к", 345_000)) is None  # within 10 %
    assert price_mismatch(make_ad("Без цены в тексте", 50_000)) is None


def test_no_warning_when_header_price_is_written_anywhere():
    text = "Негізгі бағасы 10 000 тг, 3000 ға сатамын"  # old price 10 000, sells for 3000
    assert price_mismatch(make_ad(text, 3000)) is None


def test_warning_when_text_disagrees():
    warning = price_mismatch(make_ad("Цена: 350 000 ₸ с торгом", 315_000))
    assert warning == "в шапке 315 000, в тексте: 350 000"


def test_warning_when_header_has_no_price():
    assert price_mismatch(make_ad("Бағасы 70 мың", None)) == "в шапке цены нет, в тексте: 70 000"


def write(tmp_path, text, name="inbox.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_import_reports_price_warning(tmp_path):
    inbox = "olx | техника | 30000 | Шымкент | legit | real\nLG\nотдам за 25.000 окончательно\n"
    report = import_inbox(write(tmp_path, inbox), tmp_path / "ads.jsonl")
    assert len(report.added) == 1  # a warning does not block the import
    assert len(report.price_warnings) == 1
    assert "в шапке 30 000" in report.price_warnings[0]


def test_reimport_with_fixed_price_updates_stored_ad(tmp_path):
    raw = tmp_path / "ads.jsonl"
    chair = "olx | мебель | 5000 | Тараз | legit | real\nСтул\n"
    body = "LG\nотдам за 25.000 окончательно\n---\n" + chair
    import_inbox(write(tmp_path, "olx | техника | 30000 | Шымкент | legit | real\n" + body), raw)

    fixed = "olx | техника | 25000 | Шымкент | legit | real\n" + body
    report = import_inbox(write(tmp_path, fixed, "fixed.txt"), raw)

    assert report.added == []
    assert len(report.updated) == 1 and "price_kzt" in report.updated[0]
    assert len(report.duplicates) == 1  # the chair did not change
    assert [ad.price_kzt for ad in read_ads(raw)] == [25_000, 5000]  # order kept


def test_read_price_ok_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "price_ok.txt"
    path.write_text("# header\n\nabc123   # LG: торг\n  def456\n", encoding="utf-8")
    assert read_price_ok(path) == {"abc123", "def456"}
    assert read_price_ok(tmp_path / "missing.txt") == set()


def test_checked_ads_are_hidden_from_price_problems():
    lg = make_ad("отдам за 25.000 окончательно", 30_000)
    macbook = make_ad("Цена: 350 000 ₸", 315_000)
    problems, hidden = find_price_problems([lg, macbook], price_ok={lg.id})
    assert [ad.id for ad, _ in problems] == [macbook.id]
    assert hidden == 1


def test_import_skips_price_warning_for_checked_ad(tmp_path):
    inbox = "olx | техника | 30000 | Шымкент | legit | real\nLG\nотдам за 25.000 окончательно\n"
    lg_id = make_ad_id("LG", "отдам за 25.000 окончательно")
    report = import_inbox(write(tmp_path, inbox), tmp_path / "ads.jsonl", price_ok={lg_id})
    assert report.price_warnings == []


def test_dry_run_does_not_update(tmp_path):
    raw = tmp_path / "ads.jsonl"
    import_inbox(write(tmp_path, "olx | мебель | 5000 | Тараз | legit | real\nСтул\n"), raw)
    fixed = write(tmp_path, "olx | мебель | 4000 | Тараз | legit | real\nСтул\n", "fixed.txt")
    report = import_inbox(fixed, raw, dry_run=True)
    assert len(report.updated) == 1
    assert read_ads(raw)[0].price_kzt == 5000
