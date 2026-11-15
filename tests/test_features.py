import pytest

from aldau.features import detect_lang, mask_pii


@pytest.mark.parametrize(
    "phone",
    [
        "+7 (701) 123-45-67",
        "+77011234567",
        "87011234567",
        "8 701 123 45 67",
        "8-701-123-45-67",
        "8(701)1234567",
        "701 123 45 67",
    ],
)
def test_masks_phone_formats(phone):
    masked = mask_pii(f"Звоните {phone} в любое время")
    assert masked == "Звоните [PHONE] в любое время"


@pytest.mark.parametrize(
    "text",
    [
        "Цена 8 500 000 тенге",
        "Продам за 2 500 000",
        "Пробег 87000 км, 2015 год",
        "Площадь 75 кв.м, 3 комнаты",
    ],
)
def test_does_not_mask_prices_and_numbers(text):
    assert mask_pii(text) == text


def test_masks_email():
    assert mask_pii("Пишите на seller.almaty@mail.ru") == "Пишите на [EMAIL]"


@pytest.mark.parametrize(
    "link",
    ["https://olx.kz/d/obyavlenie/abc123", "www.example.com/pay", "t.me/seller_kz", "kaspi.kz"],
)
def test_masks_links(link):
    assert mask_pii(f"Подробнее: {link}") == "Подробнее: [LINK]"


def test_whatsapp_link_with_phone_becomes_one_link():
    assert mask_pii("Пишите wa.me/77011234567") == "Пишите [LINK]"


def test_masks_username():
    assert mask_pii("Телеграм @seller_almaty") == "Телеграм [USERNAME]"


def test_masks_name_after_trigger_phrase():
    assert mask_pii("Меня зовут Жулдыз, звоните") == "Меня зовут [NAME], звоните"
    assert mask_pii("Спросить Ботагоз") == "Спросить [NAME]"


def test_masks_kazakh_name_trigger():
    assert mask_pii("Менің атым Гүлнұр") == "Менің атым [NAME]"


def test_masks_common_names_and_case_forms():
    assert mask_pii("Обращаться к Арману") == "Обращаться к [NAME]"
    assert mask_pii("Продаёт Айгуль") == "Продаёт [NAME]"


def test_does_not_mask_surname_like_words():
    # "Ивановка" starts with a listed name but is a different word.
    assert mask_pii("село Ивановка") == "село Ивановка"


def test_masks_several_things_at_once():
    text = "Меня зовут Асхат, тел 87011234567, почта a@b.kz, сайт https://x.kz"
    assert mask_pii(text) == "Меня зовут [NAME], тел [PHONE], почта [EMAIL], сайт [LINK]"


def test_text_without_pii_is_unchanged():
    text = "Продам диван, почти новый. Самовывоз из центра."
    assert mask_pii(text) == text


def test_detect_lang():
    assert detect_lang("Продам квартиру в центре") == "ru"
    assert detect_lang("Пәтер жалға беріледі") == "kk"
    assert detect_lang("ҚАЛА ОРТАЛЫҒЫ") == "kk"  # upper case works too
