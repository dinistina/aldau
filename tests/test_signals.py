import pytest

from aldau.collect import build_ad
from aldau.features import (
    SIGNAL_DESCRIPTIONS_RU,
    SIGNAL_NAMES,
    PriceReference,
    extract_features,
    find_signals,
    median,
)

# Each signal: phrases that must trigger it (Russian and Kazakh) ...
POSITIVE = {
    "prepayment": [
        "нужна предоплата 20 000 тг",
        "внесите задаток",
        "нужна бронь 50% от месячной оплаты",
        "Оплата 100% вперёд, отправка в тот же день",
        "переводите оплату за первый месяц",
        "алдын ала төлем жасайсыз",
        "30 000 теңге бронь аудару керек",
        "Kaspi-ге аударсаңыз, кілтті береді",
    ],
    "deposit": ["Депозит 35 000 теңге", "залог за ключи"],
    "messenger": ["пишите в WhatsApp", "все вопросы на ватсап", "Вацап", "ватсапқа жазыңыз"],
    "no_calls": [
        "На звонки не отвечаю",
        "пишите только в WhatsApp",
        "Қоңырауға жауап бермеймін",
        "Тек ватсапқа жазыңыздар",
    ],
    "urgency": ["СРОЧНО продаю", "уезжаю на ПМЖ", "тез арада сатылады", "Шұғыл ақша керек"],
    "pressure": [
        "Желающих много, кто первый оплатит",
        "Сегодня ещё два просмотра, решайте быстрее",
        "Количество ограничено",
        "Саны шектеулі, үлгеріп қалыңыз",
        "Қарап келетін адам көп",
    ],
    "remote_seller": [
        "Я сейчас живу в другом городе",
        "хозяин за границей",
        "Нахожусь в командировке, отправлю курьерской службой",
        "Өзім қазір басқа қалада жұмыстамын",
        "Өзім кездесе алмаймын",
    ],
    "from_abroad": [
        "Привезём автомобиль из Южной Кореи",
        "нужно доплатить за растаможку",
        "таможенный платёж",
        "Кореядан көлік әкелеміз",
        "шетелден жеткіземіз",
    ],
    "fake_delivery": [
        "вводите данные карты",
        "отправляю через OLX доставку, это безопасная сделка",
        "картаңыздың деректерін енгізіп төлейсіз",
    ],
    "sends_documents": [
        "Документы на квартиру и удостоверение скину фото",
        "Жеке куәлігім мен пәтердің құжатын жіберемін",
    ],
    "says_cheap": ["цена ниже рынка", "в два раза дешевле", "арзанға беремін", "недорого"],
    "has_link": ["Оплата по ссылке [LINK]"],
    "has_phone": ["Звоните [PHONE]"],
}

# ... and honest phrases that must NOT trigger it.
NEGATIVE = {
    "prepayment": [
        "Броне пленка на капоте",  # "бронепленка" is not a booking
        "Продам телефон, торг при встрече",
    ],
    "from_abroad": ["руль слева, растаможен в Казахстане"],  # every Kolesa ad says this
    "no_calls": ["Звоните или пишите в WhatsApp"],  # messenger is fine, refusing calls is not
    "urgency": ["Продаю в хорошем состоянии"],
    "pressure": ["Реальному покупателю торг"],
    "remote_seller": ["Работаю вахтами, ноутбук лежал дома"],
}


@pytest.mark.parametrize(
    "signal, text", [(name, text) for name, texts in POSITIVE.items() for text in texts]
)
def test_signal_found(signal, text):
    assert signal in find_signals(text)


@pytest.mark.parametrize(
    "signal, text", [(name, text) for name, texts in NEGATIVE.items() for text in texts]
)
def test_signal_not_found(signal, text):
    assert signal not in find_signals(text)


def test_every_signal_has_a_test_and_a_description():
    assert set(POSITIVE) == set(SIGNAL_NAMES)
    assert set(SIGNAL_DESCRIPTIONS_RU) == set(SIGNAL_NAMES)


def test_ordinary_ad_has_no_signals():
    assert find_signals("Продам диван, почти новый. Самовывоз из центра.") == []


def test_median():
    assert median([5, 1, 3]) == 3
    assert median([4, 1, 3, 2]) == 2.5


def ad(price, category="телефоны", city="Алматы", text="Продам телефон"):
    return build_ad("olx", category, "Заголовок", text, price_kzt=price, city=city)


def test_price_reference_uses_city_when_enough_ads():
    ads = [ad(100), ad(200), ad(300), ad(1000, city="Тараз"), ad(1000, city="Тараз")]
    prices = PriceReference(min_count=3).fit(ads)
    assert prices.market_price(ad(50)) == 200  # Алматы has 3 ads: median 200
    # Тараз has only 2 ads -> falls back to the whole category: median of 5 prices
    assert prices.market_price(ad(50, city="Тараз")) == 300


def test_price_reference_unknown_category_or_price():
    prices = PriceReference(min_count=3).fit([ad(100), ad(200), ad(300)])
    assert prices.ratio(ad(100, category="авто")) is None  # no such category
    assert prices.ratio(ad(None)) is None  # no price
    assert prices.ratio(ad(50)) == 0.25


def test_price_reference_ignores_ads_without_price():
    prices = PriceReference(min_count=3).fit([ad(None), ad(None), ad(100), ad(300)])
    assert prices.market_price(ad(1)) is None  # only 2 priced ads


def test_extract_features():
    prices = PriceReference(min_count=1).fit([ad(100_000)])
    text = "Срочно! Предоплата на карту, пишите в WhatsApp [PHONE]"
    features = extract_features(ad(25_000, text=text), prices)
    assert features["prepayment"] and features["urgency"] and features["messenger"]
    assert not features["pressure"]
    assert features["price_ratio"] == 0.25
    assert features["signal_count"] == 3  # has_phone is not counted
    assert set(SIGNAL_NAMES) <= set(features)
