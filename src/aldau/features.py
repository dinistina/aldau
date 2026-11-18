"""Hand-made text features.

Phase 1: PII masking, language detection, prices written in the text.
Phase 2: fraud signals (regular expressions, Russian + Kazakh) and the price ratio
         "asking price / typical price in this category".

PII = personally identifiable information: phones, emails, links, usernames, names.
We replace it with placeholders like [PHONE] *before* an ad is stored or sent to any
model. The placeholder keeps the useful fact ("the seller left a phone number")
without keeping the personal data itself.
"""

import re

# --- Email -------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# --- Links -------------------------------------------------------------------
# 1) anything starting with http(s):// or www.
# 2) bare domains with a common ending: t.me/xxx, wa.me/7701..., olx.kz/abc
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:kz|ru|com|me|org|net|io|info|link|site|online|xyz|top|cc)\b(?:/\S*)?",
    re.IGNORECASE,
)

# --- Usernames: @seller_almaty ------------------------------------------------
USERNAME_RE = re.compile(r"(?<!\w)@\w{3,}")

# --- Phones (Kazakhstan) -----------------------------------------------------
# Kazakhstan numbers: +7 or 8, then 10 digits, written with any mix of spaces,
# dashes, dots and brackets:  +7 (701) 123-45-67,  87011234567,  8 701 123 45 67
# (?<!\d) and (?!\d) mean "no digit right before / after", so we never cut a
# piece out of a longer number. Prices like "8 500 000" do not match because the
# digit groups must be exactly 3-3-2-2.
SEP = r"[\s\-.()]*"
PHONE_RE = re.compile(
    rf"(?<!\d)(?:\+?7|8){SEP}\d{{3}}{SEP}\d{{3}}{SEP}\d{{2}}{SEP}\d{{2}}(?!\d)"
    # the same number written without +7 / 8:  701 123 45 67
    rf"|(?<!\d)7\d{{2}}{SEP}\d{{3}}{SEP}\d{{2}}{SEP}\d{{2}}(?!\d)"
)

# --- Names -------------------------------------------------------------------
# Names cannot be found reliably with regular expressions. We use two cheap tricks
# and accept that some names slip through (so: check the text by eye before saving).
UPPER = "А-ЯЁӘҒҚҢӨҰҮҺІ"
LOWER = "а-яёәғқңөұүһі"

# Trick 1: a capitalised word right after a phrase that introduces a name.
NAME_AFTER_TRIGGER_RE = re.compile(
    r"\b((?i:меня зовут|зовут|обращаться к|обращайтесь к|спросить|спросите"
    r"|контактное лицо:?|менің атым|атым|есімім))"
    rf"\s+[{UPPER}][{LOWER}]+"
)

# Trick 2: a short list of common first names (Kazakh and Russian).
# Ambiguous ones (Роза, Вера, Дана, Алмас...) are left out on purpose: they are
# also ordinary words and would be masked by mistake.
COMMON_NAMES = [
    "Айгуль", "Айгерим", "Асель", "Асем", "Алия", "Айжан", "Гульнара", "Динара",
    "Жанар", "Мадина", "Сауле", "Камила", "Зарина", "Арман", "Асхат", "Бауыржан",
    "Берик", "Данияр", "Ерлан", "Ержан", "Жандос", "Канат", "Марат", "Нурлан",
    "Нурсултан", "Руслан", "Серик", "Тимур", "Азамат", "Бекзат", "Дархан", "Ернар",
    "Санжар", "Александр", "Алексей", "Андрей", "Дмитрий", "Евгений", "Иван",
    "Максим", "Михаил", "Николай", "Сергей", "Владимир", "Олег", "Анна", "Елена",
    "Ирина", "Мария", "Наталья", "Ольга", "Светлана", "Татьяна", "Екатерина", "Юлия",
]  # fmt: skip
# Optional ending covers simple case forms: "к Арману", "у Марата".
COMMON_NAME_RE = re.compile(r"\b(?:" + "|".join(COMMON_NAMES) + r")(?:а|у|ы|е|ом|ой|я|ю)?\b")

KAZAKH_LETTERS = set("әғқңөұүһі")
# Many people type Kazakh on a Russian keyboard, without the special letters
# ("жаксы" instead of "жақсы"). These frequent ad words catch such texts.
KAZAKH_WORDS = {
    "сатылады", "сатамын", "сатам", "жаксы", "жагдайы", "бериледи", "беремин", "берем",
    "жалга", "болме", "болмели", "жок", "керемет", "багасы", "келисим", "арзан", "жумыс",
    "уй", "ушин", "барлык", "барлыгы", "бари", "жасап", "жасайды", "озим", "касында", "емес",
}  # fmt: skip


def mask_pii(text: str, keep_links: bool = False) -> str:
    """Replace personal data in `text` with placeholders.

    Order matters: emails go first (otherwise the URL rule would eat "mail.ru"
    out of an address), links go before phones (a wa.me/7701... link contains a
    phone number and should become one [LINK]).

    keep_links=True is for our own notes (e.g. "source: a news article URL"):
    a link to a news site is not personal data and is needed to trace the source.
    """
    text = EMAIL_RE.sub("[EMAIL]", text)
    if not keep_links:
        text = URL_RE.sub("[LINK]", text)
    text = USERNAME_RE.sub("[USERNAME]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = NAME_AFTER_TRIGGER_RE.sub(r"\1 [NAME]", text)
    text = COMMON_NAME_RE.sub("[NAME]", text)
    return text


# --- Prices written in the ad text -------------------------------------------
# A number such as "25.000", "8 500 000", "9.5", optionally followed by a multiplier
# ("тыс", "мың", "млн", or "к" glued to the number: "340к") and a currency ("тг", "₸").
NUMBER = r"\d{1,3}(?:[ . ]\d{3})+|\d+(?:[.,]\d+)?"
AMOUNT_RE = re.compile(
    # not glued to a letter, digit or dash: "215-51K" is a model name, not 51 000 tenge
    rf"(?<![\w.,\-])({NUMBER})"
    r"(?:\s?(млн\w*|миллион\w*|тыс\w*|мың\w*|мын\w*)|([кk])(?!\w))?"
    r"\s?(тг|тенге|теңге|₸|tg|kzt)?",
    re.IGNORECASE,
)
# A bare number counts as a price only right after one of these words.
PRICE_WORDS_RE = re.compile(
    r"(цена|цены|бағасы|баға|стоимость|окончательно|отдам за|отдаю за)\W*$", re.IGNORECASE
)
# Amounts near these words are something else: a deposit, the old price, a discount...
NOT_PRICE_WORDS_RE = re.compile(
    r"задат|депозит|предоплат|брон|залог|взнос|аванс|покупал|стоит|рассроч|доплат|скидк"
    r"|таможен|утильсбор|арзан|дешевле|ниже",
    re.IGNORECASE,
)


def parse_number(raw: str) -> float:
    """'25.000' -> 25000, '8 500 000' -> 8500000, '9.5' -> 9.5"""
    raw = raw.replace(" ", " ")
    if re.fullmatch(r"\d{1,3}(?:[ .]\d{3})+", raw):  # dots/spaces separate thousands
        return float(raw.replace(" ", "").replace(".", ""))
    return float(raw.replace(",", "."))


def extract_prices(text: str) -> list[int]:
    """Amounts in tenge that look like the asking price.

    A number counts if it has a currency or a multiplier ("25 000 тг", "340к",
    "120 мың"), or follows a price word ("цена 60000"). Numbers next to deposit /
    old-price words are skipped. This is a heuristic for warnings, not a parser.
    """
    prices = []
    for match in AMOUNT_RE.finditer(text):
        number, big_unit, k_unit, currency = match.groups()
        before = text[max(0, match.start() - 30) : match.start()]
        after = text[match.end() : match.end() + 15]
        if not (big_unit or k_unit or currency or PRICE_WORDS_RE.search(before)):
            continue
        if NOT_PRICE_WORDS_RE.search(before) or NOT_PRICE_WORDS_RE.search(after):
            continue
        value = parse_number(number)
        if k_unit:
            if value < 10:  # "2к" usually means a 2-room flat, not 2000 tenge
                continue
            value *= 1000
        elif big_unit and big_unit.lower().startswith(("млн", "миллион")):
            if value < 1000:  # "20 млн"; but "20 000 000 млн" is already in tenge
                value *= 1_000_000
        elif big_unit and value < 1000:  # "120 мың" = 120 000; "180 000 мың" is already full
            value *= 1000
        prices.append(int(value))
    return prices


def all_numbers(text: str) -> set[int]:
    """Every number in the text, thousands separators removed."""
    return {int(parse_number(raw)) for raw in re.findall(NUMBER, text)}


def detect_lang(text: str) -> str:
    """Return "kk" if the text looks Kazakh, otherwise "ru".

    Russian and Kazakh share the Cyrillic alphabet, but Kazakh has 9 extra letters
    (ә ғ қ ң ө ұ ү һ і). One such letter is enough to call the text Kazakh.
    If there are none, we look for frequent Kazakh words typed with Russian letters.
    A mixed ad (Kazakh + Russian) counts as Kazakh.
    """
    text = text.lower()
    if KAZAKH_LETTERS & set(text):
        return "kk"
    if KAZAKH_WORDS & set(re.findall(r"\w+", text)):
        return "kk"
    return "ru"


# =============================================================================
# Phase 2 — fraud signals
# =============================================================================
#
# Each signal is one regular expression that looks for a typical piece of a scam
# scheme. A signal is *evidence*, not a verdict: honest ads also say "срочно" or ask
# for a deposit. The model (and the teacher LLM) decides from the combination.
#
# Kazakh words change their endings a lot (аудар-ыңыз, аудар-саңыз, аудар-асыз),
# so most Kazakh patterns are a stem + \w* ("any ending").
# \b = word boundary: \bтаможен matches "таможенный" but not "растаможен"
# (every Kolesa ad says "растаможен в Казахстане", which is not a signal).

SIGNAL_PATTERNS: dict[str, str] = {
    # money before the buyer has seen the thing
    "prepayment": (
        r"предоплат\w*|задат\w*|брон[ьиюя]\w*|аванс\w*|\bвперёд|\bвперед"
        r"|100\s?% оплат\w*|полн\w* оплат\w*|полн\w* сумм\w*|после (оплаты|перевода)"
        r"|переводите оплату|перевод\w* оплат\w*|на (мои )?реквизиты|после поступления"
        r"|на карту|на кошел\w*|на kaspi|kaspi-?ге"
        r"|алдын ала (төлем|аудар)\w*|аудар(ыңыз|саңыз|асыз|у керек)\w*|толық соманы"
        r"|ақшаны толық"
    ),
    # a deposit is normal for long-term rent, so it is a separate, weaker signal
    "deposit": r"депозит\w*|залог\w*",
    "messenger": (
        r"whats\s?app?|ватсап\w*|вацап\w*|вотсап\w*|ватцап\w*|telegram|телеграм\w*"
        r"|мессенджер\w*|в личные сообщения|в личку"
    ),
    # refuses to talk on the phone: harder to check the seller
    "no_calls": (
        r"на звонки не отвечаю|звонки не отвечаю|звонить не надо|не звоните"
        r"|пишите только|только в (whatsapp|ватсап\w*|телеграм\w*)|звонки не доходят"
        r"|отвечаю редко|қоңырау\w* жауап бермеймін|тек ватсап\w*"
    ),
    # the seller is in a hurry
    "urgency": r"срочн\w*|срочна|уезжа\w*|улетаю|на пмж|тез арада|шұғыл\w*|асығыс",
    # the buyer is pushed to decide fast
    "pressure": (
        r"желающих (очень )?много|покупател\w* много|звонков (очень )?много|кто первый"
        r"|решайте быстр\w*|ещё (два|несколько) просмотр\w*|осталось (одно|одна|мало)"
        r"|количество ограничен\w*|мест\w* ограничен\w*|даты не держу"
        r"|саны шектеулі|адам көп|тезірек|үлгеріп қалыңыз|сұраныс көп|тез бітеді"
    ),
    # the seller cannot meet or show the thing in person
    "remote_seller": (
        r"в другом городе|за границ\w*|за рубеж\w*|в командировке|на вахте"
        r"|приехать не могу|не могу приехать|встретиться не могу|не могу встретиться"
        r"|показать смогу только|показать не могу|осмотреть заранее не получится"
        r"|отправлю (почтой|курьер\w*)|переехал жить|басқа қала\w*|шетелде"
        r"|кездесе алмаймын|көрсете алмаймын|ауылдамын|поштамен"
    ),
    # goods "from abroad" or "stuck at customs"
    "from_abroad": (
        r"из-за границы|из (южной )?кореи|из оаэ|из эмират\w*|из дубая|из японии"
        r"|из грузии|из америки|из европы|растаможк\w*|\bтаможен\w*|\bтаможн\w*"
        r"|на границе|под заказ|кореядан|жапониядан|шетелден|дубай\w*"
    ),
    # fake "OLX delivery / safe deal" page that asks for card details
    "fake_delivery": (
        r"данные карты|карт\w* деректер\w*|безопасн\w* сделк\w*|olx доставк\w*"
        r"|olx жеткізу\w*|ссылк\w* для оплаты|по ссылке|сілтеме\w*"
    ),
    # the seller offers copies of an ID or ownership papers to win trust
    "sends_documents": (
        r"удостоверени\w*|копи\w* документ\w*|фото документ\w*"
        r"|документ\w* (скину|отправлю|пришлю)|жеке куәлі\w*|құжат\w* (суретін|жіберемін)"
    ),
    # the ad itself says it is cheap
    "says_cheap": (
        r"ниже рынка|ниже рыночной|дешевле|вдвое ниже|в (два|три|2-3) раза (ниже|дешевле)"
        r"|недорого|дёшево|дешево|арзан\w*"
    ),
    # masked contacts left by mask_pii()
    "has_link": r"\[LINK\]",
    "has_phone": r"\[PHONE\]",
}

SIGNALS = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in SIGNAL_PATTERNS.items()}
SIGNAL_NAMES = list(SIGNALS)

# Short Russian descriptions, used in reports and later in the bot's explanations.
SIGNAL_DESCRIPTIONS_RU = {
    "prepayment": "просят предоплату, задаток или бронь",
    "deposit": "упоминается депозит или залог",
    "messenger": "общение через мессенджер",
    "no_calls": "продавец не отвечает на звонки",
    "urgency": "срочность",
    "pressure": "давление на покупателя: «желающих много», «осталось мало»",
    "remote_seller": "продавец не может встретиться или показать товар",
    "from_abroad": "товар из-за границы или «на таможне»",
    "fake_delivery": "оплата по ссылке или «безопасная сделка» с вводом данных карты",
    "sends_documents": "продавец обещает прислать удостоверение или документы",
    "says_cheap": "объявление само подчёркивает низкую цену",
    "has_link": "в тексте есть ссылка",
    "has_phone": "в тексте есть номер телефона",
}


def find_signals(text: str) -> list[str]:
    """Names of the signals found in the text, in a fixed order."""
    return [name for name, regex in SIGNALS.items() if regex.search(text)]


# =============================================================================
# Phase 2 — price compared with the market
# =============================================================================


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


class PriceReference:
    """Typical ("market") price for a category, learned from a set of ads.

    market price = median price of ads in the same category and city; if there are
    fewer than `min_count` such ads, the median of the whole category; if even that
    is too small, unknown (None).

    Median, not mean: one villa for a billion would move the mean a lot, but barely
    moves the median.

    The labels are NOT used, only prices, so the feature does not peek at the
    answer. Fit it on training ads only (Phase 5): if test ads take part in the
    median, information from the test set leaks into training.
    """

    def __init__(self, min_count: int = 3):
        self.min_count = min_count
        self.by_category: dict[str, float] = {}
        self.by_category_city: dict[tuple[str, str], float] = {}

    def fit(self, ads: list) -> "PriceReference":
        by_category: dict[str, list[int]] = {}
        by_category_city: dict[tuple[str, str], list[int]] = {}
        for ad in ads:
            if not ad.price_kzt:  # None or 0: no price, nothing to learn
                continue
            by_category.setdefault(ad.category, []).append(ad.price_kzt)
            if ad.city:
                by_category_city.setdefault((ad.category, ad.city), []).append(ad.price_kzt)
        self.by_category = {
            key: median(prices)
            for key, prices in by_category.items()
            if len(prices) >= self.min_count
        }
        self.by_category_city = {
            key: median(prices)
            for key, prices in by_category_city.items()
            if len(prices) >= self.min_count
        }
        return self

    def market_price(self, ad) -> float | None:
        if (ad.category, ad.city) in self.by_category_city:
            return self.by_category_city[(ad.category, ad.city)]
        return self.by_category.get(ad.category)

    def ratio(self, ad) -> float | None:
        """price / market price. 0.3 = 70 % cheaper than usual; None = unknown."""
        market = self.market_price(ad)
        if not ad.price_kzt or not market:
            return None
        return ad.price_kzt / market


def extract_features(ad, prices: PriceReference) -> dict:
    """All hand-made features of one ad as a flat dict (input for Phase 3 and 5)."""
    found = set(find_signals(ad.title + "\n" + ad.text))
    features: dict = {name: name in found for name in SIGNAL_NAMES}
    features["price_ratio"] = prices.ratio(ad)
    # has_phone is the opposite of suspicious, so it is not counted
    features["signal_count"] = len(found - {"has_phone"})
    features["text_length"] = len(ad.text)
    return features
