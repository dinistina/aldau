"""Hand-made text features.

Phase 1: PII masking and language detection.
Phase 2 will add fraud signals (prepayment, messenger, urgency, price ratio).

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


def mask_pii(text: str) -> str:
    """Replace personal data in `text` with placeholders.

    Order matters: emails go first (otherwise the URL rule would eat "mail.ru"
    out of an address), links go before phones (a wa.me/7701... link contains a
    phone number and should become one [LINK]).
    """
    text = EMAIL_RE.sub("[EMAIL]", text)
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
