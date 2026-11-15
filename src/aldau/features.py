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


def detect_lang(text: str) -> str:
    """Return "kk" if the text has Kazakh-only letters, otherwise "ru".

    Russian and Kazakh share the Cyrillic alphabet, but Kazakh has 9 extra letters
    (ә ғ қ ң ө ұ ү һ і). One such letter is enough to call the text Kazakh.
    """
    if KAZAKH_LETTERS & set(text.lower()):
        return "kk"
    return "ru"
