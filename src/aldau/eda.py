"""Exploratory data analysis (EDA) for Phase 2: writes reports/eda.md.

Run from the project root:
    python -m aldau.eda

What it checks:
1. how often each fraud signal appears in scam vs legit ads;
2. how the price ratio (price / typical price) differs between scam and legit;
3. whether a single field (category, origin, language, ...) predicts the label on
   its own — if it does, a model can "cheat" by looking at that field instead of
   reading the ad.

No numpy / scikit-learn here on purpose: everything is plain Python, so it runs on
any machine and every number can be explained line by line.
Only ads labelled scam or legit are used; "unknown" ads are left out.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from aldau.collect import RAW_PATH, read_ads
from aldau.features import (
    SIGNAL_DESCRIPTIONS_RU,
    SIGNAL_NAMES,
    PriceReference,
    find_signals,
    median,
)
from aldau.photos import find_duplicate_photos

REPORT_PATH = Path("reports/eda.md")
PRICE_BUCKETS = [  # (upper bound, label) for price ratio = price / typical price
    (0.25, "< 0.25 (в 4+ раза дешевле)"),
    (0.5, "0.25–0.5"),
    (0.8, "0.5–0.8"),
    (1.25, "0.8–1.25 (обычная цена)"),
    (2.0, "1.25–2"),
    (float("inf"), "> 2"),
]


# --- metrics --------------------------------------------------------------------


def roc_auc(scores: list[float], is_scam: list[bool]) -> float:
    """Probability that a random scam ad gets a higher score than a random legit ad.

    0.5 = the score is useless (a coin), 1.0 = it separates the classes perfectly.
    Ties count as half. This is exactly ROC-AUC, computed by comparing every pair.
    """
    scam = [s for s, y in zip(scores, is_scam, strict=True) if y]
    legit = [s for s, y in zip(scores, is_scam, strict=True) if not y]
    if not scam or not legit:
        return float("nan")
    wins = sum(1.0 if s > lg else 0.5 if s == lg else 0.0 for s in scam for lg in legit)
    return wins / (len(scam) * len(legit))


def stratified_folds(is_scam: list[bool], folds: int = 5, seed: int = 42) -> list[int]:
    """Fold number (0..folds-1) for every ad, with the same scam share in every fold.

    Scam ads and legit ads are shuffled separately and dealt out like cards, so each
    fold gets ~1/5 of the scam ads and ~1/5 of the legit ads ("stratified").
    """
    import random  # only needed here

    fold_of = [0] * len(is_scam)
    rng = random.Random(seed)
    for label in (True, False):
        indices = [i for i, y in enumerate(is_scam) if y == label]
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            fold_of[index] = position % folds
    return fold_of


def cross_val_rate(
    values: list[str], is_scam: list[bool], folds: int = 5, prior: float = 1.0
) -> list[float]:
    """For every ad: share of scam among ads with the same value in the OTHER folds.

    This is the simplest possible classifier that looks at one field only, checked
    with k-fold cross-validation: an ad is never scored with its own label.
    `prior` pulls rare values towards the overall scam rate (a value seen twice
    should not give 0 % or 100 %).

    Why not leave-one-out: with few distinct values it is biased — removing the
    ad's own label pushes its value's rate *away* from that label, so a useless
    field shows ROC-AUC well below 0.5.
    """
    fold_of = stratified_folds(is_scam, folds)
    rates = [0.0] * len(values)
    for fold in range(folds):
        train = [i for i in range(len(values)) if fold_of[i] != fold]
        total = Counter(values[i] for i in train)
        scam = Counter(values[i] for i in train if is_scam[i])
        overall = sum(is_scam[i] for i in train) / len(train)
        for i in range(len(values)):
            if fold_of[i] == fold:
                rates[i] = (scam[values[i]] + prior * overall) / (total[values[i]] + prior)
    return rates


def auc_strength(auc: float) -> str:
    if auc != auc:  # NaN
        return "—"
    if auc >= 0.9:
        return "очень сильно — почти угадывает метку"
    if auc >= 0.75:
        return "сильно"
    if auc >= 0.6:
        return "заметно"
    return "слабо / почти случайно"


# --- report sections ------------------------------------------------------------


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f} %" if whole else "—"


def section_overview(ads: list, all_ads: list) -> list[str]:
    labels = Counter(ad.label for ad in ads)
    by_origin = Counter((ad.origin, ad.label) for ad in ads)
    lines = [
        "## 1. Данные",
        "",
        f"Всего объявлений: {len(all_ads)}; в анализе (scam или legit): {len(ads)} — "
        f"scam {labels['scam']}, legit {labels['legit']}. "
        f"Объявления с меткой unknown ({len(all_ads) - len(ads)}) не используются.",
        "",
        "| origin | scam | legit |",
        "|---|---|---|",
    ]
    for origin in sorted({ad.origin for ad in ads}):
        scam, legit = by_origin[(origin, "scam")], by_origin[(origin, "legit")]
        lines.append(f"| {origin} | {scam} | {legit} |")
    return lines


def section_signals(ads: list) -> list[str]:
    scam = [ad for ad in ads if ad.label == "scam"]
    legit = [ad for ad in ads if ad.label == "legit"]
    found = {ad.id: set(find_signals(ad.title + "\n" + ad.text)) for ad in ads}
    lines = [
        "## 2. Сигналы мошенничества",
        "",
        "Доля объявлений, в которых найден сигнал. Сигнал полезен, если доли сильно различаются.",
        "",
        f"| сигнал | что ищет | scam (n={len(scam)}) | legit (n={len(legit)}) |",
        "|---|---|---|---|",
    ]
    for name in SIGNAL_NAMES:
        in_scam = sum(name in found[ad.id] for ad in scam)
        in_legit = sum(name in found[ad.id] for ad in legit)
        lines.append(
            f"| `{name}` | {SIGNAL_DESCRIPTIONS_RU[name]} | {pct(in_scam, len(scam))} "
            f"| {pct(in_legit, len(legit))} |"
        )
    none_scam = sum(not (found[ad.id] - {"has_phone"}) for ad in scam)
    none_legit = sum(not (found[ad.id] - {"has_phone"}) for ad in legit)
    lines += [
        "",
        f"Без единого сигнала (кроме телефона): scam {pct(none_scam, len(scam))}, "
        f"legit {pct(none_legit, len(legit))}.",
    ]
    return lines


def section_prices(ads: list, prices: PriceReference) -> list[str]:
    lines = [
        "## 3. Цена относительно рынка",
        "",
        "`price_ratio` = цена объявления / медианная цена в той же категории "
        "(и городе, если там хотя бы 3 объявления). 0.3 = на 70 % дешевле обычного.",
        "",
        "| price_ratio | scam | legit |",
        "|---|---|---|",
    ]
    ratios = {label: [] for label in ("scam", "legit")}
    for ad in ads:
        ratio = prices.ratio(ad)
        if ratio is not None:
            ratios[ad.label].append(ratio)
    for i, (upper, name) in enumerate(PRICE_BUCKETS):
        lower = PRICE_BUCKETS[i - 1][0] if i else 0.0
        counts = {
            label: sum(lower <= r < upper for r in values) for label, values in ratios.items()
        }
        lines.append(f"| {name} | {counts['scam']} | {counts['legit']} |")
    lines.append("")
    for label, values in ratios.items():
        if values:
            lines.append(
                f"- {label}: цена известна у {len(values)}, медиана price_ratio "
                f"{median(values):.2f}, дешевле половины рынка — "
                f"{pct(sum(r < 0.5 for r in values), len(values))}"
            )
    return lines


def section_single_field(ads: list, prices: PriceReference) -> list[str]:
    is_scam = [ad.label == "scam" for ad in ads]

    def categorical(field: Callable) -> list[float]:
        return cross_val_rate([str(field(ad)) for ad in ads], is_scam)

    def price_score(ad) -> float:
        ratio = prices.ratio(ad)
        return -ratio if ratio is not None else -1.0  # cheaper = more suspicious

    checks = [
        ("категория", "category", categorical(lambda ad: ad.category)),
        ("источник (olx / kolesa / krisha)", "source", categorical(lambda ad: ad.source)),
        ("язык", "lang", categorical(lambda ad: ad.lang)),
        ("город", "city", categorical(lambda ad: ad.city)),
        ("origin (real / case_based)", "origin", categorical(lambda ad: ad.origin)),
        (
            "число сигналов",
            "signal_count",
            [len(set(find_signals(ad.title + "\n" + ad.text)) - {"has_phone"}) for ad in ads],
        ),
        ("price_ratio (дешевле = подозрительнее)", "price_ratio", [price_score(ad) for ad in ads]),
    ]
    lines = [
        "## 4. Предсказывает ли одно поле метку само по себе?",
        "",
        "Для каждого поля — простейший «классификатор», который видит только это поле "
        "(для категориальных полей: доля scam среди объявлений с тем же значением, посчитанная "
        "на других фолдах, 5-кратная стратифицированная кросс-валидация). "
        "ROC-AUC: 0.5 — поле бесполезно, 1.0 — поле одно угадывает метку.",
        "",
        "Для полей, не связанных со смыслом объявления (категория, origin, источник, город), "
        "высокий ROC-AUC — это **перекос данных**: модель может выучить его вместо схемы обмана.",
        "",
        "| поле | ROC-AUC | насколько выдаёт метку |",
        "|---|---|---|",
    ]
    for title, _, scores in checks:
        auc = roc_auc(scores, is_scam)
        lines.append(f"| {title} | {auc:.2f} | {auc_strength(auc)} |")

    lines += ["", "Доля scam по категориям:", "", "| категория | scam | legit | % scam |"]
    lines.append("|---|---|---|---|")
    by_category = Counter((ad.category, ad.label) for ad in ads)
    for category in sorted({ad.category for ad in ads}):
        scam, legit = by_category[(category, "scam")], by_category[(category, "legit")]
        flag = " ⚠️" if scam == 0 or legit == 0 else ""
        lines.append(f"| {category} | {scam} | {legit} | {pct(scam, scam + legit)}{flag} |")
    lines += ["", "⚠️ — в категории только один класс: по ней модель может угадывать метку."]
    return lines


def section_photos(all_ads: list) -> list[str]:
    with_photos = [ad for ad in all_ads if ad.photos]
    lines = ["## 5. Повторяющиеся фото", ""]
    if not with_photos:
        return lines + ["Фото пока не собраны — проверка не выполнялась."]
    pairs = find_duplicate_photos(with_photos)
    lines.append(f"Объявлений с фото: {len(with_photos)}. Пар с одинаковым фото: {len(pairs)}.")
    lines += [f"- `{a}` и `{b}` (расстояние {d} бит из 64)" for a, b, d in pairs]
    return lines


CAVEATS = [
    "## Оговорки",
    "",
    "- Данных мало (около 140 размеченных объявлений): каждая доля — это несколько штук, "
    "разница в 5–10 процентных пунктов может быть случайной.",
    "- Все scam-объявления сейчас `case_based` (переписаны вручную), все legit — `real`. "
    "Поэтому различия в сигналах частично отражают то, *как писались* кейсы: в пересказе "
    "схемы слово «предоплата» обычно есть явно. Проверка на настоящих мошеннических "
    "объявлениях — после того как они будут собраны.",
    "- Медианы цен посчитаны по всем объявлениям, включая само объявление. Для EDA это "
    "допустимо; при обучении (Фаза 5) медианы считаются только по train.",
]


def build_report(raw_path: Path = RAW_PATH) -> str:
    all_ads = read_ads(raw_path)
    ads = [ad for ad in all_ads if ad.label in ("scam", "legit")]
    prices = PriceReference(min_count=3).fit(all_ads)
    parts = [
        ["# EDA — Фаза 2", "", "Сгенерировано `python -m aldau.eda` по `data/raw/ads.jsonl`."],
        section_overview(ads, all_ads),
        section_signals(ads),
        section_prices(ads, prices),
        section_single_field(ads, prices),
        section_photos(all_ads),
        CAVEATS,
    ]
    return "\n\n".join("\n".join(part) for part in parts) + "\n"


def main() -> None:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Отчёт записан в {REPORT_PATH}")


if __name__ == "__main__":
    main()
