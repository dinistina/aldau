"""Metrics on the gold set (Phase 5 now, Phase 6 benchmark later).

Run from the project root:
    python -m aldau.evaluate          # check the saved baseline on gold, print the worst misses

Every number is shown for the whole gold set and separately for real and case_based
ads (see "Known limitation" in CLAUDE.md), and recall is shown with and without the
ads marked "по тексту не распознать" (t).
"""

import argparse
import math
import time

import numpy as np
from sklearn.metrics import average_precision_score

from aldau.collect import GOLD_PATH, read_ads
from aldau.eda import roc_auc
from aldau.features import SIGNAL_DESCRIPTIONS_RU, find_signals
from aldau.schema import Ad


def slice_metrics(risk: np.ndarray, is_scam: np.ndarray, threshold: float) -> dict:
    """Classification metrics for one group of ads; 'scam' is the positive class.

    precision — of the ads called scam, how many are scam;
    recall    — of the scam ads, how many were caught;
    ROC-AUC   — chance a random scam ad scores higher than a random legit ad;
    PR-AUC    — average precision over all thresholds; unlike ROC-AUC it is not
                flattered by the many easy legit ads, so it suits rare classes.
    """
    predicted = risk >= threshold
    tp = int((predicted & (is_scam == 1)).sum())
    fp = int((predicted & (is_scam == 0)).sum())
    fn = int((~predicted & (is_scam == 1)).sum())
    tn = int((~predicted & (is_scam == 0)).sum())
    both_classes = 0 < is_scam.sum() < len(is_scam)
    return {
        "n": len(is_scam),
        "scam": int(is_scam.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else math.nan,
        "recall": tp / (tp + fn) if tp + fn else math.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else math.nan,
        "roc_auc": roc_auc(list(risk), list(is_scam == 1)) if both_classes else math.nan,
        "pr_auc": average_precision_score(is_scam, risk) if both_classes else math.nan,
    }


def fmt(value: float) -> str:
    return "—" if value != value else f"{value:.2f}"


def slices(gold: list[Ad]) -> list[tuple[str, list[int]]]:
    """(name, indices) for: all gold, real only, case_based only."""
    return [("все", list(range(len(gold))))] + [
        (origin, [i for i, ad in enumerate(gold) if ad.origin == origin])
        for origin in ("real", "case_based")
    ]


def metrics_table(risk: np.ndarray, gold: list[Ad], threshold: float) -> list[str]:
    y = np.array([ad.label == "scam" for ad in gold], dtype=int)
    lines = [
        "| срез | n | scam | precision | recall | F1 | ROC-AUC | PR-AUC | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, idx in slices(gold):
        if not idx:
            continue
        m = slice_metrics(risk[idx], y[idx], threshold)
        lines.append(
            f"| {name} | {m['n']} | {m['scam']} | {fmt(m['precision'])} | {fmt(m['recall'])} "
            f"| {fmt(m['f1'])} | {fmt(m['roc_auc'])} | {fmt(m['pr_auc'])} "
            f"| {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )
    return lines


def recall_with_and_without_t(risk: np.ndarray, gold: list[Ad], threshold: float) -> list[str]:
    def recall(indices: list[int]) -> str:
        caught = sum(risk[i] >= threshold for i in indices)
        return f"{fmt(caught / len(indices)) if indices else '—'} ({caught} из {len(indices)})"

    scam = [i for i, ad in enumerate(gold) if ad.label == "scam"]
    visible = [i for i in scam if not gold[i].not_from_text]
    real_scam = [i for i in scam if gold[i].origin == "real"]
    real_visible = [i for i in real_scam if not gold[i].not_from_text]
    return [
        f"- recall по всем scam: {recall(scam)}",
        f"- recall без «по тексту не распознать» (t): {recall(visible)}",
        f"- recall по настоящим (real) scam: {recall(real_scam)}; без t: {recall(real_visible)}",
    ]


def ad_signals(ad: Ad) -> list[str]:
    return [s for s in find_signals(ad.title + "\n" + ad.text) if s != "has_phone"]


def levels_section(model, gold: list[Ad], risk: np.ndarray) -> list[str]:
    """How the bot's levels 🟢/🟡/🔴 (student only, no teacher) fall on gold,
    without and with the rule floor for known schemes (cascade.SCHEME_FLOORS)."""
    from aldau.cascade import level_for  # local import: cascade imports this package's model

    signals = [ad_signals(ad) for ad in gold]
    def pick(label: str, origin: str | None = None) -> list[int]:
        return [i for i, ad in enumerate(gold)
                if ad.label == label and (origin is None or ad.origin == origin)]  # fmt: skip

    groups = [
        ("scam, все", pick("scam")),
        ("scam, real", pick("scam", "real")),
        ("scam, case_based", pick("scam", "case_based")),
        ("legit", pick("legit")),
    ]
    lines = [
        "## Уровни ответа бота на gold: без правил и с правилами (без учителя)",
        "",
        "Правила — «пол уровня» для известных схем из промпта учителя (`cascade.SCHEME_FLOORS`): "
        "оплата по ссылке с вводом карты → 🔴; товар из-за границы / на таможне, продавец не "
        "может встретиться, обещает прислать документы, давление «желающих много» → минимум 🟡. "
        "Список взят из промпта учителя, а не из ошибок на gold.",
        "",
        "| группа | n | без правил 🟢 / 🟡 / 🔴 | с правилами 🟢 / 🟡 / 🔴 |",
        "|---|---|---|---|",
    ]
    levels = {}
    for rules in (False, True):
        levels[rules] = [level_for(r, model.threshold, s, rules) for r, s in zip(risk, signals,
                                                                                    strict=True)]
    for name, idx in groups:
        cells = []
        for rules in (False, True):
            counts = [sum(levels[rules][i] == lv for i in idx) for lv in ("green", "yellow", "red")]
            cells.append(" / ".join(map(str, counts)))
        lines.append(f"| {name} | {len(idx)} | {cells[0]} | {cells[1]} |")
    lines += ["", "| | без правил | с правилами |", "|---|---|---|"]
    scam_idx, legit_idx = groups[0][1], groups[3][1]
    real_idx = groups[1][1]

    def share(idx: list[int], rules: bool, wanted: tuple[str, ...]) -> str:
        n = sum(levels[rules][i] in wanted for i in idx)
        return f"{fmt(n / len(idx)) if idx else '—'} ({n} из {len(idx)})"

    for title, wanted in (("🔴", ("red",)), ("🟡 или 🔴", ("yellow", "red"))):
        lines.append(f"| scam с уровнем {title} | {share(scam_idx, False, wanted)} "
                     f"| {share(scam_idx, True, wanted)} |")  # fmt: skip
        lines.append(f"| настоящие scam с уровнем {title} | {share(real_idx, False, wanted)} "
                     f"| {share(real_idx, True, wanted)} |")  # fmt: skip
        lines.append(f"| legit с уровнем {title} (ложная тревога) "
                     f"| {share(legit_idx, False, wanted)} | {share(legit_idx, True, wanted)} |")
    return lines + [""]


def latency_ms(model, gold: list[Ad], repeats: int = 1) -> float:
    """Median time to score ONE ad, in milliseconds (as the bot would do it)."""
    times = []
    for ad in gold:
        for _ in range(repeats):
            start = time.perf_counter()
            model.risk([ad])
            times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def gold_report(model, gold: list[Ad], train_ads: list[Ad]) -> str:
    """Markdown report for reports/baseline.md — numbers and ids only, no ad texts."""
    from aldau.train_baseline import BaselineModel  # local import: avoid a cycle

    risk = model.risk(gold)
    info = model.info
    lines = [
        "# Baseline-ученик (Фаза 5) — прототип",
        "",
        "TF-IDF по буквенным кусочкам слов + признаки Фазы 2 → LogisticRegression.",
        "Обучен на **метках владельца из импорта** (не на метках учителя); gold в обучении не "
        "участвовал. **Это маленький прототип**, цифры ниже — не оценка готовой системы.",
        "",
        "## Данные",
        "",
        f"- обучение: {info.get('train_size')} объявлений {info.get('train_labels')}; "
        f"origin/метка: {info.get('train_origin')}",
        f"- убрано из обучения почти-копий gold (совпадение слов ≥ 80 %): "
        f"{info.get('dropped_near_copies', 0)}",
        f"- проверка (gold): {len(gold)} объявлений, scam {sum(a.label == 'scam' for a in gold)}, "
        f"из них настоящих (real) {sum(a.label == 'scam' and a.origin == 'real' for a in gold)}",
        "",
        "**Главное ограничение:** в обучении все scam — переписанные кейсы (case_based), "
        "все legit — настоящие объявления. Модель может научиться отличать стиль пересказа, "
        "а не схему обмана. "
        "Честная проверка — срез `real` в gold: этих scam модель не видела ни в каком виде.",
        "",
        "## Выбор настроек (кросс-валидация только на обучающих данных)",
        "",
        "| что видит модель | лучший C | PR-AUC (5 фолдов, train) | порог по F1 |",
        "|---|---|---|---|",
    ]
    for row in info.get("cv", []):
        lines.append(
            f"| {row['mode']} | {row['c']} | {row['pr_auc']:.2f} | {row['threshold']:.2f} |"
        )
    cv = info.get("cv_choice", {"mode": model.mode, "c": model.c, "threshold": model.threshold})
    after_gold = info.get("chosen_by") == "owner_after_gold" and cv["mode"] != model.mode
    lines += [
        "",
        f"Кросс-валидация на train выбрала: **{cv['mode']}**, C = {cv['c']}, "
        f"порог {cv['threshold']:.2f} (gold в этом выборе не участвовал).",
        "",
    ]
    if after_gold:
        cv_model = BaselineModel(mode=cv["mode"], c=cv["c"], threshold=cv["threshold"])
        cv_risk = cv_model.fit(train_ads).risk(gold)
        lines += [
            f"**Для демо сохранён другой вариант — {model.mode}** (C = {model.c}, порог "
            f"{model.threshold:.2f}; C и порог — тоже по CV на train). Его выбрал владелец "
            "**после просмотра результатов на gold**, поэтому цифры этого варианта на gold "
            "оптимистичны: gold использован для выбора. Честный результат (выбор только по CV) — "
            f"вариант **{cv['mode']}**, он ниже. Главная причина расхождения: в обучении нет "
            "настоящих scam, и текстовая модель выучила стиль пересказа, а не схему обмана. "
            "Чистая проверка демо-варианта — на новом тестовом наборе живых объявлений "
            "(план после EXPO).",
            "",
            f"## Честный результат на gold (вариант {cv['mode']}, выбран CV)",
            "",
            *metrics_table(cv_risk, gold, cv["threshold"]),
            "",
            *recall_with_and_without_t(cv_risk, gold, cv["threshold"]),
            "",
            f"## Демо-вариант на gold ({model.mode}, выбран после просмотра gold)",
        ]
    else:
        lines.append("## Результат на gold")
    lines += [
        "",
        *metrics_table(risk, gold, model.threshold),
        "",
        *recall_with_and_without_t(risk, gold, model.threshold),
        "",
        "## Сравнение вариантов на gold",
        "",
        "| вариант | PR-AUC все | recall все | precision все | PR-AUC real | recall real scam |",
        "|---|---|---|---|---|---|",
    ]
    y = np.array([ad.label == "scam" for ad in gold], dtype=int)
    real = [i for i, ad in enumerate(gold) if ad.origin == "real"]
    for row in info.get("cv", []):
        variant = BaselineModel(mode=row["mode"], c=row["c"], threshold=row["threshold"])
        variant.fit(train_ads)
        r = variant.risk(gold)
        m_all = slice_metrics(r, y, variant.threshold)
        m_real = slice_metrics(r[real], y[real], variant.threshold)
        lines.append(
            f"| {row['mode']} | {fmt(m_all['pr_auc'])} | {fmt(m_all['recall'])} "
            f"| {fmt(m_all['precision'])} | {fmt(m_real['pr_auc'])} | {fmt(m_real['recall'])} |"
        )
    misses = worst_misses(model, gold, risk)
    lines += ["", *levels_section(model, gold, risk)]
    lines += [
        f"Время на одно объявление: **{latency_ms(model, gold):.1f} мс** (медиана, CPU).",
        "",
        "## Худшие ошибки (id; тексты — в консоли, в отчёт не попадают)",
        "",
    ]
    lines += [
        f"- `{ad.id}` — человек {ad.label}{' + t' if ad.not_from_text else ''} ({ad.origin}), "
        f"риск {r:.2f}"
        for ad, r in misses[:10]
    ]
    return "\n".join(lines) + "\n"


def worst_misses(model, gold: list[Ad], risk: np.ndarray | None = None) -> list[tuple[Ad, float]]:
    """Wrong answers, most confident first (scam with the lowest risk, legit with the highest)."""
    risk = model.risk(gold) if risk is None else risk
    wrong = [
        (ad, float(r)) for ad, r in zip(gold, risk, strict=True)
        if (r >= model.threshold) != (ad.label == "scam")
    ]  # fmt: skip

    def how_wrong(pair: tuple[Ad, float]) -> float:
        ad, r = pair
        return abs(r - (1.0 if ad.label == "scam" else 0.0))

    return sorted(wrong, key=how_wrong, reverse=True)


def print_worst_misses(model, gold: list[Ad], show: int = 6) -> None:
    misses = worst_misses(model, gold)
    print(f"Ошибок на gold: {len(misses)} из {len(gold)}")
    for ad, r in misses[:show]:
        signals = [SIGNAL_DESCRIPTIONS_RU[s] for s in find_signals(ad.title + "\n" + ad.text)
                   if s != "has_phone"]  # fmt: skip
        print("\n" + "=" * 78)
        print(
            f"[{ad.id}] человек {ad.label}{' + t' if ad.not_from_text else ''} ({ad.origin}) "
            f"| модель {r:.2f} (порог {model.threshold:.2f}) | {ad.category} | {ad.price_kzt} тг"
        )
        print(f"сигналы: {'; '.join(signals) or '—'}")
        print(f"{ad.title}\n{ad.text[:500]}{' …' if len(ad.text) > 500 else ''}")


def main() -> None:
    from aldau.train_baseline import load

    argparse.ArgumentParser(description="Check the saved baseline on gold.").parse_args()
    model, gold = load(), read_ads(GOLD_PATH)
    risk = model.risk(gold)
    print("\n".join(metrics_table(risk, gold, model.threshold)))
    print("\n".join(recall_with_and_without_t(risk, gold, model.threshold)))
    print_worst_misses(model, gold)


if __name__ == "__main__":
    main()
