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
    lines += [
        "",
        f"Выбрано: **{model.mode}**, C = {model.c}, порог {model.threshold:.2f} "
        "(по PR-AUC на train; gold в выборе не участвовал).",
        "",
        "## Результат на gold",
        "",
        *metrics_table(risk, gold, model.threshold),
        "",
        *recall_with_and_without_t(risk, gold, model.threshold),
        "",
        "## Сравнение вариантов на gold (для понимания, не для выбора)",
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
    lines += [
        "",
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
