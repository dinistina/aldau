"""Phase 5 baseline student: TF-IDF + hand-made features -> LogisticRegression.

Run from the project root:
    python -m aldau.train_baseline   # train, save models/baseline.joblib, write reports/baseline.md
    python -m aldau.train_baseline --mode text   # save the CV winner instead of the demo choice

Training data: ads from data/raw with the OWNER's labels (scam / legit from the inbox
headers). Every gold ad is removed — and so is every near-copy of a gold ad — because
gold is the exam and must never be seen in training.

Everything that is "learned" (TF-IDF vocabulary, price medians, feature scaling, the
model, the decision threshold) is learned on the training ads only.
"""

import argparse
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from aldau.collect import GOLD_PATH, RAW_PATH, read_ads
from aldau.features import SIGNAL_NAMES, PriceReference, extract_features
from aldau.schema import Ad

MODEL_PATH = Path("models/baseline.joblib")
REPORT_PATH = Path("reports/baseline.md")
MODES = ["text", "features", "both"]  # what the model sees
# The owner's choice for the demo (2026-11-26), made AFTER looking at gold: the text-only
# CV winner caught 2 of 9 real scam ads, text + features 7 of 9. See reports/baseline.md.
DEMO_MODE = "both"
C_VALUES = [0.1, 0.3, 1.0, 3.0, 10.0]  # regularisation strengths tried in cross-validation
NEAR_DUPLICATE = 0.8  # word overlap (Jaccard) above which two ads count as copies
SEED = 42


# --- data ------------------------------------------------------------------------------


def words(ad: Ad) -> set[str]:
    return set(re.findall(r"\w+", (ad.title + " " + ad.text).lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Share of common words: |A ∩ B| / |A ∪ B|. 1 = same words, 0 = nothing in common."""
    return len(a & b) / len(a | b) if a | b else 0.0


def split_train_gold(raw: list[Ad], gold: list[Ad]) -> tuple[list[Ad], list[Ad]]:
    """Training ads = labelled raw ads that are neither in gold nor near-copies of a gold ad.

    Returns (train, dropped near-copies). An exact id match is simply excluded; a
    different id with nearly the same text (e.g. the same ad imported twice with a
    small edit) would leak the answer into training, so it is dropped too.
    """
    gold_ids = {ad.id for ad in gold}
    gold_words = [words(ad) for ad in gold]
    train, dropped = [], []
    for ad in raw:
        if ad.label not in ("scam", "legit") or ad.id in gold_ids:
            continue
        ad_words = words(ad)
        if any(jaccard(ad_words, g) >= NEAR_DUPLICATE for g in gold_words):
            dropped.append(ad)
        else:
            train.append(ad)
    return train, dropped


def assert_no_overlap(train: list[Ad], gold: list[Ad]) -> None:
    """The hard rule as code: training must not contain any gold ad."""
    overlap = {ad.id for ad in train} & {ad.id for ad in gold}
    if overlap:
        raise ValueError(f"gold ads in training data: {sorted(overlap)}")


# --- the model --------------------------------------------------------------------------


def ad_text(ad: Ad) -> str:
    return ad.title + "\n" + ad.text


@dataclass
class BaselineModel:
    """Everything needed to score a new ad, learned from the training ads only."""

    mode: str = "both"
    c: float = 1.0
    threshold: float = 0.5
    vectorizer: TfidfVectorizer | None = None
    scaler: StandardScaler | None = None
    prices: PriceReference = field(default_factory=PriceReference)
    classifier: LogisticRegression | None = None
    info: dict = field(default_factory=dict)  # data sizes, CV results — for the report

    def numeric(self, ads: list[Ad]) -> np.ndarray:
        """Phase 2 features as numbers: 13 signals (0/1), price, signal count, length."""
        rows = []
        for ad in ads:
            f = extract_features(ad, self.prices)
            ratio = f["price_ratio"]
            rows.append(
                [float(f[name]) for name in SIGNAL_NAMES]
                + [
                    math.log(ratio) if ratio else 0.0,  # log: 0.5 and 2.0 are equally far from 1
                    0.0 if ratio else 1.0,  # "price unknown" flag
                    float(f["signal_count"]),
                    math.log1p(f["text_length"]),
                ]
            )
        return np.array(rows)

    def matrix(self, ads: list[Ad], fit: bool = False):
        parts = []
        if self.mode in ("text", "both"):
            texts = [ad_text(ad) for ad in ads]
            if fit:
                # Character pieces of words (2-5 letters) instead of whole words: they cope
                # with Kazakh endings and typos ("жаксы"/"жақсы", "ватсап"/"вацап").
                self.vectorizer = TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
                )
                parts.append(self.vectorizer.fit_transform(texts))
            else:
                parts.append(self.vectorizer.transform(texts))
        if self.mode in ("features", "both"):
            numbers = self.numeric(ads)
            if fit:
                self.scaler = StandardScaler()
                numbers = self.scaler.fit_transform(numbers)
            else:
                numbers = self.scaler.transform(numbers)
            parts.append(csr_matrix(numbers))
        return hstack(parts).tocsr()

    def fit(self, ads: list[Ad]) -> "BaselineModel":
        self.prices = PriceReference(min_count=3).fit(ads)  # medians from training ads only
        x = self.matrix(ads, fit=True)
        y = np.array([ad.label == "scam" for ad in ads], dtype=int)
        # class_weight="balanced": scam is the minority class, so its mistakes count more
        self.classifier = LogisticRegression(
            C=self.c, class_weight="balanced", max_iter=2000, random_state=SEED
        )
        self.classifier.fit(x, y)
        return self

    def risk(self, ads: list[Ad]) -> np.ndarray:
        """Probability of scam for each ad (0..1)."""
        return self.classifier.predict_proba(self.matrix(ads))[:, 1]


# --- choosing settings with cross-validation on the training ads ---------------------------


def out_of_fold_risk(ads: list[Ad], mode: str, c: float, folds: int = 5) -> np.ndarray:
    """Risk for every training ad from a model that did NOT see that ad (k-fold)."""
    y = np.array([ad.label == "scam" for ad in ads], dtype=int)
    folds = min(folds, int(y.sum()), int(len(y) - y.sum()))  # each fold needs both classes
    risk = np.zeros(len(ads))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        model = BaselineModel(mode=mode, c=c).fit([ads[i] for i in train_idx])
        risk[test_idx] = model.risk([ads[i] for i in test_idx])
    return risk


def best_threshold(risk: np.ndarray, is_scam: np.ndarray) -> float:
    """The threshold with the best F1 on the given scores (ties: the lower one, more recall)."""
    best, best_f1 = 0.5, -1.0
    for t in sorted(set(np.round(risk, 4))):
        predicted = risk >= t
        tp = int((predicted & (is_scam == 1)).sum())
        fp = int((predicted & (is_scam == 0)).sum())
        fn = int((~predicted & (is_scam == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def train(
    train_ads: list[Ad], modes: list[str] = MODES, force_mode: str | None = None
) -> tuple[BaselineModel, list[dict]]:
    """Pick mode and C by out-of-fold PR-AUC on the training ads, then fit on all of them.

    Returns the final model and the CV table. Gold is not involved in any choice here.

    force_mode: use this mode instead of the CV winner (C and threshold still come from
    CV on train). The owner chose "both" for the demo AFTER looking at gold, so the
    model records both choices and the report states it honestly.
    """
    y = np.array([ad.label == "scam" for ad in train_ads], dtype=int)
    table = []
    for mode in modes:
        for c in C_VALUES:
            risk = out_of_fold_risk(train_ads, mode, c)
            table.append({"mode": mode, "c": c, "pr_auc": average_precision_score(y, risk),
                          "risk": risk})  # fmt: skip
    cv_best = max(table, key=lambda row: row["pr_auc"])
    best = cv_best
    if force_mode:
        best = max((row for row in table if row["mode"] == force_mode),
                   key=lambda row: row["pr_auc"])  # fmt: skip
    model = BaselineModel(mode=best["mode"], c=best["c"])
    model.threshold = best_threshold(best["risk"], y)
    model.fit(train_ads)
    best_per_mode = {}
    for row in table:
        if row["pr_auc"] > best_per_mode.get(row["mode"], {"pr_auc": -1})["pr_auc"]:
            best_per_mode[row["mode"]] = row
    summary = [
        {"mode": m, "c": r["c"], "pr_auc": r["pr_auc"],
         "threshold": best_threshold(r["risk"], y)}
        for m, r in best_per_mode.items()
    ]  # fmt: skip
    model.info = {
        "train_size": len(train_ads),
        "train_labels": dict(Counter(ad.label for ad in train_ads)),
        "train_origin": dict(Counter(f"{ad.origin}/{ad.label}" for ad in train_ads)),
        "cv": summary,
        "chosen_by": "owner_after_gold" if force_mode else "cv",
        "cv_choice": {"mode": cv_best["mode"], "c": cv_best["c"], "pr_auc": cv_best["pr_auc"],
                      "threshold": best_threshold(cv_best["risk"], y)},
    }  # fmt: skip
    return model, summary


def save(model: BaselineModel, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path = MODEL_PATH) -> BaselineModel:
    return joblib.load(path)


# --- command line ----------------------------------------------------------------------


def main() -> None:
    from aldau.evaluate import gold_report, print_worst_misses  # avoid an import cycle

    parser = argparse.ArgumentParser(description="Train the baseline student and check it on gold.")
    parser.add_argument(
        "--mode", choices=MODES, default=DEMO_MODE,
        help=f"what the saved model sees (default {DEMO_MODE}: the owner's demo choice, made "
             "after looking at gold; the CV winner is reported next to it)",
    )  # fmt: skip
    args = parser.parse_args()
    raw, gold = read_ads(RAW_PATH), read_ads(GOLD_PATH)
    train_ads, dropped = split_train_gold(raw, gold)
    assert_no_overlap(train_ads, gold)
    print(
        f"Обучение: {len(train_ads)} объявлений {dict(Counter(ad.label for ad in train_ads))}; "
        f"убрано почти-копий gold: {len(dropped)}; gold: {len(gold)}"
    )
    model, summary = train(train_ads, force_mode=args.mode)
    model.info["dropped_near_copies"] = len(dropped)
    for row in summary:
        print(f"  CV {row['mode']:<8} C={row['c']:<4} PR-AUC {row['pr_auc']:.2f}")
    cv = model.info["cv_choice"]
    print(f"Выбрано по CV на train: mode={cv['mode']}, C={cv['c']}, порог {cv['threshold']:.2f}")
    print(f"Сохраняется: mode={model.mode}, C={model.c}, порог {model.threshold:.2f}")
    save(model)
    report = gold_report(model, gold, train_ads)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nМодель: {MODEL_PATH}   Отчёт: {REPORT_PATH}\n")
    print(report)
    print_worst_misses(model, gold)


if __name__ == "__main__":
    # Run main() from the module imported under its real name. Otherwise the saved model
    # would be pickled as "__main__.BaselineModel" and no other program (bot, evaluate,
    # offline demo) could load it.
    from aldau.train_baseline import main as real_main

    real_main()
