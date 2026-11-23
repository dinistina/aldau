"""Second annotator and inter-annotator agreement (Cohen's kappa).

Usage (run from the project root):
    python -m aldau.agreement sample            # pick 30 random gold ads for the second person
    python -m aldau.agreement label             # the second person labels them (resumable)
    python -m aldau.agreement kappa             # agreement between the two people

Why: one person's labels can be biased or simply wrong. If a second person, without
seeing the first person's answers, labels the same ads, we can measure how much
the two agree. Low agreement means the task itself is ambiguous — and then no model
can be expected to score high on it either.

Cohen's kappa = (p_o - p_e) / (1 - p_e)
    p_o — observed agreement: share of ads where both gave the same label;
    p_e — agreement expected by pure chance, from how often each person says "scam".
kappa = 1 perfect agreement, 0 no better than chance, < 0 worse than chance.
Plain p_o is misleading with unbalanced classes: if 75 % of ads are legit, two people
who always say "legit" agree on 75 % of ads without reading anything — kappa shows 0.
"""

import argparse
import json
import math
import random
from collections.abc import Callable
from pathlib import Path

from aldau.collect import GOLD_PATH, parse_gold_answer, read_ads
from aldau.schema import Ad

SAMPLE_PATH = Path("data/gold/second_sample.txt")
SECOND_PATH = Path("data/gold/second_annotator.jsonl")
SAMPLE_SIZE = 30


# --- sample ---------------------------------------------------------------------------


def draw_sample(gold: list[Ad], size: int = SAMPLE_SIZE, seed: int = 7) -> list[str]:
    """Random gold ids. A fixed seed: the same sample every time (reproducible)."""
    ids = [ad.id for ad in gold]
    return random.Random(seed).sample(ids, min(size, len(ids)))


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- second person's labels ------------------------------------------------------------


def read_second(path: Path = SECOND_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {row["id"]: row for row in rows}


def label_second(
    gold: list[Ad],
    sample_ids: list[str],
    out_path: Path = SECOND_PATH,
    ask: Callable[[str], str] = input,
) -> int:
    """Show the sampled ads WITHOUT the first person's label, origin or source note.

    Answers are the same as in gold labelling (s, l, st, lt, n, q) and are saved one
    by one, so the session can be stopped and continued.
    """
    by_id = {ad.id: ad for ad in gold}
    done = read_second(out_path)
    todo = [i for i in sample_ids if i not in done and i in by_id]
    labelled = 0
    for n, ad_id in enumerate(todo, start=len(done) + 1):
        ad = by_id[ad_id]
        print("\n" + "=" * 70)
        print(f"[{n}/{len(sample_ids)}]  {ad.source} | {ad.category} | {ad.city or '-'}")
        print(f"Цена: {ad.price_kzt if ad.price_kzt is not None else 'не указана'} тг")
        print(f"\n{ad.title}\n\n{ad.text}")
        print("=" * 70)
        answer = ask(
            "s = scam, l = legit (+t: по тексту не распознать), n = пропустить, q = выйти > "
        )
        if answer.strip().lower() == "q":
            break
        parsed = parse_gold_answer(answer.strip().lower())
        if parsed is None:
            continue
        label, not_from_text = parsed
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            row = {"id": ad_id, "label": label, "not_from_text": not_from_text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        labelled += 1
    return labelled


# --- agreement ------------------------------------------------------------------------


def cohen_kappa(first: list[str], second: list[str]) -> tuple[float, float, float]:
    """(kappa, observed agreement p_o, chance agreement p_e) for two lists of labels."""
    n = len(first)
    if n == 0:
        return math.nan, math.nan, math.nan
    p_o = sum(a == b for a, b in zip(first, second, strict=True)) / n
    classes = set(first) | set(second)
    p_e = sum((first.count(c) / n) * (second.count(c) / n) for c in classes)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else math.nan
    return kappa, p_o, p_e


def kappa_strength(kappa: float) -> str:
    """Common reading scale (Landis & Koch, 1977)."""
    if kappa != kappa:
        return "не определена (у обоих одна и та же единственная метка)"
    for bound, text in [
        (0.0, "хуже случайного"),
        (0.2, "незначительное"),
        (0.4, "слабое"),
        (0.6, "умеренное"),
        (0.8, "существенное"),
    ]:
        if kappa < bound:
            return text
    return "почти полное"


def agreement_report(gold: list[Ad], second: dict[str, dict]) -> list[str]:
    by_id = {ad.id: ad for ad in gold}
    pairs = [(by_id[i], row) for i, row in second.items() if i in by_id]
    first_labels = [ad.label for ad, _ in pairs]
    second_labels = [row["label"] for _, row in pairs]
    kappa, p_o, p_e = cohen_kappa(first_labels, second_labels)
    table = {
        (a, b): sum(x == a and y == b for x, y in zip(first_labels, second_labels, strict=True))
        for a in ("scam", "legit")
        for b in ("scam", "legit")
    }
    t_same = sum(ad.not_from_text == row["not_from_text"] for ad, row in pairs)
    lines = [
        f"Объявлений, размеченных обоими: {len(pairs)}",
        f"Совпадение меток (p_o): {p_o:.2f}" if pairs else "Совпадение меток: —",
        f"Случайное совпадение (p_e): {p_e:.2f}" if pairs else "",
        f"Cohen's kappa: {kappa:.2f} — {kappa_strength(kappa)}" if pairs else "",
        "",
        "                  второй: scam  второй: legit",
        f"  первый: scam    {table[('scam', 'scam')]:>12} {table[('scam', 'legit')]:>14}",
        f"  первый: legit   {table[('legit', 'scam')]:>12} {table[('legit', 'legit')]:>14}",
        "",
        f"Пометка t совпала в {t_same} из {len(pairs)}",
    ]
    disagreements = [(ad, row) for ad, row in pairs if ad.label != row["label"]]
    if disagreements:
        lines += ["", f"Расхождения ({len(disagreements)}):"]
        lines += [
            f"  [{ad.id}] первый {ad.label}, второй {row['label']} — {ad.title[:50]}"
            for ad, row in disagreements
        ]
    return lines


# --- command line ---------------------------------------------------------------------


def cmd_sample(args: argparse.Namespace) -> None:
    if SAMPLE_PATH.exists() and not args.force:
        raise SystemExit(f"{SAMPLE_PATH} уже есть. Новая выборка — с --force.")
    ids = draw_sample(read_ads(GOLD_PATH), args.n)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"Выбрано {len(ids)} объявлений из gold -> {SAMPLE_PATH}")


def cmd_label(args: argparse.Namespace) -> None:
    if not SAMPLE_PATH.exists():
        raise SystemExit("Сначала: python -m aldau.agreement sample")
    count = label_second(read_ads(GOLD_PATH), read_ids(SAMPLE_PATH))
    print(f"\nРазмечено сейчас: {count}. Всего у второго разметчика: {len(read_second())}")


def cmd_kappa(args: argparse.Namespace) -> None:
    print("\n".join(agreement_report(read_ads(GOLD_PATH), read_second())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Second annotator and Cohen's kappa.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_sample = sub.add_parser("sample", help="pick random gold ads for the second person")
    p_sample.add_argument("--n", type=int, default=SAMPLE_SIZE)
    p_sample.add_argument("--force", action="store_true", help="replace an existing sample")
    p_sample.set_defaults(func=cmd_sample)
    sub.add_parser("label", help="second person labels the sample").set_defaults(func=cmd_label)
    sub.add_parser("kappa", help="agreement between the two").set_defaults(func=cmd_kappa)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
