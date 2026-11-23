"""Gold set maintenance: review, logged changes, freezing.

Usage (run from the project root):
    python -m aldau.gold diff                     # gold labels that differ from the import labels
    python -m aldau.gold change <id> --label legit --reason "..."    # fix one label (logged)
    python -m aldau.gold change <id> --label scam --t --reason "..." # ... with the "t" mark
    python -m aldau.gold freeze                   # lock gold before training
    python -m aldau.gold check                    # is gold still exactly as frozen?

Why log every change: gold is the exam for every model. If labels change after you
have seen a model's mistakes, the exam stops measuring anything ("fitting the test").
A change log with a reason for each change keeps such edits honest and visible.

Why freeze: from Phase 5 on, gold must not change at all. `freeze` stores a SHA-256
fingerprint of gold.jsonl; `check` (and later the training code) compares the file
with it, so even an accidental edit is noticed.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from aldau.collect import GOLD_FREEZE_PATH, GOLD_PATH, RAW_PATH, read_ads, write_ads
from aldau.schema import Ad

CHANGES_PATH = Path("data/gold_changes.jsonl")


# --- review --------------------------------------------------------------------------


def label_text(label: str, not_from_text: bool) -> str:
    return label + (" + t" if not_from_text else "")


def gold_vs_import(gold: list[Ad], raw: list[Ad]) -> list[tuple[int, Ad, Ad]]:
    """(position in gold, gold ad, raw ad) for every ad whose gold label differs from
    the label it got at import (from the inbox header)."""
    raw_by_id = {ad.id: ad for ad in raw}
    return [
        (number, ad, raw_by_id[ad.id])
        for number, ad in enumerate(gold, start=1)
        if ad.id in raw_by_id and ad.label != raw_by_id[ad.id].label
    ]


# --- logged changes -------------------------------------------------------------------


def is_frozen(freeze_path: Path = GOLD_FREEZE_PATH) -> bool:
    return freeze_path.exists()


def change_label(
    ad_id: str,
    label: str,
    not_from_text: bool,
    reason: str,
    gold_path: Path = GOLD_PATH,
    changes_path: Path = CHANGES_PATH,
    freeze_path: Path = GOLD_FREEZE_PATH,
) -> dict:
    """Change one gold label and append the change to the log. Returns the log record."""
    if is_frozen(freeze_path):
        raise ValueError("gold заморожен — менять метки нельзя")
    if label not in ("scam", "legit"):
        raise ValueError("метка в gold — только scam или legit")
    if not reason.strip():
        raise ValueError("нужна причина изменения (--reason)")
    gold = read_ads(gold_path)
    positions = [i for i, ad in enumerate(gold) if ad.id == ad_id]
    if not positions:
        raise ValueError(f"в gold нет объявления {ad_id}")
    i = positions[0]
    old = gold[i]
    if (old.label, old.not_from_text) == (label, not_from_text):
        raise ValueError("метка уже такая — изменять нечего")

    gold[i] = old.model_copy(update={"label": label, "not_from_text": not_from_text})
    write_ads(gold_path, gold)
    record = {
        "id": ad_id,
        "was": {"label": old.label, "not_from_text": old.not_from_text},
        "now": {"label": label, "not_from_text": not_from_text},
        "reason": reason.strip(),
        "date": datetime.now(UTC).date().isoformat(),
    }
    changes_path.parent.mkdir(parents=True, exist_ok=True)
    with changes_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# --- freezing -------------------------------------------------------------------------


def fingerprint(path: Path) -> str:
    """SHA-256 of the file bytes: any change, even one character, gives a new value."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(gold_path: Path = GOLD_PATH, freeze_path: Path = GOLD_FREEZE_PATH) -> dict:
    if is_frozen(freeze_path):
        raise ValueError("gold уже заморожен")
    gold = read_ads(gold_path)
    info = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ads": len(gold),
        "scam": sum(ad.label == "scam" for ad in gold),
        "legit": sum(ad.label == "legit" for ad in gold),
        "sha256": fingerprint(gold_path),
    }
    freeze_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def check_unchanged(gold_path: Path = GOLD_PATH, freeze_path: Path = GOLD_FREEZE_PATH) -> bool:
    """True if gold.jsonl is byte-for-byte what was frozen. Training code calls this."""
    if not is_frozen(freeze_path):
        raise ValueError("gold ещё не заморожен")
    info = json.loads(freeze_path.read_text(encoding="utf-8"))
    return fingerprint(gold_path) == info["sha256"]


# --- command line ---------------------------------------------------------------------


def cmd_diff(args: argparse.Namespace) -> None:
    rows = gold_vs_import(read_ads(GOLD_PATH), read_ads(RAW_PATH))
    print(f"Метка в gold отличается от метки при импорте: {len(rows)}\n")
    for number, ad, raw in rows:
        print("=" * 78)
        print(
            f"№{number} в gold  [{ad.id}]  {ad.origin} | {ad.category} | {ad.city or '-'} "
            f"| {ad.price_kzt} тг"
        )
        print(f"при импорте: {raw.label}   ->   в gold: {label_text(ad.label, ad.not_from_text)}")
        if ad.source_note:
            print(f"источник: {ad.source_note}")
        print(f"\n{ad.title}\n{ad.text[:700]}{' …' if len(ad.text) > 700 else ''}")
    print("=" * 78)
    print(
        "Метку в gold можно поменять: python -m aldau.gold change <id> --label scam|legit "
        '[--t] --reason "..."'
    )


def cmd_change(args: argparse.Namespace) -> None:
    try:
        record = change_label(args.id, args.label, args.t, args.reason)
    except ValueError as error:
        raise SystemExit(f"Не изменено: {error}") from error
    was, now = record["was"], record["now"]
    print(
        f"{args.id}: {label_text(was['label'], was['not_from_text'])} -> "
        f"{label_text(now['label'], now['not_from_text'])}. Записано в {CHANGES_PATH}"
    )


def cmd_freeze(args: argparse.Namespace) -> None:
    try:
        info = freeze()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Gold заморожен: {info['ads']} объявлений (scam {info['scam']}, legit {info['legit']}).")
    print(f"Отпечаток SHA-256: {info['sha256'][:16]}… -> {GOLD_FREEZE_PATH}")


def cmd_check(args: argparse.Namespace) -> None:
    try:
        same = check_unchanged()
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(
        "Gold не изменился с момента заморозки."
        if same
        else "ВНИМАНИЕ: gold изменён после заморозки!"
    )
    if not same:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold set: review, logged changes, freeze.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("diff", help="gold labels that differ from import labels").set_defaults(
        func=cmd_diff
    )
    p_change = sub.add_parser("change", help="change one gold label (logged)")
    p_change.add_argument("id")
    p_change.add_argument("--label", required=True, choices=["scam", "legit"])
    p_change.add_argument("--t", action="store_true", help="mark: cannot be seen from the text")
    p_change.add_argument("--reason", required=True)
    p_change.set_defaults(func=cmd_change)
    sub.add_parser("freeze", help="lock gold").set_defaults(func=cmd_freeze)
    sub.add_parser("check", help="is gold unchanged since freezing?").set_defaults(func=cmd_check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
