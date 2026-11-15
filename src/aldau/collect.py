"""Tools for hand-collecting and hand-labelling ads. No scraping: you paste ads yourself.

Usage (run from the project root):
    python -m aldau.collect add                  # add one ad, typing/pasting the text
    python -m aldau.collect add --text-file a.txt  # ... or read the ad text from a UTF-8 file
    python -m aldau.collect import data/inbox.txt  # add many ads at once (see "batch import")
    python -m aldau.collect import data/inbox.txt --dry-run   # only check, save nothing
    python -m aldau.collect gold                 # label ads by hand into the gold set
    python -m aldau.collect stats                # how many ads are collected / labelled

Files:
    data/raw/ads.jsonl    every collected ad (PII already masked)
    data/gold/gold.jsonl  ads you labelled by hand — the honest test set

JSONL = one JSON object per line. It is easy to append to and survives a crash:
a broken last line does not destroy the lines before it.
"""

import argparse
import hashlib
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aldau.features import detect_lang, mask_pii
from aldau.schema import Ad

RAW_PATH = Path("data/raw/ads.jsonl")
GOLD_PATH = Path("data/gold/gold.jsonl")
GOLD_TARGET = 100
SOURCES = ["olx", "kolesa", "krisha"]
LABELS = ["scam", "legit", "unknown"]
ORIGINS = ["real", "case_based", "synthetic"]


# --- reading and writing JSONL ------------------------------------------------


def read_ads(path: Path) -> list[Ad]:
    """Read all ads from a JSONL file. A missing file means "no ads yet"."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Ad.model_validate_json(line) for line in f if line.strip()]


def append_ad(path: Path, ad: Ad) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(ad.model_dump_json() + "\n")


# --- adding an ad -------------------------------------------------------------


def make_ad_id(title: str, text: str) -> str:
    """Id = short hash of the ad's content.

    The same ad pasted twice gets the same id, so duplicates are easy to refuse.
    Whitespace and letter case are ignored.
    """
    normalized = " ".join((title + " " + text).lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def build_ad(
    source: str,
    category: str,
    title: str,
    text: str,
    price_kzt: int | None = None,
    city: str | None = None,
    label: str = "unknown",
    origin: str = "real",
) -> Ad:
    """Mask PII and build an Ad. Nothing is saved here."""
    title = mask_pii(title.strip())
    text = mask_pii(text.strip())
    return Ad(
        id=make_ad_id(title, text),
        source=source,
        category=category.strip().lower(),
        title=title,
        text=text,
        price_kzt=price_kzt,
        city=city.strip() if city else None,
        lang=detect_lang(title + " " + text),
        label=label,
        origin=origin,
    )


def save_new_ad(ad: Ad, raw_path: Path = RAW_PATH) -> bool:
    """Append the ad to the raw file. Returns False if it is already there."""
    if any(existing.id == ad.id for existing in read_ads(raw_path)):
        return False
    append_ad(raw_path, ad)
    return True


# --- batch import from an inbox file ------------------------------------------
#
# Inbox format: ads separated by a line "---". Each ad is
#     source | category | price | city | label | origin      <- header
#     Title                                                    <- first line after it
#     Text, any number of lines...
# Price, city, label and origin may be left empty (label -> unknown, origin -> real).


@dataclass
class ImportReport:
    """What happened during one import run."""

    added: list[Ad] = field(default_factory=list)
    masked: list[str] = field(default_factory=list)  # titles of ads where PII was masked
    duplicates: list[str] = field(default_factory=list)  # human-readable messages
    errors: list[str] = field(default_factory=list)  # human-readable messages


def split_inbox(text: str) -> list[tuple[int, list[str]]]:
    """Cut the inbox into blocks. Returns (line number of the header, lines of the block)."""
    blocks = []
    current: list[str] = []
    start = 0
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "---":
            if current:
                blocks.append((start, current))
            current = []
        elif line.strip() or current:  # skip blank lines before the header
            if not current:
                start = number
            current.append(line)
    if current:
        blocks.append((start, current))
    return blocks


def parse_block(lines: list[str]) -> Ad:
    """Turn one inbox block into an Ad. Raises ValueError with a readable message."""
    parts = [part.strip() for part in lines[0].split("|")]
    if len(parts) != 6:
        raise ValueError(
            f"в шапке должно быть 6 полей через |, а найдено {len(parts)}: {lines[0]!r}"
        )
    source, category, price_raw, city, label, origin = parts
    source, label, origin = source.lower(), label.lower() or "unknown", origin.lower() or "real"

    if source not in SOURCES:
        raise ValueError(f"источник {source!r} — нужно одно из {SOURCES}")
    if not category:
        raise ValueError("не указана категория")
    if label not in LABELS:
        raise ValueError(f"метка {label!r} — нужно одно из {LABELS}")
    if origin not in ORIGINS:
        raise ValueError(f"origin {origin!r} — нужно одно из {ORIGINS}")
    price_digits = price_raw.replace(" ", "")
    if price_digits and not price_digits.isdigit():
        raise ValueError(f"цена {price_raw!r} — нужно число или пусто")

    body = "\n".join(lines[1:]).strip()
    if not body:
        raise ValueError("после шапки нет текста объявления")
    # First line is the title, the rest is the text. A one-line ad uses the title as its text.
    title, _, rest = body.partition("\n")
    text = rest.strip() or title

    return build_ad(
        source=source,
        category=category,
        title=title,
        text=text,
        price_kzt=int(price_digits) if price_digits else None,
        city=city or None,
        label=label,
        origin=origin,
    )


def import_inbox(
    inbox_path: Path, raw_path: Path = RAW_PATH, dry_run: bool = False
) -> ImportReport:
    """Add every valid ad from the inbox file to the raw file.

    Bad ads are reported and skipped; they never stop the import.
    With dry_run=True nothing is written — only the report is produced.
    """
    report = ImportReport()
    # utf-8-sig also accepts files that Windows Notepad saved with a BOM mark
    text = inbox_path.read_text(encoding="utf-8-sig")
    known_ids = {ad.id for ad in read_ads(raw_path)}

    for number, (line_no, lines) in enumerate(split_inbox(text), start=1):
        where = f"№{number} (строка {line_no})"
        try:
            ad = parse_block(lines)
        except ValueError as error:
            report.errors.append(f"{where}: {error}")
            continue
        if ad.id in known_ids:
            report.duplicates.append(f"{where}: {ad.title[:60]}")
            continue
        known_ids.add(ad.id)

        raw_body = "\n".join(lines[1:])
        if mask_pii(raw_body) != raw_body:
            report.masked.append(f"{where}: {ad.title[:60]}")
        if not dry_run:
            append_ad(raw_path, ad)
        report.added.append(ad)

    return report


# --- labelling the gold set ---------------------------------------------------


def pick_unlabelled(raw_ads: list[Ad], gold_ads: list[Ad], seed: int = 42) -> list[Ad]:
    """Ads that are not in the gold set yet, in a fixed random order.

    Random order matters: if gold were simply "the first 100 ads collected", it
    would reflect whatever you happened to collect first (one category, one day).
    A fixed seed makes the order the same on every run.
    """
    gold_ids = {ad.id for ad in gold_ads}
    todo = [ad for ad in raw_ads if ad.id not in gold_ids]
    random.Random(seed).shuffle(todo)
    return todo


def label_gold(
    raw_path: Path = RAW_PATH,
    gold_path: Path = GOLD_PATH,
    target: int = GOLD_TARGET,
    ask: Callable[[str], str] = input,
) -> int:
    """Show ads one by one and save your answers to the gold file.

    Each answer is written immediately, so you can stop (q) and continue later.
    `ask` is `input` in real use; tests pass a fake function instead.
    Returns how many ads were labelled in this run.
    """
    gold_ads = read_ads(gold_path)
    todo = pick_unlabelled(read_ads(raw_path), gold_ads)
    done = len(gold_ads)
    labelled_now = 0

    for ad in todo:
        if done >= target:
            print(f"Gold set is complete: {done}/{target}.")
            break
        print("\n" + "=" * 70)
        print(f"[{done + 1}/{target}]  {ad.source} | {ad.category} | {ad.city or '-'}")
        print(f"Цена: {ad.price_kzt if ad.price_kzt is not None else 'не указана'} тг")
        print(f"\n{ad.title}\n\n{ad.text}")
        print("=" * 70)

        answer = ask("s = scam, l = legit, n = пропустить, q = выйти > ").strip().lower()
        if answer == "q":
            break
        if answer not in ("s", "l"):
            continue  # skipped: the ad stays outside gold
        labelled = ad.model_copy(update={"label": "scam" if answer == "s" else "legit"})
        append_ad(gold_path, labelled)
        done += 1
        labelled_now += 1

    return labelled_now


# --- command line -------------------------------------------------------------


def read_multiline(prompt: str) -> str:
    """Read pasted text until a line that contains only END."""
    print(prompt + " (вставь текст, в конце отдельной строкой напиши END)")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def cmd_add(args: argparse.Namespace) -> None:
    source = input(f"Источник {SOURCES} > ").strip().lower()
    category = input("Категория (например: телефоны, авто, аренда квартир) > ")
    title = input("Заголовок > ")
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    else:
        text = read_multiline("Текст объявления")
    price_raw = input("Цена в тенге (пусто = не указана) > ").replace(" ", "")
    city = input("Город (пусто = не указан) > ")
    label_raw = input("Метка, если точно знаешь: s = scam, l = legit, пусто = unknown > ")
    label = {"s": "scam", "l": "legit"}.get(label_raw.strip().lower(), "unknown")

    ad = build_ad(
        source=source,
        category=category,
        title=title,
        text=text,
        price_kzt=int(price_raw) if price_raw else None,
        city=city or None,
        label=label,
    )

    print("\n--- Так объявление будет сохранено (после маскировки) ---")
    print(f"{ad.title}\n\n{ad.text}")
    print(f"--- язык: {ad.lang}, метка: {ad.label} ---")
    print("Проверь глазами: не осталось ли имён, телефонов, адресов?")
    if input("Сохранить? (y/n) > ").strip().lower() != "y":
        print("Не сохранено.")
        return
    if save_new_ad(ad):
        print(f"Сохранено. Всего объявлений: {len(read_ads(RAW_PATH))}")
    else:
        print("Такое объявление уже есть — пропущено.")


def cmd_import(args: argparse.Namespace) -> None:
    report = import_inbox(Path(args.inbox), dry_run=args.dry_run)
    added = report.added

    print("ПРОВЕРКА (ничего не сохранено)" if args.dry_run else "ИМПОРТ ЗАВЕРШЁН")
    print(f"Добавлено: {len(added)}")
    print(f"  по меткам:     {dict(Counter(ad.label for ad in added))}")
    print(f"  по origin:     {dict(Counter(ad.origin for ad in added))}")
    print(f"  по языку:      {dict(Counter(ad.lang for ad in added))}")
    print(f"  по источникам: {dict(Counter(ad.source for ad in added))}")
    print(f"  по категориям: {dict(Counter(ad.category for ad in added))}")

    print(f"\nЗамаскированы личные данные в {len(report.masked)} объявлениях:")
    for line in report.masked:
        print(f"  {line}")
    print(f"\nДубликаты (пропущены): {len(report.duplicates)}")
    for line in report.duplicates:
        print(f"  {line}")
    print(f"\nНе прошли проверку: {len(report.errors)}")
    for line in report.errors:
        print(f"  {line}")
    if not args.dry_run:
        print(f"\nВсего в {RAW_PATH}: {len(read_ads(RAW_PATH))}")


def cmd_gold(args: argparse.Namespace) -> None:
    count = label_gold(target=args.target)
    print(f"\nРазмечено сейчас: {count}. Всего в gold: {len(read_ads(GOLD_PATH))}")


def cmd_stats(args: argparse.Namespace) -> None:
    raw = read_ads(RAW_PATH)
    gold = read_ads(GOLD_PATH)
    print(f"Собрано объявлений: {len(raw)}")
    print(f"  по источникам: {dict(Counter(ad.source for ad in raw))}")
    print(f"  по языку:      {dict(Counter(ad.lang for ad in raw))}")
    print(f"  по категориям: {dict(Counter(ad.category for ad in raw))}")
    print(f"Gold (ручная разметка): {len(gold)}/{GOLD_TARGET}")
    print(f"  по меткам: {dict(Counter(ad.label for ad in gold))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and label ads by hand.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="add one ad to data/raw/ads.jsonl")
    p_add.add_argument("--text-file", help="read the ad text from this UTF-8 file")
    p_add.set_defaults(func=cmd_add)

    p_import = sub.add_parser("import", help="add many ads at once from an inbox file")
    p_import.add_argument("inbox", help="path to the inbox file, e.g. data/inbox.txt")
    p_import.add_argument("--dry-run", action="store_true", help="check only, save nothing")
    p_import.set_defaults(func=cmd_import)

    p_gold = sub.add_parser("gold", help="label ads by hand into data/gold/gold.jsonl")
    p_gold.add_argument("--target", type=int, default=GOLD_TARGET, help="gold set size")
    p_gold.set_defaults(func=cmd_gold)

    p_stats = sub.add_parser("stats", help="show how much is collected and labelled")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
