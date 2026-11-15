"""Tools for hand-collecting and hand-labelling ads. No scraping: you paste ads yourself.

Usage (run from the project root):
    python -m aldau.collect add                  # add one ad, typing/pasting the text
    python -m aldau.collect add --text-file a.txt  # ... or read the ad text from a UTF-8 file
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
from pathlib import Path

from aldau.features import detect_lang, mask_pii
from aldau.schema import Ad

RAW_PATH = Path("data/raw/ads.jsonl")
GOLD_PATH = Path("data/gold/gold.jsonl")
GOLD_TARGET = 100
SOURCES = ["olx", "kolesa", "krisha"]


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
    )


def save_new_ad(ad: Ad, raw_path: Path = RAW_PATH) -> bool:
    """Append the ad to the raw file. Returns False if it is already there."""
    if any(existing.id == ad.id for existing in read_ads(raw_path)):
        return False
    append_ad(raw_path, ad)
    return True


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

    p_gold = sub.add_parser("gold", help="label ads by hand into data/gold/gold.jsonl")
    p_gold.add_argument("--target", type=int, default=GOLD_TARGET, help="gold set size")
    p_gold.set_defaults(func=cmd_gold)

    p_stats = sub.add_parser("stats", help="show how much is collected and labelled")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
