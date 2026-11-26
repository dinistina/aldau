"""Check an ad from the command line — the offline backup demo (no Telegram, no internet).

    python -m aldau.check                         # paste an ad, end it with a line "---"
    python -m aldau.check --file docs/demo_ads.txt   # several ads separated by "---" lines
    python -m aldau.check --price 25000 --teacher    # also ask the teacher in the grey zone

Offline by default: only the local student and the regex signals. With --teacher a grey-zone
ad costs at most ONE API request (cost guard), and only within the daily limit.
"""

import argparse
import sys
from pathlib import Path

from aldau.cascade import TEACHER_DAILY_LIMIT, calls_today, check
from aldau.explain import format_answer
from aldau.student import load_student

SEPARATOR = "---"


def split_ads(text: str) -> list[str]:
    """Ads separated by lines that contain only '---' (the same as in the inbox files)."""
    ads, current = [], []
    for line in text.splitlines():
        if line.strip() == SEPARATOR:
            ads.append("\n".join(current))
            current = []
        else:
            current.append(line)
    ads.append("\n".join(current))
    return [ad.strip() for ad in ads if ad.strip()]


def read_pasted_ad() -> str | None:
    """Read lines until '---' (end of the ad) or 'q' / end of input (quit)."""
    print(f"\nВставьте объявление и закончите строкой {SEPARATOR}  (q — выход):")
    lines = []
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not lines and line.strip().lower() == "q":
            return None
        if line.strip() == SEPARATOR:
            return "\n".join(lines)
        lines.append(line)
    return "\n".join(lines) if lines else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check an ad offline (student + regex signals).")
    parser.add_argument("--file", type=Path, help="text file with ads separated by '---'")
    parser.add_argument("--price", type=int, help="price in tenge (default: taken from the text)")
    parser.add_argument("--teacher", action="store_true",
                        help="ask the teacher for grey-zone ads (max 1 request per ad)")
    args = parser.parse_args()  # fmt: skip

    try:
        model = load_student()
    except FileNotFoundError:
        raise SystemExit("Нет модели: сначала python -m aldau.train_baseline") from None
    if args.teacher:
        print(f"Учитель: только для 🟡, не больше 1 запроса на объявление; сегодня уже "
              f"{calls_today()} из {TEACHER_DAILY_LIMIT}.")  # fmt: skip

    def show(text: str) -> None:
        result = check(model, text, price_kzt=args.price, use_teacher=args.teacher)
        print("\n" + "=" * 70)
        print(text[:300] + (" …" if len(text) > 300 else ""))
        print("-" * 70)
        print(format_answer(result))

    if args.file:
        for text in split_ads(args.file.read_text(encoding="utf-8")):
            show(text)
        return
    while (text := read_pasted_ad()) is not None:
        if text.strip():
            show(text)


if __name__ == "__main__":
    main()
