"""A tiny local web page for the offline backup demo: paste an ad, get the answer.

    python -m aldau.web               # http://127.0.0.1:8000, offline (student + regex only)
    python -m aldau.web --teacher     # also ask the teacher in the grey zone (needs internet)

Built on the standard library (http.server) — no new dependencies. It listens on
127.0.0.1 only, so nobody else on the expo Wi-Fi can open it.
"""

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from aldau.cascade import check
from aldau.explain import format_answer
from aldau.student import load_student

MAX_TEXT = 5000  # characters; a longer paste is cut

PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aldau — проверка объявления</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
         padding: 0 16px; background: #f6f7f9; color: #1d2330; }}
  h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }}
  .sub {{ color: #5b6475; margin-top: 0; }}
  textarea {{ width: 100%; min-height: 180px; font: inherit; padding: .7rem;
             border: 1px solid #c8cdd6; border-radius: 8px; box-sizing: border-box; }}
  input[type=number] {{ font: inherit; padding: .4rem; width: 12rem; }}
  button {{ font: inherit; padding: .6rem 1.4rem; border: 0; border-radius: 8px;
           background: #2856d8; color: #fff; cursor: pointer; margin-top: .6rem; }}
  .answer {{ white-space: pre-wrap; background: #fff; border-radius: 8px; padding: 1rem;
            margin-top: 1.4rem; border-left: 6px solid {color}; line-height: 1.45; }}
</style></head>
<body>
<h1>Aldau</h1>
<p class="sub">Оценка риска мошенничества в объявлении (OLX / Kolesa / Krisha), русский и казахский.
Режим: {mode}.</p>
<form method="post" action="/">
  <textarea name="text" placeholder="Вставьте текст объявления">{text}</textarea><br>
  <label>Цена, тг (необязательно): <input type="number" name="price" value="{price}"></label><br>
  <button type="submit">Проверить</button>
</form>
{answer}
</body></html>
"""
COLORS = {"green": "#2e9d5b", "yellow": "#e0a800", "red": "#d93636"}


def render(text: str = "", price: str = "", answer: str = "", level: str = "",
           mode: str = "офлайн") -> str:  # fmt: skip
    """The whole page. Everything the user typed is HTML-escaped before it goes back."""
    block = f'<div class="answer">{html.escape(answer)}</div>' if answer else ""
    return PAGE.format(
        text=html.escape(text), price=html.escape(price), answer=block,
        color=COLORS.get(level, "#c8cdd6"), mode=html.escape(mode),
    )  # fmt: skip


def parse_price(value: str) -> int | None:
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else None


def make_handler(model, use_teacher: bool):
    mode = "с углублённой проверкой" if use_teacher else "офлайн, только локальная модель"

    class Handler(BaseHTTPRequestHandler):
        def send_page(self, page: str) -> None:
            body = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (the name is fixed by http.server)
            self.send_page(render(mode=mode))

        def do_POST(self) -> None:  # noqa: N802
            length = min(int(self.headers.get("Content-Length", 0)), 4 * MAX_TEXT)
            form = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
            text = form.get("text", [""])[0][:MAX_TEXT]
            price = form.get("price", [""])[0]
            if not text.strip():
                self.send_page(render(price=price, mode=mode))
                return
            result = check(model, text, price_kzt=parse_price(price), use_teacher=use_teacher)
            self.send_page(render(text, price, format_answer(result), result.level, mode))

        def log_message(self, format: str, *args) -> None:
            pass  # do not print requests (they could contain the pasted ad)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local web page for the offline demo.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--teacher", action="store_true", help="ask the teacher in the grey zone")
    args = parser.parse_args()
    try:
        model = load_student()
    except FileNotFoundError:
        raise SystemExit("Нет модели: сначала python -m aldau.train_baseline") from None
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, args.teacher))
    print(f"Откройте http://127.0.0.1:{args.port}  (Ctrl+C — остановить)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")


if __name__ == "__main__":
    main()
