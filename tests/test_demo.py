"""Offline demo: the CLI helpers and the local web page."""

import threading
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

from test_baseline import toy_data

from aldau.check import split_ads
from aldau.explain import DISCLAIMER
from aldau.train_baseline import BaselineModel
from aldau.web import make_handler, parse_price, render


def test_split_ads_by_separator_lines():
    text = "Первое\nобъявление\n---\n\nВторое\n ---\n---\n"
    assert split_ads(text) == ["Первое\nобъявление", "Второе"]


def test_render_escapes_what_the_user_typed():
    page = render(text="<script>alert(1)</script>", answer="🔴 <b>риск</b>", level="red")
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page and "&lt;b&gt;риск&lt;/b&gt;" in page
    assert "#d93636" in page  # red border


def test_parse_price():
    assert parse_price("25 000") == 25000
    assert parse_price("") is None


def test_web_page_answers_a_post_offline():
    model = BaselineModel(mode="both", threshold=0.73).fit(toy_data())
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(model, use_teacher=False))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        assert "Проверить" in urllib.request.urlopen(url).read().decode("utf-8")
        form = urllib.parse.urlencode({"text": "Предоплата на карту, хозяин в другом городе",
                                       "price": ""}).encode()  # fmt: skip
        page = urllib.request.urlopen(url, data=form).read().decode("utf-8")
        assert "Оценка локальной модели" in page
        assert DISCLAIMER.split(":")[0] in page
    finally:
        server.shutdown()
