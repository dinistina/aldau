import pytest
from pydantic import BaseModel

from aldau import llm


class Answer(BaseModel):
    value: int


def fake_provider(answers):
    """A provider that returns / raises the prepared items one by one and counts calls."""
    calls = []

    def call(prompt, schema, model):
        calls.append((prompt, schema, model))
        item = answers[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    call.calls = calls
    return call


def test_settings_and_model_id():
    assert llm.get_settings() == ("gemini", "test-model")
    assert llm.model_id() == "gemini:test-model"


@pytest.mark.parametrize(
    "env, message",
    [({"LLM_PROVIDER": "gpt"}, "LLM_PROVIDER"), ({"LLM_MODEL": ""}, "LLM_MODEL")],
)
def test_bad_settings_give_readable_error(monkeypatch, env, message):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(llm.LLMError, match=message):
        llm.get_settings()


def test_complete_json_returns_parsed_answer(monkeypatch):
    provider = fake_provider(['{"value": 7}'])
    monkeypatch.setitem(llm.PROVIDERS, "gemini", provider)
    assert llm.complete_json("prompt", Answer) == {"value": 7}
    assert provider.calls == [("prompt", Answer, "test-model")]


def test_retries_temporary_errors_with_growing_pause(monkeypatch):
    provider = fake_provider([llm.RetryableError("429"), llm.RetryableError("503"), '{"value": 1}'])
    monkeypatch.setitem(llm.PROVIDERS, "gemini", provider)
    waits = []
    assert llm.complete_json("p", Answer, sleep=waits.append, log=lambda m: None) == {"value": 1}
    assert waits == [10, 20]  # exponential backoff
    assert len(provider.calls) == 3


def test_gives_up_after_max_retries(monkeypatch):
    errors = [llm.RetryableError("429")] * (llm.MAX_RETRIES + 1)
    monkeypatch.setitem(llm.PROVIDERS, "gemini", fake_provider(errors))
    with pytest.raises(llm.LLMError, match="не помогли"):
        llm.complete_json("p", Answer, sleep=lambda s: None, log=lambda m: None)


def test_permanent_error_is_not_retried(monkeypatch):
    provider = fake_provider([llm.LLMError("ошибка 400: API key not valid")])
    monkeypatch.setitem(llm.PROVIDERS, "gemini", provider)
    with pytest.raises(llm.LLMError, match="400"):
        llm.complete_json("p", Answer, sleep=lambda s: None)
    assert len(provider.calls) == 1


def test_non_json_answer_is_an_error(monkeypatch):
    monkeypatch.setitem(llm.PROVIDERS, "gemini", fake_provider(["Конечно! Вот ответ: ..."]))
    with pytest.raises(llm.LLMError, match="не JSON"):
        llm.complete_json("p", Answer)


def test_missing_gemini_key_is_reported_without_calling_the_api():
    # the real Gemini function (not the fake): it must stop before any network call
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm._call_gemini("p", Answer, "test-model")


def test_network_problems_are_recognised_by_class_name():
    class ReadTimeout(Exception):  # e.g. httpx.ReadTimeout, httpx2.ReadTimeout
        pass

    class ConnectError(Exception):
        pass

    assert llm.is_network_problem(ReadTimeout())
    assert llm.is_network_problem(ConnectError())
    assert llm.is_network_problem(TimeoutError())
    assert not llm.is_network_problem(ValueError("bad schema"))


def test_other_providers_are_not_connected_yet():
    with pytest.raises(llm.LLMError, match="не подключён"):
        llm._not_implemented("anthropic")("p", Answer, "m")
