"""Shared test setup.

Safety net for the rule "tests must not call the real API":
- .env is never loaded in tests (the real key stays out of reach);
- every real provider function is replaced by one that fails the test loudly.
Tests that need an LLM pass a fake function or put a fake into llm.PROVIDERS.
"""

import pytest

from aldau import llm


def _real_api_forbidden(prompt, schema, model):
    raise AssertionError("A test tried to call a real LLM API")


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    monkeypatch.setattr(llm, "load_dotenv", lambda: None)
    for provider in list(llm.PROVIDERS):
        monkeypatch.setitem(llm.PROVIDERS, provider, _real_api_forbidden)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
