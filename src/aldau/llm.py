"""One door to every LLM: complete_json(prompt, schema) -> dict.

The rest of the project never imports a provider SDK directly. Which provider and
model to use comes from .env:
    LLM_PROVIDER=gemini          # gemini | openrouter | ollama | anthropic
    LLM_MODEL=gemini-2.5-flash
So switching the teacher from a free model to Claude later is a change in .env,
not in code.

Only Gemini is implemented for now; the other providers raise a clear error.

Retries: free tiers answer "429 Too Many Requests" when the per-minute limit is
hit, and servers sometimes answer 5xx. Those are temporary, so we wait and try
again with a growing pause (exponential backoff: 10 s, 20 s, 40 s, 80 s).
Any other error (wrong key, wrong model name) is permanent: retrying would only
burn the daily limit, so it is raised at once.
"""

import json
import os
import time
from collections.abc import Callable
from functools import cache

from dotenv import load_dotenv
from pydantic import BaseModel

PROVIDERS_KNOWN = ["gemini", "openrouter", "ollama", "anthropic"]
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
FIRST_WAIT_SECONDS = 10


class LLMError(Exception):
    """A permanent problem: bad settings, bad key, unusable answer, or retries exhausted."""


class RetryableError(Exception):
    """A temporary problem (rate limit, server overload): worth waiting and trying again."""


def get_settings() -> tuple[str, str]:
    """(provider, model) from .env. Raises LLMError with a readable message."""
    load_dotenv()  # does not override variables that are already set
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    model = os.environ.get("LLM_MODEL", "").strip()
    if provider not in PROVIDERS_KNOWN:
        raise LLMError(f"LLM_PROVIDER в .env должен быть одним из {PROVIDERS_KNOWN}")
    if not model:
        raise LLMError("LLM_MODEL в .env не задан")
    return provider, model


def model_id() -> str:
    """'provider:model' — part of every cache key, so labels of different teachers never mix."""
    provider, model = get_settings()
    return f"{provider}:{model}"


# --- providers -------------------------------------------------------------------
# Each provider function takes (prompt, schema, model) and returns the answer text.


@cache
def _gemini_client(api_key: str):
    from google import genai  # imported here so the rest of the project works without it

    return genai.Client(api_key=api_key)


def _call_gemini(prompt: str, schema: type[BaseModel], model: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise LLMError("GEMINI_API_KEY в .env не задан")
    try:
        from google.genai import errors, types
    except ImportError as error:
        raise LLMError('Пакет google-genai не установлен: pip install -e ".[llm]"') from error
    config = types.GenerateContentConfig(
        response_mime_type="application/json",  # ask for JSON only, no prose around it
        response_json_schema=schema.model_json_schema(),  # ... in exactly this shape
        temperature=0,  # the most likely answer every time -> repeatable labels
    )
    try:
        response = _gemini_client(api_key).models.generate_content(
            model=model, contents=prompt, config=config
        )
    except errors.APIError as error:
        # Only the status code and the API's message are shown: never the key.
        message = f"Gemini: ошибка {error.code}: {error.message}"
        if error.code in RETRY_STATUS:
            raise RetryableError(message) from error
        raise LLMError(message) from error
    if not response.text:
        raise LLMError("Gemini вернул пустой ответ (возможно, сработал фильтр безопасности)")
    return response.text


def _not_implemented(provider: str) -> Callable[[str, type[BaseModel], str], str]:
    def call(prompt: str, schema: type[BaseModel], model: str) -> str:
        raise LLMError(f"Провайдер {provider!r} пока не подключён (есть только gemini)")

    return call


PROVIDERS: dict[str, Callable[[str, type[BaseModel], str], str]] = {
    "gemini": _call_gemini,
    "openrouter": _not_implemented("openrouter"),
    "ollama": _not_implemented("ollama"),
    "anthropic": _not_implemented("anthropic"),
}


# --- the one public function -------------------------------------------------------


def complete_json(
    prompt: str,
    schema: type[BaseModel],
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict:
    """Send the prompt, get back a dict shaped like `schema`.

    The answer is parsed as JSON but NOT validated here: the caller validates it
    piece by piece, so one broken item does not throw away a whole batch.
    `sleep` and `log` are parameters so tests can run without waiting.
    """
    provider, model = get_settings()
    call = PROVIDERS[provider]
    for attempt in range(MAX_RETRIES + 1):
        try:
            text = call(prompt, schema, model)
            break
        except RetryableError as error:
            if attempt == MAX_RETRIES:
                raise LLMError(f"{error} — не помогли {MAX_RETRIES} повтора") from error
            wait = FIRST_WAIT_SECONDS * 2**attempt
            log(f"  {error}; жду {wait} с и пробую снова ({attempt + 1}/{MAX_RETRIES})")
            sleep(wait)
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMError(f"Модель вернула не JSON: {text[:200]!r}") from error
