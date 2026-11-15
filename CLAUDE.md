# CLAUDE.md — Aldau: scam-ad detector for Kazakhstan marketplaces

"Aldau" (алдау) means "deception" in Kazakh.

## What this project is
An ML system that estimates the fraud risk of a classified ad (OLX.kz / Kolesa.kz / Krisha.kz style)
in Russian and Kazakh, and explains *why* it looks suspicious.

Core idea — **knowledge distillation (teacher → student)**:
1. **Teacher**: a large LLM deeply analyses each ad and returns a label, risk score, list of fraud
   signals and an explanation (structured JSON). Development starts with a **free** model;
   Claude Fable (`claude-fable-5-2`) is plugged in later as the strongest teacher (see "LLM providers").
2. **Student**: a small local classifier (TF-IDF + LogisticRegression baseline, then fine-tuned
   `cointegrated/rubert-tiny2`) trained on the teacher's labels. Fast, free, runs offline.
3. **Cascade at inference**: student first; if it is uncertain (score in the grey zone), escalate to the teacher.
4. **Benchmark**: compare teacher vs student vs `claude-haiku-5-5` zero-shot on a human-labelled gold set.

The full roadmap is in `docs/PLAN.md`. Work **one phase at a time**; do not start the next phase
until the current one has passing tests and the user has confirmed.

## The author
The repo owner is a 2nd-year IT student (machine-learning minor). He knows Python up to OOP,
basic Java, has built Telegram bots, knows basic PostgreSQL. He must be able to explain every part
of this project at a job interview, so:
- Prefer simple, readable code over clever code. No unnecessary abstractions.
- After finishing each phase, write a short explanation (in Russian) of what was built and why,
  including the key ML concepts involved, and append it to `docs/LEARNING_LOG.md`.
- When there is a design choice, briefly explain the trade-off before choosing.

## LLM providers
All LLM calls go through one module, `src/aldau/llm.py`, with a single function like
`complete_json(prompt, schema) -> dict`. The provider is chosen by `LLM_PROVIDER` in `.env`:
- `gemini` — Google AI Studio free tier, Flash models (default for development)
- `openrouter` — free models with the `:free` suffix (fallback)
- `ollama` — fully local model, no limits, needs a decent GPU/CPU
- `anthropic` — Claude Fable / Haiku, paid; used for the final teacher run and the benchmark
The rest of the code must not know which provider is used. Model names live in `.env`, not in code.
Free tiers have small daily request limits, so:
- label **several ads per request** (batch of ~10, configurable), with JSON output per ad;
- respect rate limits (sleep + retry on 429), never crash mid-run, resume from cache.
- the cache key includes the provider + model name, so labels from different teachers never mix.

## Stack
- Python 3.11+, dependencies in `pyproject.toml`
- `anthropic`, `google-genai`, `openai` (for OpenRouter/Ollama OpenAI-compatible APIs) SDKs
- `pandas`, `scikit-learn`, `transformers`, `torch`, `datasets` for ML
- `imagehash` + `Pillow` for duplicate-photo detection
- `aiogram` 3.x for the Telegram bot
- SQLite (via the standard `sqlite3` module) for storing ads, labels and bot logs
- `pytest` for tests, `ruff` for linting

## Project layout (target)
```
aldau/
  data/            # raw/ (hand-collected), synthetic/, labeled/, gold/  — gitignored except samples
  src/aldau/
    schema.py      # Ad dataclass / pydantic model, label schema
    collect.py     # helpers to add hand-collected ads to data/raw
    features.py    # price-vs-market, regex signals (prepayment, messenger, urgency), PII masking
    photos.py      # perceptual hashing, duplicate lookup
    llm.py         # provider-agnostic LLM client (gemini / openrouter / ollama / anthropic)
    teacher.py     # LLM labelling with structured JSON output + on-disk cache
    synth.py       # synthetic scam / legit ad generation (train only!)
    train_baseline.py
    train_student.py
    evaluate.py    # metrics + benchmark table
    cascade.py     # student → teacher escalation
    bot/           # aiogram bot
  tests/
  docs/PLAN.md, docs/LEARNING_LOG.md
  notebooks/       # optional EDA
```

## Hard rules
- **Secrets**: API keys (`GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`) and the bot token live only in `.env` (loaded with `python-dotenv`).
  `.env` is in `.gitignore`. Provide `.env.example`. Never print or log keys.
- **Never re-pay for the same API call**: every teacher/Haiku response is cached on disk
  (`data/cache/*.jsonl`, keyed by a hash of model + prompt version + ad text). Scripts must resume
  from the cache after a crash.
- **Cost guard**: every script that calls the API takes `--limit N` and prints how many calls it
  will make before starting. Default to a small limit.
- **No data leakage**: synthetic ads are used for *training only*. The gold test set contains only
  real, human-labelled ads and is never seen by training code. Split by ad, deduplicate near-identical
  texts before splitting. Add a test that asserts no overlap between train and gold.
- **Privacy**: mask phone numbers, emails, names and links in ad text before storing or sending to
  any model (`features.mask_pii`). Do not store personal data of sellers.
- **No mass scraping** of OLX/Kolesa/Krisha. Data comes from manual collection, small samples and
  synthetic generation. Do not write crawlers.
- The bot is advisory: every answer ends with a short disclaimer that the result is a probability,
  not a verdict.
- Tests must not call the real API — mock the Anthropic client.

## Commands
- Install: `pip install -e ".[dev]"`
- Tests: `pytest -q`
- Lint: `ruff check src tests`

## Git
Small commits with clear messages, one logical change each. Commit at the end of every phase.

## Working agreement with the owner (added 2026-11-15)
- After every phase, report to the owner in Russian how it went: what was built, what went wrong
  or was skipped, test/lint results.
- In the same report, recommend which Claude model and which effort level to use for the next phase
  (and a cheaper model that would probably also work).
- Keep the "Progress log" below up to date: one entry per finished phase or other notable piece of
  work, updated in the same commit as the work itself.
- Backups: the owner wants occasional backups; details are not agreed yet — ask before setting up.

## Progress log
| Date | Work | Status |
|------|------|--------|
| 2026-11-15 | Handoff files (CLAUDE.md, START.md, docs/PLAN.md) placed in repo, git initialised, public GitHub repo `dinistina/aldau` created | done |
