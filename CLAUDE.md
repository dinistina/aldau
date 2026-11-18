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
  human-written, human-labelled ads (`origin` = `real` or `case_based`, never `synthetic`) and is
  never seen by training code. Split by ad, deduplicate near-identical
  texts before splitting. Add a test that asserts no overlap between train and gold.
- **Privacy**: mask phone numbers, emails, names and links in ad text before storing or sending to
  any model (`features.mask_pii`). Do not store personal data of sellers.
- **No mass scraping** of OLX/Kolesa/Krisha. Data comes from manual collection, small samples and
  synthetic generation. Do not write crawlers.
- The bot is advisory: every answer ends with a short disclaimer that the result is a probability,
  not a verdict.
- Tests must not call the real API — mock `llm.py` (any provider client).

## Known limitation: case_based ads in gold (owner's decision, 2026-11-17)
- There are no real scam ads yet, so the gold set may include `case_based` ads (scam schemes
  rewritten by hand from news, complaints and safety memos). Real scam ads (e.g. from victims'
  screenshots, `origin=real` with a `source_note`) are added to gold as they appear.
- Every evaluation report (teacher check in Phase 3, benchmark in Phase 6) shows metrics
  **separately for `real` and `case_based`**, plus the overall number.
- Phase 6 includes a style-leak check (can a classifier predict `origin` from the text?).
- This limitation is also stated in `README.md`.

## Commands
- Install: `pip install -e ".[dev,photos]"` (add groups as phases need them: `llm`, `ml`
  (scikit-learn, matplotlib, pandas), `deep` (torch, transformers), `bot`).
  Use a local virtualenv in `.venv/` (gitignored).
- torch with GPU support (Windows): `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  (plain `pip install torch` on Windows gives a CPU-only build).
- EDA report: `python -m aldau.eda` -> `reports/eda.md` (aggregates only, no ad texts; committed).
- Tests: `pytest -q`
- Lint: `ruff check src tests`
- On the owner's Windows laptop, Application Control blocks the `pytest.exe` launcher in `.venv`;
  run tools through the interpreter instead: `.venv\Scripts\python.exe -m pytest -q` and
  `.venv\Scripts\python.exe -m ruff check src tests`.
- Windows Smart App Control / Application Control is active on the owner's laptop and blocks
  DLLs by reputation; the blocks can change over time. Do not try to bypass the policy (the owner
  decided not to touch it, 2026-11-17). If an import fails with "An Application Control policy has
  blocked this file", report it and find a way around in code (e.g. drop the dependency).

## Machines (checked 2026-11-19)
- **MSI Katana GF76 (host STINA), NVIDIA RTX 3060 Laptop GPU, driver 596.21** — the main machine.
  Repo in `F:\aldau`, `.venv` with all groups. Works: numpy, scipy, scikit-learn, pandas,
  matplotlib, torch 2.11 + CUDA 12.8 (`torch.cuda.is_available()` is True, matmul on cuda:0),
  transformers, google-genai, aiogram. Blocked: `pyarrow._dataset` -> the `datasets` library does
  not import (not needed). On 2026-11-17 numpy was blocked on this same machine; on 2026-11-19 it
  loads. **Phase 5 (training, rubert-tiny2 on GPU) is done here**; Google Colab is the fallback.
- `data/` and `.env` are not in git; they live in `F:\aldau\data` and `F:\aldau\.env` on the MSI.

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
| 2026-11-15 | Phase 0 — skeleton: `pyproject.toml` (heavy deps in optional groups `llm`/`ml`/`photos`/`bot`), `schema.py` (pydantic `Ad`, `TeacherLabel`), tests, GitHub Actions CI, README stub, `.env.example`, `docs/LEARNING_LOG.md` | done |
| 2026-11-15 | Phase 1 (tools only) — `features.py` (`mask_pii`, `detect_lang`), `collect.py` CLI (`add`, `gold`, `stats`), tests, `docs/DATA_GUIDE.md`; CI actions bumped to checkout@v5 / setup-python@v6 | tools done, **waiting for the owner's data** |
| 2026-11-15 | Phase 1 addition — batch import `python -m aldau.collect import <file> [--dry-run]` (inbox format: `source \| category \| price \| city \| label \| origin`, ads separated by `---`); `Ad.origin` field (`real` / `case_based` / `synthetic`); `detect_lang` also recognises Kazakh typed without special letters. First import: 146 ads in `data/raw/ads.jsonl` (95 legit real, 45 scam case_based, 5 unknown, +1 earlier test ad) | done |

| 2026-11-17 | Data quality tools — `stats` shows category/origin/lang × label tables; `import` warns when header price disagrees with the text (>10 %) and updates stored ads whose header changed; new `collect check-prices`; optional 7th header field `source_note`; gold never offers synthetic ads. Owner decisions recorded: case_based allowed in gold (see "Known limitation"), metrics split by origin, category check added to Phase 2 and style-leak check to Phase 6 in `docs/PLAN.md` | done |
| 2026-11-17 | Header prices fixed on the owner's instruction (HP 255 G7 -> 70 000, MacBook -> 350 000, Темиртау flat -> 10 500 000); `data/price_ok.txt` list of checked mismatches (LG washer: bargaining, not an error); new category `посуточная аренда` (7 daily-rent scam ads moved); `import --verbose` lists duplicates | done |
| 2026-11-17 | Phase 2 — 13 regex fraud signals ru + kk (`features.SIGNAL_PATTERNS`, `find_signals`), `PriceReference` (median price by category/city, label-free) and `extract_features`; `photos.py` (hand-written pHash on Pillow, duplicate search); `eda.py` -> `reports/eda.md` (signal rates, price ratio, single-field ROC-AUC with stratified 5-fold CV). Dependency groups split into `ml` / `deep`; `photos` = Pillow only. Started on the owner's go-ahead on current data (146 ads) | done |
| 2026-11-19 | Data import `data/inbox2.txt`: +31 ads (30 legit real from dealers, parts sellers, agencies, shops, 9 daily rents with legal booking/deposit in `посуточная аренда`; 1 scam real from a news screenshot, `source_note` = link). 5 checked price mismatches added to `data/price_ok.txt`. Now 177 ads; `запчасти` 38 % scam, `посуточная аренда` 44 % scam (were 100 %); origin still confounded (only 1 real scam) | done |

Data findings (2026-11-17, `reports/eda.md`): origin predicts the label perfectly (ROC-AUC 1.00);
category 0.78 (`запчасти`, `посуточная аренда` 100 % scam, `одежда` 0 %); city 0.72; signal count
0.98 but confounded with origin (rewritten cases name the scheme explicitly); price ratio 0.48
(no signal yet). No photos collected. The owner is collecting more data in parallel (honest
booking/deposit ads, legit parts, real scam texts); gold labelling is postponed.

| 2026-11-19 | MSI environment checked: all dependency groups installed, torch with CUDA sees the RTX 3060, 173 tests pass; `datasets` dropped from the `deep` group (pyarrow DLL blocked); `GEMINI_API_KEY` placed in `.env` (owner's request) | done |

Next: Phase 3 (teacher, `llm.py`) — plan shown to the owner 2026-11-19, waiting for "ok". Order agreed
with the owner: (1) code + tests with mocks, no real requests; (2) trial run on 10 ads, show 3–4
full teacher answers; (3) **no full labelling run until the owner has collected more data and
labelled gold.** Before any real API run, state the number of requests and wait for confirmation.
