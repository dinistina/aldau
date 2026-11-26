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
  Repo in `F:\projects\aldau` (moved from `F:\aldau` on 2026-11-23), `.venv` with all groups. Works: numpy, scipy, scikit-learn, pandas,
  matplotlib, torch 2.11 + CUDA 12.8 (`torch.cuda.is_available()` is True, matmul on cuda:0),
  transformers, google-genai, aiogram. Blocked: `pyarrow._dataset` -> the `datasets` library does
  not import (not needed). On 2026-11-17 numpy was blocked on this same machine; on 2026-11-19 it
  loads. **Phase 5 (training, rubert-tiny2 on GPU) is done here**; Google Colab is the fallback.
- `data/` and `.env` are not in git; they live in `F:\projects\aldau\data` and
  `F:\projects\aldau\.env` on the MSI. The old copy `F:\aldau` is kept until the owner deletes it.

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

| 2026-11-19 | Phase 3, part 1 — `llm.py` (`complete_json`, Gemini via google-genai with JSON schema + temperature 0, backoff 10/20/40/80 s on 429/5xx, other providers stubbed), `teacher.py` (prompt `PROMPT_VERSION = "v1"` with Kazakhstan scam schemes and "not a scam by itself" list; batches of 10; per-ad cache `data/cache/teacher.jsonl`; per-item pydantic check; `label` with y/n cost guard and `--show`; `compare` -> `reports/teacher_check.md` split by origin), `tests/conftest.py` blocks real API calls and .env in tests. No real API request made yet | done |

Trial run prepared (waiting for the owner's go): `python -m aldau.teacher label --show --ids
39acf6eda68e,4c02874893cb,702429d930e2,e7244b0ccd31,4551e689bf8e,edef976ad909,5a9b726f10b8,c4483ded2f31,80255078a661,450887c1738e`
= 1 request to Gemini (up to 5 with retries).

| 2026-11-19 | Phase 3 trial run, approved by the owner: 10 chosen ads, model switched to `gemini-3.8-flash` (`gemini-2.5-flash` returned 404 "no longer available to new users"). Attempts: 404, then a request that hung >10 min (no client timeout), then a local import bug; 4th attempt succeeded — 1 request, 10/10 valid answers cached. Fixes: 180 s request timeout, network errors recognised by exception class name (google-genai 3.5 no longer uses httpx), `source_note` keeps links (a news URL had been masked to [LINK]) | done |

Trial findings: 9 of 9 judgeable ads — 8 correct. All 3 case_based scams caught (0.92–0.99), all 5 legit
correct incl. hard negatives (daily rent with deposit, imported engines, "СРОЧНО" MacBook). The only real
scam (kitchen set, from a Tengrinews story) was called legit 0.05 — its text has no visible scheme; the fraud
happened outside the ad. Risk scores are bimodal (0.05 or >0.9): poor calibration for the cascade grey
zone; candidate for prompt v2.

| 2026-11-19 | Prompt v2 (owner's request): explicit risk scale with examples (0–0.15 / 0.15–0.4 / 0.4–0.7 / 0.7–1), `PROMPT_VERSION = "v2"`; `teacher versions --old v1` side-by-side table; gold key `t` (`st`/`lt`) -> `Ad.not_from_text`, `compare` shows recall with and without such ads; PLAN: cascade grey zone uses the **student's** risk, teacher `risk_score` is the soft label in Phase 5 | done |
| 2026-11-19 | v2 trial on the same 10 ads — done on the owner's go. Batch of 10 kept failing (503 "high demand" / 504 "deadline expired"); with `--batch-size 5` (owner's rule on 504) it worked: 2 requests (the first on its 4th attempt), 10/10 valid answers | done |

v1 vs v2 (same model `gemini-3.8-flash`): labels unchanged (8 of 9 correct; the kitchen-set real scam is
still legit). Risk spread improved but stays narrow: distinct values 4 -> 7, in 0.15–0.70: 0 -> 2 of 10
(kitchen set 0.20 "messenger", 4Runner 0.20 "urgency"; daily rent with deposit 0.15; scams 0.90–0.98).
Gemini's free tier is unstable for this model: expect 503/504; use `--batch-size 5` for real runs.

| 2026-11-23 | Gold labelled by the owner: 100 ads (73 legit / 27 scam after the #19 fix; 28 case_based, 72 real; 30 kk), 77 skipped, 7 with `t`, no synthetic. #19 (c7352d73766c) changed st -> l on the owner's request (it was `st`, not `lt`), logged. Backup `data/backup/gold_2026-11-23_before_fix19.jsonl` | done |
| 2026-11-23 | `gold.py`: `diff` (gold vs import labels), `change` (logged to `data/gold_changes.jsonl`: id, was, now, reason, date), `freeze` (`data/gold/FROZEN.json` with SHA-256; `gold` labelling and `change` refuse afterwards), `check`. `agreement.py`: `sample` (30 random gold ids, seed 7), `label` (second annotator, blind to the first labels), `kappa` (Cohen's kappa, confusion table, t agreement) | done |

Gold issues raised with the owner 2026-11-23: scam 27 < 30; only 2 real scam in gold, both `t`;
8 gold labels differ from import labels — incl. 3 case_based schemes labelled legit (two of them
`lt`, which contradicts the meaning of `t`) and the kitchen-set real scam labelled plain `l`.
The owner rereads them once (`python -m aldau.gold diff`), then gold is frozen. The full teacher
run (34 requests, `--batch-size 5`) waits for the owner's "ok".

| 2026-11-23 | Project moved to `F:\projects\aldau` on the owner's request (robocopy without `.venv`/caches; new `.venv` with all groups, torch CUDA OK, 233 tests pass); the session works there now | done |
| 2026-11-23 | Gold fixes by the owner (logged in `data/gold_changes.jsonl`, reason "уточнён смысл t: метка = правда, t = по тексту не видно"): fb6bd5f1445d, 34f42134036a, 39acf6eda68e -> scam + t. Gold now 30 scam / 70 legit, 8 with t. Kept legit: 197a0fa10b9c, 450887c1738e. Pending the owner's decision: b951640564f2 (фары), 23bb71cac21d, 94ef27cd90a1 | done |
| 2026-11-23 | Skip counter: "n" in gold labelling writes the id to `data/gold/skipped.txt` (the first 77 rebuilt from the labelling order); skipped ads are not offered again (`--include-skipped` to revisit); unclear answers are asked again; `collect gold --only-inbox data/inboxN.txt` labels only the ads of one inbox (no size limit); `stats` shows skipped and t counts | done |

| 2026-11-23 | Owner's last 3 gold decisions (b951640564f2 -> scam; 23bb71cac21d, 94ef27cd90a1 -> legit) were made in the OLD folder `F:\aldau` by mistake; re-applied in `F:\projects\aldau` with the owner's reasons verbatim. Gold now 29 scam / 71 legit, 6 with t, 7 logged changes. Only 1 real scam in gold (kitchen set, t): origin still decides the label (case_based 100 % scam, real 1 %). **Work only in `F:\projects\aldau`; `F:\aldau` is stale** | done |

| 2026-11-24 | Data import `data/inbox3.txt`: +20 live suspicious OLX ads (label `unknown`, origin `real`, `source_note` = ad link; 9 daily rents, 4 phones, 2 laptops, 2 rents, 3 cars «рассрочка с предоплатой»; 3 kk; Krisha blocked by captcha). 4 header/text price gaps are bait prices, added to `data/price_ok.txt`. Now 197 ads | done |

Full teacher run rule (owner, 2026-11-23, teacher updated 2026-11-24): all labels must come from ONE
teacher — now `gemini-3.5-flash`, prompt v2; never mix models in one labelling. If a run fails on 503, do not retry the same day; retry in the
morning, 08:00–11:00 Almaty time (night in the US, less load). The run resumes from the cache.
2026-11-23 evening: two attempts (22:15, 22:38) stopped on the first request with 503; 10 of 177
ads labelled (the trial ads). Scheduled app task `aldau-teacher-morning-run` resumes the run on
2026-11-24 at 08:30 Almaty, retrying every 30 min until 11:00 (runs only if the Claude app is open).

2026-11-24: the scheduled task never labelled anything — the 08:30 and 22:47 runs stopped on a
permission prompt nobody answered (task ran in default mode); the 01:13 run got 503 x5. The 22:47
and 01:13 runs broke the "not the same day" rule (likely triggered by "Run now", which I had
suggested for pre-approval). **Owner's decision 2026-11-24: no more scheduled tasks; the owner
starts runs himself.**

**Teacher change (owner, 2026-11-24):** `gemini-3.8-flash` gave 503 on every attempt since
2026-11-19 evening, so the teacher becomes `gemini-3.7-flash`, or `gemini-3.5-flash` if that is
overloaded — and ALL ads are relabelled with the new teacher (one teacher for all labels still
holds; the 20 cached answers of 3.8-flash stay in the cache only as history). Same day: 3.7-flash
503 x5 on a 5-ad probe -> `.env` now `LLM_MODEL=gemini-3.5-flash`; 3.5-flash 503 x5 on a 5-ad
batch, but diagnostic requests succeeded (tiny prompt 2 s; tiny prompt + schema 8 s; real prompt
for 1 ad + schema 9 s, ~400 thinking tokens). The 503s hit heavy (multi-ad) requests; smaller
batches are the next step (needs the owner's ok: more requests).

2026-11-25: owner chose `--batch-size 1` (more reliable; each ad judged independently, no
influence from neighbouring ads in the prompt). First run: 16 of 197 ads labelled, then one ad
hit 503 x5 and the whole run stopped. Fix: `llm` now raises `QuotaError` (429 after retries:
stop) or `OverloadedError` (5xx/timeouts after retries); `teacher.label_ads` skips an overloaded
batch (it stays uncached, asked again next run) and stops only after 3 overloads in a row,
on quota, or on permanent errors.

2026-11-25, second run: the very first request got 503 x2, then 429 "You exceeded your current
quota" (free-tier daily request limit for gemini-3.5-flash; exact number unknown, it seems small —
about 20 requests had been sent that day incl. retries). Stopped per the owner's rule; 16 of 197
ads labelled. At ~20 requests/day, batch 1 needs ~10 days -> raised with the owner (options:
billing on the Google project, batch 5, another provider). `TELEGRAM_BOT_TOKEN` placed in `.env`
(owner's request).

**Owner's decisions 2026-11-25 (no billing; stay on the free key):**
- Phase 5 starts now without waiting for the teacher: the baseline trains on the OWNER's import
  labels; gold is never used for training, only for evaluation. Report it as a small prototype.
- Teacher labelling continues in the background, ~20 requests/day, started by the owner; it is
  for the "teacher vs human" comparison after the EXPO.
- In the bot the teacher is used only for the cascade grey zone, with a timeout; if the quota is
  used up, the bot answers with the student + regex signals and never crashes.
Phase 5 plan shown 2026-11-25, "ok" given 2026-11-26.

| 2026-11-26 | Phase 5 baseline prototype — `train_baseline.py` (train = owner's import labels minus gold and near-copies of gold (Jaccard ≥ 0.8), char_wb TF-IDF 2–5 + Phase 2 features, LogisticRegression balanced; mode/C/threshold chosen by 5-fold CV on train only), `evaluate.py` (gold metrics by slice, recall with/without t, variant comparison, latency, worst misses; `reports/baseline.md` ids only), `student.py` (`predict_text`). CI installs the `ml` group. 254 tests pass | done |

Phase 5 results (train 74 ads: 17 scam all case_based, 57 legit all real; gold 120: 37 scam, 9 of
them real): CV chose `text` (PR-AUC 0.97, C=10, threshold 0.40). On gold: all — precision 0.97,
recall 0.81, PR-AUC 0.96; **case_based recall 1.00 but real scam recall 0.22 (2 of 9)** — the
text model learned the rewriting style (shortcut). Variant `features` (regex + price): real scam
recall 0.89 but overall precision 0.67; `both`: real recall 0.78, precision 0.79. Latency ~1 ms.
Choosing the variant for the demo after looking at gold would use gold for selection — the
owner decides; the pre-registered (CV-chosen) result is `text`.

| 2026-11-26 | Demo student chosen by the owner: `both` (text + features), chosen AFTER looking at gold — stated in `reports/baseline.md` and README next to the honest CV result (`text`); reason: no real scam in training, the text model learned the style. `train_baseline --mode` (default `both`), report shows both. Bug fixed: `python -m aldau.train_baseline` pickled `__main__.BaselineModel`, so no other program could load the model. PLAN: clean test set of 20–30 live ads after the EXPO | done |

Demo model on gold (optimistic, gold used for the choice): all — P 0.79, R 0.84; real scam recall
0.78 (7 of 9), real precision 0.47. All 8 false alarms are legit ads with a booking prepayment
(6 daily rents, a car dealer, phone numbers), risk 0.93–1.00 — the bot must handle this hard negative.

| 2026-11-26 | Week 3 (Phase 7) code — `cascade.py` (student levels 🟢 < 0.35 ≤ 🟡 < threshold 0.73 ≤ 🔴; teacher only for 🟡: one attempt (`llm.complete_json(retries=0)`), 20 s wait in a worker thread, 10 requests/day in `data/cache/teacher_budget.json`, own cache `data/cache/live_teacher.jsonl`; never raises), `explain.py` (level, 2–4 reasons, advice per signal at every level, soft prepayment advice, disclaimer), `check.py` CLI and `web.py` (stdlib http.server on 127.0.0.1) — offline by default, `bot/` (aiogram 3 long polling, `/start` `/help` `/stats`, 👍/👎 buttons, pending line then edit with the teacher answer, SQLite `data/bot.db` with masked text and no user data). `docs/DEMO.md`, draft `docs/demo_ads.txt`. CI installs `bot`. 306 tests pass. Bot not yet started with the real token, no real teacher request made | done |

Smoke test 2026-11-26 on 6 new texts (written by Claude, not in data): 3 of 6 right. Misses: honest
daily rent from an ИП with 50 % booking -> 🔴 0.91; iPhone "OLX delivery link, enter card data" -> 🟢
0.06 (`fake_delivery` is in only 2 training ads, weight +0.49; the only strong learned signal is
`prepayment` +3.13); Kazakh "алдын ала 30 % төлем" -> 🟢 0.01 (the kk prepayment regex does not allow
a number inside the phrase). Proposed to the owner: a rule floor for known schemes + the kk regex fix.

Gold is NOT frozen yet (owner, 2026-11-23): 15–20 real scam texts from victims' screenshots will be
added via a new inbox and labelled with `--only-inbox`; then the owner rereads and freezes.

Next: Phase 3 (teacher, `llm.py`) — plan shown to the owner 2026-11-19, "ok" given. Order agreed
with the owner: (1) code + tests with mocks, no real requests; (2) trial run on 10 ads, show 3–4
full teacher answers; (3) **no full labelling run until the owner has collected more data and
labelled gold.** Before any real API run, state the number of requests and wait for confirmation.
