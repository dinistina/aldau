# Aldau

[![CI](https://github.com/dinistina/aldau/actions/workflows/ci.yml/badge.svg)](https://github.com/dinistina/aldau/actions/workflows/ci.yml)

*Aldau* (алдау) means "deception" in Kazakh.

A scam-ad detector for Kazakhstan marketplaces (OLX.kz / Kolesa.kz / Krisha.kz style ads in
Russian and Kazakh). It estimates the fraud risk of an ad and explains why it looks suspicious.

The idea is knowledge distillation: a large LLM (the *teacher*) labels ads, a small local model
(the *student*) learns from those labels, and at inference the student asks the teacher only when
it is unsure.

**Status:** work in progress — project skeleton only. See [docs/PLAN.md](docs/PLAN.md) (in Russian)
for the roadmap.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
pytest -q
```

## Data and known limitations

Ads are collected by hand (no scraping) and personal data is masked before storage. Each ad
records its origin: `real` (copied from a marketplace or a victim's screenshot), `case_based`
(a scam scheme rewritten by hand from news, complaints or safety memos) or `synthetic`
(LLM-generated, training only).

- **The test set includes `case_based` scam ads.** Real scam ads are rare and get removed quickly,
  so part of the scam examples in the human-labelled gold set are rewritten cases. Rewritten text
  can differ in style from real ads, which may make detection look easier than it is. Every
  reported metric is therefore given separately for `real` and `case_based` ads, and the
  benchmark includes a style-leak check (how well a simple classifier predicts the origin of an
  ad from its text alone).
- The dataset is small (a few hundred hand-collected ads) and not a random sample of any
  marketplace.

## Disclaimer

The output is a probability, not a verdict. No personal data of sellers is stored and the project
does not scrape marketplaces.
