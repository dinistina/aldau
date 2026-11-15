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

## Disclaimer

The output is a probability, not a verdict. No personal data of sellers is stored and the project
does not scrape marketplaces.
