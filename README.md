# LBM — Large Behavioural Model (research prototype)

Standalone research project exploring "Large Behavioural Models": models
that predict the next *action* from a sequence of behavior, instead of the
next *word*.

## v1 scope

Predict, from an e-commerce session's click/buy history:
1. The next action (next item viewed/bought).
2. The session outcome (does it end in a purchase?).

A small causal Transformer is compared against a classical baseline
(item co-occurrence + XGBoost) on the YOOCHOOSE / RecSys Challenge 2015
dataset. See `docs/superpowers/specs/2026-07-07-lbm-prototype-design.md`
for the full design.

## Setup

```bash
uv sync --extra dev
```

## Run the full pipeline

```bash
uv run python -m lbm.pipeline --config configs/default.yaml
```

First run downloads the dataset via the Kaggle CLI (`kaggle` must be
configured with credentials) into `data/raw/`.

## Run tests

```bash
uv run pytest
```
