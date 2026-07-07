# LBM Prototype Design

**Date**: 2026-07-07
**Status**: Approved for implementation

## Background

The core idea (see original concept note below) is that the next stage of AI
is about predicting *actions*, not just words — a "Large Behavioural Model"
(LBM) learns from sequences of user/system behavior (navigation, clicks,
outcomes) and predicts what should happen next, rather than what should be
said next.

This is a standalone research effort, not tied to any existing GetSetAI
product. The goal of this first phase is to build a small, concrete
prototype that demonstrates the LLM-vs-LBM distinction on real behavioral
data, before deciding whether/how to scale it further.

> Beyond LLMs: Why the Future of AI Is About Predicting Actions, Not Just
> Words. LLMs predict words — they take text as input and generate the most
> likely next word. Great for knowledge, search, content, chat. LBMs predict
> actions — they learn patterns from behavior: how users navigate, where
> they click, how systems respond, what decisions lead to success or
> failure. Instead of "What should I say next?", LBMs answer "What should I
> do next?"

## Goal of this phase (v1)

Build an end-to-end pipeline — data → features → model → evaluation — that
predicts, from a user's session behavior:
1. **Next action**: what item/event comes next in the session.
2. **Outcome**: whether the session ends in success (purchase) or failure
   (abandonment).

And show that a learned sequence model (the "LBM" prototype) beats a
classical baseline on both, as evidence the framing has legs before
investing further.

## Constraints

- CPU-only training environment (no GPU available yet). Model size and
  dataset subsample must be chosen so a full run finishes in minutes, not
  hours.
- Standalone / domain-agnostic research — not wired into any existing
  product or live data source.

## 1. Data & Task Definition

- **Source**: YOOCHOOSE dataset (RecSys Challenge 2015) —
  `yoochoose-clicks.dat` (session_id, timestamp, item_id, category) and
  `yoochoose-buys.dat` (session_id, timestamp, item_id, price, quantity).
  Chosen over richer alternatives (e.g. REES46) because it's small enough to
  subsample for CPU training and is the standard benchmark used by the
  GRU4Rec/SASRec papers, so results are directly comparable to published
  numbers.
- **Unit of modeling**: one session = one behavior sequence. Each event
  becomes a token: `(item_id, category, action_type ∈ {view, buy},
  log_time_gap_bucket)`.
- **Filtering**: drop sessions with length 1 (nothing to learn from); truncate
  sessions longer than 50 events to the last 50 — matches standard
  preprocessing in the GRU4Rec/SASRec literature and keeps sequences
  CPU-tractable.
- **Labels**:
  - *Next-action*: at each position, the next item_id in the session
    (causal, next-item prediction).
  - *Outcome*: session-level binary label — did this session ID appear in
    `yoochoose-buys.dat` at all?
- **Split**: temporal split (train on earlier days, evaluate on the last N
  days), not a random split — random splits leak future item popularity into
  training on this dataset.
- **Subsampling for CPU**: start with a fixed subsample (~200k–500k
  sessions) rather than the full ~9M. Scale up only once a small run looks
  healthy.

## 2. Model Architecture

**LBM prototype — small causal Transformer:**
- **Embeddings**: separate tables for `item_id` (top-K by frequency + one
  `<unk>` bucket — YOOCHOOSE has ~50k items and the long tail would
  otherwise blow up the vocab), `category`, `action_type`, and
  `time_gap_bucket`. Concatenated/summed into one event vector.
- **Backbone**: causal (masked) self-attention Transformer, 2–4 layers,
  model dim 64–128, 2–4 heads, learned positional embeddings over the short
  (≤50) sequence.
- **Heads**:
  - *Next-action*: linear + softmax over item vocab at every position,
    trained with causal cross-entropy (teacher forcing).
  - *Outcome*: representation at the last position → linear + sigmoid →
    P(session ends in purchase).
- **Loss**: weighted sum of next-action cross-entropy + outcome BCE. Start
  with equal weighting.

**Baseline (built alongside, for comparison):**
- Item-item co-occurrence counts for next-action (recommend the
  most-frequently co-occurring next item).
- XGBoost on hand-engineered session features (length, distinct
  items/categories, view count, avg time-gap) for the outcome head.

## 3. Training, Evaluation & Repo Structure

**Metrics** (on the held-out temporal split, model vs. baseline):
- Next-action: Recall@20, MRR@20 (standard metrics in the GRU4Rec/SASRec
  papers on this dataset).
- Outcome: ROC-AUC, PR-AUC (outcome is imbalanced — most sessions don't end
  in a buy).

**Training loop**: plain PyTorch on CPU, small batch size, a handful of
epochs over the subsample, checkpoint best model by validation Recall@20. No
training framework needed at this scale.

**Repo structure**:
```
LBM/
  data/                # raw + processed YOOCHOOSE subsample (gitignored)
  src/
    data/              # download, filter, tokenize, temporal split
    baseline/          # co-occurrence + XGBoost
    model/             # embeddings, transformer backbone, heads
    train.py
    evaluate.py
  notebooks/           # exploratory analysis
  configs/             # hyperparams (vocab size, dims, epochs)
  README.md
  TASK.md
  CHANGELOG.md
```

**Testing**: unit tests for the data pipeline (tokenization, temporal split
correctness — no leakage) and for one forward/backward pass of the model on
a tiny synthetic batch (catches shape bugs before a full run). Full training
runs are checked by eyeballing metrics, not asserted in CI.

## Definition of done for v1

Baseline and Transformer both train end-to-end on the subsample, and the
Transformer beats the baseline on Recall@20 and ROC-AUC — that's the
evidence the "LBM" framing has legs before scaling data or model size.

## Explicitly out of scope for v1

- GPU training / large-scale data (full ~9M sessions).
- Any tie-in to WisdomChat or other GetSetAI products.
- RL/agentic environments (WebArena, Mind2Web) — noted as a possible future
  testbed once the sequence-modeling core is proven.
- Multi-modal inputs (images, text) — behavior tokens only for v1.
