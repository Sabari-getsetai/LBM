# LBM Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end CPU-trainable pipeline that predicts next-action and session outcome (purchase vs. abandon) from e-commerce clickstream sessions, and show a small causal Transformer ("LBM") beats a classical baseline.

**Architecture:** A data pipeline (download → parse → session-build → tokenize → temporal split) feeds two consumers: a classical baseline (item co-occurrence + XGBoost on session features) and a small causal-Transformer sequence model with a next-action head and an outcome head. Both are evaluated with the same metrics so their results are directly comparable.

**Tech Stack:** Python 3.12, uv (dependency management), PyTorch (CPU), pandas, scikit-learn, XGBoost, pytest, Kaggle CLI (data download).

## Global Constraints

- CPU-only: no GPU available. Model size and dataset subsample must keep a full run to minutes, not hours.
- Standalone research: no tie-in to WisdomChat or any other GetSetAI product.
- Dataset: YOOCHOOSE / RecSys Challenge 2015, via Kaggle dataset `chadgostopp/recsys-challenge-2015` (Kaggle CLI is already configured with credentials in this environment).
- Sessions: drop length < 2, truncate length > 50 (`min_session_len=2`, `max_session_len=50` in config).
- Split: temporal (train on earlier days, test on the last N days) — never random split.
- Metrics: Recall@k / MRR@k for next-action, ROC-AUC / PR-AUC for outcome, `k=20` default.
- Definition of done: baseline and LBM both train end-to-end on a subsample, and the LBM beats the baseline on Recall@k and ROC-AUC.
- Every task's commit step also updates `TASK.md` (mark the task done) and `CHANGELOG.md` (one line under `[Unreleased]`).

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `configs/default.yaml`
- Create: `src/lbm/__init__.py`
- Create: `src/lbm/data/__init__.py`
- Create: `src/lbm/model/__init__.py`
- Create: `src/lbm/baseline/__init__.py`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `README.md`
- Create: `TASK.md`
- Create: `CHANGELOG.md`

**Interfaces:**
- Produces: an installable `lbm` package under `src/lbm`, a `configs/default.yaml` with keys `data`, `model`, `train`, `eval` (consumed by `lbm.pipeline.run_pipeline` in Task 16), and a working `uv` environment with `pytest` available via `uv run pytest`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "lbm"
version = "0.1.0"
description = "Large Behavioural Model research prototype"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "pandas>=2.2",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "xgboost>=2.0",
    "pyyaml>=6.0",
    "kaggle>=1.6",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/lbm"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
*.pt
uv.lock
```

- [ ] **Step 3: Create package directories and empty `__init__.py` files**

```bash
mkdir -p src/lbm/data src/lbm/model src/lbm/baseline tests configs data/raw data/processed
touch src/lbm/__init__.py src/lbm/data/__init__.py src/lbm/model/__init__.py src/lbm/baseline/__init__.py
touch data/raw/.gitkeep data/processed/.gitkeep
```

- [ ] **Step 4: Write `configs/default.yaml`**

```yaml
data:
  raw_dir: data/raw
  processed_dir: data/processed
  n_sessions_subsample: 300000
  max_session_len: 50
  min_session_len: 2
  top_k_items: 20000
  test_days: 1

model:
  embed_dim: 64
  n_layers: 2
  n_heads: 2
  ffn_dim: 128
  dropout: 0.1

train:
  batch_size: 64
  epochs: 3
  lr: 0.001
  outcome_loss_weight: 1.0
  seed: 42

eval:
  k: 20
```

- [ ] **Step 5: Sync the environment and verify imports**

Run: `uv sync --extra dev`
Expected: creates `.venv/` and `uv.lock`, installs torch/pandas/numpy/scikit-learn/xgboost/pyyaml/kaggle/pytest without errors.

Run: `uv run python -c "import torch, pandas, numpy, sklearn, xgboost, yaml, pytest; print('ok')"`
Expected: prints `ok`

- [ ] **Step 6: Write `README.md`**

```markdown
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
```

- [ ] **Step 7: Write `TASK.md`**

```markdown
# LBM Tasks

## Completed
- [x] Task 1: Project scaffolding

## Not Started
- [ ] Task 2: Vocab module
- [ ] Task 3: Parse raw YOOCHOOSE files
- [ ] Task 4: Session events, filtering, outcomes
- [ ] Task 5: Tokenize sessions
- [ ] Task 6: Temporal train/test split
- [ ] Task 7: PyTorch Dataset + collate
- [ ] Task 8: Metrics (Recall@k, MRR@k, ROC-AUC, PR-AUC)
- [ ] Task 9: Co-occurrence baseline
- [ ] Task 10: Outcome feature engineering
- [ ] Task 11: Train baseline (co-occurrence + XGBoost)
- [ ] Task 12: LBM model (causal Transformer)
- [ ] Task 13: Training loop
- [ ] Task 14: Evaluation + comparison report
- [ ] Task 15: Dataset download (Kaggle)
- [ ] Task 16: End-to-end pipeline + real run
```

- [ ] **Step 8: Write `CHANGELOG.md`**

```markdown
# Changelog

## [Unreleased]
### Added
- Project scaffolding: `pyproject.toml`, package layout under `src/lbm`,
  default config, README/TASK/CHANGELOG.
```

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .gitignore configs/ src/ data/raw/.gitkeep data/processed/.gitkeep README.md TASK.md CHANGELOG.md
git commit -m "chore: scaffold LBM project"
```

---

### Task 2: Vocab module

**Files:**
- Create: `src/lbm/data/vocab.py`
- Test: `tests/test_vocab.py`

**Interfaces:**
- Produces: `Vocab` class with `from_counts(counter, top_k=None)` classmethod, `.encode(token) -> int`, `__len__`; `PAD_TOKEN`/`UNK_TOKEN` constants; `time_gap_bucket(seconds: float) -> int` returning an id in `[1, 5]` (id `0` reserved for padding everywhere). Consumed by Tasks 4, 5, 16.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vocab.py
from collections import Counter

from lbm.data.vocab import Vocab, time_gap_bucket


def test_vocab_reserves_pad_and_unk():
    vocab = Vocab.from_counts(Counter({"a": 3, "b": 1}))
    assert vocab.encode("<pad>") == 0
    assert vocab.token2id["<unk>"] == 1


def test_vocab_encodes_known_tokens_by_frequency():
    vocab = Vocab.from_counts(Counter({"a": 3, "b": 1}))
    assert vocab.encode("a") == 2
    assert vocab.encode("b") == 3


def test_vocab_encodes_unknown_token_as_unk():
    vocab = Vocab.from_counts(Counter({"a": 3}))
    assert vocab.encode("never_seen") == vocab.encode("<unk>")


def test_vocab_respects_top_k():
    vocab = Vocab.from_counts(Counter({"a": 3, "b": 2, "c": 1}), top_k=1)
    assert len(vocab) == 3  # pad + unk + top-1 token
    assert vocab.encode("c") == vocab.encode("<unk>")


def test_time_gap_bucket_boundaries():
    assert time_gap_bucket(5) == 1
    assert time_gap_bucket(30) == 2
    assert time_gap_bucket(120) == 3
    assert time_gap_bucket(1800) == 4
    assert time_gap_bucket(7200) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vocab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.data.vocab'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/data/vocab.py
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocab:
    """Maps hashable tokens to contiguous integer ids. Reserves id 0 for
    padding and id 1 for unknown tokens, so every embedding table built
    from a Vocab can use id 0 as its padding_idx."""

    def __init__(self, tokens: list):
        self.token2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token in tokens:
            self.token2id[token] = len(self.token2id)
        self.id2token = {i: t for t, i in self.token2id.items()}

    def __len__(self) -> int:
        return len(self.token2id)

    def encode(self, token) -> int:
        return self.token2id.get(token, self.token2id[UNK_TOKEN])

    @classmethod
    def from_counts(cls, counter: Counter, top_k: int | None = None) -> "Vocab":
        most_common = counter.most_common(top_k)
        tokens = [token for token, _ in most_common]
        return cls(tokens)


def time_gap_bucket(seconds: float) -> int:
    """Bucket a time gap (in seconds) since the previous event in a
    session into a discrete id in [1, 5]. Id 0 is reserved for padding,
    matching the convention used by Vocab-encoded fields."""
    if seconds < 10:
        return 1
    if seconds < 60:
        return 2
    if seconds < 300:
        return 3
    if seconds < 3600:
        return 4
    return 5
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vocab.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

In `TASK.md`, move `Task 2` from "Not Started" to "Completed".
In `CHANGELOG.md`, add under `### Added`: `- Vocab module with padding/unknown-token handling and time-gap bucketing.`

```bash
git add src/lbm/data/vocab.py tests/test_vocab.py TASK.md CHANGELOG.md
git commit -m "feat: add vocab and time-gap bucketing"
```

---

### Task 3: Parse raw YOOCHOOSE files

**Files:**
- Create: `src/lbm/data/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Produces: `load_clicks(path) -> pd.DataFrame` with columns `session_id, timestamp, item_id, category`; `load_buys(path) -> pd.DataFrame` with columns `session_id, timestamp, item_id, price, quantity`. Consumed by Task 4 and Task 16.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_parse.py
import pandas as pd

from lbm.data.parse import load_buys, load_clicks


def test_load_clicks_parses_expected_columns(tmp_path):
    path = tmp_path / "clicks.dat"
    path.write_text(
        "1,2014-04-07T10:51:09.277Z,214536502,0\n"
        "1,2014-04-07T10:54:09.868Z,214536500,0\n"
    )
    df = load_clicks(path)
    assert list(df.columns) == ["session_id", "timestamp", "item_id", "category"]
    assert df.loc[0, "session_id"] == 1
    assert df.loc[0, "item_id"] == 214536502
    assert isinstance(df.loc[0, "timestamp"], pd.Timestamp)


def test_load_buys_parses_expected_columns(tmp_path):
    path = tmp_path / "buys.dat"
    path.write_text("1,2014-04-07T10:54:09.868Z,214536500,12345,1\n")
    df = load_buys(path)
    assert list(df.columns) == ["session_id", "timestamp", "item_id", "price", "quantity"]
    assert df.loc[0, "price"] == 12345.0
    assert df.loc[0, "quantity"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_parse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.data.parse'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/data/parse.py
from pathlib import Path

import pandas as pd


def load_clicks(path: Path) -> pd.DataFrame:
    """Parses yoochoose-clicks.dat into session_id, timestamp, item_id,
    category columns."""
    return pd.read_csv(
        path,
        header=None,
        names=["session_id", "timestamp", "item_id", "category"],
        dtype={"session_id": "int64", "item_id": "int64", "category": "str"},
        parse_dates=["timestamp"],
    )


def load_buys(path: Path) -> pd.DataFrame:
    """Parses yoochoose-buys.dat into session_id, timestamp, item_id,
    price, quantity columns."""
    return pd.read_csv(
        path,
        header=None,
        names=["session_id", "timestamp", "item_id", "price", "quantity"],
        dtype={
            "session_id": "int64",
            "item_id": "int64",
            "price": "float64",
            "quantity": "int64",
        },
        parse_dates=["timestamp"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_parse.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/data/parse.py tests/test_parse.py TASK.md CHANGELOG.md
git commit -m "feat: parse raw YOOCHOOSE clicks/buys files"
```

---

### Task 4: Session events, filtering, outcomes

**Files:**
- Create: `src/lbm/data/sessions.py`
- Test: `tests/test_sessions.py`

**Interfaces:**
- Consumes: DataFrames shaped like `load_clicks`/`load_buys` output (Task 3).
- Produces: `build_session_events(clicks, buys) -> pd.DataFrame` with columns `session_id, timestamp, item_id, category, action_type` (`action_type` in `{"view", "buy"}`); `session_outcomes(buys) -> set[int]`; `filter_sessions(events, min_len, max_len) -> pd.DataFrame`. Consumed by Task 5 and Task 16.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sessions.py
import pandas as pd

from lbm.data.sessions import build_session_events, filter_sessions, session_outcomes


def _clicks():
    return pd.DataFrame(
        {
            "session_id": [1, 1, 1, 2],
            "timestamp": pd.to_datetime(
                [
                    "2014-04-07T10:00:00Z",
                    "2014-04-07T10:01:00Z",
                    "2014-04-07T10:02:00Z",
                    "2014-04-07T11:00:00Z",
                ]
            ),
            "item_id": [100, 101, 102, 200],
            "category": ["0", "0", "1", "0"],
        }
    )


def _buys():
    return pd.DataFrame(
        {
            "session_id": [1],
            "timestamp": pd.to_datetime(["2014-04-07T10:03:00Z"]),
            "item_id": [102],
            "price": [999.0],
            "quantity": [1],
        }
    )


def test_build_session_events_merges_and_sorts():
    events = build_session_events(_clicks(), _buys())
    session_1 = events[events["session_id"] == 1]
    assert list(session_1["action_type"]) == ["view", "view", "view", "buy"]
    assert session_1["timestamp"].is_monotonic_increasing


def test_session_outcomes_returns_sessions_with_buys():
    assert session_outcomes(_buys()) == {1}


def test_filter_sessions_drops_short_and_truncates_long():
    events = build_session_events(_clicks(), _buys())
    filtered = filter_sessions(events, min_len=2, max_len=2)
    assert 2 not in filtered["session_id"].values
    assert len(filtered[filtered["session_id"] == 1]) == 2
    assert list(filtered[filtered["session_id"] == 1]["item_id"]) == [102, 102]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sessions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.data.sessions'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/data/sessions.py
import pandas as pd


def build_session_events(clicks: pd.DataFrame, buys: pd.DataFrame) -> pd.DataFrame:
    """Combines click and buy events into one chronological event stream
    per session. Buy events don't carry a category in the raw YOOCHOOSE
    data, so they're tagged with the sentinel category "NA"."""
    click_events = clicks[["session_id", "timestamp", "item_id", "category"]].copy()
    click_events["action_type"] = "view"

    buy_events = buys[["session_id", "timestamp", "item_id"]].copy()
    buy_events["category"] = "NA"
    buy_events["action_type"] = "buy"

    events = pd.concat([click_events, buy_events], ignore_index=True)
    events = events.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    return events


def session_outcomes(buys: pd.DataFrame) -> set:
    """Session ids that had at least one buy event."""
    return set(buys["session_id"].unique())


def filter_sessions(events: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    """Drops sessions shorter than min_len events; truncates sessions
    longer than max_len to their last max_len events."""
    counts = events.groupby("session_id").size()
    keep_ids = counts[counts >= min_len].index
    events = events[events["session_id"].isin(keep_ids)]

    def _truncate(group: pd.DataFrame) -> pd.DataFrame:
        return group.tail(max_len)

    events = (
        events.groupby("session_id", group_keys=False)
        .apply(_truncate)
        .reset_index(drop=True)
    )
    return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sessions.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/data/sessions.py tests/test_sessions.py TASK.md CHANGELOG.md
git commit -m "feat: build session event streams, filtering, outcome labels"
```

---

### Task 5: Tokenize sessions

**Files:**
- Create: `src/lbm/data/tokenize.py`
- Test: `tests/test_tokenize.py`

**Interfaces:**
- Consumes: `Vocab` (Task 2), events DataFrame shaped like `build_session_events` output (Task 4).
- Produces: `SessionRecord` dataclass with fields `session_id: int, item_ids: list[int], category_ids: list[int], action_ids: list[int], time_gap_ids: list[int], outcome: int`; `tokenize_sessions(events, outcome_ids, item_vocab, category_vocab, action_vocab) -> list[SessionRecord]`. Consumed by Tasks 6, 7, 9, 10, 11, 16.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tokenize.py
from collections import Counter

import pandas as pd

from lbm.data.tokenize import tokenize_sessions
from lbm.data.vocab import Vocab


def _events():
    return pd.DataFrame(
        {
            "session_id": [1, 1, 1],
            "timestamp": pd.to_datetime(
                ["2014-04-07T10:00:00Z", "2014-04-07T10:00:05Z", "2014-04-07T10:05:05Z"]
            ),
            "item_id": [100, 101, 100],
            "category": ["0", "0", "0"],
            "action_type": ["view", "view", "buy"],
        }
    )


def _vocabs(events):
    return (
        Vocab.from_counts(Counter(events["item_id"])),
        Vocab.from_counts(Counter(events["category"])),
        Vocab.from_counts(Counter(events["action_type"])),
    )


def test_tokenize_sessions_produces_one_record_per_session():
    events = _events()
    item_vocab, category_vocab, action_vocab = _vocabs(events)

    records = tokenize_sessions(
        events, outcome_ids={1}, item_vocab=item_vocab,
        category_vocab=category_vocab, action_vocab=action_vocab,
    )

    assert len(records) == 1
    record = records[0]
    assert record.session_id == 1
    assert record.outcome == 1
    assert len(record.item_ids) == 3
    assert len(record.time_gap_ids) == 3
    assert record.time_gap_ids[0] == 1  # first gap is 0 seconds -> bucket 1


def test_tokenize_sessions_marks_outcome_zero_when_no_buy():
    events = _events()
    item_vocab, category_vocab, action_vocab = _vocabs(events)

    records = tokenize_sessions(
        events, outcome_ids=set(), item_vocab=item_vocab,
        category_vocab=category_vocab, action_vocab=action_vocab,
    )
    assert records[0].outcome == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tokenize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.data.tokenize'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/data/tokenize.py
from dataclasses import dataclass

import pandas as pd

from lbm.data.vocab import Vocab, time_gap_bucket


@dataclass
class SessionRecord:
    session_id: int
    item_ids: list
    category_ids: list
    action_ids: list
    time_gap_ids: list
    outcome: int


def tokenize_sessions(
    events: pd.DataFrame,
    outcome_ids: set,
    item_vocab: Vocab,
    category_vocab: Vocab,
    action_vocab: Vocab,
) -> list:
    """Converts the chronological event stream into one SessionRecord
    per session, with every field encoded as an integer id ready for
    embedding lookups."""
    records = []
    for session_id, group in events.groupby("session_id"):
        group = group.sort_values("timestamp")
        timestamps = group["timestamp"].astype("int64").to_numpy() // 10**9
        prev_ts = timestamps[0]
        time_gap_ids = []
        for ts in timestamps:
            time_gap_ids.append(time_gap_bucket(float(ts - prev_ts)))
            prev_ts = ts

        records.append(
            SessionRecord(
                session_id=int(session_id),
                item_ids=[item_vocab.encode(i) for i in group["item_id"]],
                category_ids=[category_vocab.encode(c) for c in group["category"]],
                action_ids=[action_vocab.encode(a) for a in group["action_type"]],
                time_gap_ids=time_gap_ids,
                outcome=int(session_id in outcome_ids),
            )
        )
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tokenize.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/data/tokenize.py tests/test_tokenize.py TASK.md CHANGELOG.md
git commit -m "feat: tokenize sessions into integer-id SessionRecords"
```

---

### Task 6: Temporal train/test split

**Files:**
- Create: `src/lbm/data/split.py`
- Test: `tests/test_split.py`

**Interfaces:**
- Consumes: `list[SessionRecord]` (Task 5), events DataFrame (Task 4, for per-session start timestamps).
- Produces: `temporal_split(records, events, test_days) -> tuple[list[SessionRecord], list[SessionRecord]]` (train, test). Consumed by Task 16.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split.py
import pandas as pd

from lbm.data.split import temporal_split
from lbm.data.tokenize import SessionRecord


def _record(session_id):
    return SessionRecord(
        session_id=session_id, item_ids=[2, 3], category_ids=[2, 2],
        action_ids=[2, 2], time_gap_ids=[1, 1], outcome=0,
    )


def test_temporal_split_by_session_start_time():
    events = pd.DataFrame(
        {
            "session_id": [1, 2],
            "timestamp": pd.to_datetime(["2014-04-01T00:00:00Z", "2014-04-09T00:00:00Z"]),
        }
    )
    records = [_record(1), _record(2)]
    train, test = temporal_split(records, events, test_days=1)

    assert [r.session_id for r in train] == [1]
    assert [r.session_id for r in test] == [2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.data.split'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/data/split.py
import pandas as pd


def temporal_split(records: list, events: pd.DataFrame, test_days: int) -> tuple:
    """Splits sessions by start time: sessions starting within the last
    test_days days of the dataset go to test, everything earlier goes
    to train. Avoids a random split, which would leak future item
    popularity into training."""
    start_times = events.groupby("session_id")["timestamp"].min()
    cutoff = start_times.max() - pd.Timedelta(days=test_days)

    train, test = [], []
    for record in records:
        if start_times.loc[record.session_id] >= cutoff:
            test.append(record)
        else:
            train.append(record)
    return train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/data/split.py tests/test_split.py TASK.md CHANGELOG.md
git commit -m "feat: add temporal train/test split"
```

---

### Task 7: PyTorch Dataset + collate

**Files:**
- Create: `src/lbm/data/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `SessionRecord` (Task 5).
- Produces: `PAD_ID = 0`; `SessionDataset(records)` (`torch.utils.data.Dataset`) yielding dicts with keys `item_ids, category_ids, action_ids, time_gap_ids, target_items, outcome`; `collate_sessions(batch) -> dict` adding `lengths` and `padding_mask` (`bool`, `True` = padded). Consumed by Tasks 13, 14, 16.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
import torch

from lbm.data.dataset import SessionDataset, collate_sessions
from lbm.data.tokenize import SessionRecord


def _record(session_id, length):
    return SessionRecord(
        session_id=session_id,
        item_ids=list(range(2, 2 + length)),
        category_ids=[2] * length,
        action_ids=[2] * length,
        time_gap_ids=[1] * length,
        outcome=1,
    )


def test_dataset_shifts_targets_by_one_position():
    dataset = SessionDataset([_record(1, length=4)])
    example = dataset[0]
    assert torch.equal(example["item_ids"], torch.tensor([2, 3, 4]))
    assert torch.equal(example["target_items"], torch.tensor([3, 4, 5]))


def test_collate_pads_variable_length_batch():
    dataset = SessionDataset([_record(1, length=4), _record(2, length=2)])
    batch = collate_sessions([dataset[0], dataset[1]])

    assert batch["item_ids"].shape == (2, 3)
    assert batch["lengths"].tolist() == [3, 1]
    assert batch["item_ids"][1].tolist() == [2, 0, 0]
    assert batch["padding_mask"][1].tolist() == [False, True, True]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.data.dataset'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/data/dataset.py
import torch
from torch.utils.data import Dataset

PAD_ID = 0


class SessionDataset(Dataset):
    """Wraps SessionRecords for next-action + outcome prediction. Each
    example uses all but the last event as input; position i's
    next-action target is the item at position i+1 in the original
    session."""

    def __init__(self, records: list):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "item_ids": torch.tensor(r.item_ids[:-1], dtype=torch.long),
            "category_ids": torch.tensor(r.category_ids[:-1], dtype=torch.long),
            "action_ids": torch.tensor(r.action_ids[:-1], dtype=torch.long),
            "time_gap_ids": torch.tensor(r.time_gap_ids[:-1], dtype=torch.long),
            "target_items": torch.tensor(r.item_ids[1:], dtype=torch.long),
            "outcome": torch.tensor(float(r.outcome), dtype=torch.float),
        }


def collate_sessions(batch: list) -> dict:
    """Pads a batch of variable-length session examples with PAD_ID (0)
    and builds a boolean padding mask (True where padded) plus each
    example's true length."""
    lengths = torch.tensor([len(b["item_ids"]) for b in batch], dtype=torch.long)
    max_len = int(lengths.max())

    def _pad(key: str) -> torch.Tensor:
        out = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
        for i, b in enumerate(batch):
            seq = b[key]
            out[i, : len(seq)] = seq
        return out

    padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)

    return {
        "item_ids": _pad("item_ids"),
        "category_ids": _pad("category_ids"),
        "action_ids": _pad("action_ids"),
        "time_gap_ids": _pad("time_gap_ids"),
        "target_items": _pad("target_items"),
        "outcome": torch.stack([b["outcome"] for b in batch]),
        "lengths": lengths,
        "padding_mask": padding_mask,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/data/dataset.py tests/test_dataset.py TASK.md CHANGELOG.md
git commit -m "feat: add SessionDataset and padding collate function"
```

---

### Task 8: Metrics

**Files:**
- Create: `src/lbm/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `recall_at_k(logits, targets, k) -> float`, `mrr_at_k(logits, targets, k) -> float` (both ignore positions where `target == PAD_ID`), `roc_auc(probs, labels) -> float`, `pr_auc(probs, labels) -> float`. Consumed by Tasks 11, 13, 14.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import torch

from lbm.metrics import mrr_at_k, pr_auc, recall_at_k, roc_auc


def test_recall_at_k_counts_hit_within_top_k():
    logits = torch.tensor([[[0.1, 0.9, 0.2, 0.3]]])
    targets = torch.tensor([[1]])
    assert recall_at_k(logits, targets, k=1) == 1.0


def test_recall_at_k_ignores_padding():
    logits = torch.zeros(1, 2, 4)
    targets = torch.tensor([[1, 0]])
    assert recall_at_k(logits, targets, k=4) == 1.0


def test_mrr_at_k_ranks_correctly():
    logits = torch.tensor([[[0.9, 0.1, 0.2, 0.3]]])
    targets = torch.tensor([[2]])
    assert abs(mrr_at_k(logits, targets, k=4) - 1 / 3) < 1e-6


def test_roc_auc_and_pr_auc_perfect_separation():
    probs = [0.1, 0.9]
    labels = [0, 1]
    assert roc_auc(probs, labels) == 1.0
    assert pr_auc(probs, labels) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.metrics'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/metrics.py
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PAD_ID = 0


def recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Fraction of non-padded positions where the true next item is
    among the model's top-k predictions."""
    mask = targets != PAD_ID
    if mask.sum() == 0:
        return 0.0
    topk = logits.topk(k, dim=-1).indices
    hits = (topk == targets.unsqueeze(-1)).any(dim=-1)
    return float(hits[mask].float().mean())


def mrr_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Mean reciprocal rank of the true next item within the top-k
    predictions (0 if it doesn't appear in the top-k)."""
    mask = targets != PAD_ID
    if mask.sum() == 0:
        return 0.0
    topk = logits.topk(k, dim=-1).indices
    matches = topk == targets.unsqueeze(-1)
    ranks = matches.float().argmax(dim=-1) + 1
    reciprocal = torch.where(
        matches.any(dim=-1), 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float)
    )
    return float(reciprocal[mask].mean())


def roc_auc(probs, labels) -> float:
    return float(roc_auc_score(labels, probs))


def pr_auc(probs, labels) -> float:
    return float(average_precision_score(labels, probs))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/metrics.py tests/test_metrics.py TASK.md CHANGELOG.md
git commit -m "feat: add recall@k, mrr@k, roc-auc, pr-auc metrics"
```

---

### Task 9: Co-occurrence baseline

**Files:**
- Create: `src/lbm/baseline/cooccurrence.py`
- Test: `tests/test_cooccurrence.py`

**Interfaces:**
- Consumes: `list[SessionRecord]` (Task 5).
- Produces: `CooccurrenceBaseline` with `.fit(records)`, `.predict_topk(item_id, k) -> list[int]`, `.evaluate(records, k) -> dict{"recall@k": float, "mrr@k": float}`. Consumed by Task 11.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cooccurrence.py
from lbm.baseline.cooccurrence import CooccurrenceBaseline
from lbm.data.tokenize import SessionRecord


def _record(item_ids):
    return SessionRecord(
        session_id=1, item_ids=item_ids, category_ids=[2] * len(item_ids),
        action_ids=[2] * len(item_ids), time_gap_ids=[1] * len(item_ids), outcome=0,
    )


def test_fit_learns_most_common_transition():
    train = [_record([2, 3]), _record([2, 3]), _record([2, 4])]
    model = CooccurrenceBaseline()
    model.fit(train)
    assert model.predict_topk(2, k=1) == [3]


def test_evaluate_computes_recall_and_mrr():
    train = [_record([2, 3])]
    model = CooccurrenceBaseline()
    model.fit(train)
    metrics = model.evaluate([_record([2, 3])], k=1)
    assert metrics["recall@k"] == 1.0
    assert metrics["mrr@k"] == 1.0


def test_predict_falls_back_to_global_popularity_for_unseen_item():
    train = [_record([2, 3]), _record([2, 3]), _record([5, 3])]
    model = CooccurrenceBaseline()
    model.fit(train)
    assert model.predict_topk(999, k=1) == [3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cooccurrence.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.baseline.cooccurrence'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/baseline/cooccurrence.py
from collections import Counter, defaultdict

PAD_ID = 0


class CooccurrenceBaseline:
    """Recommends the next item as the most frequent item observed to
    follow the current item in training sessions. Falls back to global
    most-popular items for items never seen in training."""

    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.global_popularity = Counter()

    def fit(self, records: list) -> None:
        for r in records:
            items = [i for i in r.item_ids if i != PAD_ID]
            for i in range(len(items) - 1):
                self.transitions[items[i]][items[i + 1]] += 1
            self.global_popularity.update(items)

    def predict_topk(self, item_id: int, k: int) -> list:
        if item_id in self.transitions:
            return [i for i, _ in self.transitions[item_id].most_common(k)]
        return [i for i, _ in self.global_popularity.most_common(k)]

    def evaluate(self, records: list, k: int) -> dict:
        hits = 0
        reciprocal_sum = 0.0
        total = 0
        for r in records:
            items = [i for i in r.item_ids if i != PAD_ID]
            for i in range(len(items) - 1):
                total += 1
                preds = self.predict_topk(items[i], k)
                if items[i + 1] in preds:
                    hits += 1
                    reciprocal_sum += 1.0 / (preds.index(items[i + 1]) + 1)
        if total == 0:
            return {"recall@k": 0.0, "mrr@k": 0.0}
        return {"recall@k": hits / total, "mrr@k": reciprocal_sum / total}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cooccurrence.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/baseline/cooccurrence.py tests/test_cooccurrence.py TASK.md CHANGELOG.md
git commit -m "feat: add item co-occurrence next-action baseline"
```

---

### Task 10: Outcome feature engineering

**Files:**
- Create: `src/lbm/baseline/outcome_features.py`
- Test: `tests/test_outcome_features.py`

**Interfaces:**
- Consumes: `list[SessionRecord]` (Task 5).
- Produces: `extract_session_features(record, buy_action_id) -> list[float]` (5 features: length, distinct_items, distinct_categories, buy_action_count, avg_time_gap — computed over `record`'s input events, i.e. excluding the last event, matching what the LBM sees); `build_feature_matrix(records, buy_action_id) -> tuple[np.ndarray, np.ndarray]` (X, y). Consumed by Task 11.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_outcome_features.py
from lbm.baseline.outcome_features import build_feature_matrix, extract_session_features
from lbm.data.tokenize import SessionRecord


def _record(outcome, buy_count):
    action_ids = [2] * (3 - buy_count) + [3] * buy_count
    return SessionRecord(
        session_id=1, item_ids=[10, 11, 12], category_ids=[2, 2, 2],
        action_ids=action_ids, time_gap_ids=[1, 2, 3], outcome=outcome,
    )


def test_extract_session_features_shape_and_length():
    record = _record(outcome=1, buy_count=1)
    features = extract_session_features(record, buy_action_id=3)
    assert len(features) == 5
    assert features[0] == 2  # excludes last event -> input length 2


def test_extract_session_features_counts_buy_actions_in_input_only():
    # buy_count=1 puts the single buy action last, which is excluded
    # from the input (record.*[:-1]) -> buy_action_count should be 0
    record = _record(outcome=1, buy_count=1)
    features = extract_session_features(record, buy_action_id=3)
    assert features[3] == 0


def test_build_feature_matrix_returns_X_and_y():
    records = [_record(outcome=1, buy_count=0), _record(outcome=0, buy_count=0)]
    X, y = build_feature_matrix(records, buy_action_id=3)
    assert X.shape == (2, 5)
    assert y.tolist() == [1, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_outcome_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.baseline.outcome_features'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/baseline/outcome_features.py
import numpy as np

FEATURE_NAMES = [
    "length",
    "distinct_items",
    "distinct_categories",
    "buy_action_count",
    "avg_time_gap_bucket",
]


def extract_session_features(record, buy_action_id: int) -> list:
    """Hand-engineered features describing a session's behavior so far
    (all events except the last, matching what the LBM sees when it
    predicts the outcome)."""
    items = record.item_ids[:-1]
    categories = record.category_ids[:-1]
    actions = record.action_ids[:-1]
    time_gaps = record.time_gap_ids[:-1]

    length = len(items)
    distinct_items = len(set(items))
    distinct_categories = len(set(categories))
    buy_action_count = sum(1 for a in actions if a == buy_action_id)
    avg_time_gap = sum(time_gaps) / length if length else 0.0

    return [length, distinct_items, distinct_categories, buy_action_count, avg_time_gap]


def build_feature_matrix(records: list, buy_action_id: int) -> tuple:
    X = np.array([extract_session_features(r, buy_action_id) for r in records], dtype=float)
    y = np.array([r.outcome for r in records], dtype=int)
    return X, y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_outcome_features.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/baseline/outcome_features.py tests/test_outcome_features.py TASK.md CHANGELOG.md
git commit -m "feat: add session outcome feature engineering"
```

---

### Task 11: Train baseline (co-occurrence + XGBoost)

**Files:**
- Create: `src/lbm/baseline/train_baseline.py`
- Test: `tests/test_train_baseline.py`

**Interfaces:**
- Consumes: `CooccurrenceBaseline` (Task 9), `build_feature_matrix` (Task 10), `roc_auc`/`pr_auc` (Task 8).
- Produces: `run_baseline(train_records, test_records, buy_action_id, k) -> dict` shaped `{"next_action": {"recall@k": float, "mrr@k": float}, "outcome": {"roc_auc": float, "pr_auc": float}}`. Consumed by Task 16; this exact shape is matched by `evaluate_lbm` in Task 14 so results compare directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_baseline.py
from lbm.baseline.train_baseline import run_baseline
from lbm.data.tokenize import SessionRecord


def _record(session_id, outcome):
    return SessionRecord(
        session_id=session_id, item_ids=[2, 3, 4], category_ids=[2, 2, 2],
        action_ids=[2, 2, 3 if outcome else 2], time_gap_ids=[1, 2, 3], outcome=outcome,
    )


def test_run_baseline_returns_expected_metric_shape():
    train = [_record(i, outcome=i % 2) for i in range(10)]
    test = [_record(i + 100, outcome=i % 2) for i in range(4)]

    metrics = run_baseline(train, test, buy_action_id=3, k=5)

    assert set(metrics.keys()) == {"next_action", "outcome"}
    assert set(metrics["next_action"].keys()) == {"recall@k", "mrr@k"}
    assert set(metrics["outcome"].keys()) == {"roc_auc", "pr_auc"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.baseline.train_baseline'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/baseline/train_baseline.py
from xgboost import XGBClassifier

from lbm.baseline.cooccurrence import CooccurrenceBaseline
from lbm.baseline.outcome_features import build_feature_matrix
from lbm.metrics import pr_auc, roc_auc


def run_baseline(train_records: list, test_records: list, buy_action_id: int, k: int) -> dict:
    """Fits the co-occurrence next-action baseline and the XGBoost
    outcome baseline on train_records, evaluates both on test_records,
    and returns metrics in the same shape produced by
    evaluate.evaluate_lbm so results can be compared directly."""
    cooc = CooccurrenceBaseline()
    cooc.fit(train_records)
    next_action_metrics = cooc.evaluate(test_records, k)

    X_train, y_train = build_feature_matrix(train_records, buy_action_id)
    X_test, y_test = build_feature_matrix(test_records, buy_action_id)

    clf = XGBClassifier(n_estimators=100, max_depth=4, eval_metric="logloss", random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    return {
        "next_action": {
            "recall@k": next_action_metrics["recall@k"],
            "mrr@k": next_action_metrics["mrr@k"],
        },
        "outcome": {"roc_auc": roc_auc(probs, y_test), "pr_auc": pr_auc(probs, y_test)},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train_baseline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/baseline/train_baseline.py tests/test_train_baseline.py TASK.md CHANGELOG.md
git commit -m "feat: wire up combined baseline (co-occurrence + XGBoost)"
```

---

### Task 12: LBM model (causal Transformer)

**Files:**
- Create: `src/lbm/model/lbm.py`
- Test: `tests/test_model_lbm.py`

**Interfaces:**
- Produces: `LBM(nn.Module)` with constructor `LBM(n_items, n_categories, n_actions, n_time_gaps, max_len, embed_dim=64, n_layers=2, n_heads=2, ffn_dim=128, dropout=0.1)` and `forward(item_ids, category_ids, action_ids, time_gap_ids, lengths, padding_mask) -> tuple[Tensor[B,T,n_items], Tensor[B]]` (next-action logits, outcome logits). Consumed by Tasks 13, 14, 16.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model_lbm.py
import torch

from lbm.model.lbm import LBM


def _model(max_len=10):
    return LBM(
        n_items=20, n_categories=5, n_actions=3, n_time_gaps=6, max_len=max_len,
        embed_dim=16, n_layers=1, n_heads=2, ffn_dim=32, dropout=0.0,
    )


def test_forward_produces_expected_shapes():
    model = _model()
    batch_size, seq_len = 4, 6
    item_ids = torch.randint(1, 20, (batch_size, seq_len))
    category_ids = torch.randint(1, 5, (batch_size, seq_len))
    action_ids = torch.randint(1, 3, (batch_size, seq_len))
    time_gap_ids = torch.randint(1, 6, (batch_size, seq_len))
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    next_action_logits, outcome_logits = model(
        item_ids, category_ids, action_ids, time_gap_ids, lengths, padding_mask
    )

    assert next_action_logits.shape == (batch_size, seq_len, 20)
    assert outcome_logits.shape == (batch_size,)


def test_backward_pass_computes_gradients():
    model = _model()
    item_ids = torch.randint(1, 20, (2, 5))
    category_ids = torch.randint(1, 5, (2, 5))
    action_ids = torch.randint(1, 3, (2, 5))
    time_gap_ids = torch.randint(1, 6, (2, 5))
    lengths = torch.full((2,), 5, dtype=torch.long)
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)

    next_action_logits, outcome_logits = model(
        item_ids, category_ids, action_ids, time_gap_ids, lengths, padding_mask
    )
    (next_action_logits.sum() + outcome_logits.sum()).backward()

    assert model.item_embed.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_lbm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.model.lbm'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/model/lbm.py
import torch
from torch import nn


class LBM(nn.Module):
    """Causal Transformer over behavior events, with a next-action head
    (predicts the next item at every position) and an outcome head
    (predicts whether the session ends in a purchase, read off each
    example's last real position)."""

    def __init__(
        self,
        n_items: int,
        n_categories: int,
        n_actions: int,
        n_time_gaps: int,
        max_len: int,
        embed_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_embed = nn.Embedding(n_items, embed_dim, padding_idx=0)
        self.category_embed = nn.Embedding(n_categories, embed_dim, padding_idx=0)
        self.action_embed = nn.Embedding(n_actions, embed_dim, padding_idx=0)
        self.time_gap_embed = nn.Embedding(n_time_gaps, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.next_action_head = nn.Linear(embed_dim, n_items)
        self.outcome_head = nn.Linear(embed_dim, 1)

    def forward(self, item_ids, category_ids, action_ids, time_gap_ids, lengths, padding_mask):
        batch_size, seq_len = item_ids.shape
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = (
            self.item_embed(item_ids)
            + self.category_embed(category_ids)
            + self.action_embed(action_ids)
            + self.time_gap_embed(time_gap_ids)
            + self.pos_embed(positions)
        )

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        hidden = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        next_action_logits = self.next_action_head(hidden)

        last_idx = (lengths - 1).clamp(min=0)
        last_hidden = hidden[torch.arange(batch_size, device=x.device), last_idx]
        outcome_logits = self.outcome_head(last_hidden).squeeze(-1)

        return next_action_logits, outcome_logits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_lbm.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/model/lbm.py tests/test_model_lbm.py TASK.md CHANGELOG.md
git commit -m "feat: add LBM causal Transformer with dual heads"
```

---

### Task 13: Training loop

**Files:**
- Create: `src/lbm/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `LBM` (Task 12), `SessionDataset`/`collate_sessions` (Task 7), `recall_at_k` (Task 8).
- Produces: `train_lbm(model, train_records, val_records, epochs, batch_size, lr, outcome_loss_weight, eval_k) -> dict` (history with keys `train_loss`, `val_recall@k`; trains `model` in place, restoring the best-by-val-recall state at the end). Consumed by Task 16.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
from lbm.data.tokenize import SessionRecord
from lbm.model.lbm import LBM
from lbm.train import train_lbm


def _record(session_id):
    return SessionRecord(
        session_id=session_id, item_ids=[2, 3, 4, 2], category_ids=[2, 2, 3, 2],
        action_ids=[2, 2, 2, 3], time_gap_ids=[1, 2, 1, 3], outcome=session_id % 2,
    )


def test_train_lbm_runs_and_returns_history():
    records = [_record(i) for i in range(8)]
    train_records, val_records = records[:6], records[6:]

    model = LBM(
        n_items=10, n_categories=5, n_actions=5, n_time_gaps=6, max_len=10,
        embed_dim=8, n_layers=1, n_heads=2, ffn_dim=16, dropout=0.0,
    )

    history = train_lbm(
        model, train_records, val_records, epochs=2, batch_size=2, lr=0.01,
        outcome_loss_weight=1.0, eval_k=5,
    )

    assert len(history["train_loss"]) == 2
    assert len(history["val_recall@k"]) == 2
    assert all(loss >= 0 for loss in history["train_loss"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.train'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/train.py
import torch
from torch import nn
from torch.utils.data import DataLoader

from lbm.data.dataset import SessionDataset, collate_sessions
from lbm.metrics import recall_at_k

PAD_ID = 0


def train_lbm(
    model,
    train_records: list,
    val_records: list,
    epochs: int,
    batch_size: int,
    lr: float,
    outcome_loss_weight: float,
    eval_k: int,
) -> dict:
    """Trains model in place for `epochs` epochs, evaluating recall@k on
    val_records after each epoch. Restores the best-by-val-recall state
    at the end and returns a history dict of per-epoch train loss and
    val recall@k."""
    train_loader = DataLoader(
        SessionDataset(train_records), batch_size=batch_size, shuffle=True, collate_fn=collate_sessions
    )
    val_loader = DataLoader(
        SessionDataset(val_records), batch_size=batch_size, shuffle=False, collate_fn=collate_sessions
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    next_action_loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    outcome_loss_fn = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_recall@k": []}
    best_recall = -1.0
    best_state = None

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            next_action_logits, outcome_logits = model(
                batch["item_ids"], batch["category_ids"], batch["action_ids"],
                batch["time_gap_ids"], batch["lengths"], batch["padding_mask"],
            )
            next_action_loss = next_action_loss_fn(
                next_action_logits.reshape(-1, next_action_logits.size(-1)),
                batch["target_items"].reshape(-1),
            )
            outcome_loss = outcome_loss_fn(outcome_logits, batch["outcome"])
            loss = next_action_loss + outcome_loss_weight * outcome_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        recalls = []
        with torch.no_grad():
            for batch in val_loader:
                next_action_logits, _ = model(
                    batch["item_ids"], batch["category_ids"], batch["action_ids"],
                    batch["time_gap_ids"], batch["lengths"], batch["padding_mask"],
                )
                recalls.append(recall_at_k(next_action_logits, batch["target_items"], eval_k))

        avg_loss = total_loss / max(len(train_loader), 1)
        avg_recall = sum(recalls) / max(len(recalls), 1)
        history["train_loss"].append(avg_loss)
        history["val_recall@k"].append(avg_recall)

        if avg_recall > best_recall:
            best_recall = avg_recall
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/train.py tests/test_train.py TASK.md CHANGELOG.md
git commit -m "feat: add LBM training loop with best-checkpoint restore"
```

---

### Task 14: Evaluation + comparison report

**Files:**
- Create: `src/lbm/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `LBM` (Task 12), `SessionDataset`/`collate_sessions` (Task 7), metrics (Task 8).
- Produces: `evaluate_lbm(model, records, k, batch_size=64) -> dict` (same shape as `run_baseline`'s return from Task 11); `print_comparison(baseline_metrics, lbm_metrics) -> None`. Consumed by Task 16.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
from lbm.data.tokenize import SessionRecord
from lbm.evaluate import evaluate_lbm
from lbm.model.lbm import LBM


def _record(session_id):
    return SessionRecord(
        session_id=session_id, item_ids=[2, 3, 4, 2], category_ids=[2, 2, 3, 2],
        action_ids=[2, 2, 2, 3], time_gap_ids=[1, 2, 1, 3], outcome=session_id % 2,
    )


def test_evaluate_lbm_returns_expected_metric_shape():
    records = [_record(i) for i in range(4)]
    model = LBM(
        n_items=10, n_categories=5, n_actions=5, n_time_gaps=6, max_len=10,
        embed_dim=8, n_layers=1, n_heads=2, ffn_dim=16, dropout=0.0,
    )

    metrics = evaluate_lbm(model, records, k=5, batch_size=2)

    assert set(metrics.keys()) == {"next_action", "outcome"}
    assert set(metrics["next_action"].keys()) == {"recall@k", "mrr@k"}
    assert set(metrics["outcome"].keys()) == {"roc_auc", "pr_auc"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.evaluate'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/evaluate.py
import torch
from torch.utils.data import DataLoader

from lbm.data.dataset import SessionDataset, collate_sessions
from lbm.metrics import mrr_at_k, pr_auc, recall_at_k, roc_auc


def evaluate_lbm(model, records: list, k: int, batch_size: int = 64) -> dict:
    """Evaluates a trained LBM on next-action (recall@k, mrr@k) and
    outcome (roc_auc, pr_auc), in the same metrics shape produced by
    baseline.train_baseline.run_baseline so results compare directly."""
    loader = DataLoader(SessionDataset(records), batch_size=batch_size, shuffle=False, collate_fn=collate_sessions)

    model.eval()
    recalls, mrrs = [], []
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            next_action_logits, outcome_logits = model(
                batch["item_ids"], batch["category_ids"], batch["action_ids"],
                batch["time_gap_ids"], batch["lengths"], batch["padding_mask"],
            )
            recalls.append(recall_at_k(next_action_logits, batch["target_items"], k))
            mrrs.append(mrr_at_k(next_action_logits, batch["target_items"], k))
            all_probs.extend(torch.sigmoid(outcome_logits).tolist())
            all_labels.extend(batch["outcome"].tolist())

    return {
        "next_action": {
            "recall@k": sum(recalls) / max(len(recalls), 1),
            "mrr@k": sum(mrrs) / max(len(mrrs), 1),
        },
        "outcome": {"roc_auc": roc_auc(all_probs, all_labels), "pr_auc": pr_auc(all_probs, all_labels)},
    }


def print_comparison(baseline_metrics: dict, lbm_metrics: dict) -> None:
    print(f"{'Metric':<20}{'Baseline':>12}{'LBM':>12}")
    print(
        f"{'Recall@k':<20}"
        f"{baseline_metrics['next_action']['recall@k']:>12.4f}"
        f"{lbm_metrics['next_action']['recall@k']:>12.4f}"
    )
    print(
        f"{'MRR@k':<20}"
        f"{baseline_metrics['next_action']['mrr@k']:>12.4f}"
        f"{lbm_metrics['next_action']['mrr@k']:>12.4f}"
    )
    print(
        f"{'ROC-AUC':<20}"
        f"{baseline_metrics['outcome']['roc_auc']:>12.4f}"
        f"{lbm_metrics['outcome']['roc_auc']:>12.4f}"
    )
    print(
        f"{'PR-AUC':<20}"
        f"{baseline_metrics['outcome']['pr_auc']:>12.4f}"
        f"{lbm_metrics['outcome']['pr_auc']:>12.4f}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/evaluate.py tests/test_evaluate.py TASK.md CHANGELOG.md
git commit -m "feat: add LBM evaluation and baseline-vs-LBM comparison report"
```

---

### Task 15: Dataset download (Kaggle)

**Files:**
- Create: `src/lbm/data/download.py`
- Test: `tests/test_download.py`

**Interfaces:**
- Produces: `ensure_yoochoose_data(raw_dir) -> tuple[Path, Path]` (clicks_path, buys_path) — returns existing files under `raw_dir` if present, otherwise downloads+extracts the Kaggle dataset `chadgostopp/recsys-challenge-2015` via the `kaggle` CLI. Consumed by Task 16.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_download.py
from unittest.mock import MagicMock, patch

from lbm.data.download import ensure_yoochoose_data


def test_skips_download_when_files_already_exist(tmp_path):
    (tmp_path / "yoochoose-clicks.dat").write_text("existing")
    (tmp_path / "yoochoose-buys.dat").write_text("existing")

    with patch("subprocess.run") as mock_run:
        clicks_path, buys_path = ensure_yoochoose_data(tmp_path)

    mock_run.assert_not_called()
    assert clicks_path == tmp_path / "yoochoose-clicks.dat"
    assert buys_path == tmp_path / "yoochoose-buys.dat"


def test_downloads_and_extracts_when_files_missing(tmp_path):
    def fake_run(cmd, check):
        import zipfile

        zip_path = tmp_path / "recsys-challenge-2015.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("yoochoose-clicks.dat", "click data")
            zf.writestr("yoochoose-buys.dat", "buy data")
        return MagicMock()

    with patch("subprocess.run", side_effect=fake_run) as mock_run:
        clicks_path, buys_path = ensure_yoochoose_data(tmp_path)

    mock_run.assert_called_once()
    assert clicks_path.read_text() == "click data"
    assert buys_path.read_text() == "buy data"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.data.download'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/data/download.py
import subprocess
import zipfile
from pathlib import Path


def ensure_yoochoose_data(raw_dir: Path) -> tuple:
    """Ensures yoochoose-clicks.dat and yoochoose-buys.dat exist under
    raw_dir, downloading and unzipping the Kaggle "RecSys Challenge
    2015" dataset via the kaggle CLI if they're missing. Returns their
    paths."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    clicks_path = raw_dir / "yoochoose-clicks.dat"
    buys_path = raw_dir / "yoochoose-buys.dat"

    if clicks_path.exists() and buys_path.exists():
        return clicks_path, buys_path

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "chadgostopp/recsys-challenge-2015", "-p", str(raw_dir)],
        check=True,
    )

    zip_path = raw_dir / "recsys-challenge-2015.zip"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    if not (clicks_path.exists() and buys_path.exists()):
        raise FileNotFoundError(
            f"Expected {clicks_path} and {buys_path} after extracting {zip_path}, "
            "but they weren't found. Check the archive contents."
        )

    return clicks_path, buys_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_download.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/data/download.py tests/test_download.py TASK.md CHANGELOG.md
git commit -m "feat: add Kaggle-backed YOOCHOOSE dataset download"
```

---

### Task 16: End-to-end pipeline + real run

**Files:**
- Create: `src/lbm/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 3–15.
- Produces: `load_config(path) -> dict`; `run_pipeline(config) -> dict` (`{"baseline": {...}, "lbm": {...}}`, each shaped like Task 11/14's outputs); `main()` CLI entry point (`python -m lbm.pipeline --config configs/default.yaml`).

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_pipeline.py
from pathlib import Path

from lbm.pipeline import run_pipeline


def _write_synthetic_yoochoose(raw_dir: Path, n_sessions: int = 40) -> None:
    click_lines = []
    buy_lines = []
    for session_id in range(1, n_sessions + 1):
        day = 1 + (session_id % 10)
        base_ts = f"2014-04-{day:02d}T10:00:00"
        n_events = 3 + (session_id % 4)
        for i in range(n_events):
            item_id = 100 + (session_id % 5) * 10 + i
            click_lines.append(f"{session_id},{base_ts}.{i:03d}Z,{item_id},{session_id % 3}\n")
        if session_id % 2 == 0:
            buy_item = 100 + (session_id % 5) * 10 + (n_events - 1)
            buy_lines.append(f"{session_id},{base_ts}.{n_events:03d}Z,{buy_item},999,1\n")

    (raw_dir / "yoochoose-clicks.dat").write_text("".join(click_lines))
    (raw_dir / "yoochoose-buys.dat").write_text("".join(buy_lines))


def test_run_pipeline_end_to_end_on_synthetic_data(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_synthetic_yoochoose(raw_dir)

    config = {
        "data": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "processed"),
            "n_sessions_subsample": None,
            "max_session_len": 50,
            "min_session_len": 2,
            "top_k_items": 100,
            "test_days": 3,
        },
        "model": {"embed_dim": 8, "n_layers": 1, "n_heads": 2, "ffn_dim": 16, "dropout": 0.0},
        "train": {"batch_size": 4, "epochs": 1, "lr": 0.01, "outcome_loss_weight": 1.0, "seed": 42},
        "eval": {"k": 5},
    }

    results = run_pipeline(config)

    assert set(results.keys()) == {"baseline", "lbm"}
    for metrics in results.values():
        assert set(metrics["next_action"].keys()) == {"recall@k", "mrr@k"}
        assert set(metrics["outcome"].keys()) == {"roc_auc", "pr_auc"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lbm.pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# src/lbm/pipeline.py
import argparse
from collections import Counter
from pathlib import Path

import yaml

from lbm.baseline.train_baseline import run_baseline
from lbm.data.download import ensure_yoochoose_data
from lbm.data.parse import load_buys, load_clicks
from lbm.data.sessions import build_session_events, filter_sessions, session_outcomes
from lbm.data.split import temporal_split
from lbm.data.tokenize import tokenize_sessions
from lbm.data.vocab import Vocab
from lbm.evaluate import evaluate_lbm, print_comparison
from lbm.model.lbm import LBM
from lbm.train import train_lbm

N_TIME_GAP_BUCKETS = 6  # id 0 = pad, ids 1-5 = buckets from time_gap_bucket()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(config: dict) -> dict:
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    eval_cfg = config["eval"]

    clicks_path, buys_path = ensure_yoochoose_data(Path(data_cfg["raw_dir"]))
    clicks = load_clicks(clicks_path)
    buys = load_buys(buys_path)

    subsample = data_cfg.get("n_sessions_subsample")
    if subsample:
        keep_ids = clicks["session_id"].drop_duplicates().sample(
            n=min(subsample, clicks["session_id"].nunique()),
            random_state=train_cfg["seed"],
        )
        clicks = clicks[clicks["session_id"].isin(keep_ids)]
        buys = buys[buys["session_id"].isin(keep_ids)]

    events = build_session_events(clicks, buys)
    events = filter_sessions(events, data_cfg["min_session_len"], data_cfg["max_session_len"])
    outcome_ids = session_outcomes(buys)

    item_vocab = Vocab.from_counts(Counter(events["item_id"]), top_k=data_cfg["top_k_items"])
    category_vocab = Vocab.from_counts(Counter(events["category"]))
    action_vocab = Vocab.from_counts(Counter(events["action_type"]))

    records = tokenize_sessions(events, outcome_ids, item_vocab, category_vocab, action_vocab)
    train_records, test_records = temporal_split(records, events, data_cfg["test_days"])

    buy_action_id = action_vocab.encode("buy")
    baseline_metrics = run_baseline(train_records, test_records, buy_action_id, eval_cfg["k"])

    model = LBM(
        n_items=len(item_vocab),
        n_categories=len(category_vocab),
        n_actions=len(action_vocab),
        n_time_gaps=N_TIME_GAP_BUCKETS,
        max_len=data_cfg["max_session_len"],
        embed_dim=model_cfg["embed_dim"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=model_cfg["dropout"],
    )
    train_lbm(
        model,
        train_records,
        test_records,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        outcome_loss_weight=train_cfg["outcome_loss_weight"],
        eval_k=eval_cfg["k"],
    )
    lbm_metrics = evaluate_lbm(model, test_records, eval_cfg["k"], batch_size=train_cfg["batch_size"])

    print_comparison(baseline_metrics, lbm_metrics)
    return {"baseline": baseline_metrics, "lbm": lbm_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_pipeline(load_config(args.config))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests across every task pass.

- [ ] **Step 6: Update TASK.md and CHANGELOG.md, then commit**

```bash
git add src/lbm/pipeline.py tests/test_pipeline.py TASK.md CHANGELOG.md
git commit -m "feat: wire up end-to-end LBM pipeline (data -> baseline -> LBM -> comparison)"
```

- [ ] **Step 7: Run the real pipeline on actual YOOCHOOSE data**

This is the "definition of done" check from the spec — not a unit test, a real run.

Run: `uv run python -m lbm.pipeline --config configs/default.yaml`

Expected: downloads the dataset via Kaggle on first run (may take a few minutes depending on connection), then prints a `Baseline` vs `LBM` comparison table. Confirm the LBM beats the baseline on `Recall@k` and `ROC-AUC` — if it doesn't, that's a signal to revisit hyperparameters (`configs/default.yaml`) or model capacity before concluding v1 is done, not a reason to change the metrics being compared.

- [ ] **Step 8: Record the real-run result and finalize TASK.md/CHANGELOG.md**

Add the printed comparison table (or a summary of it) as a new `## Results` section at the bottom of `README.md`. Mark all remaining items in `TASK.md` complete. Add a final `CHANGELOG.md` entry noting the v1 milestone is reached.

```bash
git add README.md TASK.md CHANGELOG.md
git commit -m "docs: record v1 baseline-vs-LBM results"
```

---

## Self-Review Notes

- **Spec coverage**: Task 1 covers scaffolding/repo structure; Tasks 2-6 cover Data & Task Definition; Tasks 7, 12 cover Model Architecture (dataset framing + Transformer); Tasks 8-11, 13-14 cover Training/Evaluation; Task 15 covers the Kaggle data source; Task 16 covers the definition-of-done check. All spec sections have a corresponding task.
- **Deviation from spec's illustrative repo structure**: the spec sketched separate `embeddings.py`/`transformer.py`/`heads.py` files under `model/`; this plan consolidates them into one `src/lbm/model/lbm.py` (~80 lines) since splitting a single cohesive, small module adds indirection without benefit. Noted here rather than silently diverging.
- **Type/signature consistency checked**: `SessionRecord` fields (Task 5) are used identically in Tasks 6, 7, 9, 10; `PAD_ID = 0` is consistent across Tasks 7, 8, 9, 13; the `{"next_action": {...}, "outcome": {...}}` metrics shape is identical between Task 11 (`run_baseline`) and Task 14 (`evaluate_lbm`), which is what makes Task 16's final comparison valid.
