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
