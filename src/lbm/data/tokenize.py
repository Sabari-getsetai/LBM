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
