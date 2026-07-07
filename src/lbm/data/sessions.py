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
        result = group.tail(max_len)
        result = result.copy()
        result["session_id"] = group.name
        return result

    events = (
        events.groupby("session_id", group_keys=False)
        .apply(_truncate)
        .reset_index(drop=True)
    )
    return events
