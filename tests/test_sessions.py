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
