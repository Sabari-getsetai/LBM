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
