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
