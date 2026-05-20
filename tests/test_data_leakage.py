from __future__ import annotations

import pandas as pd

from imtsa.data.loader import _assert_no_leakage


def test_timestamp_must_be_monotonic() -> None:
    good = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01"])})
    _assert_no_leakage(good, "timestamp")

    bad = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:01", "2024-01-01 00:00"])})
    try:
        _assert_no_leakage(bad, "timestamp")
        assert False, "expected ValueError"
    except ValueError:
        assert True
