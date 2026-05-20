from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from imtsa.utils import ensure_dir


def main() -> None:
    out = ensure_dir("outputs/data")
    n = 2000
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    close = np.cumprod(1 + rng.normal(0, 0.0008, size=n)) * 100
    volume = rng.lognormal(10, 0.4, size=n)

    price = pd.DataFrame({
        "timestamp": ts,
        "symbol": "SYNTH",
        "open": close * (1 + rng.normal(0, 0.0002, size=n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.0004, size=n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.0004, size=n))),
        "close": close,
        "volume": volume,
    })
    price["ret_1"] = price["close"].pct_change().fillna(0.0)
    price["ret_5"] = price["close"].pct_change(5).fillna(0.0)
    price["vol_5"] = price["ret_1"].rolling(5).std().fillna(0.0)

    sentiment = np.tanh(price["ret_5"].to_numpy() * 50 + rng.normal(0, 0.3, size=n))
    text = pd.DataFrame({
        "timestamp": ts,
        "symbol": "SYNTH",
        "sentiment": sentiment,
        "relevance": rng.uniform(0.5, 1.0, size=n),
        "event_strength": np.abs(sentiment) * rng.uniform(0.8, 1.2, size=n),
    })

    price.to_csv(Path(out) / "price.csv", index=False)
    text.to_csv(Path(out) / "text.csv", index=False)


if __name__ == "__main__":
    main()
