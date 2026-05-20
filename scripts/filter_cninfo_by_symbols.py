from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import load_universe, normalize_symbol_code  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter CNINFO announcement csv to a symbol universe")
    parser.add_argument("--input-csv", default="raw/cninfo_announcements.csv")
    parser.add_argument("--output-csv", default="raw/cninfo_announcements_universe.csv")
    parser.add_argument("--universe-file", default="data/universe_paper_50.txt")
    args = parser.parse_args()

    universe = set(load_universe(Path(args.universe_file)))
    df = pd.read_csv(args.input_csv)
    if "symbol" not in df.columns:
        raise ValueError("input csv must contain symbol column")

    df = df.copy()
    df["_code"] = df["symbol"].astype(str).map(normalize_symbol_code)
    out = df[df["_code"].isin(universe)].drop(columns=["_code"]).reset_index(drop=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"saved filtered announcements: {out_path}")
    print(f"rows={len(out)}, symbols={out['symbol'].astype(str).map(normalize_symbol_code).nunique()}")
    if len(out) > 0 and "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"])
        print(f"date_range={ts.min()} .. {ts.max()}")


if __name__ == "__main__":
    main()
