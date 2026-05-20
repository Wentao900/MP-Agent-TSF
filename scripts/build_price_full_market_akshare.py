from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import akshare as ak
import pandas as pd


def log(msg: str) -> None:
    print(msg, flush=True)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"next_batch": 0, "done": False}
    return json.loads(path.read_text())


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def get_all_a_symbols() -> list[str]:
    spot = ak.stock_zh_a_spot_em()
    code_col = "代码" if "代码" in spot.columns else spot.columns[0]
    codes = spot[code_col].astype(str).str.extract(r"(\d{6})", expand=False).dropna().unique().tolist()

    def to_symbol(c: str) -> str:
        return f"{c}.SH" if c.startswith("6") else f"{c}.SZ"

    symbols = sorted({to_symbol(c) for c in codes})
    if not symbols:
        raise RuntimeError("failed to fetch A-share symbols from akshare")
    return symbols


def split_batches(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def run_batch(
    python_exec: str,
    batch_symbols: list[str],
    batch_idx: int,
    total_batches: int,
    out_csv: Path,
    freq: str,
    adjust: str,
    start_date: str,
    end_date: str,
    minute_period: str,
) -> None:
    cmd = [
        python_exec,
        "-u",
        "scripts/build_price_from_akshare.py",
        "--symbols",
        ",".join(batch_symbols),
        "--freq",
        freq,
        "--adjust",
        adjust,
        "--output-csv",
        str(out_csv),
    ]
    if freq == "daily":
        cmd.extend(["--start-date", start_date, "--end-date", end_date])
    else:
        cmd.extend(["--minute-period", minute_period])

    log(f"[batch] {batch_idx + 1}/{total_batches} symbols={len(batch_symbols)} out={out_csv.name}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"batch failed idx={batch_idx} code={proc.returncode}")


def merge_batches(batch_dir: Path, merged_csv: Path) -> None:
    files = sorted(batch_dir.glob("price_batch_*.csv"))
    if not files:
        raise RuntimeError(f"no batch files found in {batch_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("all batch files are empty")

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp", "symbol"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    merged_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(merged_csv, index=False)
    log(f"[merge-done] files={len(files)} rows={len(out)} output={merged_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch full A-share price data with AkShare in resumable batches")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--freq", choices=["daily", "minute"], default="daily")
    parser.add_argument("--start-date", default="20240101", help="YYYYMMDD, used for daily")
    parser.add_argument("--end-date", default="20241231", help="YYYYMMDD, used for daily")
    parser.add_argument("--minute-period", choices=["1", "5", "15", "30", "60"], default="1", help="used for minute")
    parser.add_argument("--adjust", default="qfq")

    parser.add_argument("--symbols-csv", default="raw/all_a_symbols.csv", help="cached symbol list path")
    parser.add_argument("--refresh-symbols", action="store_true", help="force refresh full A-share symbols from AkShare")

    parser.add_argument("--batch-dir", default="raw/price_batches")
    parser.add_argument("--output-csv", default="outputs/data/price.csv")
    parser.add_argument("--checkpoint", default="raw/.checkpoints/full_market_akshare.json")
    parser.add_argument("--stop-after-batches", type=int, default=0, help="0 means run all; >0 means stop after N batches for testing")
    parser.add_argument("--merge-only", action="store_true", help="only merge existing batch files")
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    checkpoint_path = Path(args.checkpoint)
    symbols_csv = Path(args.symbols_csv)
    output_csv = Path(args.output_csv)

    batch_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_batches(batch_dir, output_csv)
        return

    if args.refresh_symbols or (not symbols_csv.exists()):
        symbols = get_all_a_symbols()
        symbols_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"symbol": symbols}).to_csv(symbols_csv, index=False)
        log(f"[symbols] refreshed count={len(symbols)} saved={symbols_csv}")
    else:
        symbols = pd.read_csv(symbols_csv)["symbol"].dropna().astype(str).tolist()
        log(f"[symbols] loaded cached count={len(symbols)} from={symbols_csv}")

    if not symbols:
        raise RuntimeError("no symbols available")

    batches = split_batches(symbols, args.batch_size)
    total_batches = len(batches)

    ckpt = load_checkpoint(checkpoint_path)
    start_batch = int(ckpt.get("next_batch", 0))
    if start_batch < 0 or start_batch > total_batches:
        start_batch = 0

    log(f"[start] total_symbols={len(symbols)} batch_size={args.batch_size} total_batches={total_batches} start_batch={start_batch}")

    ran = 0
    for i in range(start_batch, total_batches):
        out_csv = batch_dir / f"price_batch_{i:04d}.csv"
        run_batch(
            python_exec=args.python,
            batch_symbols=batches[i],
            batch_idx=i,
            total_batches=total_batches,
            out_csv=out_csv,
            freq=args.freq,
            adjust=args.adjust,
            start_date=args.start_date,
            end_date=args.end_date,
            minute_period=args.minute_period,
        )

        save_checkpoint(
            checkpoint_path,
            {
                "next_batch": i + 1,
                "total_batches": total_batches,
                "batch_size": args.batch_size,
                "freq": args.freq,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "minute_period": args.minute_period,
                "adjust": args.adjust,
                "batch_dir": str(batch_dir),
                "output_csv": str(output_csv),
                "done": False,
            },
        )

        ran += 1
        if args.stop_after_batches > 0 and ran >= args.stop_after_batches:
            log(f"[stop] reached stop-after-batches={args.stop_after_batches}")
            return

    merge_batches(batch_dir, output_csv)
    save_checkpoint(
        checkpoint_path,
        {
            "next_batch": total_batches,
            "total_batches": total_batches,
            "done": True,
            "output_csv": str(output_csv),
            "batch_dir": str(batch_dir),
        },
    )
    log("[done] full market fetch completed")


if __name__ == "__main__":
    main()
