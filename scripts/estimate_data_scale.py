from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import load_universe, to_akshare_symbol  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate training samples and disk for IMTSA data plan")
    parser.add_argument("--universe-file", default="data/universe_paper_50.txt")
    parser.add_argument("--years", type=float, default=3.0)
    parser.add_argument("--freq", choices=["daily", "minute"], default="daily")
    parser.add_argument("--trading-days-per-year", type=int, default=242)
    parser.add_argument("--minutes-per-day", type=int, default=240)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--ann-per-symbol-month", type=float, default=6.0, help="avg announcements per symbol per month")
    parser.add_argument("--pdf-mib-per-ann", type=float, default=0.35, help="avg PDF size (from your Jan sample)")
    parser.add_argument("--download-pdf", action="store_true")
    args = parser.parse_args()

    n_sym = len(load_universe(Path(args.universe_file)))
    months = int(args.years * 12)

    if args.freq == "daily":
        bars_per_sym = int(args.trading_days_per_year * args.years)
    else:
        bars_per_sym = int(args.trading_days_per_year * args.years * args.minutes_per_day)

    total_bars = n_sym * bars_per_sym
    total_seq = max(0, total_bars - n_sym * args.seq_len)
    train_seq = int(total_seq * args.train_ratio)

    total_ann = int(n_sym * months * args.ann_per_symbol_month)
    pdf_gib = total_ann * args.pdf_mib_per_ann / 1024 if args.download_pdf else 0.0

    reflector_params = 74_314
    ratio = train_seq / reflector_params if reflector_params else 0.0

    print("=== IMTSA data scale estimate ===")
    print(f"symbols={n_sym}, years={args.years}, freq={args.freq}")
    print(f"price_bars≈{total_bars:,}")
    print(f"sequence_samples≈{total_seq:,} (train≈{train_seq:,})")
    print(f"train_seq / reflector_params≈{ratio:.2f}x  (pilot建议≥50x, 当前目标≈{reflector_params*50:,} train_seq)")
    print(f"announcements≈{total_ann:,} (assumed {args.ann_per_symbol_month}/sym/month)")
    if args.download_pdf:
        print(f"pdf_disk≈{pdf_gib:.1f} GiB")
    else:
        print("pdf_disk≈0 (title-only / --skip-content-fetch)")
    print(f"price_csv≈negligible (<100 MiB for daily panel)")

    print("\n=== verdict ===")
    if train_seq < 3_000:
        print("不足：仅适合跑通 pipeline，不宜写「有预测力」。")
    elif train_seq < 10_000:
        print("偏弱：可做 pilot 消融，但需控制结论表述（小样本/单市场）。")
    else:
        print("可达论文 pilot：满足最小样本量，仍需 walk-forward / 成本 / 多 seed 验证。")

    print("\n=== suggested akshare symbols string (first 10) ===")
    codes = load_universe(Path(args.universe_file))[:10]
    print(",".join(to_akshare_symbol(c) for c in codes), "...")

if __name__ == "__main__":
    main()
