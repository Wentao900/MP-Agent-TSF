from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def log(msg: str) -> None:
    print(msg, flush=True)


def run_step(cmd: list[str], name: str) -> None:
    log(f"\n[step] {name}")
    log(f"[cmd] {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"step failed: {name}, code={proc.returncode}")
    log(f"[ok] {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot Scheme C data pipeline: crawl announcements -> fetch files -> build price/text -> validate"
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable path")

    # Crawl window
    parser.add_argument("--start-date", required=True, help="Announcement crawl start date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Announcement crawl end date, YYYY-MM-DD")

    # Price source mode
    parser.add_argument("--price-source", choices=["akshare", "tushare_csv"], default="akshare")
    parser.add_argument("--symbols", default="000001.SZ,600036.SH", help="Used when --price-source akshare")
    parser.add_argument("--freq", choices=["daily", "minute"], default="daily", help="Used when --price-source akshare")
    parser.add_argument("--ak-start-date", default="20240101", help="YYYYMMDD, used for akshare daily")
    parser.add_argument("--ak-end-date", default="20241231", help="YYYYMMDD, used for akshare daily")
    parser.add_argument("--minute-period", choices=["1", "5", "15", "30", "60"], default="1", help="Used for akshare minute")
    parser.add_argument("--adjust", default="qfq", help="adjust flag for akshare/tushare builder")

    parser.add_argument("--tushare-input-csv", default="raw/price_tushare.csv", help="Used when --price-source tushare_csv")

    # Paths
    parser.add_argument("--raw-ann-csv", default="raw/cninfo_announcements.csv")
    parser.add_argument("--raw-ann-content-csv", default="raw/cninfo_announcements_with_content.csv")
    parser.add_argument("--download-dir", default="raw/cninfo_files")
    parser.add_argument("--price-csv", default="outputs/data/price.csv")
    parser.add_argument("--text-csv", default="outputs/data/text.csv")

    # Crawler tuning
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--crawl-sleep-min", type=float, default=0.8)
    parser.add_argument("--crawl-sleep-max", type=float, default=1.6)
    parser.add_argument("--crawl-retries", type=int, default=3)
    parser.add_argument("--crawl-timeout", type=int, default=20)

    parser.add_argument("--fetch-sleep-min", type=float, default=0.6)
    parser.add_argument("--fetch-sleep-max", type=float, default=1.4)
    parser.add_argument("--fetch-retries", type=int, default=3)
    parser.add_argument("--fetch-timeout", type=int, default=25)
    parser.add_argument("--fetch-max-rows", type=int, default=0, help="0 means no limit")

    parser.add_argument("--skip-content-fetch", action="store_true", help="Skip file download step, build text from announcement list directly")
    parser.add_argument("--no-extract-html-text", action="store_true", help="Disable HTML/TXT text extraction when fetching files")

    args = parser.parse_args()

    py = args.python
    Path("raw").mkdir(parents=True, exist_ok=True)
    Path("outputs/data").mkdir(parents=True, exist_ok=True)

    # 1) Incremental crawl
    run_step(
        [
            py,
            "scripts/crawl_cninfo_incremental.py",
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--output-csv",
            args.raw_ann_csv,
            "--page-size",
            str(args.page_size),
            "--sleep-min",
            str(args.crawl_sleep_min),
            "--sleep-max",
            str(args.crawl_sleep_max),
            "--retries",
            str(args.crawl_retries),
            "--timeout",
            str(args.crawl_timeout),
        ],
        "crawl_cninfo_incremental",
    )

    # 2) Download attachment files (optional)
    text_input_csv = args.raw_ann_csv
    if not args.skip_content_fetch:
        cmd = [
            py,
            "scripts/crawl_cninfo_content_fetch.py",
            "--input-csv",
            args.raw_ann_csv,
            "--download-dir",
            args.download_dir,
            "--output-csv",
            args.raw_ann_content_csv,
            "--sleep-min",
            str(args.fetch_sleep_min),
            "--sleep-max",
            str(args.fetch_sleep_max),
            "--retries",
            str(args.fetch_retries),
            "--timeout",
            str(args.fetch_timeout),
            "--max-rows",
            str(args.fetch_max_rows),
        ]
        if not args.no_extract_html_text:
            cmd.append("--extract-html-text")

        run_step(cmd, "crawl_cninfo_content_fetch")
        text_input_csv = args.raw_ann_content_csv

    # 3) Build price.csv
    if args.price_source == "akshare":
        price_cmd = [
            py,
            "scripts/build_price_from_akshare.py",
            "--symbols",
            args.symbols,
            "--freq",
            args.freq,
            "--adjust",
            args.adjust,
            "--output-csv",
            args.price_csv,
        ]
        if args.freq == "daily":
            price_cmd.extend(["--start-date", args.ak_start_date, "--end-date", args.ak_end_date])
        else:
            price_cmd.extend(["--minute-period", args.minute_period])
        run_step(price_cmd, "build_price_from_akshare")
    else:
        run_step(
            [
                py,
                "scripts/build_price_from_tushare.py",
                "--input-csv",
                args.tushare_input_csv,
                "--output-csv",
                args.price_csv,
                "--freq",
                "minute" if args.freq == "minute" else "daily",
            ],
            "build_price_from_tushare",
        )

    # 4) Build text.csv
    run_step(
        [
            py,
            "scripts/build_text_from_cn_sources.py",
            "--input-csv",
            text_input_csv,
            "--output-csv",
            args.text_csv,
        ],
        "build_text_from_cn_sources",
    )

    # 5) Validate contract
    run_step(
        [
            py,
            "scripts/validate_dataset_contract.py",
            "--price-csv",
            args.price_csv,
            "--text-csv",
            args.text_csv,
        ],
        "validate_dataset_contract",
    )

    log("\n[done] full data pipeline completed")
    log(f"[artifact] price={args.price_csv}")
    log(f"[artifact] text={args.text_csv}")


if __name__ == "__main__":
    main()
