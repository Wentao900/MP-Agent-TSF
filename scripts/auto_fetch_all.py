from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from datetime import datetime
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


def _import_fetch_helpers():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from crawl_cninfo_content_fetch import compute_resume_index  # noqa: PLC0415

    return compute_resume_index


def content_fetch_done(input_csv: Path, download_dir: Path, output_csv: Path) -> bool:
    if not input_csv.exists():
        return True
    import pandas as pd

    compute_resume_index = _import_fetch_helpers()
    df = pd.read_csv(input_csv)
    if len(df) == 0:
        return True
    resume_idx = compute_resume_index(df, download_dir, output_csv, skip_existing=True)
    return resume_idx >= len(df)


def run_content_fetch_batches(
    py: str,
    *,
    input_csv: str,
    download_dir: str,
    output_csv: str,
    checkpoint: str,
    batch_size: int,
    max_rounds: int,
    retry_passes: int,
    extract_html_text: bool,
    fetch_args: list[str],
) -> None:
    in_path = Path(input_csv)
    out_path = Path(output_csv)
    dl_path = Path(download_dir)

    for round_idx in range(1, max_rounds + 1):
        if content_fetch_done(in_path, dl_path, out_path):
            log(f"[fetch] round={round_idx} already complete")
            break

        cmd = [
            py,
            "scripts/crawl_cninfo_content_fetch.py",
            "--input-csv",
            input_csv,
            "--download-dir",
            download_dir,
            "--output-csv",
            output_csv,
            "--checkpoint",
            checkpoint,
            "--sync-checkpoint",
            "--skip-existing",
            "--max-rows",
            str(batch_size),
            *fetch_args,
        ]
        if extract_html_text:
            cmd.append("--extract-html-text")

        log(f"[fetch] round={round_idx}/{max_rounds} batch_size={batch_size}")
        run_step(cmd, f"content_fetch_round_{round_idx}")

        if content_fetch_done(in_path, dl_path, out_path):
            log("[fetch] download pass complete")
            break

        time.sleep(1.0)
    else:
        log("[warn] reached max fetch rounds; re-run this script to continue")

    for pass_idx in range(1, retry_passes + 1):
        cmd = [
            py,
            "scripts/crawl_cninfo_content_fetch.py",
            "--input-csv",
            input_csv,
            "--download-dir",
            download_dir,
            "--output-csv",
            output_csv,
            "--checkpoint",
            checkpoint,
            "--skip-existing",
            "--retry-errors",
            *fetch_args,
        ]
        if extract_html_text:
            cmd.append("--extract-html-text")
        log(f"[fetch] retry-errors pass={pass_idx}/{retry_passes}")
        run_step(cmd, f"content_fetch_retry_{pass_idx}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable one-shot pipeline: CNINFO crawl -> batch download -> price/text build -> validate"
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable path")

    parser.add_argument("--start-date", required=True, help="Announcement crawl start date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Announcement crawl end date, YYYY-MM-DD")

    parser.add_argument("--price-source", choices=["akshare", "tushare_csv", "none"], default="akshare")
    parser.add_argument("--symbols", default="000001.SZ,600036.SH", help="akshare symbols; overridden when --universe-file is set")
    parser.add_argument("--universe-file", default="", help="e.g. data/universe_paper_50.txt (per-symbol CNINFO crawl)")
    parser.add_argument("--crawl-window", choices=["day", "month"], default="month", help="date window for universe crawl")
    parser.add_argument("--freq", choices=["daily", "minute"], default="daily")
    parser.add_argument("--ak-start-date", default="20240101", help="YYYYMMDD")
    parser.add_argument("--ak-end-date", default="20241231", help="YYYYMMDD")
    parser.add_argument("--minute-period", choices=["1", "5", "15", "30", "60"], default="1")
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--tushare-input-csv", default="raw/price_tushare.csv")

    parser.add_argument("--raw-ann-csv", default="raw/cninfo_announcements.csv")
    parser.add_argument("--raw-ann-content-csv", default="raw/cninfo_announcements_with_content.csv")
    parser.add_argument("--download-dir", default="raw/cninfo_files")
    parser.add_argument("--fetch-checkpoint", default="raw/.checkpoints/cninfo_content_fetch.json")
    parser.add_argument(
        "--crawl-checkpoint",
        default="",
        help="defaults to cninfo_incremental.json or cninfo_incremental_universe.json when using --universe-file",
    )
    parser.add_argument("--price-csv", default="outputs/data/price.csv")
    parser.add_argument("--text-csv", default="outputs/data/text.csv")
    parser.add_argument("--log-dir", default="logs")

    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--crawl-sleep-min", type=float, default=0.3)
    parser.add_argument("--crawl-sleep-max", type=float, default=0.6)
    parser.add_argument("--crawl-retries", type=int, default=5)
    parser.add_argument("--crawl-timeout", type=int, default=30)

    parser.add_argument("--fetch-sleep-min", type=float, default=0.35)
    parser.add_argument("--fetch-sleep-max", type=float, default=0.7)
    parser.add_argument("--fetch-retries", type=int, default=5)
    parser.add_argument("--fetch-timeout", type=int, default=60)
    parser.add_argument("--fetch-batch-size", type=int, default=500, help="rows per download subprocess (断线友好)")
    parser.add_argument("--fetch-max-rounds", type=int, default=2000, help="safety cap for batch loops")
    parser.add_argument("--fetch-retry-passes", type=int, default=2, help="extra passes for failed downloads")

    parser.add_argument("--skip-crawl", action="store_true", help="skip announcement list crawl")
    parser.add_argument("--skip-content-fetch", action="store_true", help="skip attachment download")
    parser.add_argument("--skip-price", action="store_true", help="skip price.csv build")
    parser.add_argument("--skip-validate", action="store_true", help="skip dataset contract validation")
    parser.add_argument("--no-extract-html-text", action="store_true")

    args = parser.parse_args()
    py = args.python

    if args.universe_file:
        scripts_dir = Path(__file__).resolve().parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from universe import load_universe, to_akshare_symbol  # noqa: PLC0415

        codes = load_universe(Path(args.universe_file))
        args.symbols = ",".join(to_akshare_symbol(c) for c in codes)
        if args.raw_ann_csv == "raw/cninfo_announcements.csv":
            args.raw_ann_csv = "raw/cninfo_announcements_universe.csv"
        if args.raw_ann_content_csv == "raw/cninfo_announcements_with_content.csv":
            args.raw_ann_content_csv = "raw/cninfo_announcements_with_content_universe.csv"
        if args.download_dir == "raw/cninfo_files":
            args.download_dir = "raw/cninfo_files_universe"
        if args.fetch_checkpoint == "raw/.checkpoints/cninfo_content_fetch.json":
            args.fetch_checkpoint = "raw/.checkpoints/cninfo_content_fetch_universe.json"
        if not args.crawl_checkpoint:
            args.crawl_checkpoint = "raw/.checkpoints/cninfo_incremental_universe.json"
        log(f"[universe] symbols={len(codes)} ann_csv={args.raw_ann_csv}")

    if not args.crawl_checkpoint:
        args.crawl_checkpoint = "raw/.checkpoints/cninfo_incremental.json"

    Path("raw").mkdir(parents=True, exist_ok=True)
    Path("raw/.checkpoints").mkdir(parents=True, exist_ok=True)
    Path("outputs/data").mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log(f"[log-dir] {log_dir.resolve()} (redirect with: ... > {log_dir}/auto_fetch_{datetime.now():%Y%m%d_%H%M%S}.log 2>&1)")

    fetch_args = [
        "--sleep-min",
        str(args.fetch_sleep_min),
        "--sleep-max",
        str(args.fetch_sleep_max),
        "--retries",
        str(args.fetch_retries),
        "--timeout",
        str(args.fetch_timeout),
        "--flush-every",
        "1",
    ]

    if not args.skip_crawl:
        crawl_cmd = [
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
        ]
        crawl_cmd.extend(["--checkpoint", args.crawl_checkpoint])
        if args.universe_file:
            crawl_cmd.extend(["--universe-file", args.universe_file, "--window", args.crawl_window])
        run_step(crawl_cmd, "crawl_cninfo_incremental")

    text_input_csv = args.raw_ann_csv
    if not args.skip_content_fetch:
        run_content_fetch_batches(
            py,
            input_csv=args.raw_ann_csv,
            download_dir=args.download_dir,
            output_csv=args.raw_ann_content_csv,
            checkpoint=args.fetch_checkpoint,
            batch_size=args.fetch_batch_size,
            max_rounds=args.fetch_max_rounds,
            retry_passes=args.fetch_retry_passes,
            extract_html_text=not args.no_extract_html_text,
            fetch_args=fetch_args,
        )
        text_input_csv = args.raw_ann_content_csv

    if not args.skip_price and args.price_source != "none":
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

    if not args.skip_validate and not args.skip_price and args.price_source != "none":
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

    log("\n[done] auto_fetch_all completed")
    log(f"[artifact] announcements={args.raw_ann_csv}")
    if not args.skip_content_fetch:
        log(f"[artifact] announcements_with_content={args.raw_ann_content_csv}")
    log(f"[artifact] text={args.text_csv}")
    if not args.skip_price and args.price_source != "none":
        log(f"[artifact] price={args.price_csv}")


if __name__ == "__main__":
    main()
