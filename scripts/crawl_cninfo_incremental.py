from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from universe import load_universe  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


CNINFO_QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_FILE_PREFIX = "https://static.cninfo.com.cn/"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.cninfo.com.cn",
    "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
}


def daterange(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    spans: list[tuple[datetime, datetime]] = []
    cur = start
    while cur <= end:
        next_day = cur + timedelta(days=1)
        day_end = min(next_day - timedelta(seconds=1), end)
        spans.append((cur, day_end))
        cur = next_day
    return spans


def monthrange(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    spans: list[tuple[datetime, datetime]] = []
    cur = datetime(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            next_month = datetime(cur.year + 1, 1, 1)
        else:
            next_month = datetime(cur.year, cur.month + 1, 1)
        month_end = min(next_month - timedelta(seconds=1), end)
        span_start = max(cur, start)
        if span_start <= month_end:
            spans.append((span_start, month_end))
        cur = next_month
    return spans


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_end": None}
    return json.loads(path.read_text())


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def request_with_retry(
    session: requests.Session,
    payload: dict[str, Any],
    headers: dict[str, str],
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session.post(CNINFO_QUERY_URL, data=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt >= retries:
                break
            backoff = min(8.0, 1.2 * (2**attempt)) + random.uniform(0.1, 0.6)
            log(f"[retry] attempt={attempt + 1}/{retries + 1} backoff={backoff:.2f}s err={err}")
            time.sleep(backoff)
    raise RuntimeError(f"request failed after retries: {last_err}")


def normalize_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in raw_rows:
        adjunct = a.get("adjunctUrl") or ""
        file_url = CNINFO_FILE_PREFIX + adjunct if adjunct else ""
        ts_ms = a.get("announcementTime")
        if ts_ms is None:
            published_at = None
        else:
            published_at = pd.to_datetime(ts_ms, unit="ms", errors="coerce")

        rows.append(
            {
                "timestamp": published_at,
                "symbol": a.get("secCode"),
                "symbol_name": a.get("secName"),
                "title": a.get("announcementTitle"),
                "announcement_id": a.get("announcementId"),
                "adjunct_type": a.get("adjunctType"),
                "adjunct_size": a.get("adjunctSize"),
                "url": file_url,
                "org_id": a.get("orgId"),
                "column_code": a.get("columnCode"),
                "source": "cninfo",
            }
        )
    return rows


def fetch_window(
    session: requests.Session,
    start: datetime,
    end: datetime,
    page_size: int,
    sleep_min: float,
    sleep_max: float,
    retries: int,
    timeout: int,
    stock_code: str = "",
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    page_num = 1

    date_str = f"{start.strftime('%Y-%m-%d')}~{end.strftime('%Y-%m-%d')}"
    # CNINFO 按股票过滤需用 searchkey（6 位代码），stock 字段无效
    searchkey = stock_code if stock_code else ""
    sym_tag = f" searchkey={searchkey}" if searchkey else ""
    log(f"[window-start] {date_str}{sym_tag}")
    while True:
        payload = {
            "pageNum": page_num,
            "pageSize": page_size,
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "",
            "searchkey": searchkey,
            "secid": "",
            "category": "",
            "trade": "",
            "seDate": date_str,
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        js = request_with_retry(session, payload, DEFAULT_HEADERS, retries=retries, timeout=timeout)
        anns = js.get("announcements", [])
        if not anns:
            log(f"[window-end] {date_str} pages={page_num - 1} rows={len(all_rows)}")
            break

        all_rows.extend(normalize_rows(anns))
        log(f"[page] {date_str} page={page_num} rows={len(anns)} total_window_rows={len(all_rows)}")
        page_num += 1
        wait = random.uniform(sleep_min, sleep_max)
        log(f"[sleep] {wait:.2f}s")
        time.sleep(wait)

    return all_rows


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["announcement_id"]).reset_index(drop=True)
        merged.to_csv(path, index=False)
    else:
        df.drop_duplicates(subset=["announcement_id"]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental CNINFO announcement crawler (resumable)")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-csv", default="raw/cninfo_announcements.csv")
    parser.add_argument("--checkpoint", default="raw/.checkpoints/cninfo_incremental.json")
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--sleep-min", type=float, default=0.8)
    parser.add_argument("--sleep-max", type=float, default=1.6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--universe-file", default="", help="if set, crawl per symbol in this file")
    parser.add_argument("--window", choices=["day", "month"], default="day", help="use month windows with --universe-file")
    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    if start > end:
        raise ValueError("start-date must be <= end-date")

    out_csv = Path(args.output_csv)
    ckpt = Path(args.checkpoint)

    symbols: list[str] = []
    if args.universe_file:
        symbols = load_universe(Path(args.universe_file))
        if args.window == "month":
            window_fn = monthrange
        else:
            window_fn = daterange
    else:
        window_fn = daterange

    full_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    checkpoint = load_checkpoint(ckpt)
    resume_symbol_idx = 0
    if symbols and checkpoint.get("last_symbol"):
        try:
            resume_symbol_idx = symbols.index(str(checkpoint["last_symbol"]))
        except ValueError:
            resume_symbol_idx = 0

    symbol_loop = symbols if symbols else [""]
    total_rows = 0
    log(
        f"[start] symbols={len(symbols) or 'ALL'} window={args.window if symbols else 'day'} "
        f"output={out_csv} checkpoint={ckpt}"
    )

    with requests.Session() as session:
        for sym_idx, sym in enumerate(symbol_loop):
            if sym_idx < resume_symbol_idx:
                continue

            if sym_idx == resume_symbol_idx and checkpoint.get("last_end"):
                sym_start = datetime.strptime(checkpoint["last_end"], "%Y-%m-%d")
            else:
                sym_start = full_start
            windows = window_fn(sym_start, end)
            if not windows:
                continue

            for ws, we in windows:
                rows = fetch_window(
                    session=session,
                    start=ws,
                    end=we,
                    page_size=args.page_size,
                    sleep_min=args.sleep_min,
                    sleep_max=args.sleep_max,
                    retries=args.retries,
                    timeout=args.timeout,
                    stock_code=sym,
                )
                append_csv(out_csv, rows)
                total_rows += len(rows)

                payload = {
                    "last_end": we.strftime("%Y-%m-%d"),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_rows": len(rows),
                }
                if sym:
                    payload["last_symbol"] = sym
                save_checkpoint(ckpt, payload)
                log(f"[window-saved] {ws.date()}~{we.date()} sym={sym or 'ALL'} rows={len(rows)} total_rows={total_rows}")

    log(f"[done] appended_rows={total_rows}, output={out_csv}")


if __name__ == "__main__":
    main()
