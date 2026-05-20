from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests


def log(msg: str) -> None:
    print(msg, flush=True)


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": "https://www.cninfo.com.cn/",
}


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_index": -1}
    return json.loads(path.read_text())


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def safe_filename(symbol: str, ts: str, announcement_id: str, url: str) -> str:
    suffix = Path(urlparse(url).path).suffix or ".bin"
    sym = (symbol or "UNK").replace("/", "_")
    t = (ts or "unknown").replace(":", "-").replace(" ", "_")
    aid = str(announcement_id or "na")
    return f"{sym}__{t}__{aid}{suffix}"


def file_path_for_row(row: pd.Series, out_dir: Path) -> Path:
    symbol = str(row.get("symbol", "UNK"))
    ts = str(row.get("timestamp", "unknown"))
    announcement_id = str(row.get("announcement_id", "na"))
    url = str(row.get("url", "") or "")
    return out_dir / safe_filename(symbol, ts, announcement_id, url)


def load_ok_records(output_csv: Path) -> dict[str, dict[str, Any]]:
    if not output_csv.exists():
        return {}
    out = pd.read_csv(output_csv)
    if "announcement_id" not in out.columns:
        return {}
    ok_mask = out.get("download_status", pd.Series(dtype=str)).astype(str) == "ok"
    ok_df = out.loc[ok_mask]
    return {str(r["announcement_id"]): dict(r) for _, r in ok_df.iterrows()}


def compute_resume_index(
    df: pd.DataFrame,
    out_dir: Path,
    output_csv: Path,
    skip_existing: bool,
) -> int:
    """First input index that still needs work (file missing and no ok record)."""
    ok_records = load_ok_records(output_csv)
    for i in range(len(df)):
        row = df.iloc[i]
        aid = str(row.get("announcement_id", ""))
        url = str(row.get("url", "") or "")
        if not url:
            continue
        fp = file_path_for_row(row, out_dir)
        has_file = fp.exists() and fp.stat().st_size > 0
        has_ok = aid in ok_records
        if skip_existing and has_file:
            continue
        if has_ok and has_file:
            continue
        if has_ok and not has_file:
            return i
        if not has_ok:
            return i
    return len(df)


def sync_checkpoint(
    df: pd.DataFrame,
    out_dir: Path,
    output_csv: Path,
    ckpt_path: Path,
    skip_existing: bool,
) -> int:
    resume_idx = compute_resume_index(df, out_dir, output_csv, skip_existing=skip_existing)
    last_index = resume_idx - 1
    save_checkpoint(
        ckpt_path,
        {
            "last_index": last_index,
            "updated_at": pd.Timestamp.now().isoformat(),
            "status": "synced",
            "resume_index": resume_idx,
        },
    )
    log(f"[sync-checkpoint] resume_index={resume_idx} last_index={last_index}")
    return resume_idx


def request_with_retry(session: requests.Session, url: str, retries: int, timeout: int) -> requests.Response:
    last_err: Exception | None = None
    for i in range(retries + 1):
        try:
            r = session.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as err:  # noqa: BLE001
            last_err = err
            if i >= retries:
                break
            wait = min(10.0, 1.2 * (2**i)) + random.uniform(0.1, 0.7)
            log(f"[retry] attempt={i + 1}/{retries + 1} backoff={wait:.2f}s url={url} err={err}")
            time.sleep(wait)
    raise RuntimeError(f"download failed: {url}, err={last_err}")


def html_to_text(html: str) -> str:
    text = html
    for token in ["<br>", "<br/>", "<br />", "</p>", "</div>", "</li>", "</tr>"]:
        text = text.replace(token, "\n")
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def flush_output(records: list[dict[str, Any]], output_csv: Path) -> None:
    if not records:
        return
    out_df = pd.DataFrame(records)
    if "announcement_id" in out_df.columns:
        out_df = out_df.drop_duplicates(subset=["announcement_id"], keep="last")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)


def build_retry_indices(df: pd.DataFrame, output_csv: Path) -> list[int]:
    if not output_csv.exists():
        return list(range(len(df)))
    out = pd.read_csv(output_csv)
    if "announcement_id" not in out.columns or "download_status" not in out.columns:
        return list(range(len(df)))
    err_ids = set(
        out.loc[out["download_status"].astype(str) == "error", "announcement_id"].astype(str).tolist()
    )
    if not err_ids:
        return []
    id_to_idx = {str(row["announcement_id"]): i for i, row in df.iterrows()}
    return sorted(id_to_idx[aid] for aid in err_ids if aid in id_to_idx)


def process_row(
    session: requests.Session,
    row: pd.Series,
    out_dir: Path,
    *,
    skip_existing: bool,
    extract_html_text: bool,
    retries: int,
    timeout: int,
) -> dict[str, Any]:
    url = str(row.get("url", "") or "")
    symbol = str(row.get("symbol", "UNK"))
    ts = str(row.get("timestamp", "unknown"))
    announcement_id = str(row.get("announcement_id", "na"))
    file_path = file_path_for_row(row, out_dir)

    status = "ok"
    error_msg = ""
    content_text = ""
    http_status = None

    if skip_existing and file_path.exists() and file_path.stat().st_size > 0:
        log(f"[skip-existing] ann_id={announcement_id} file={file_path.name}")
        if extract_html_text and file_path.suffix.lower() in {".html", ".htm", ".txt"}:
            try:
                content_text = html_to_text(file_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception as e:  # noqa: BLE001
                status = "partial"
                error_msg = f"extract_text_failed:{e}"
        rec = dict(row)
        rec.update(
            {
                "download_status": status,
                "download_error": error_msg,
                "download_http_status": http_status,
                "local_file": str(file_path),
                "content_text": content_text,
            }
        )
        return rec

    try:
        resp = request_with_retry(session, url, retries=retries, timeout=timeout)
        http_status = resp.status_code
        file_path.write_bytes(resp.content)

        ctype = (resp.headers.get("Content-Type") or "").lower()
        if extract_html_text and ("text/html" in ctype or file_path.suffix.lower() in {".html", ".htm", ".txt"}):
            try:
                content_text = html_to_text(resp.text)
            except Exception as e:  # noqa: BLE001
                status = "partial"
                error_msg = f"extract_text_failed:{e}"
    except Exception as e:  # noqa: BLE001
        status = "error"
        error_msg = str(e)

    rec = dict(row)
    rec.update(
        {
            "download_status": status,
            "download_error": error_msg,
            "download_http_status": http_status,
            "local_file": str(file_path) if file_path.exists() else "",
            "content_text": content_text,
        }
    )
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CNINFO announcement files from raw list")
    parser.add_argument("--input-csv", default="raw/cninfo_announcements.csv")
    parser.add_argument("--download-dir", default="raw/cninfo_files")
    parser.add_argument("--output-csv", default="raw/cninfo_announcements_with_content.csv")
    parser.add_argument("--checkpoint", default="raw/.checkpoints/cninfo_content_fetch.json")
    parser.add_argument("--sleep-min", type=float, default=0.6)
    parser.add_argument("--sleep-max", type=float, default=1.4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means no limit")
    parser.add_argument("--flush-every", type=int, default=1, help="flush output csv every N processed rows")
    parser.add_argument("--extract-html-text", action="store_true", help="extract text for html/txt files")
    parser.add_argument("--skip-existing", action="store_true", help="skip download when local file already exists")
    parser.add_argument(
        "--sync-checkpoint",
        action="store_true",
        help="align checkpoint to first missing item (fixes checkpoint ahead of downloaded files)",
    )
    parser.add_argument(
        "--sync-checkpoint-only",
        action="store_true",
        help="run --sync-checkpoint then exit (repair progress without downloading)",
    )
    parser.add_argument("--retry-errors", action="store_true", help="only retry rows with download_status=error")
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"input csv not found: {in_path}")

    df = pd.read_csv(in_path)
    if "url" not in df.columns:
        raise ValueError("input csv must contain url column")

    out_dir = Path(args.download_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    existing_out = Path(args.output_csv)

    if args.sync_checkpoint or args.sync_checkpoint_only:
        sync_checkpoint(df, out_dir, existing_out, ckpt_path, skip_existing=args.skip_existing)
    if args.sync_checkpoint_only:
        log("[done] checkpoint synced only")
        return

    records: list[dict[str, Any]] = []
    if existing_out.exists():
        old = pd.read_csv(existing_out)
        records = old.to_dict(orient="records")

    if args.retry_errors:
        indices = build_retry_indices(df, existing_out)
        log(f"[retry-errors] indices={len(indices)}")
    else:
        ckpt = load_checkpoint(ckpt_path)
        start_idx = int(ckpt.get("last_index", -1)) + 1
        if args.max_rows > 0:
            end_idx = min(len(df), start_idx + args.max_rows)
        else:
            end_idx = len(df)
        indices = list(range(start_idx, end_idx))
        log(f"[start] start_idx={start_idx} end_idx={end_idx} input_rows={len(df)} output={existing_out}")

    if not indices:
        log("[done] nothing to process")
        return

    processed = 0
    with requests.Session() as session:
        for i in indices:
            row = df.iloc[i]
            url = str(row.get("url", "") or "")
            if not url:
                log(f"[skip] idx={i} empty url")
                save_checkpoint(
                    ckpt_path,
                    {
                        "last_index": i,
                        "updated_at": pd.Timestamp.now().isoformat(),
                        "status": "skip_empty_url",
                    },
                )
                continue

            rec = process_row(
                session,
                row,
                out_dir,
                skip_existing=args.skip_existing,
                extract_html_text=args.extract_html_text,
                retries=args.retries,
                timeout=args.timeout,
            )
            symbol = str(row.get("symbol", "UNK"))
            announcement_id = str(row.get("announcement_id", "na"))
            status = rec.get("download_status", "")
            log(f"[item] idx={i} symbol={symbol} ann_id={announcement_id} status={status}")

            # upsert by announcement_id
            aid = str(row.get("announcement_id", ""))
            replaced = False
            for j, old in enumerate(records):
                if str(old.get("announcement_id", "")) == aid:
                    records[j] = rec
                    replaced = True
                    break
            if not replaced:
                records.append(rec)

            if not args.retry_errors:
                save_checkpoint(
                    ckpt_path,
                    {
                        "last_index": i,
                        "updated_at": pd.Timestamp.now().isoformat(),
                        "status": status,
                        "last_file": rec.get("local_file", ""),
                    },
                )

            processed += 1
            if processed % max(1, args.flush_every) == 0:
                flush_output(records, existing_out)
                log(f"[flush] processed={processed} saved_partial={existing_out}")

            wait = random.uniform(args.sleep_min, args.sleep_max)
            log(f"[sleep] {wait:.2f}s")
            time.sleep(wait)

    flush_output(records, existing_out)

    out_df = pd.read_csv(existing_out) if existing_out.exists() else pd.DataFrame()
    ok = int((out_df.get("download_status", pd.Series(dtype=str)) == "ok").sum()) if len(out_df) else 0
    err = int((out_df.get("download_status", pd.Series(dtype=str)) == "error").sum()) if len(out_df) else 0
    log(f"[done] rows={len(out_df)}, ok={ok}, error={err}, output={existing_out}")


if __name__ == "__main__":
    main()
