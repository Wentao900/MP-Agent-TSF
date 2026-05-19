from __future__ import annotations

import argparse
import json
import random
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
    # 轻量提取（不依赖 bs4）；用于可选摘要，不追求完美
    text = html
    for token in ["<br>", "<br/>", "<br />", "</p>", "</div>", "</li>", "</tr>"]:
        text = text.replace(token, "\n")
    import re

    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    parser.add_argument("--extract-html-text", action="store_true", help="extract text for html/txt files")
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
    ckpt = load_checkpoint(ckpt_path)
    start_idx = int(ckpt.get("last_index", -1)) + 1

    if args.max_rows > 0:
        end_idx = min(len(df), start_idx + args.max_rows)
    else:
        end_idx = len(df)

    records: list[dict[str, Any]] = []
    existing_out = Path(args.output_csv)
    if existing_out.exists():
        old = pd.read_csv(existing_out)
        records = old.to_dict(orient="records")

    log(f"[start] start_idx={start_idx} end_idx={end_idx} input_rows={len(df)} output={existing_out}")

    with requests.Session() as session:
        for i in range(start_idx, end_idx):
            row = df.iloc[i]
            url = str(row.get("url", "") or "")
            if not url:
                log(f"[skip] idx={i} empty url")
                save_checkpoint(ckpt_path, {"last_index": i, "updated_at": pd.Timestamp.now().isoformat(), "status": "skip_empty_url"})
                continue

            symbol = str(row.get("symbol", "UNK"))
            ts = str(row.get("timestamp", "unknown"))
            announcement_id = str(row.get("announcement_id", "na"))
            file_name = safe_filename(symbol, ts, announcement_id, url)
            file_path = out_dir / file_name

            status = "ok"
            error_msg = ""
            content_text = ""
            http_status = None

            try:
                resp = request_with_retry(session, url, retries=args.retries, timeout=args.timeout)
                http_status = resp.status_code
                file_path.write_bytes(resp.content)

                ctype = (resp.headers.get("Content-Type") or "").lower()
                if args.extract_html_text and ("text/html" in ctype or file_path.suffix.lower() in {".html", ".htm", ".txt"}):
                    try:
                        content_text = html_to_text(resp.text)
                    except Exception as e:  # noqa: BLE001
                        status = "partial"
                        error_msg = f"extract_text_failed:{e}"
            except Exception as e:  # noqa: BLE001
                status = "error"
                error_msg = str(e)

            log(f"[item] idx={i} symbol={symbol} ann_id={announcement_id} status={status} http={http_status}")

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
            records.append(rec)

            save_checkpoint(
                ckpt_path,
                {
                    "last_index": i,
                    "updated_at": pd.Timestamp.now().isoformat(),
                    "status": status,
                    "last_file": str(file_path),
                },
            )

            if (i - start_idx + 1) % 50 == 0:
                pd.DataFrame(records).drop_duplicates(subset=["announcement_id"], keep="last").to_csv(existing_out, index=False)
                log(f"[flush] processed={i - start_idx + 1} saved_partial={existing_out}")

            wait = random.uniform(args.sleep_min, args.sleep_max)
            log(f"[sleep] {wait:.2f}s")
            time.sleep(wait)

    out_df = pd.DataFrame(records)
    if "announcement_id" in out_df.columns:
        out_df = out_df.drop_duplicates(subset=["announcement_id"], keep="last")
    out_df.to_csv(existing_out, index=False)

    ok = int((out_df.get("download_status", pd.Series(dtype=str)) == "ok").sum())
    err = int((out_df.get("download_status", pd.Series(dtype=str)) == "error").sum())
    log(f"[done] rows={len(out_df)}, ok={ok}, error={err}, output={existing_out}")


if __name__ == "__main__":
    main()
