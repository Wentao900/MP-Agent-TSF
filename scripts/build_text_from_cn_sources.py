from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POS_WORDS = {"增长", "上调", "中标", "回购", "增持", "突破", "盈利", "利好", "改善", "创新高"}
NEG_WORDS = {"下滑", "下调", "减持", "亏损", "违约", "问询", "处罚", "风险", "利空", "暴跌"}


def score_sentiment(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    pos = sum(1 for w in POS_WORDS if w in text)
    neg = sum(1 for w in NEG_WORDS if w in text)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)


def score_relevance(title: str, content: str, symbol: str) -> float:
    t = title if isinstance(title, str) else ""
    c = content if isinstance(content, str) else ""
    s = symbol if isinstance(symbol, str) else ""

    score = 0.4
    if s and (s in t or s in c):
        score += 0.3
    if len(t) >= 10:
        score += 0.1
    if len(c) >= 50:
        score += 0.1

    for kw in ["业绩", "公告", "预告", "分红", "并购", "回购", "问询", "监管"]:
        if kw in t or kw in c:
            score += 0.02

    return float(np.clip(score, 0.0, 1.0))


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ts_code": "symbol",
        "code": "symbol",
        "published_at": "timestamp",
        "pub_time": "timestamp",
        "datetime": "timestamp",
        "headline": "title",
        "news_title": "title",
        "article": "content",
        "news_content": "content",
    }
    out = df.rename(columns=rename_map).copy()

    required = ["timestamp", "symbol"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing required columns in source csv: {missing}")

    if "title" not in out.columns:
        out["title"] = ""
    if "content" not in out.columns:
        out["content"] = ""

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["symbol"] = out["symbol"].astype(str)
    out["title"] = out["title"].astype(str)
    out["content"] = out["content"].astype(str)

    out = out.dropna(subset=["timestamp", "symbol"]).reset_index(drop=True)

    # 去重：symbol + timestamp + title
    out["_dup_key"] = (
        out["symbol"].astype(str)
        + "|"
        + out["timestamp"].astype(str)
        + "|"
        + out["title"].str.slice(0, 80)
    )
    out = out.drop_duplicates("_dup_key").drop(columns=["_dup_key"]).reset_index(drop=True)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    merged_text = (out["title"].fillna("") + " " + out["content"].fillna(""))
    out["sentiment"] = merged_text.map(score_sentiment)
    out["relevance"] = [score_relevance(t, c, s) for t, c, s in zip(out["title"], out["content"], out["symbol"])]
    out["event_strength"] = (out["sentiment"].abs() * out["relevance"]).clip(0.0, 1.0)

    out = out[["timestamp", "symbol", "sentiment", "relevance", "event_strength"]].copy()
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pipeline-ready text.csv from CN announcements/news exports")
    parser.add_argument("--input-csv", required=True, help="Raw text csv path (公告/快讯聚合结果)")
    parser.add_argument("--output-csv", default="outputs/data/text.csv")
    args = parser.parse_args()

    src = Path(args.input_csv)
    if not src.exists():
        raise FileNotFoundError(f"input csv not found: {src}")

    raw = pd.read_csv(src)
    text = normalize_text_columns(raw)
    text = build_features(text)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text.to_csv(out_path, index=False)

    print(f"saved text dataset: {out_path}")
    print(f"rows={len(text)}, symbols={text['symbol'].nunique()}")


if __name__ == "__main__":
    main()
