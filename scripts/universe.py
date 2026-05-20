from __future__ import annotations

from pathlib import Path


def normalize_symbol_code(raw: str) -> str:
    s = raw.strip().upper()
    if not s or s.startswith("#"):
        return ""
    # 000001.SZ / 000001,平安银行 / 000001
    code = s.split(",")[0].split()[0]
    if "." in code:
        code = code.split(".")[0]
    return code.zfill(6)


def load_universe(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"universe file not found: {path}")
    codes: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        code = normalize_symbol_code(line)
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    if not codes:
        raise ValueError(f"no symbols parsed from universe file: {path}")
    return codes


def to_akshare_symbol(code: str) -> str:
    """000001 -> 000001.SZ / 600036.SH"""
    c = normalize_symbol_code(code)
    if c.startswith(("5", "6", "9")):
        return f"{c}.SH"
    return f"{c}.SZ"
