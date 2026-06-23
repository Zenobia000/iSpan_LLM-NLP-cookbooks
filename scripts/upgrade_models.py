#!/usr/bin/env python3
"""升級 prompt-engineering notebook 內的過時 model id(引號錨定、外科手術式)。

dry-run(預設)只報告;加 --apply 才寫檔。僅替換引號包裹的完整 id,
markdown 內未加引號的歷史提及不受影響。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "prompt-engineering"

MODEL_MAP: dict[str, str] = {
    "gpt-3.5-turbo-16k": "gpt-4o-mini",
    "gpt-3.5-turbo-1106": "gpt-4o-mini",
    "gpt-3.5-turbo-0613": "gpt-4o-mini",
    "gpt-3.5-turbo-0301": "gpt-4o-mini",
    "gpt-3.5-turbo-0125": "gpt-4o-mini",
    "gpt-3.5-turbo": "gpt-4o-mini",
    "gpt-4-0125-preview": "gpt-4o",
    "gpt-4-1106-preview": "gpt-4o",
    "gpt-4-turbo-preview": "gpt-4o",
    "gpt-4-turbo": "gpt-4o",
    "gpt-4-0613": "gpt-4o",
}

# 機械基線跳過:需人工處理或即將刪除。
EXCLUDE = {
    "810-fine-tune-with-synthetic-data.ipynb",
    "960--LLM-workshop-swarm.ipynb",
}

# 原始 .ipynb 中:雙引號為 \" ;單引號為 ' 。兩端必為同型,且 id 不含引號,
# 故以「開頭分隔符 + id + 同型結尾分隔符」精準匹配,長短 id 順序不影響正確性。
_DELIMS = ('\\"', "'")


def upgrade_text(raw: str) -> tuple[str, int]:
    count = 0
    for old, new in MODEL_MAP.items():
        for delim in _DELIMS:
            pattern = re.compile(re.escape(delim) + re.escape(old) + re.escape(delim))
            raw, n = pattern.subn(delim + new + delim, raw)
            count += n
    return raw, count


def process(path: Path, apply: bool) -> int:
    raw = path.read_text(encoding="utf-8")
    new, count = upgrade_text(raw)
    if apply and count:
        path.write_text(new, encoding="utf-8")
    return count


def main() -> int:
    apply = "--apply" in sys.argv
    grand = 0
    for path in sorted(NB_DIR.glob("*.ipynb")):
        if path.name in EXCLUDE:
            continue
        n = process(path, apply)
        if n:
            verb = "UPDATED" if apply else "WOULD UPDATE"
            print(f"{verb} {path.name}: {n} 處")
            grand += n
    tail = "" if apply else "(dry-run;加 --apply 才寫檔)"
    print(f"合計:{grand} 處 {tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
