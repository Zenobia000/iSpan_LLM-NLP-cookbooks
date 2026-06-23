"""守門測試:prompt-engineering 的 code cell 不得殘留已淘汰 model id。"""
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "prompt-engineering"

# 與 scripts/upgrade_models.py 的 MODEL_MAP keys 一致(裸 gpt-4 不列入,刻意保留)。
FORBIDDEN = [
    "gpt-3.5-turbo-16k", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301", "gpt-3.5-turbo-0125", "gpt-3.5-turbo",
    "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-turbo-preview",
    "gpt-4-turbo", "gpt-4-0613",
]

# 豁免:需人工處理(人工 model-aware 改寫),不在機械基線範圍。
EXEMPT = {
    "810-fine-tune-with-synthetic-data.ipynb",
}


def _code_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            parts.append("".join(cell.get("source", [])))
    return "\n".join(parts)


def test_no_legacy_models_in_code_cells():
    offenders: dict[str, list[str]] = {}
    for path in sorted(NB_DIR.glob("*.ipynb")):
        if path.name in EXEMPT:
            continue
        text = _code_text(path)
        hits = [m for m in FORBIDDEN if f'"{m}"' in text or f"'{m}'" in text]
        if hits:
            offenders[path.name] = sorted(set(hits))
    assert not offenders, f"仍殘留已淘汰 model:{offenders}"
