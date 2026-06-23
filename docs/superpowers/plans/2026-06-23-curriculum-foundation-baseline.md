# 課程基線整備 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 為《AI 可控性工程》課程建立乾淨、現代化、重組過的基線——升級過時 model 字串、移除已淘汰範例、封存非主軸內容、同步 README。

**Architecture:** 以引號錨定的「原始文字替換」升級 notebook 中的 model id(避免 JSON round-trip 造成全檔 diff),用一支可乾跑(dry-run)的 Python 腳本批次處理,並以 pytest 守門測試保證程式碼 cell 無殘留舊 model。封存採 `git mv` 保留歷史,只處理「整棵且無保留子項」的子樹,部分目錄封存延至對應模組計畫。

**Tech Stack:** Python 3.11(僅用標準庫 `json`/`re`/`pathlib`)、pytest 9.x、git。不新增第三方相依。

## Global Constraints

- **僅修改 code cell**,不動 markdown cell(歷史性提及具教學價值)。
- **Model 升級對照表(逐字)**:
  - `gpt-3.5-turbo`、`gpt-3.5-turbo-1106`、`gpt-3.5-turbo-0613`、`gpt-3.5-turbo-0301`、`gpt-3.5-turbo-0125`、`gpt-3.5-turbo-16k` → `gpt-4o-mini`
  - `gpt-4-0125-preview`、`gpt-4-1106-preview`、`gpt-4-turbo-preview`、`gpt-4-turbo`、`gpt-4-0613` → `gpt-4o`
  - **維持不動**:`gpt-4o`、`gpt-4o-mini`、`gpt-4.1`、`gpt-4.1-mini`、`gpt-4.1-nano`、`o4-mini`、`text-embedding-3-small`、`text-embedding-3-large`、`whisper-1`、裸 `gpt-4`
- **豁免檔(機械替換跳過)**:`810-fine-tune-with-synthetic-data.ipynb`(微調基底模型語意需人工處理)、`960--LLM-workshop-swarm.ipynb`(將刪除)。
- **人工改寫(非機械)**:`101-start-openai.ipynb` 的 `gpt-3.5-turbo-instruct`(completions 端點已淘汰)。
- **引號錨定**:替換僅作用於引號包裹的完整 id(原始位元組中雙引號為 `\"`、單引號為 `'`),確保 `gpt-3.5-turbo-instruct`、`gpt-4o` 等不被部分改寫。
- 每個 task 結束都要 commit,使用 Conventional Commits。
- 工作分支:`dev`。

---

### Task 1: 建立 model 基線守門測試(失敗測試)

**Files:**
- Create: `tests/test_model_baseline.py`

**Interfaces:**
- Consumes: 無(只讀 `prompt-engineering/*.ipynb`)
- Produces: 測試 `test_no_legacy_models_in_code_cells`,供 Task 3 驗證替換完成。

- [ ] **Step 1: 撰寫失敗測試**

```python
# tests/test_model_baseline.py
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

# 豁免:需人工處理,不在機械基線範圍。
EXEMPT = {"810-fine-tune-with-synthetic-data.ipynb"}


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
```

- [ ] **Step 2: 執行測試,確認失敗**

Run: `cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks && python3 -m pytest tests/test_model_baseline.py -v`
Expected: FAIL,assert 訊息列出多個 notebook(如 `702-function-calling-basic.ipynb: ['gpt-3.5-turbo']`)。

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_baseline.py
git commit -m "test(curriculum): 新增 model 基線守門測試"
```

---

### Task 2: 撰寫 model 升級腳本(可乾跑)

**Files:**
- Create: `scripts/upgrade_models.py`

**Interfaces:**
- Consumes: `prompt-engineering/*.ipynb` 原始檔。
- Produces: CLI 腳本;`--apply` 寫入,不帶旗標為 dry-run。函式 `upgrade_text(raw: str) -> tuple[str, int]`。

- [ ] **Step 1: 撰寫腳本**

```python
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
```

- [ ] **Step 2: 乾跑驗證報告**

Run: `python3 scripts/upgrade_models.py`
Expected: 列出多支 notebook 與替換數,結尾 `合計:N 處 (dry-run;加 --apply 才寫檔)`,N > 50。**不寫檔**。

- [ ] **Step 3: 確認 dry-run 未改動工作區**

Run: `git status --porcelain prompt-engineering/`
Expected: 僅可能的無輸出(乾跑不寫檔);若有輸出代表誤寫,需檢查。

- [ ] **Step 4: Commit**

```bash
git add scripts/upgrade_models.py
git commit -m "build(curriculum): 新增 model 升級腳本(引號錨定、可乾跑)"
```

---

### Task 3: 套用升級並讓守門測試轉綠

**Files:**
- Modify: `prompt-engineering/*.ipynb`(腳本自動處理,排除豁免檔)

**Interfaces:**
- Consumes: Task 2 的 `scripts/upgrade_models.py`、Task 1 的測試。
- Produces: 已升級的 notebook 基線。

- [ ] **Step 1: 套用替換**

Run: `python3 scripts/upgrade_models.py --apply`
Expected: 同 Task 2 列表,但前綴為 `UPDATED`。

- [ ] **Step 2: 執行守門測試,確認轉綠**

Run: `python3 -m pytest tests/test_model_baseline.py -v`
Expected: PASS。

- [ ] **Step 3: 抽查 diff 為外科手術式(只動 model id 行)**

Run: `git diff --stat prompt-engineering/ | tail -5` 與 `git diff prompt-engineering/702-function-calling-basic.ipynb`
Expected: 變更僅落在含 model id 的行,無大規模重排;`"gpt-3.5-turbo"` → `"gpt-4o-mini"`。

- [ ] **Step 4: 確認 notebook 仍為合法 JSON**

Run: `python3 -c "import json,glob; [json.load(open(f,encoding='utf-8')) for f in glob.glob('prompt-engineering/*.ipynb')]; print('all valid')"`
Expected: `all valid`。

- [ ] **Step 5: Commit**

```bash
git add prompt-engineering/
git commit -m "refactor(curriculum): 全面升級 notebook model 字串至現行模型"
```

---

### Task 4: 人工改寫 101 的 instruct/completions 範例

**Files:**
- Modify: `prompt-engineering/101-start-openai.ipynb`(含 `gpt-3.5-turbo-instruct` 的 code cell)

**Interfaces:**
- Consumes: 無
- Produces: 以現行 chat API 取代已淘汰的 completions/instruct 範例。

- [ ] **Step 1: 定位 instruct 範例 cell**

Run: `python3 -c "import json; nb=json.load(open('prompt-engineering/101-start-openai.ipynb',encoding='utf-8')); [print(i, repr(''.join(c['source'])[:200])) for i,c in enumerate(nb['cells']) if c['cell_type']=='code' and 'instruct' in ''.join(c['source'])]"`
Expected: 印出含 `gpt-3.5-turbo-instruct` 的 cell 索引與內容(通常為 `client.completions.create(model="gpt-3.5-turbo-instruct", prompt=...)`)。

- [ ] **Step 2: 將該 cell 改寫為 chat API**

把 completions 寫法改為現行 chat 寫法(以實際變數名對齊原 cell)。範本:

```python
# 舊式 completions/instruct 已淘汰,改用 chat completions
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "請用一句話介紹大型語言模型"}],
)
print(response.choices[0].message.content)
```

並在同 cell 上方或相鄰 markdown 補一句說明:`completions/instruct 端點已於 2025 後淘汰,統一改用 chat completions`。

- [ ] **Step 3: 確認無殘留 instruct 且 JSON 合法**

Run: `python3 -c "import json; t=''.join(s for c in json.load(open('prompt-engineering/101-start-openai.ipynb',encoding='utf-8'))['cells'] if c['cell_type']=='code' for s in c['source']); print('instruct殘留' if 'gpt-3.5-turbo-instruct' in t else 'clean')"`
Expected: `clean`。

- [ ] **Step 4: Commit**

```bash
git add prompt-engineering/101-start-openai.ipynb
git commit -m "refactor(curriculum): 101 改用 chat API 取代淘汰的 instruct 端點"
```

---

### Task 5: 刪除 960 Swarm 並解決 610 重複檔

**Files:**
- Delete: `prompt-engineering/960--LLM-workshop-swarm.ipynb`
- Delete(其一): `prompt-engineering/610__LLM_workshop_RAG_evaluation.ipynb` 或 `610--LLM_workshop_RAG_evaluation.ipynb`

**Interfaces:**
- Consumes: 無
- Produces: 移除已淘汰 Swarm 範例與重複的 RAG evaluation notebook。

- [ ] **Step 1: 刪除 960 Swarm**

Run: `git rm prompt-engineering/960--LLM-workshop-swarm.ipynb`
Expected: `rm 'prompt-engineering/960--LLM-workshop-swarm.ipynb'`。
理由:Swarm 為實驗性、已被 Agents SDK(`970`)取代。

- [ ] **Step 2: 比對 610 兩個重複檔**

Run: `diff <(python3 -c "import json;print(json.dumps(json.load(open('prompt-engineering/610--LLM_workshop_RAG_evaluation.ipynb',encoding='utf-8'))['cells']))") <(python3 -c "import json;print(json.dumps(json.load(open('prompt-engineering/610__LLM_workshop_RAG_evaluation.ipynb',encoding='utf-8'))['cells']))") && echo IDENTICAL || echo DIFFER`
Expected: `IDENTICAL` 或 `DIFFER`。

- [ ] **Step 3: 移除多餘者(保留 `--` 命名,與其他 notebook 一致)**

若 Step 2 為 `IDENTICAL`:
Run: `git rm prompt-engineering/610__LLM_workshop_RAG_evaluation.ipynb`
若為 `DIFFER`:先 `git diff --no-index` 檢視差異,保留內容較完整者、`git rm` 另一個,並在 commit message 註明判斷依據。

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(curriculum): 刪除已淘汰 960 Swarm 與重複的 610 RAG evaluation"
```

---

### Task 6: 封存非主軸內容至 `_archive/`

**Files:**
- Move(git mv,保留歷史):
  - `HuggingFace_scratch/05-Distributed Training/` → `_archive/HuggingFace_scratch/05-Distributed Training/`
  - `Langchain_scratch/Slides/` → `_archive/Langchain_scratch/Slides/`
  - `Langchain_scratch/langchain_framework/Course/` → `_archive/Langchain_scratch/langchain_framework/Course/`

**Interfaces:**
- Consumes: 無
- Produces: 主結構移除「整棵且無保留子項」的非主軸內容。

> **範圍說明(非無聲縮減)**:spec 第 9 節另列 `HuggingFace_scratch/01-Component`(僅留 `02tokenizer`)與 `02-Adv-tasks`(僅留 `01-finetune_optimize`)的「部分封存」。因其需保留個別子項,留待 **M1 / M7 模組計畫**在實際驗證該 notebook 時一併處理,避免此處碎片化。本 task 只封存無保留子項的整棵子樹。

- [ ] **Step 1: 建立 _archive 並移動(路徑含空白需引號)**

```bash
mkdir -p "_archive/HuggingFace_scratch" "_archive/Langchain_scratch/langchain_framework"
git mv "HuggingFace_scratch/05-Distributed Training" "_archive/HuggingFace_scratch/05-Distributed Training"
git mv "Langchain_scratch/Slides" "_archive/Langchain_scratch/Slides"
git mv "Langchain_scratch/langchain_framework/Course" "_archive/Langchain_scratch/langchain_framework/Course"
```

- [ ] **Step 2: 確認移動結果與歷史保留**

Run: `git status --short | head -20 && echo "---" && ls _archive/HuggingFace_scratch _archive/Langchain_scratch`
Expected: 顯示 `R`(renamed)項目;`_archive` 下出現三個被封存目錄。

- [ ] **Step 3: 加一份封存說明**

Create `_archive/README.md`:

```markdown
# _archive

此目錄存放與《AI 可控性工程》課程主軸無關、但保留歷史的內容(以 `git mv` 移入,可隨時還原)。

- `HuggingFace_scratch/05-Distributed Training/` — 分散式訓練(基礎設施,超出課程範圍)
- `Langchain_scratch/Slides/` — 框架導向投影片
- `Langchain_scratch/langchain_framework/Course/` — LangChain 框架本身課程(與「不以框架為重」原則相衝)

課程設計見 `docs/superpowers/specs/2026-06-23-ai-controllability-engineering-curriculum-design.md`。
```

- [ ] **Step 4: Commit**

```bash
git add _archive/README.md
git commit -m "chore(curriculum): 封存非主軸內容至 _archive(分散式訓練/slides/框架課)"
```

---

### Task 7: 同步 README(外科手術式)

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 5(960 已刪)、Task 6(封存路徑)
- Produces: README 反映新狀態並連結課程 spec。完整改寫留待全模組落地後的收尾。

- [ ] **Step 1: 概述後加課程設計指引**

在第 5 行(MECE 段落)之後插入:

```markdown

> 📐 **課程重構中**:本庫正依《AI 可控性工程:從使用者到指揮官》重新編排,主軸為「可控性」。設計文件見 [`docs/superpowers/specs/2026-06-23-ai-controllability-engineering-curriculum-design.md`](docs/superpowers/specs/2026-06-23-ai-controllability-engineering-curriculum-design.md)。部分非主軸內容已移至 [`_archive/`](_archive/)。
```

- [ ] **Step 2: 更新 prompt-engineering 多智能體行(移除已刪的 960)**

舊(README line 101):
```
│   └── 多智能體 (960, 970)             # Swarm, SDK
```
新:
```
│   └── 多智能體 (970)                  # OpenAI Agents SDK
```

- [ ] **Step 3: 從目錄樹移除已封存子樹,改列封存區**

刪除 README 中以下三段樹狀列(分散式訓練 line 61-63、langchain Course line 67-70、Slides line 103),並在「專案目錄結構」程式碼區塊結尾(line 104 的 ```` ``` ```` 之前)加入:

```
│
└── 🗄️ _archive/                         # 已封存(非主軸,保留歷史)
    ├── HuggingFace_scratch/05-Distributed Training/
    ├── Langchain_scratch/Slides/
    └── Langchain_scratch/langchain_framework/Course/
```

- [ ] **Step 4: 更新「最後更新」區塊**

把 README 末段「🔄 最後更新」日期改為 `2026-06-23`,內容條列改為:
```markdown
- 啟動《AI 可控性工程》課程重構(spec 見 docs/superpowers/specs/)
- 升級全 notebook model 字串、移除淘汰範例(960 Swarm、610 重複檔)
- 封存非主軸內容至 _archive/
```

- [ ] **Step 5: 驗證 README 內部連結與一致性**

Run: `cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks && test -f docs/superpowers/specs/2026-06-23-ai-controllability-engineering-curriculum-design.md && test -d _archive && rg -n "960" README.md || echo "OK: 960 已從 README 移除"`
Expected: spec 與 `_archive` 皆存在;README 不再出現 `960`(或僅出現在無關處,需人工確認)。

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs(curriculum): README 反映封存與課程重構，移除 960 引用"
```

---

## Self-Review

**Spec coverage(對照 spec 各節):**
- spec §10 全面程式碼更新 → 本計畫僅做「機械基線」(model 字串)那一層;SDK 寫法深改屬各模組計畫,已於 spec §11 Step 2 明列,**非本計畫範圍**(刻意)。
- spec §8 刪除 960 → Task 5 ✓;208 ToT 降級、720 Assistants 改寫屬 M2/M5 模組計畫(非基線)。
- spec §9 封存 → Task 6 ✓(整棵子樹);部分目錄封存明確延至 M1/M7,已註記非無聲縮減。
- spec §11 Step 0(機械基線)+ Step 1(封存+README)→ 本計畫 Task 1-7 完整覆蓋 ✓。

**Placeholder scan:** 無 TBD/TODO;所有 code step 附完整程式碼或精確指令。Task 4/7 含「依實際變數/行號對齊」屬必要的人工判斷,已提供定位指令與範本。

**Type consistency:** `FORBIDDEN`(測試)與 `MODEL_MAP` keys(腳本)一致(裸 `gpt-4` 兩處皆刻意排除);`upgrade_text` 簽章一致;封存路徑於 Task 6 與 Task 7 一致。

## 後續計畫(非本計畫)

基線落地後,逐一撰寫模組改寫計畫:M1(含 HF tokenizer 部分封存)、M2(CoT 改寫、ToT 降級)、M3、M4(期中專題)、M5(harness、720→Responses)、M6(多 agent、cross-model review)、M7(含 HF 微調部分封存)、M8(capstone)。每份計畫含 spec §10 的全面程式碼更新。
