# prompt-engineering 分層與重新命名 Implementation Plan (Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 把扁平的 `prompt-engineering/` 依「控制權階梯」重組成 8 個英文編號子資料夾,所有 notebook 與資料夾**檔名一律英文**,資料檔隨模組搬移並保持路徑可用,離主軸內容封存。

**Architecture:** 以 `git mv`(保留歷史)把既有 notebook 移入 8 個模組資料夾並改英文名;本地資料檔(PDF/jsonl/.py)隨「引用它的 notebook」搬入同資料夾,使裸檔名相對路徑仍可解析;`!wget` 型免動;⭐新章只建佔位 stub;同步守門測試/腳本的 glob(改遞迴)與 README。

**Tech Stack:** git、Python 3.11 標準庫、pytest 9.x。不新增相依。

## Global Constraints

- **檔名一律英文**(資料夾與 .ipynb 皆是);編號前綴 `NN-`,小寫、連字號分隔。
- **8 個模組資料夾**(於 `prompt-engineering/` 下):
  `01-uncontrollability` `02-intent-convergence` `03-structured-output` `04-knowledge-rag` `05-agent-harness` `06-multi-agent` `07-calibration-eval` `08-capstone`
- **移動一律用 `git mv`**(保留歷史),不得 delete+add。
- **資料檔隨模組**:notebook 用裸檔名引用的本地檔,搬入同一模組資料夾即可解析(Jupyter cwd = notebook 所在目錄);`!wget` 下載型不需本地檔。
- **不改 notebook 程式邏輯**:本計畫只動檔案位置/檔名與必要的「本地檔路徑」一致性;model 升級已於 Plan 1 完成,不在此重做。
- **守門測試必須維持有效**:notebook 進子資料夾後,`prompt-engineering/*.ipynb` 非遞迴 glob 會掃不到(假綠)→ 必須改 `rglob`,且 EXEMPT/EXCLUDE 改為重新命名後的微調基底 notebook 檔名。
- 每個 task 結束 commit(Conventional Commits)。分支:`dev`。

## 完整對照(old → new)

| 模組資料夾 | new 檔名 | ← 來源 |
|---|---|---|
| 01-uncontrollability | 01-openai-api-intro.ipynb | 101-start-openai.ipynb |
| | 02-sampling-and-uncertainty.ipynb | ⭐ stub |
| | 03-reasoning-thinking.ipynb | 130--LLM-workshop-thinking.ipynb |
| | 04-multimodal.ipynb | 110--LLM-workshop-multimodal.ipynb |
| | 05-gemini-model-boundaries.ipynb | 115--LLM-workshop-gemini.ipynb |
| 02-intent-convergence | 01-prompt-basics.ipynb | 102-prompt-engineering.ipynb |
| | 02-cot-reasoning.ipynb | 201-CoT-prompt.ipynb |
| | 03-prompt-chaining.ipynb | 207-chaining-prompt.ipynb |
| | 04-prompt-integration-usecase.ipynb | 206-prompt-integration-usecase.ipynb |
| | 05-spec-writing-template.ipynb | ⭐ stub |
| | 06-tree-of-thought-deprecated.ipynb | 208-ToT-prompt.ipynb |
| 03-structured-output | 01-json-mode.ipynb | 103-json-mode.ipynb |
| | 02-function-calling-basics.ipynb | 702-function-calling-basic.ipynb |
| | 03-structured-extraction.ipynb | 706-function-calling-extract.ipynb |
| | 04-classification-gradio.ipynb | 402-gradio_query_classification.ipynb |
| 04-knowledge-rag | 01-embedding.ipynb | 601--LLM-workshop-embedding.ipynb |
| | 02-vanilla-rag.ipynb | 602-varnilla-RAG.ipynb |
| | 03-similarity-and-relevance.ipynb | 605--LLM-workshop-similarity-and-relevance.ipynb |
| | 04-vector-db-rag.ipynb | 604-vector-db-RAG.ipynb |
| | 05-dynamic-few-shot.ipynb | 606--LLM-workshop-dynamic-few-shot.ipynb |
| | 06-advanced-rag.ipynb | 607-advance-RAG.ipynb |
| | 07-pdf-parsing.ipynb | 612--LLM-workshop-pdf-parsing-v2.ipynb |
| 05-agent-harness | 01-function-calling-agents.ipynb | 703-function-calling-agents.ipynb |
| | 02-react-loop.ipynb | 711--LLM-workshop-react.ipynb |
| | 03-langchain-agents.ipynb | 701-langchain-agents.ipynb |
| | 04-function-calling-rag.ipynb | 705-function-calling-rag.ipynb |
| | 05-shop-guardrails.ipynb | 712-function-calling-shop.ipynb |
| | 06-prompt-injection.ipynb | 202--LLM-workshop-prompt-injection.ipynb |
| | 07-plugin-tools.ipynb | 205-plugin-tools.ipynb |
| | 08-chatbot.ipynb | 401-chatbot.ipynb |
| | 09-responses-api.ipynb | 720-assistants-api.ipynb |
| | 10-agent-md-mcp-skills.ipynb | ⭐ stub |
| 06-multi-agent | 01-openai-agents-sdk.ipynb | 970--LLM-workshop-openai-agents-sdk.ipynb |
| | 02-deep-search.ipynb | 721--LLM-workshop-agent-deep-search.ipynb |
| | 03-cross-model-review.ipynb | ⭐ stub |
| 07-calibration-eval | 01-rag-evaluation.ipynb | 610--LLM_workshop_RAG_evaluation.ipynb |
| | 02-feedback-loop.ipynb | ⭐ stub |
| | 03-fine-tuning-synthetic-data.ipynb | 810-fine-tune-with-synthetic-data.ipynb |
| 08-capstone | 01-capstone-overview.ipynb | ⭐ stub |

**封存到 `_archive/prompt-engineering/`**:`501-whisper-summarization.ipynb`、`502-whisper-summarization_longtext.ipynb`、`11.ipynb`、`test.ipynb`。

**資料檔歸屬(隨引用的 notebook)**:
- → `04-knowledge-rag/`:`ntu-111-2.pdf`(602)、`2023台灣產業AI化大調查完整報告.pdf`(604)、`pdfs_data/` 內 `1130205.pdf`/`1130215.pdf`/`1130226.pdf`(607,攤平進資料夾)
- → `05-agent-harness/`:`C11201717_1.pdf`(720)、`function-calling-rag-dependency.py`(705 同資料夾,待確認用途)
- → `07-calibration-eval/`:`1121113.pdf`+`training_data.jsonl`(810)、`1140224.pdf`(610)
- → `_archive/prompt-engineering/data/`:未被任何 notebook 以本地路徑引用、或為 `!wget` 重複下載產物者(`1130219.pdf`、`pdfs_data/1130219.pdf`、`1130513.pdf`、`1140224.pdf.1`、`1140224.pdf.2`、其餘無引用 PDF)。

---

### Task 1: 建立模組資料夾並搬移/改名所有主線 notebook

**Files:** 建立 8 個資料夾;`git mv` 約 30 個 notebook(見對照表)。

- [ ] **Step 1: 建立資料夾**

```bash
cd prompt-engineering
mkdir -p 01-uncontrollability 02-intent-convergence 03-structured-output 04-knowledge-rag 05-agent-harness 06-multi-agent 07-calibration-eval 08-capstone
```

- [ ] **Step 2: git mv 主線 notebook(逐條,保留歷史)**

```bash
cd prompt-engineering
# M1
git mv 101-start-openai.ipynb 01-uncontrollability/01-openai-api-intro.ipynb
git mv 130--LLM-workshop-thinking.ipynb 01-uncontrollability/03-reasoning-thinking.ipynb
git mv 110--LLM-workshop-multimodal.ipynb 01-uncontrollability/04-multimodal.ipynb
git mv 115--LLM-workshop-gemini.ipynb 01-uncontrollability/05-gemini-model-boundaries.ipynb
# M2
git mv 102-prompt-engineering.ipynb 02-intent-convergence/01-prompt-basics.ipynb
git mv 201-CoT-prompt.ipynb 02-intent-convergence/02-cot-reasoning.ipynb
git mv 207-chaining-prompt.ipynb 02-intent-convergence/03-prompt-chaining.ipynb
git mv 206-prompt-integration-usecase.ipynb 02-intent-convergence/04-prompt-integration-usecase.ipynb
git mv 208-ToT-prompt.ipynb 02-intent-convergence/06-tree-of-thought-deprecated.ipynb
# M3
git mv 103-json-mode.ipynb 03-structured-output/01-json-mode.ipynb
git mv 702-function-calling-basic.ipynb 03-structured-output/02-function-calling-basics.ipynb
git mv 706-function-calling-extract.ipynb 03-structured-output/03-structured-extraction.ipynb
git mv 402-gradio_query_classification.ipynb 03-structured-output/04-classification-gradio.ipynb
# M4
git mv 601--LLM-workshop-embedding.ipynb 04-knowledge-rag/01-embedding.ipynb
git mv 602-varnilla-RAG.ipynb 04-knowledge-rag/02-vanilla-rag.ipynb
git mv 605--LLM-workshop-similarity-and-relevance.ipynb 04-knowledge-rag/03-similarity-and-relevance.ipynb
git mv 604-vector-db-RAG.ipynb 04-knowledge-rag/04-vector-db-rag.ipynb
git mv 606--LLM-workshop-dynamic-few-shot.ipynb 04-knowledge-rag/05-dynamic-few-shot.ipynb
git mv 607-advance-RAG.ipynb 04-knowledge-rag/06-advanced-rag.ipynb
git mv 612--LLM-workshop-pdf-parsing-v2.ipynb 04-knowledge-rag/07-pdf-parsing.ipynb
# M5
git mv 703-function-calling-agents.ipynb 05-agent-harness/01-function-calling-agents.ipynb
git mv 711--LLM-workshop-react.ipynb 05-agent-harness/02-react-loop.ipynb
git mv 701-langchain-agents.ipynb 05-agent-harness/03-langchain-agents.ipynb
git mv 705-function-calling-rag.ipynb 05-agent-harness/04-function-calling-rag.ipynb
git mv 712-function-calling-shop.ipynb 05-agent-harness/05-shop-guardrails.ipynb
git mv 202--LLM-workshop-prompt-injection.ipynb 05-agent-harness/06-prompt-injection.ipynb
git mv 205-plugin-tools.ipynb 05-agent-harness/07-plugin-tools.ipynb
git mv 401-chatbot.ipynb 05-agent-harness/08-chatbot.ipynb
git mv 720-assistants-api.ipynb 05-agent-harness/09-responses-api.ipynb
# M6
git mv 970--LLM-workshop-openai-agents-sdk.ipynb 06-multi-agent/01-openai-agents-sdk.ipynb
git mv 721--LLM-workshop-agent-deep-search.ipynb 06-multi-agent/02-deep-search.ipynb
# M7
git mv 610--LLM_workshop_RAG_evaluation.ipynb 07-calibration-eval/01-rag-evaluation.ipynb
git mv 810-fine-tune-with-synthetic-data.ipynb 07-calibration-eval/03-fine-tuning-synthetic-data.ipynb
```

- [ ] **Step 3: 確認皆為 rename 且根目錄已無散落 notebook(除待封存的 4 個)**

```bash
cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks
git status --short prompt-engineering | rg '^R' | wc -l   # 應為 31
ls prompt-engineering/*.ipynb 2>/dev/null                  # 應只剩 11.ipynb 501 502 test.ipynb
```
Expected: 31 個 rename;根目錄僅剩 4 個待封存 notebook。

- [ ] **Step 4: Commit**

```bash
git add -A prompt-engineering
git commit -m "refactor(curriculum): prompt-engineering 主線 notebook 分層並改英文名"
```

---

### Task 2: 搬移本地資料檔至所屬模組,封存無引用者

**Files:** `git mv` PDF/jsonl/.py 至模組資料夾或 `_archive/prompt-engineering/data/`。

- [ ] **Step 1: 搬移被引用的本地資料檔到所屬模組**

```bash
cd prompt-engineering
# M4 RAG 用
git mv ntu-111-2.pdf 04-knowledge-rag/
git mv "2023台灣產業AI化大調查完整報告.pdf" 04-knowledge-rag/
git mv pdfs_data/1130205.pdf 04-knowledge-rag/
git mv pdfs_data/1130215.pdf 04-knowledge-rag/
git mv pdfs_data/1130226.pdf 04-knowledge-rag/
# M5 用
git mv C11201717_1.pdf 05-agent-harness/
git mv function-calling-rag-dependency.py 05-agent-harness/
# M7 用
git mv 1121113.pdf 07-calibration-eval/
git mv training_data.jsonl 07-calibration-eval/
git mv 1140224.pdf 07-calibration-eval/
```

- [ ] **Step 2: 封存無本地引用 / 重複下載產物的資料檔**

```bash
cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks
mkdir -p _archive/prompt-engineering/data
cd prompt-engineering
git mv 1130219.pdf ../_archive/prompt-engineering/data/ 2>/dev/null || true
git mv pdfs_data/1130219.pdf ../_archive/prompt-engineering/data/1130219-pdfs_data.pdf 2>/dev/null || true
git mv 1130513.pdf ../_archive/prompt-engineering/data/ 2>/dev/null || true
git mv 1140224.pdf.1 ../_archive/prompt-engineering/data/ 2>/dev/null || true
git mv 1140224.pdf.2 ../_archive/prompt-engineering/data/ 2>/dev/null || true
# 若 pdfs_data 已空則移除空目錄
rmdir pdfs_data 2>/dev/null || true
```

- [ ] **Step 3: 驗證每個移動後 notebook 的「本地檔」引用都能在同資料夾解析**

```bash
cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks/prompt-engineering
for d in 0*/; do
  for nb in "$d"*.ipynb; do
    [ -e "$nb" ] || continue
    # 抽出 notebook 中引用的本地 .pdf/.jsonl 裸檔名(排除 http/wget 行)
    refs=$(python3 -c "
import json,sys,re
nb=json.load(open('$nb',encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type']!='code': continue
    src=''.join(c['source'])
    for m in re.findall(r'[\"\x27]([^\"\x27]+\.(?:pdf|jsonl))[\"\x27]', src):
        if m.startswith('http') or '/' in m: continue
        print(m)
")
    for f in $refs; do
      [ -e "$d$f" ] || echo "MISSING: $nb 引用 $f 但 $d$f 不存在"
    done
  done
done
echo "檢查完成(無 MISSING 行即為通過)"
```
Expected: 無 `MISSING` 行。若有,該 notebook 的本地檔需補搬入其資料夾,或將該裸路徑改為 `_archive/...` 相對路徑;在報告中列出處理方式。

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(curriculum): 資料檔隨模組搬移，無引用者封存"
```

---

### Task 3: 封存離主軸 notebook

**Files:** `git mv` 4 個 notebook 至 `_archive/prompt-engineering/`。

- [ ] **Step 1: 搬移**

```bash
cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks
mkdir -p _archive/prompt-engineering
cd prompt-engineering
git mv 501-whisper-summarization.ipynb ../_archive/prompt-engineering/
git mv 502-whisper-summarization_longtext.ipynb ../_archive/prompt-engineering/
git mv 11.ipynb ../_archive/prompt-engineering/
git mv test.ipynb ../_archive/prompt-engineering/
```

- [ ] **Step 2: 確認根目錄已無散落 notebook**

```bash
cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks
ls prompt-engineering/*.ipynb 2>/dev/null && echo "仍有殘留" || echo "根目錄已淨空 ✓"
```
Expected: `根目錄已淨空 ✓`。

- [ ] **Step 3: Commit**

```bash
git commit -am "chore(curriculum): 封存離主軸 notebook(whisper/草稿/測試)"
```

---

### Task 4: 建立 ⭐ 佔位 stub notebook

**Files:** 建立 6 個 stub:
`01-uncontrollability/02-sampling-and-uncertainty.ipynb`、`02-intent-convergence/05-spec-writing-template.ipynb`、`05-agent-harness/10-agent-md-mcp-skills.ipynb`、`06-multi-agent/03-cross-model-review.ipynb`、`07-calibration-eval/02-feedback-loop.ipynb`、`08-capstone/01-capstone-overview.ipynb`。

- [ ] **Step 1: 以腳本產生 stub**

```bash
cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks
python3 - <<'PY'
import json
from pathlib import Path

SPEC = "docs/superpowers/specs/2026-06-23-ai-controllability-engineering-curriculum-design.md"
stubs = {
    "prompt-engineering/01-uncontrollability/02-sampling-and-uncertainty.ipynb":
        ("Sampling 與不確定性 (Sampling & Uncertainty)", "M1"),
    "prompt-engineering/02-intent-convergence/05-spec-writing-template.ipynb":
        ("Spec 撰寫模板 (Spec Writing Template)", "M2"),
    "prompt-engineering/05-agent-harness/10-agent-md-mcp-skills.ipynb":
        ("agent.md、MCP 與 Skills (Harness Context)", "M5"),
    "prompt-engineering/06-multi-agent/03-cross-model-review.ipynb":
        ("跨模型審核 (Cross-Model Review)", "M6"),
    "prompt-engineering/07-calibration-eval/02-feedback-loop.ipynb":
        ("自動化 Feedback Loop", "M7"),
    "prompt-engineering/08-capstone/01-capstone-overview.ipynb":
        ("整合專題 Capstone 說明", "M8"),
}
for path, (title, mod) in stubs.items():
    md = (
        f"# {title}\n\n"
        f"> **本章內容待建（placeholder）。**\n>\n"
        f"> 對應課程設計 {mod}，詳見 [`{SPEC}`](/{SPEC})。\n>\n"
        f"> 此 stub 僅保留章節編號與順序；實際教學內容於該模組的改寫計畫中建立。\n"
    )
    nb = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)}],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("created", path)
PY
```

- [ ] **Step 2: 驗證 stub 為合法 notebook**

```bash
python3 -c "import json,glob; [json.load(open(f,encoding='utf-8')) for f in glob.glob('prompt-engineering/0*/0*.ipynb')]; print('all valid')"
```
Expected: `all valid`。

- [ ] **Step 3: Commit**

```bash
git add prompt-engineering
git commit -m "feat(curriculum): 新增 6 個待建章節佔位 stub"
```

---

### Task 5: 修復守門測試與升級腳本的 glob(改遞迴)

**Files:**
- Modify: `tests/test_model_baseline.py`
- Modify: `scripts/upgrade_models.py`

**Interfaces:** notebook 已進子資料夾,非遞迴 glob 會掃不到(假綠);需改遞迴,且豁免清單改為重新命名後的微調 notebook 路徑名。

- [ ] **Step 1: 測試改遞迴 + 更新豁免**

`tests/test_model_baseline.py`:
- 將 `NB_DIR.glob("*.ipynb")` 改為 `NB_DIR.rglob("*.ipynb")`。
- 將 `EXEMPT` 改為比對重新命名後的微調 notebook 檔名:把 `"810-fine-tune-with-synthetic-data.ipynb"` 換成 `"03-fine-tuning-synthetic-data.ipynb"`(微調基底仍刻意保留 gpt-3.5 系列,故維持豁免)。移除已不存在的 `960` 條目(若仍殘留)。
- 由於 stub 與封存目錄也會被 `rglob` 掃到:stub 無 model 字串、`_archive` 不在 `NB_DIR`(`NB_DIR` 指向 `prompt-engineering`,`_archive` 在 repo 根,不受影響),無需額外排除。

- [ ] **Step 2: 腳本改遞迴 + 更新 EXCLUDE**

`scripts/upgrade_models.py`:
- 將 `NB_DIR.glob("*.ipynb")` 改為 `NB_DIR.rglob("*.ipynb")`。
- `EXCLUDE` 的 `"810-fine-tune-with-synthetic-data.ipynb"` 改為 `"03-fine-tuning-synthetic-data.ipynb"`;移除已不存在的 `960` 條目(若殘留)。

- [ ] **Step 3: 跑守門測試(必須綠)與腳本乾跑(應 0 處)**

```bash
cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks
python3 -m pytest tests/test_model_baseline.py -v
python3 scripts/upgrade_models.py
```
Expected: 測試 PASS;乾跑 `合計:0 處`(基線已升級且微調豁免)。若測試紅,代表某子資料夾 notebook 殘留舊 model 或豁免名未對齊——依輸出修正。

- [ ] **Step 4: Commit**

```bash
git add tests/test_model_baseline.py scripts/upgrade_models.py
git commit -m "test(curriculum): 守門測試與升級腳本改遞迴 glob 並對齊改名後豁免"
```

---

### Task 6: 更新 README 反映 prompt-engineering 新分層

**Files:** Modify `README.md`

- [ ] **Step 1: 讀 README,定位 prompt-engineering 區塊**

先 `Read README.md`,找到目錄樹中 `📝 prompt-engineering/` 段(Plan 1 已將其多智能體行改為 `(970)`)。

- [ ] **Step 2: 以新的 8 模組分層取代舊的 (1xx-9xx) 條列**

把舊的 `├── 基礎技術 (101-103) ...` 等粗分類行,替換為 8 個英文模組資料夾結構:

```
├── 📝 prompt-engineering/               # AI 可控性工程主課程（控制權階梯）
│   ├── 01-uncontrollability/            # 不可控的根源
│   ├── 02-intent-convergence/           # 意圖收斂：prompt → spec
│   ├── 03-structured-output/            # 結構收斂：JSON / function calling
│   ├── 04-knowledge-rag/                # 知識收斂：RAG
│   ├── 05-agent-harness/                # 行為收斂：agent harness
│   ├── 06-multi-agent/                  # 協作收斂：多 agent
│   ├── 07-calibration-eval/             # 校準層：驗證與評估
│   └── 08-capstone/                     # 整合專題
```

- [ ] **Step 3: 更新「最後更新」區塊**

日期改為 `2026-06-24`,新增一條:`prompt-engineering 依控制權階梯分層為 8 個英文模組資料夾、notebook 全面英文化命名`。

- [ ] **Step 4: 驗證並 Commit**

```bash
cd /home/sunny/python_workstation/github/iSpan_LLM-NLP-cookbooks
rg -n '01-uncontrollability|08-capstone' README.md && echo "README 已含新結構 ✓"
git add README.md
git commit -m "docs(curriculum): README 反映 prompt-engineering 8 模組分層"
```

---

## Self-Review

**Spec coverage:** 對照 spec §7 各模組——M1-M8 皆有對應資料夾與 notebook 對照(Task 1);⭐新章以 stub 保留槽位(Task 4);資料檔路徑一致性(Task 2);離主軸封存(Task 3);守門測試在 reorg 後維持有效(Task 5,關鍵);README 同步(Task 6)。

**Placeholder scan:** 無 TBD;所有 step 附完整指令或腳本。Task 6 因 README 內容需依實際行定位,已要求先 Read 再精準替換。

**Type/一致性:** 微調 notebook 改名後檔名 `03-fine-tuning-synthetic-data.ipynb` 在 Task 1(rename)、Task 5(EXEMPT/EXCLUDE)一致;資料夾英文名在對照表、Task 1、Task 6 一致。

**風險備註:** Task 1-4 期間守門測試暫時失效(非遞迴 glob 掃不到子資料夾),於 Task 5 還原並驗證綠——這是刻意排序,Task 5 為安全網修復點。

## 後續(非本計畫)
基線(Plan 1)+ 分層(Plan 2)完成後,逐一進行各模組改寫計畫(M1-M8),建立 ⭐ stub 的實際內容並做 spec §10 的全面 SDK 程式碼更新。
