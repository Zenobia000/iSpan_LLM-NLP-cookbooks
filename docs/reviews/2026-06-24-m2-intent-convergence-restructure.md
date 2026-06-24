# M2「意圖收斂」模組重構紀錄（2026 / gpt-5+）

Date: 2026-06-24

## 背景

模型升到 gpt-5+ 後，2023–24 的「咒語型」prompt 技巧已過時甚至有害：推理模型已內化 CoT，
顯式「Let's think step by step」冗餘；few-shot 不再提升推理（只剩格式對齊）；推理深度改由
API `reasoning_effort`/`verbosity` 控制。典範從 Prompt Engineering → Context → Harness Engineering。

課程設計 spec 早已欽定 M2 的「減法哲學／寫規格而非長提示詞」，但舊 notebook 仍停在技巧目錄。
本次把 M2 對齊 spec 既定意圖。**使用者決策**：深度重構 + 過時咒語技巧直接移除（不留 deprecated）。

## 變更：6 本 → 4 本

| 新檔 | 處置 | 重點 |
|------|------|------|
| `01-prompt-basics.ipynb` | 重寫 | CTCO（Context→Task→Constraints→Output）取代 5W1H 主架構；減法哲學；長 prompt 退化對照 demo；避免 ALL-CAPS/"YOU MUST" |
| `02-few-shot-and-reasoning.ipynb` | 由 `02-cot-reasoning` 改名瘦身 | few-shot=格式對齊（非推理增強）；推理交給 `reasoning_effort`（cross-link M1）；內心 OS/XML 保留；列出已淘汰咒語 |
| `03-prompt-chaining.ipynb` | 補強 | 定位為 spec 分解前奏 + agent(M5) 橋；新增程式化「分而治之」取代 least-to-most 咒語 |
| `04-spec-writing.ipynb` | 由 `05-spec-writing-template` 升格 | 模組高潮；新增 business rule vs model rule、prompt→context→harness 演進敘事、銜接 M5 `AGENT.md` |

**直接刪除**：
- `06-tree-of-thought-deprecated.ipynb`（整本；ToT 已被推理模型取代）。
- `04-prompt-integration-usecase.ipynb`（分類屬 M3、LLM-judge 屬 M7，留 M2 稀釋主軸並重複；few-shot 價值已由新 02 自足 demo 承接）。
- `05-spec-writing-template.ipynb` / `02-cot-reasoning.ipynb`（被上述新檔取代）。

## 直接移除的過時技巧（不再教）
顯式 CoT 當推薦咒語、"Take a deep breath"、詳盡 5-why 連鎖、self-consistency-via-prompt、Tree-of-Thought。
各在 `02-few-shot-and-reasoning` 以一張「為何 2026 過時」對照表標示，供辨識。

## 保留並重框的核心技巧（2026 仍有效）
明確指示/CTCO 結構、角色/system、delimiter/XML、輸出格式、few-shot（格式對齊）、
chaining（工作流/spec 分解）、spec 撰寫（business vs model rule）。

## 推理(CoT)分工
- **M1 `03-reasoning-thinking`**：推理深度的 API 控制（`reasoning_effort`/`budget_tokens`/`thinking_budget`）與不可見性風險（三家對照表）。
- **M2 `02-few-shot-and-reasoning`**：意圖層要不要手動引導推理（few-shot 格式對齊、內心 OS、何時別寫「一步一步思考」）。
- 兩本互加 cross-link，不重複。

## 同步更新
- `tests/test_openai_2026_static.py`（`MODERN_OPENAI_NOTEBOOKS`）：移除 `02-cot-reasoning`、`04-prompt-integration-usecase`；新增 `02-few-shot-and-reasoning`、`04-spec-writing`。
- `prompt-engineering/README.md` M2 模組地圖。
- 課程 spec `docs/superpowers/specs/2026-06-23-...curriculum-design.md`：M2 實作清單（ToT 由「降級」改為「刪除」、CoT 併入 few-shot 章）。

## 驗證
- `python -m pytest tests/ -q` → 11 passed（含 30 本 responses-first guardrail）。
- 全 02 模組 notebook JSON 有效、code cell 語法零錯。
- 無內嵌圖片受影響（02 模組原無含圖 cell；ToT 含圖隨整本刪除，已確認無他處引用）。
- 全程未實際呼叫 API。
