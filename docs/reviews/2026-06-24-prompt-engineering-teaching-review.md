# prompt-engineering 教學 Code Review 報告（Phase 1 審查）

**日期**: 2026-06-24
**範圍**: `prompt-engineering/` 全 8 模組、39 本 notebook、1 個 `.py`
**方法**: Sonnet 多 agent workflow — 線上查證 19 套件 2026 版本/SDK 寫法 → 逐本審查 → 資料夾彙整
**基準**: 2026 年 SDK 現況；多模型並陳（OpenAI + Claude + Gemini）
**狀態**: 唯讀審查完成，等待核可後進 Phase 2 編輯（圖片不動）

統計：**305 個教學盲點、194 處過時程式碼、109 個高嚴重度、6 本空殼 notebook**。

詳細逐本 findings 見 `scratchpad/findings/*.json`，逐資料夾報告見 `scratchpad/findings/_report__*.md`。

---

## 一、最關鍵判斷（Linus 式）

✅ **值得做**。這份教材結構（控制權階梯）本身是好的，但程式碼層已嚴重 library rot：**至少 12 本 notebook 在 2026 全新環境直接跑不起來**，6 本是空殼。問題不是「理論不完美」，是「學生 clone 下來第一個 cell 就爆」。

【核心數據結構問題】
- 全庫沒有依賴管理（無 `pyproject.toml`/`uv.lock`），靠散落的 `!pip install`（且多半不定版）。這是所有「跑不起來」的根。**先建 uv 定版環境，是所有修正的前提。**

【最大破壞性風險】
- Phase 2 編輯 39 本 notebook 必須**逐 cell 精準改、絕不碰 `[HAS-EMBEDDED-IMAGE]` cell**。已在抽取階段標記出所有含圖 cell。

---

## 二、跨模組系統性問題（每個都是「不合時宜」程式碼）

| # | 問題 | 影響範圍 | 後果 | 2026 正解 |
|---|------|---------|------|-----------|
| S1 | 手刻 `requests.post()` 打 OpenAI 端點，繞過官方 SDK | 01,02,03,04 幾乎全部 | 教壞習慣、無重試、寫法不一 | `from openai import OpenAI; client.responses.create()` |
| S2 | 前 v1.0 `openai.ChatCompletion.create()` / 模組級 `openai.api_key` | 03,05(08,09) | `openai>=2.0` 直接 `AttributeError` | client 實例 + `client.chat.completions.create()` |
| S3 | `from langchain.agents import initialize_agent` / `langchain.document_loaders` | 04(04,06),05(01,03),07(03) | `langchain>=1.0` 移除，`ImportError` | `langchain>=1.3` `create_agent`、`langchain_openai` |
| S4 | ChromaDB 0.x：`Client()` / `create_collection()` / `configure()` | 04(04,05,06),05(04) | `chromadb>=1.0` 崩潰 | `EphemeralClient()`/`PersistentClient()` + `get_or_create_collection` |
| S5 | `import fitz` | 04(02,07),07(01) | PyMuPDF 1.24.3 起棄用 | `import pymupdf` |
| S6 | `tiktoken.get_encoding('cl100k_base')` 硬編碼 | 01(01),04(02) | gpt-4o/5/o 系列用 `o200k_base`，token 數靜默錯誤 | `encoding_for_model('gpt-4o')` |
| S7 | Gemini `gemini-2.0-flash`（2026-06-01 已下線）+ `google-generativeai`（EOL） | 01(05) 每個 cell | 全 404 | `google-genai>=2.0` + `gemini-3.5-flash` / `gemini-2.5-flash` |
| S8 | 棄用 Claude model ID `claude-3-7-sonnet-20250219` | 01(03) 等 | 將被淘汰 | `claude-sonnet-4-6` / `claude-opus-4-8` |
| S9 | Gradio tuple 對話歷史 `for human, assistant in history` | 03(04),05(05,08) | Gradio 6 移除 | dict 格式 `[{'role','content'}]` |
| S10 | Assistants API 標成 Responses API | 05(09-responses-api) | 2026-08-26 sunset、標題文不對題 | 真正的 `client.responses.create()` |
| **S11** | **零多模型並陳**（全 OpenAI-only） | **全 8 模組** | 與「多模型對照」教學目標相違 | 關鍵概念加 Claude/Gemini 對照 cell |
| S12 | WHY 框架普遍缺失、無評估、無收尾 | 全模組 | 學生「會跑不懂為何」 | 每本加動機 cell + 收尾 takeaways |
| S13 | 安全：`eval()` 吃 LLM 輸出、few-shot 資料洩漏、敏感人物生圖 prompt | 05(07)、02(04)、01(04 cell13) | 安全反模式/評估失真/教材不當 | `json.loads()`、train/test 分離、換中性題材 |
| S14 | 雜訊：Colab `userdata` 死碼、硬編碼 Windows 路徑、散落未定版 pip | 全模組 | 非 Colab 學生困惑、路徑爆掉 | 移除死碼、相對路徑、單一 setup cell |

---

## 三、6 本空殼 notebook（P0，全需補寫）

| Notebook | 應補內容 |
|----------|---------|
| `01-uncontrollability/02-sampling-and-uncertainty` | 模組概念核心：溫度/top-p/top-k、隨機取樣為何造成非確定性、緩解策略、三家 provider 參數對照 |
| `02-intent-convergence/05-spec-writing-template` | 模組收斂的 capstone：prompt→spec 模板 |
| `05-agent-harness/10-agent-md-mcp-skills` | agent.md 結構、MCP 協定、skill 生命週期、trust boundary |
| `06-multi-agent/03-cross-model-review` | 跨模型 review pipeline：Claude 寫作 → GPT+Gemini 評審 → 綜合 |
| `07-calibration-eval/02-feedback-loop` | 評估結果回饋改進的閉環 |
| `08-capstone/01-capstone-overview` | 串接 01–07 的整合專題規格與交付標準 |

---

## 四、各模組重點摘要

- **01-uncontrollability**：主題（為何不可控）從未在任何 notebook 建立；05 全 cell Gemini 已下線；02 空殼。**主題框架 + 跑得起來**是重點。
- **02-intent-convergence**：結構連貫但全 raw-HTTP；02 Self-Consistency 聚合把 Python 物件塞進 prompt（壞）；04 few-shot 資料洩漏；05 空殼。
- **03-structured-output**：02、04 在 2026 跑不起來；缺輸出驗證 arc；缺 JSON mode vs tool-calling vs Structured Outputs 決策框架。
- **04-knowledge-rag**：最完整但技術債最重；04–06 全新安裝即壞（langchain+chromadb）；全模組零評估（無 Recall@k/RAGAS）；硬編碼 Windows 路徑。
- **05-agent-harness**：最複雜也腐蝕最重；5 本遞迴 agent 無 `max_turns` 上限（危險）；01/02/03 順序教反（現代→歷史）；09 文不對題；06 缺間接 prompt injection。
- **06-multi-agent**：01 有型別註記 silent bug；03 空殼；全 OpenAI 壟斷（與跨模型前提矛盾）。
- **07-calibration-eval**：01 `fitz`+autoevals import 壞；03 硬編碼私有 ft 模型 ID（404）+ EOL 模型 + 死連結；02 空殼。
- **08-capstone**：純空殼，整個要建。

---

## 五、uv 定版草案（2026 已查證版本）

建議在 `prompt-engineering/` 根建立單一 `pyproject.toml`（uv 管理），各 notebook 開頭只留一句註解指向它，移除散落的 `!pip install`。

```toml
[project]
name = "ispan-prompt-engineering"
version = "2026.6.0"
description = "iSpan AI 可控性工程課程 — prompt-engineering 模組"
requires-python = ">=3.10"
dependencies = [
  # --- LLM providers（多模型並陳）---
  "openai>=2.0,<3",
  "anthropic>=0.111,<1",
  "google-genai>=2.0,<3",
  "openai-agents>=0.17,<1",
  "cohere>=7.0,<8",
  # --- LangChain（1.x，create_agent 跑在 langgraph 上）---
  "langchain>=1.3,<2",
  "langchain-openai>=0.3,<1",
  "langchain-text-splitters>=0.3",
  "langgraph>=1.2,<2",
  # --- RAG / 向量庫 ---
  "chromadb>=1.5,<2",
  "sentence-transformers>=3.0",
  "tiktoken>=0.13,<1",
  # --- 文件解析 ---
  "pymupdf>=1.24,<2",
  "pypdf>=5.0,<7",
  # --- 結構化 / 驗證 ---
  "pydantic>=2.10,<3",
  # --- 工具 / 搜尋 ---
  "tavily-python>=0.7,<1",
  "googlesearch-python>=1.3,<2",
  "numexpr>=2.8,<3",
  # --- UI ---
  "gradio>=6.0,<7",
  # --- 評估 / 觀測 ---
  "braintrust>=0.24,<1",
  "autoevals>=0.0.130,<0.1",
  # --- 基礎 ---
  "python-dotenv>=1.0",
  "requests>=2.31",
  "numpy",
  "pandas>=2.0",
  "scikit-learn",
  "matplotlib",
  "pillow",
  "nest-asyncio",
  "ipykernel",
]
```

**移除/禁用**（不合時宜）：`pysqlite3`/`pysqlite3-binary`（chromadb≥1.5 不需要）、`PyPDF2`（→ pypdf）、`google-generativeai`（EOL → google-genai）、`fitz`（假套件 → pymupdf）。

---

## 六、建議 Phase 2 執行順序

1. **建立 uv 環境**（`pyproject.toml` + `uv.lock`），確立可重現基準 — 所有修正的前提。
2. **P0 跑不起來/空殼**：修 S2–S10 的 import/model 崩潰；補 6 本空殼。
3. **P1 教學缺口**：補 WHY 框架、多模型對照 cell（S11）、評估、安全修正（S13）。
4. **P2 打磨**：移除 Colab 死碼、統一風格、收尾 takeaways、單一 setup cell。

> 每本 notebook 文字+程式碼都升級到 2026 寫法；`[HAS-EMBEDDED-IMAGE]` cell 一律不動，僅改周邊敘述。
