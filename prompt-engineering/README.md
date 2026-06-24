# prompt-engineering — AI 可控性工程主課程

依「控制權階梯」編排的 8 個模組，主軸為**把不可控的 LLM 逐層收斂成可控、可驗收的系統**。
範例採**多模型並陳**（OpenAI / Claude / Gemini），程式碼以 2026 年 SDK 寫法為準。

## 模組地圖（控制權階梯）

| 模組 | 收斂層 | 重點 |
|------|--------|------|
| `01-uncontrollability` | — | 取樣/溫度為何不可控、推理、多模態 |
| `02-intent-convergence` | prompt → spec | 明確指示/CTCO、few-shot 與推理、chaining、spec 撰寫（高潮） |
| `03-structured-output` | 結構收斂 | JSON mode、function calling、Structured Outputs |
| `04-knowledge-rag` | 知識收斂 | embedding、RAG、向量庫、reranking、進階 RAG、Agentic RAG |
| `05-agent-harness` | 行為收斂 | function calling agent、guardrails、prompt injection、Responses API |
| `06-multi-agent` | 協作收斂 | OpenAI Agents SDK、deep search、跨模型互審 |
| `07-calibration-eval` | 校準層 | RAG 評估、回饋迴圈、微調合成資料 |
| `08-capstone` | 整合 | 串接 01–07 的可控問答助理 |

## 環境設定（uv）

本模組以 [uv](https://docs.astral.sh/uv/) 管理依賴，版本已鎖定於 `uv.lock`（2026-06 查證）。

```bash
cd prompt-engineering

# 1. 建立虛擬環境並安裝鎖定版本依賴
uv sync

# 2. 設定金鑰
cp .env.example .env   # 編輯填入 OPENAI/ANTHROPIC/GEMINI 等金鑰

# 3. 啟動 Jupyter（於 uv 環境內）
uv run jupyter lab
```

> notebook 內的 `!pip install` 已改為註解（套件由 `pyproject.toml` 統一定版）；
> 若在 Google Colab 單獨執行某本 notebook，再按註解指示手動安裝該本所需套件即可。

## 主要技術版本（2026）

`openai>=2.26` · `anthropic>=0.111` · `google-genai>=2.0`（取代已 EOL 的 google-generativeai）·
`openai-agents>=0.17` · `langchain>=1.3` / `langgraph>=1.2`（`create_agent`）·
`chromadb>=1.5` · `pymupdf`（取代 `fitz`）· `gradio>=6` · `pydantic>=2.10`。

鎖定版本見 [`pyproject.toml`](pyproject.toml) 與 [`uv.lock`](uv.lock)。
