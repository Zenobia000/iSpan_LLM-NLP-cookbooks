# iSpan LLM-NLP Cookbooks

<p align="center">
  <img src="./assets/hero.png" alt="AI 可控性工程：把不可控的 LLM 沿控制權階梯逐層收斂成可驗收的系統" width="100%">
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Course: 2026](https://img.shields.io/badge/curriculum-2026-green.svg)](./prompt-engineering/)

**把不可控的 LLM，逐層收斂成可驗收、可維護的 AI 系統** — iSpan 學院 LLM / NLP 實戰教材庫。

以 Jupyter Notebook 為主，涵蓋 HuggingFace 微調、LangChain / LangGraph 應用、RAG、Agent 與多智能體編排。主課程軸線為 [**AI 可控性工程**](./prompt-engineering/)（控制權階梯 01–08），其餘目錄為並行或補充學習路徑。

---

## 目錄

- [快速開始](#快速開始)
- [課程地圖](#課程地圖)
- [學習路徑](#學習路徑)
- [目錄結構](#目錄結構)
- [技術棧](#技術棧)
- [貢獻與授權](#貢獻與授權)

---

## 快速開始

主課程 `prompt-engineering/` 以 [uv](https://docs.astral.sh/uv/) 管理依賴，5 分鐘內可跑起第一個 notebook：

```bash
git clone https://github.com/Zenobia000/iSpan_LLM-NLP-cookbooks.git
cd iSpan_LLM-NLP-cookbooks/prompt-engineering

uv sync
cp .env.example .env    # 填入 OPENAI / ANTHROPIC / GEMINI 等 API 金鑰
uv run jupyter lab
```

從 [`01-uncontrollability/`](./prompt-engineering/01-uncontrollability/) 依序進入控制權階梯。模組說明與版本鎖定見 [`prompt-engineering/README.md`](./prompt-engineering/README.md)。

**前置需求**：Python 3.11+、至少一組 LLM API 金鑰。其餘子專案（HuggingFace、LangChain 舊線）依各自 README 以 `pip install` 安裝。

---

## 課程地圖

```
                    AI 可控性工程（主軸）
                   prompt-engineering/
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  HuggingFace         LangChain ×         多智能體 &
  模型與微調          LangGraph 2026       長文寫作
  HuggingFace_        LangChain_           Multi-agent-
  scratch/            LangGraph_2026/      system/
```

| 路徑 | 定位 | 入口 |
|------|------|------|
| [`prompt-engineering/`](./prompt-engineering/) | **主課程** — 控制權階梯：意圖 → 結構 → 知識 → 行為 → 協作 → 校準 | [README](./prompt-engineering/README.md) |
| [`Langchain_scratch/LangChain_LangGraph_2026/`](./Langchain_scratch/LangChain_LangGraph_2026/) | LangChain v1 + LangGraph v1 兩冊實戰（2026 API） | [README](./Langchain_scratch/LangChain_LangGraph_2026/README.md) |
| [`HuggingFace_scratch/`](./HuggingFace_scratch/) | Transformers 組件、NLP 任務、PEFT、量化微調 | 從 `01-Component/` 開始 |
| [`Langchain_scratch/Multi-agent-system/`](./Langchain_scratch/Multi-agent-system/) | LangGraph 深度研究、多智能體長文寫作（2026 主線） | [README](./Langchain_scratch/Multi-agent-system/README.md) |

### 主課程：控制權階梯

| 模組 | 收斂層 | 重點 |
|------|--------|------|
| `01-uncontrollability` | — | 取樣、不確定性、多模態 |
| `02-intent-convergence` | prompt → spec | 指示工程、few-shot、spec 撰寫 |
| `03-structured-output` | 結構 | JSON mode、function calling |
| `04-knowledge-rag` | 知識 | embedding、RAG、reranking |
| `05-agent-harness` | 行為 | function calling、guardrails、prompt injection、Responses API |
| `06-multi-agent` | 協作 | Agents SDK、deep search |
| `07-calibration-eval` | 校準 | 評估、回饋迴圈 |
| `08-capstone` | 整合 | 可控問答助理 |

---

## 學習路徑

依目標選一條主線，不必一次讀完整個 repo。

| 你是… | 建議路徑 | 預估 |
|--------|----------|------|
| LLM 應用開發者 | `prompt-engineering/` 01 → 08 | 4–6 週 |
| LangChain / Agent 工程師 | `LangChain_LangGraph_2026/` 第一冊 → 第二冊 | 4–6 週 |
| NLP / 微調工程師 | `HuggingFace_scratch/` 01-Component → PEFT → kbits | 6–8 週 |
| 多智能體 / 研究寫作 | `Multi-agent-system/` LangGraph 深度研究 + 長文寫作 | 2–4 週 |

進階串接：完成主課程 04（RAG）後接 LangGraph 第二冊 M06；完成 05（agent harness）後接 Multi-agent-system。

---

## 目錄結構

```
iSpan_LLM-NLP-cookbooks/
├── prompt-engineering/          # 主課程（uv + uv.lock）
├── HuggingFace_scratch/         # HF 組件、任務、PEFT、量化
├── Langchain_scratch/
│   ├── LangChain_LangGraph_2026/   # 2026 LangChain / LangGraph 兩冊
│   ├── Multi-agent-system/         # Deep Research、長文寫作
│   └── streamlit_resource/
├── Slides/                      # 課程投影片與評估資源
├── scripts/                     # 維護工具（如 model 字串升級）
└── tests/                       # 2026 API 靜態守門測試
```

完整子目錄說明請見各模組 README，不在此展開整棵樹。

---

## 技術棧

| 類別 | 主要依賴 |
|------|----------|
| LLM SDK | OpenAI、Anthropic、Google GenAI（2026 寫法） |
| 框架 | LangChain ≥1.3、LangGraph ≥1.2、HuggingFace Transformers |
| Agent | OpenAI Agents SDK、function calling、guardrails |
| RAG | ChromaDB、embedding、reranking |
| 微調 | PEFT（LoRA / IA3）、QLoRA 4-bit |
| UI | Streamlit、Gradio |

主課程鎖定版本見 [`prompt-engineering/pyproject.toml`](./prompt-engineering/pyproject.toml) 與 [`uv.lock`](./prompt-engineering/uv.lock)。

---

## 貢獻與授權

**貢獻**：Fork → 特性分支 → PR。Notebook 變更請先做 JSON 與 code cell syntax 驗證再提交。

**問題回報**：[GitHub Issues](https://github.com/Zenobia000/iSpan_LLM-NLP-cookbooks/issues)

**授權**：[MIT License](./LICENSE)

**維護**：iSpan 資訊教育中心

---

<details>
<summary>更新紀錄</summary>

**2026-06-24**

- 啟動《AI 可控性工程》課程重構，確立控制權階梯 01–08 主軸
- `prompt-engineering` 依控制權階梯分為 8 模組、notebook 英文化命名
- 升級全 notebook model 字串，確立 `prompt-engineering/` 為單一主課程
- 新增 `LangChain_LangGraph_2026/` 2026 版兩冊課程

</details>
