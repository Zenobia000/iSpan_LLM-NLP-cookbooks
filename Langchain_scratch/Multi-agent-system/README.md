# Multi-Agent System 教學專案

本資料夾已重構為 **2026 LangChain / LangGraph 主線**。

長文寫作課程已改成 LangChain / LangGraph notebooks。新的可維護實作請從：

```text
framework/LangGraph_2026/
```

## 新主線：LangGraph Deep Research

`framework/LangGraph_2026` 參考 LangChain 官方
[`open_deep_research`](https://github.com/langchain-ai/open_deep_research) 的設計，但不直接 vendoring 外部專案。課程版保留核心結構：

1. `plan_research`：把使用者問題轉成 research brief 與子任務。
2. `researcher`：針對每個子任務搜尋、整理 evidence。
3. `write_report`：把 findings 合成最終報告。

這比 CrewAI 更適合作為 2026 課程主線，原因是：

- 使用 LangChain v1 `init_chat_model`，模型供應商切換集中管理。
- 使用 LangGraph `StateGraph` 與 `Send` 表達 fan-out/fan-in。
- 每個節點都是普通 Python function，方便測試、替換與上線。
- 可以逐步擴充 Tavily、MCP、LangSmith、human-in-the-loop，而不是被框架黑箱綁住。

## 資料夾定位

```text
Multi-agent-system/
├── framework/
│   └── LangGraph_2026/     # 新主線：LangChain v1 + LangGraph
└── 應用專案-多智能體長文寫作/  # Notebook 教學：LCEL → LangGraph → Deep Research
```

## 快速開始

```bash
cd Langchain_scratch/Multi-agent-system/framework/LangGraph_2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python cli.py "請研究 2026 年企業採用 LangGraph 做多智能體系統的優缺點" --offline-search
```

`--offline-search` 會使用 placeholder evidence，適合課堂講 graph 流程。正式研究請設定 `TAVILY_API_KEY`。
