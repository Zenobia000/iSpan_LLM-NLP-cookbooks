# LangGraph 2026 Deep Research

這是多智能體課程的 2026 版主線實作：用 LangChain v1 與 LangGraph 建立 deep research workflow。

設計參考 LangChain 官方 `open_deep_research`：

- 官方版：完整產品級 deep research agent，支援多模型、多 search API、MCP、LangGraph Studio 與 benchmark。
- 課程版：保留核心架構，拿掉產品級複雜度，讓學生看得懂、改得動、能延伸。

## 架構

```text
START
  ↓
plan_research
  ↓ fan-out with Send
researcher(topic A) ┐
researcher(topic B) ├─ findings reducer
researcher(topic C) ┘
  ↓
write_report
  ↓
END
```

## 對應 Open Deep Research 概念

| Open Deep Research | 課程版 |
|---|---|
| `clarify_with_user` | 先省略，保留為延伸練習 |
| `write_research_brief` | `plan_research` |
| `research_supervisor` | `plan_research` + `Send` |
| researcher subgraph | `researcher` 節點 |
| compression | `ResearchFinding` 結構化整理 |
| final report generation | `write_report` |
| MCP / native web search | 先用 Tavily 或 offline fallback |

## 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` 至少設定一個模型供應商金鑰，例如：

```bash
OPENAI_API_KEY=sk-...
COURSE_MODEL=openai:gpt-4o-mini
```

若要使用真實搜尋：

```bash
TAVILY_API_KEY=tvly-...
```

## 執行

離線流程測試：

```bash
python cli.py "請研究 LangGraph 在多智能體系統中的優缺點" --offline-search
```

真實搜尋：

```bash
python cli.py "請研究 LangGraph 在多智能體系統中的優缺點"
```

指定模型：

```bash
python cli.py "請比較 CrewAI 與 LangGraph 的教學取捨" --model openai:gpt-4o-mini
```

## 核心教學點

- `init_chat_model(model_name)`：維持 LangChain 原生模型介面。
- `with_structured_output(ResearchPlan)`：讓 planner 產出可驗證資料。
- `StateGraph(ResearchState)`：把多步流程顯式建模。
- `Send("researcher", {"unit": unit})`：把 research units 並行分派。
- reducer：`findings` 使用 `operator.add` 收斂多個 researcher 結果。

## 下一步延伸

1. 加回 clarification node，讓模糊題目先問使用者。
2. 把 `researcher` 拆成子圖：search → reflect → compress。
3. 加 MCP search tools。
4. 加 LangSmith tracing 與 evaluation dataset。
5. 用 LangGraph checkpointer 做 human-in-the-loop 審核。
