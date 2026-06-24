# 第二冊 — LangGraph 編排：把積木組成有狀態的系統

第一冊的 LCEL 管線是**單向、一次性**的：輸入進去、輸出出來。但真實應用需要
**迴圈**（Agent 反覆思考）、**分支**（依情況走不同路）、**記憶**（記得上一輪對話）、
**人介入**（危險操作前先讓人審核）、**多 Agent 協作**。

這些都需要一個**有狀態的編排層**——這就是 LangGraph。

> 關鍵認知：第一冊用的 `create_agent`，底層就是一張 LangGraph 圖。
> 本冊帶你掀開蓋子，自己畫圖、控制每一步。

## 模組進度與依賴關係

```
M00 為什麼需要 LangGraph
      │  （從 chain 到 graph 的心智轉換）
      ▼
M01 StateGraph 基礎
      │  State / node / edge / START / END
      ▼
M02 狀態與 Reducer
      │  Annotated / add_messages / MessagesState
      ▼
M03 條件路由與 Command
      │  條件邊 / 迴圈 / Command(goto+update)
      ▼
M04 持久化與記憶  ──────────┐
      │  checkpointer / thread │ 有了記憶才談得上
      ▼                        │ 中斷與多輪
M05 Human-in-the-loop  ───────┘
      │  interrupt / 時光旅行
      ▼
M06 多智能體系統
      │  subgraph / supervisor / handoff
      ▼
M07 串流、可觀測與整合專案
         stream_mode / LangSmith / Capstone
```

## 各模組一句話

| 模組 | 你會學到 | 核心 API |
|------|----------|----------|
| **M00** | 何時該用 graph 而非 chain；圖的心智模型 | （概念為主） |
| **M01** | 用 State + node + edge 畫第一張圖 | `StateGraph` / `add_node` / `add_edge` |
| **M02** | state 如何更新；reducer 為何能消除特殊情況 | `Annotated` / `add_messages` |
| **M03** | 讓圖會分支、會迴圈 | 條件邊 / `Command` |
| **M04** | 讓圖記得對話、可中斷續跑 | `InMemorySaver` / `thread_id` |
| **M05** | 在關鍵步驟暫停等人類審核 / 修改 | `interrupt` / `Command(resume=...)` |
| **M06** | 多個 Agent 分工協作 | subgraph / supervisor 模式 |
| **M07** | 串流輸出、用 LangSmith 觀測、整合成 Capstone | `stream_mode` / tracing |

## 先備知識

請先完成第一冊（至少 M00～M04、M06）。本冊假設你已熟悉 `init_chat_model`、
訊息物件、`@tool` 與 `create_agent`。

## 學習方式

LangGraph 的核心是「**資料結構（State）對了，流程就簡單了**」。每個模組請特別
留意 `README.md` 裡對 **State 設計** 的說明——這是寫好圖的關鍵，比記 API 更重要。
