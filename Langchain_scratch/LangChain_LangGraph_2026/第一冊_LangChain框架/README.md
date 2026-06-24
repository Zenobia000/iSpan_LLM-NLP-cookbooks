# 第一冊 — LangChain 框架：認識每一塊積木

本冊目標：把 LangChain v1 提供的核心組件**逐一拆開**，理解每個組件的職責，
並學會用 LCEL 把它們**疊起來**。學完本冊，你能獨立做出 RAG 問答與會用工具的 Agent。

## 模組進度與依賴關係

```
M00 環境與心智模型
      │  （建好環境、理解 LangChain 在解決什麼）
      ▼
M01 對話模型與訊息  ──────────────┐
      │  init_chat_model / Messages │
      ▼                             │
M02 PromptTemplate 與結構化輸出     │
      │  ChatPromptTemplate         │ 這三塊是「呼叫一次模型」的基本功
      ▼                             │
M03 LCEL 與 Runnable 管線  ─────────┘
      │  prompt | model | parser
      ▼
M04 工具與工具呼叫
      │  @tool / bind_tools（讓模型能呼叫外部能力）
      ▼
M05 Embedding、檢索與 RAG
      │  把外部知識餵給模型
      ▼
M06 create_agent 與 Middleware
         整合上述全部，做出自動決策的 Agent
```

## 各模組一句話

| 模組 | 你會學到 | 核心 API |
|------|----------|----------|
| **M00** | 環境設定、LangChain 心智模型、舊版對照 | `init_chat_model` |
| **M01** | 跟模型對話、訊息種類、多模態、串流 | `HumanMessage` / `.invoke` / `.stream` |
| **M02** | 把 prompt 模板化、強制模型回傳結構化資料 | `ChatPromptTemplate` / `with_structured_output` |
| **M03** | 用 `\|` 把組件串成可組合、可並行的管線 | LCEL / `RunnableParallel` |
| **M04** | 定義工具、讓模型決定何時呼叫 | `@tool` / `bind_tools` |
| **M05** | 向量化、檢索、建一條 RAG 問答管線 | `init_embeddings` / VectorStore / Retriever |
| **M06** | 一行做出會自己用工具的 Agent，並用 Middleware 控制行為 | `create_agent` |

## 學習方式

每個模組先讀 `README.md`（搞懂「為什麼」），再打開 `lab.ipynb` 逐格執行。
遇到 `🧪 練習` 區塊請動手改程式碼，這是吸收的關鍵。

> 學完本冊，第二冊會告訴你：當「呼叫一次」不夠、需要迴圈 / 分支 / 記憶 /
> 多 Agent 協作時，怎麼用 LangGraph 把這些積木組成有狀態的系統。
