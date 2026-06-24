# LangChain × LangGraph 實戰課程（2026 版）

> 從零開始，循序漸進掌握 **LangChain v1** 與 **LangGraph v1** —
> 用 Jupyter Notebook 一個組件一個組件地疊起一套可上線的 AI 應用。

本課程站在 **2026 年的視角** 重新組織教學：拋棄已淘汰的 `LLMChain`、
`initialize_agent`、`AgentExecutor` 等舊寫法，全程採用 v1 的標準 API
（`init_chat_model`、LCEL、`create_agent`、`StateGraph`）。

---

## 課程設計理念

| 原則 | 做法 |
|------|------|
| **循序漸進** | 每個模組只引入 1～2 個新組件，先理解再疊加，不跳步。 |
| **先懂概念再寫碼** | 每個模組都有 `README.md`（教學文件）講「這是什麼、為什麼需要、何時用」，再進 `lab.ipynb` 動手。 |
| **供應商無關** | 所有範例用 `init_chat_model` 抽象；改 `.env` 一行就能切 OpenAI / Claude / 本地模型，學生不被綁定。 |
| **由下而上** | 先學 LangChain 的「積木」（模型、訊息、工具、檢索），再學 LangGraph 的「編排」（狀態圖、記憶、多智能體）。 |

---

## 兩冊地圖

```
LangChain_LangGraph_2026/
├── 第一冊_LangChain框架/      ← 積木：組件本身
│   M00 環境設定與心智模型
│   M01 對話模型與訊息
│   M02 PromptTemplate 與結構化輸出
│   M03 LCEL 與 Runnable 管線
│   M04 工具與工具呼叫
│   M05 Embedding、檢索與 RAG
│   M06 create_agent 與 Middleware
│
└── 第二冊_LangGraph編排/      ← 骨架：把積木組成有狀態的系統
    M00 為什麼需要 LangGraph
    M01 StateGraph 基礎
    M02 狀態與 Reducer
    M03 條件路由與 Command
    M04 持久化與記憶
    M05 Human-in-the-loop
    M06 多智能體系統
    M07 串流、可觀測與整合專案
```

### 為什麼是「兩冊」而不是一堆教學？

- **第一冊**回答：「LangChain 給了我哪些零件，每個零件怎麼用？」
  你會學會把 `prompt | model | parser` 串成管線，呼叫工具，做 RAG，
  並用 `create_agent` 快速做出一個會自己用工具的 Agent。
- **第二冊**回答：「當流程有分支、迴圈、需要記憶、需要人介入、需要多個
  Agent 協作時，怎麼辦？」這就是 LangGraph 存在的理由——它是 LangChain
  之上的**有狀態編排層**。`create_agent` 其實就是跑在 LangGraph 上的。

> 一句話：**第一冊教你做出一次性的呼叫鏈，第二冊教你做出能持續運行、
> 有記憶、可控制的系統。**

---

## 學習路徑建議

| 對象 | 路徑 |
|------|------|
| **完全新手** | 第一冊 M00 → M06 全做完，再進第二冊。約 2～3 週。 |
| **已會 LangChain 0.x** | 先看第一冊 M00（看 API 變更對照表）→ 直接跳第二冊。 |
| **只想做 Agent** | 第一冊 M01、M04、M06 → 第二冊全冊。 |

每個模組資料夾固定兩個檔案：

- `README.md` — **教學文件**：組件用途、心智模型、與前一模組如何銜接、常見陷阱。
- `lab.ipynb` — **動手實作**：可逐格執行的範例，每格都有說明。

---

## 環境設定（只需做一次）

```bash
# 1. Python 3.10+（v1 不再支援 3.9）
python --version

# 2. 建議用虛擬環境
python -m venv .venv && source .venv/bin/activate

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 設定金鑰
cp .env.example .env      # 然後編輯 .env，至少填一個供應商的金鑰

# 5. 啟動
jupyter notebook
```

詳細的環境檢查與第一個 Hello World，請進 **第一冊 / M00**。

---

## 2026 年的 LangChain 長什麼樣？（舊版使用者快速對照）

| 任務 | 舊寫法（0.x，已淘汰） | 新寫法（v1，本課程採用） |
|------|----------------------|--------------------------|
| 建立模型 | `ChatOpenAI(model=...)` 散落各處 | `init_chat_model("openai:gpt-4o-mini")` |
| 串接 Chain | `LLMChain(llm, prompt)` | `prompt \| model \| parser`（LCEL） |
| 建立 Agent | `initialize_agent(...)` / `AgentExecutor` | `create_agent(model, tools=...)` |
| 工具裝飾器 | `from langchain.agents import tool` | `from langchain.tools import tool` |
| 結構化輸出 | 自己解析 JSON | `model.with_structured_output(Schema)` |
| 複雜流程 | 一堆 if/else 包 Chain | **LangGraph `StateGraph`** |

> 舊的 `chains`、`AgentExecutor` 等並未消失，被移到 `langchain-classic`
> 套件以維持向後相容。本課程一律教 v1 主線寫法。

---

## 延伸：本資料夾的其他資源

- `../langchain_framework/project/` — 5 個應用專案（Custom ChatGPT、私有文件
  QA、摘要、Streamlit 前端），可作為學完本課程後的**實戰練習素材**。
- `../Multi-agent-system/` — 多智能體長文寫作專案（STORM / CrewAI），與第二冊
  M06「多智能體」可互相對照。

---

*本課程程式碼註解使用英文、教學說明使用繁體中文。所有 API 對齊 2026 年的
LangChain v1.x / LangGraph v1.x。*
