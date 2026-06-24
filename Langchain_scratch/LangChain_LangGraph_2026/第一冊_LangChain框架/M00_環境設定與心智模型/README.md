# M00 — 環境設定與心智模型

> 整套課程的起點：把開發環境裝好，並建立「LangChain 到底是什麼」的正確心智模型，避免一路用錯方向。

## 學習目標
- 確認 Python 3.10+、安裝課程套件、複製 `.env` 並設定 `COURSE_MODEL`，用 `get_model()` 跑出你的第一個模型呼叫。
- 建立 LangChain 的心智模型：它是「一套標準介面 + 可組合組件」，**本身不是模型**。
- 用一句話說清楚 LangChain 與 LangGraph 的分工，以及本課程兩冊的結構。
- 看懂舊版（0.x）→ v1 的 API 變更對照，知道為什麼 `init_chat_model` 是新標準。

## 為什麼需要這個？

很多人第一次碰 LangChain 會有兩個誤解，這兩個誤解會讓你後面整套課都學歪：

1. **以為 LangChain 是一個模型**。不是。它不會幫你生成文字，真正生成文字的是 OpenAI、Anthropic、Ollama 那些供應商的模型。LangChain 做的是「在這些模型外面包一層統一介面」，讓你換供應商時程式碼幾乎不用動。
2. **以為要記一大堆 class**。在 0.x 時代確實如此（每換一家就要 `ChatOpenAI`、`ChatAnthropic`…，還有一堆 `LLMChain`、`AgentExecutor`）。v1 把這些收斂成少數幾個標準入口，最重要的就是 `init_chat_model`。

所以這個模組要解決的痛點是：**先把方向校正好，再開始疊積木**。環境裝錯、心智模型錯，後面每個模組都會卡。

## 核心概念

### 1. LangChain 是「標準介面 + 可組合組件」

把 LangChain 想成一個樂高底板。它定義了每種積木的「接口長什麼樣」（例如：模型都有 `.invoke()`、`.stream()`），然後提供一組各司其職的組件：

| 組件 | 職責（一句話） | 對應模組 |
|------|----------------|----------|
| **model** | 對話模型本體，輸入訊息、輸出回覆 | M01 |
| **prompt** | 把變數套進固定模板，產生要餵給模型的訊息 | M02 |
| **output parser** | 把模型的原始回覆轉成你要的格式（純文字、結構化物件） | M02 / M03 |
| **tool** | 讓模型能呼叫外部能力（算數、查資料、call API） | M04 |
| **retriever** | 從向量庫撈出相關文件，餵給模型當作背景知識 | M05 |
| **agent** | 整合上述全部，讓模型自己決定何時用工具、何時收尾 | M06 |

關鍵心智：**這些組件接口一致，所以可以用 `|` 串起來**（這就是 M03 要學的 LCEL）。你現在不需要懂全部，只要記得「每塊積木各司其職，而且能互相疊」。

### 2. 供應商無關：一個環境變數切換所有模型

本課程所有 notebook 都不直接寫 `ChatOpenAI(...)`，而是透過 `_shared/course_utils.py` 的 `get_model()`：

```python
from course_utils import get_model
model = get_model()          # 讀 .env 裡的 COURSE_MODEL
```

`get_model()` 底層只做一件事：呼叫 v1 的 `init_chat_model("openai:gpt-4o-mini")`。它吃 `provider:name` 格式，所以你只要改 `.env` 裡的一行 `COURSE_MODEL`，整套課程就從 OpenAI 換到 Claude 或本地 Ollama，**程式碼一字不動**。這就是「標準介面」的價值。

### 3. LangChain vs LangGraph：一句話分工

- **LangChain（第一冊）**：提供**組件**與把組件串成「呼叫一次」管線的能力。輸入 → 經過 prompt / model / parser → 輸出，是一條直線。
- **LangGraph（第二冊）**：當「呼叫一次」不夠用——你需要**迴圈、分支、記憶、多 Agent 協作**——就用 LangGraph 做**有狀態的編排**。它把第一冊的積木組成一張可以反覆走、會記住狀態的圖。

先學會單塊積木怎麼用（第一冊），再學怎麼把它們組成有狀態的系統（第二冊）。

## 與前一模組的銜接

這是整套課程的**第一個模組**，沒有前一個。它的角色是「地基」：之後每個模組（M01 對話模型、M02 Prompt、…）都會用到這裡建好的 `get_model()` 樣板與 `.env` 設定。把這格跑通，後面才有東西可疊。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱

- **直接 `pip install langchain` 後 import 不到東西**：v1 把組件拆進多個套件（`langchain`、`langchain-core`、各供應商套件）。請用課程根目錄的 `requirements.txt` 一次裝齊，不要單裝。
- **忘了複製 `.env`**：`get_model()` 讀不到 `COURSE_MODEL` 會回退到預設值，金鑰沒填則呼叫時才報錯。請先 `cp .env.example .env` 並填好對應供應商的金鑰。
- **沿用舊版寫法**：v1 之後**不要**再用 `ChatOpenAI(...)` 散落各處、`LLMChain`、`AgentExecutor`、`ConversationBufferMemory`。這些在本課程一律以 `init_chat_model` / LCEL / LangGraph checkpointer 取代（詳見下表）。
- **Python 版本太舊**：v1 需要 Python 3.10+。3.9 以下會在 import 階段就掛掉。

### 舊版（0.x）→ v1 對照

| 你想做的事 | ❌ 舊版（0.x） | ✅ v1（本課程） |
|------------|----------------|------------------|
| 建立模型 | `ChatOpenAI(...)` 各家各一個 class | `init_chat_model("openai:gpt-4o-mini")` |
| 串 prompt + model | `LLMChain(llm=..., prompt=...)` | `prompt \| model`（LCEL） |
| 做 Agent | `initialize_agent` / `AgentExecutor` | `create_agent(...)`（跑在 LangGraph 上） |
| 對話記憶 | `ConversationBufferMemory` | LangGraph checkpointer（第二冊） |

`init_chat_model` 之所以是新標準，是因為它把「換供應商」這件事收斂成改字串，消除了「每家一個 class」這個特殊情況——這正是好設計：**讓特殊情況變成正常情況**。

## 小結 & 下一步

你已經：裝好環境、確認供應商無關的 `get_model()` 能跑、並建立了「LangChain = 標準介面 + 可組合組件」的心智模型。

下一個模組 **M01 — 對話模型與訊息**，會把鏡頭拉近到 `model` 這塊積木：認識 `HumanMessage` / `SystemMessage` / `AIMessage` 等訊息種類、`.invoke()` 與 `.stream()` 的差別，以及多模態輸入。那是所有後續組件的共同基礎。
