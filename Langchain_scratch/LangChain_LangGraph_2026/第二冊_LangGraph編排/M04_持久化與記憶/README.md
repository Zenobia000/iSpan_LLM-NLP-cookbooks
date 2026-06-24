# M04 — 持久化與記憶

> 給圖裝上「記憶」：每一步存成 checkpoint，靠 `thread_id` 區分對話，讓多輪 invoke 接得起來。

## 學習目標

- 理解 **checkpointer** 是什麼：每執行一步就把 state 存成一個 checkpoint，圖因此有了記憶、能續跑。
- 學會用 config 裡的 **`thread_id`** 區分不同對話；同一個 thread 跨多次 `invoke` 會記得歷史。
- 分清 **短期記憶**（同一 thread 內的對話）與 **長期記憶**（跨 thread 的 store）兩種概念。
- 會用 **`InMemorySaver`**（教學用，程式結束就忘）與 **`SqliteSaver`**（寫進本機檔案，重啟仍記得）。
- 會用 **`get_state` / `get_state_history`** 觀察當前與歷史 checkpoint（為 M05 時光旅行鋪路）。

## 為什麼需要這個？

到 M03 為止，我們畫的圖每次 `invoke` 都是**從零開始**：圖跑完，state 就被丟掉。

這對單次任務沒問題，但對「對話」是致命的。你問第一句「我叫小明」，圖回完話；你再問
第二句「我叫什麼名字？」——圖完全不記得第一輪發生過什麼，因為上一輪的 state
早就消失了。沒有記憶，就沒有多輪對話、沒有中斷續跑、更沒有 M05 的人類介入。

舊版 LangChain 是用 `ConversationBufferMemory` 這類元件硬塞歷史進 prompt。本課程
**不教這套**——在 LangGraph 裡，記憶是編排層的內建能力，由 **checkpointer** 提供。
你只要 compile 時掛一個 checkpointer，圖就自動把每一步的 state 存下來。

## 核心概念

### 1. checkpointer：每一步存檔

把 checkpointer 想成圖的「**自動存檔系統**」。圖每執行完一個 super-step，就把
當下完整的 state 序列化成一個 **checkpoint** 存起來。下次用**同一個 thread** 再
`invoke`，圖會先把最後一個 checkpoint 載回來當起點，於是新的輸入會**疊在舊歷史上**。

```python
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())
```

掛上 checkpointer 後，`invoke` 一定要帶 config，否則圖不知道要存到哪個對話。

### 2. thread_id：對話的身分證

checkpoint 不是亂存一堆，而是按 **`thread_id`** 分組。每個 `thread_id` 就是一條
獨立的對話時間線。

```python
config = {"configurable": {"thread_id": "alice"}}
graph.invoke({"messages": [...]}, config)   # 存進 alice 這條線
```

- **同一個 `thread_id`** 連續 invoke：第二輪會看到第一輪的訊息 → 記得名字。
- **換一個 `thread_id`**：等於開一條全新對話，看不到別條線的歷史 → 忘記名字。

這就是「記得」與「忘記」的全部祕密——不是模型變聰明，而是 checkpointer 把歷史
餵了回來。

### 3. 短期記憶 vs 長期記憶

| 類型 | 範圍 | 由誰提供 | 例子 |
|------|------|----------|------|
| **短期記憶** | 單一 `thread_id` 內 | **checkpointer** | 同一場對話記得你剛說的話 |
| **長期記憶** | 跨 thread、跨對話 | **store**（`BaseStore`） | 記住「使用者偏好繁中」橫跨所有對話 |

本模組聚焦**短期記憶**（checkpointer + thread）。長期記憶（store）是另一個獨立機制，
本模組只建立概念，細節留待後續：checkpointer 管「這場對話的狀態」，store 管
「跨對話要永久記住的知識」。兩者可以並存。

### 4. 兩種 checkpointer：教學用 vs 本機持久化

| Checkpointer | 存哪裡 | 程式結束後 | 適用 |
|--------------|--------|------------|------|
| `InMemorySaver` | RAM | **消失** | 教學、測試、Notebook 內示範 |
| `SqliteSaver` | 本機 `.sqlite` 檔 | **保留**，重啟仍記得 | 本機開發、單機應用 |

兩者 API 完全一樣，差別只在「存哪裡」。先用 `InMemorySaver` 把概念跑通，要持久化
就換成 `SqliteSaver`，圖的程式碼一行都不用改——這就是抽象做對的好處。

```python
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("memory.sqlite") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke({"messages": [...]}, config)
```

### 5. get_state / get_state_history：看存了什麼

掛了 checkpointer，你就能隨時把某個 thread 的狀態挖出來看：

```python
graph.get_state(config)            # 當前最新 checkpoint（含 values / next）
graph.get_state_history(config)    # 從新到舊的所有 checkpoint
```

`get_state` 回傳一個 **`StateSnapshot`**：`.values` 是當下的 state、`.next` 是
接下來要跑哪個 node、`.config` 帶著這個 checkpoint 的 id。M05「時光旅行」就是靠
歷史 checkpoint 的 id，讓圖從過去某一步重新跑起。

## 與前一模組的銜接

本模組直接疊在 **M02 的 `MessagesState`** 上：

- M02：用 `add_messages` reducer 讓 `messages` 是**累加**而非覆蓋。
- M03：用條件邊 / `Command` 控制流程走向。
- **M04（本模組）**：在同一張 `MessagesState` 圖上 `compile(checkpointer=...)`，
  讓「累加的 messages」能**跨多次 invoke** 持續累加下去。

換句話說：`add_messages` 解決的是「單次執行內，訊息怎麼累加」；checkpointer 解決的是
「多次執行之間，state 怎麼接續」。兩者合起來，才構成完整的多輪對話。新增的東西只有
一個：`compile(checkpointer=...)` 與 `invoke(..., config)` 裡的 `thread_id`。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱

- **掛了 checkpointer 卻不帶 `thread_id`**：`graph.invoke(inputs)` 沒有 config 會報錯。
  只要 compile 時掛了 checkpointer，每次 invoke 都**必須**帶 `{"configurable": {"thread_id": ...}}`。
- **以為換 `thread_id` 還會記得**：不會。不同 thread 是完全隔離的時間線。要延續對話就用
  同一個 id；要開新對話就換 id。這是 feature，不是 bug。
- **拿 `InMemorySaver` 當持久化**：它只活在 RAM，程式一結束記憶就蒸發。要重啟後仍記得，
  必須用 `SqliteSaver`（或其他 DB 版 saver）。
- **誤用舊版 memory 元件**：`ConversationBufferMemory`、`ConversationChain` 等在 v1
  已不是記憶的正解。LangGraph 裡，記憶 = checkpointer，別再往 prompt 裡硬塞歷史。

## 小結 & 下一步

checkpointer 把每一步的 state 存成 checkpoint，`thread_id` 把這些 checkpoint
按對話分組——這兩個機制合起來，圖就有了「短期記憶」，能跨多輪 invoke 接續對話。
`InMemorySaver` 拿來學，`SqliteSaver` 拿來真正持久化。`get_state` /
`get_state_history` 則讓你隨時觀察存了什麼。

有了 checkpoint，圖不只能「記得」，還能「**回到過去某一步重新跑**」——這正是
**M05 Human-in-the-loop** 的基礎：在關鍵步驟用 `interrupt` 暫停、等人類審核或修改，
再從那個 checkpoint 續跑。下一站見。
