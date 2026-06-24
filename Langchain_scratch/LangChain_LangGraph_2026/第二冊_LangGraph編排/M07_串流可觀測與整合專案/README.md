# M07 — 串流、可觀測與整合專案（Capstone）

> 全課程的收尾：學會把圖的執行「串流」出來、用 LangSmith「看見」每一步，最後把整套課程（RAG＋工具＋記憶＋人介入）整合成一張會自主運作的小助理圖。

## 學習目標
- 用 `graph.stream(...)` 的三種 `stream_mode`：`"values"`（看每步後的完整 state）、`"updates"`（看每步的增量更新）、`"messages"`（看 token 級的逐字輸出）。
- 知道什麼情境該用哪一種串流模式，以及它們各自吐出什麼形狀的資料。
- 開啟 LangSmith 追蹤（只需設環境變數），並讀懂 UI 上能看到的執行軌跡。
- 把第一冊的 RAG、第一冊 M04 的工具、第二冊 M04 的 checkpointer 記憶、第二冊 M05 的 `interrupt` 人介入，整合進**同一張圖**，並用串流跑一個完整情境。
- 在腦中建立整套課程的學習地圖，知道接下來往哪走（部署、LangGraph Platform、評估）。

## 為什麼需要這個？

到 M06 為止，你的圖都是用 `graph.invoke(...)` **一次跑到底、最後才回一坨結果**。兩個痛點浮現：

- **黑盒感**：一個多節點的圖跑了 5 秒，使用者面前一片空白，直到最後才看到答案。你也不知道中途卡在哪個節點、哪一步把 state 改成什麼樣。
- **難除錯**：圖出錯時，`invoke` 只給你最終狀態或一個 traceback。你看不到「每個節點實際收到什麼、吐出什麼、模型被餵了什麼 prompt」。

`stream` 解決第一個痛點：把圖的執行**邊跑邊吐**，可以做打字機效果、可以即時顯示「正在檢索…正在呼叫工具…」。LangSmith 解決第二個痛點：自動把每一步的輸入輸出、模型呼叫、token 用量錄下來，在 UI 上攤平成一條可點擊的軌跡。

最後，本模組的 Capstone 把前面所有零件拼成一個完整應用——這正是真實專案的樣子：很少有系統只用一個組件，價值來自把它們**疊起來**。

## 核心概念

### 1. 串流：同一張圖，三種觀看角度

`graph.stream(inputs, config, stream_mode=...)` 回傳一個 iterator，每次 `for` 拿到一個 chunk。差別只在 `stream_mode`——它決定每個 chunk 是什麼：

| `stream_mode` | 每個 chunk 是什麼 | 什麼時候用 |
|---------------|------------------|-----------|
| `"values"` | 每個節點執行**後**的**完整 state**（整張快照） | 想看 state 一步步演化成什麼樣；除錯狀態 |
| `"updates"` | 每個節點**只回傳的增量**，形如 `{節點名: 該節點 return 的 dict}` | 想知道「是哪個節點、改了什麼」；做進度提示 |
| `"messages"` | `(token, metadata)` tuple，模型**逐字**吐出的 token | 想做打字機效果、即時顯示模型生成 |

心智模型：`values` 看「現在整體長怎樣」，`updates` 看「剛剛這一步做了什麼」，`messages` 看「模型正在打字」。同一張圖、同一個輸入，換 `stream_mode` 就換一個觀看角度。

```python
# 看完整 state 快照
for chunk in graph.stream(inputs, config, stream_mode="values"):
    print(chunk)

# 只看每步增量：{node_name: {...}}
for chunk in graph.stream(inputs, config, stream_mode="updates"):
    print(chunk)

# 看 token 級輸出（打字機）
for token, meta in graph.stream(inputs, config, stream_mode="messages"):
    print(token.content, end="", flush=True)
```

也可以一次傳一個 list（如 `stream_mode=["updates", "messages"]`）同時收多種，chunk 會變成 `(mode, data)` tuple，但教學上先分開看最清楚。

### 2. 可觀測性：LangSmith 自動追蹤

LangSmith 是 LangChain 官方的觀測平台。最棒的一點是：**你不用改任何程式碼**。只要在環境設好這幾個變數，所有 LangChain / LangGraph 的執行就會自動上傳追蹤：

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=ls__你的金鑰
export LANGSMITH_PROJECT=langgraph-capstone   # 選填，分組用
```

設好之後，照常 `graph.invoke(...)` 或 `graph.stream(...)`，到 [smith.langchain.com](https://smith.langchain.com) 就能看到這次執行的 **trace**。一條 trace 是一棵樹，你能看到：

- 圖跑了哪些節點、各自的**輸入與輸出**、耗時。
- 每次模型呼叫的**完整 prompt、回應、token 數、成本**。
- 工具呼叫的參數與回傳值。
- 出錯時，是哪一步、丟了什麼 exception。

這就是把「黑盒」變「玻璃盒」。本 notebook 不需要真的連線；我們只示範如何開啟，並說明你會看到什麼。

### 3. Capstone：把全課程疊成一張圖

整合目標是一個小助理圖，資料流如下：

```
START → retrieve（RAG 檢索）→ agent（帶工具的模型）→ human_approve（人工核准）→ END
```

各節點對應你已經學過的組件：

| 節點 | 用到的組件 | 來自 |
|------|-----------|------|
| `retrieve` | `InMemoryVectorStore` + retriever | 第一冊 M05（RAG） |
| `agent` | `bind_tools` + 工具迴圈 | 第一冊 M04（工具） |
| `human_approve` | `interrupt(...)` 暫停等人 | 第二冊 M05（人介入） |
| 整張圖 | `compile(checkpointer=...)` 記住對話 | 第二冊 M04（記憶） |

`State` 是把這些零件串起來的「共用資料結構」：它要同時裝得下檢索到的 `context`、對話 `messages`、以及核准結果。compile 時掛上 `checkpointer`，整張圖就有了跨輪記憶；`interrupt` 讓圖能在核准節點停下、等人回覆再續跑。最後用 `stream` 跑，就能逐步看到「檢索 → 思考 → 停下等核准 → 續跑 → 完成」的全貌。

## 與前一模組的銜接

這個模組疊在**整個第二冊**之上，是收斂而非新增大量概念：

- **M01～M03** 給你 `StateGraph`、reducer、條件路由——Capstone 的圖就是這樣搭起來的。
- **M04（持久化）** 的 `checkpointer` + `thread_id`：Capstone `compile(checkpointer=...)` 直接沿用，讓助理記得整段對話。
- **M05（HumanInTheLoop）** 的 `interrupt` / `Command(resume=...)`：Capstone 的核准節點就是它。
- **M06（多智能體）** 讓你習慣「節點裡可以是一個完整的子能力」——Capstone 的 `agent` 節點就是一個會呼叫工具的小 agent。

M07 唯一真正的新東西是 `stream` 與 LangSmith——兩者都是**觀測手段**，不改變圖本身，只是讓你「看得見」圖在做什麼。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱
- **搞混 `values` 與 `updates`**：`values` 每個 chunk 是**完整 state**（會越來越大、key 都在）；`updates` 每個 chunk 只是 `{節點名: 那步的回傳}`。除錯看 `values`，做進度條看 `updates`，別拿錯。
- **`messages` 模式忘了解包 tuple**：它吐的是 `(token, metadata)`，要 `for token, meta in ...`，且取文字是 `token.content`。直接 `print(chunk)` 會印出一坨 tuple。
- **以為 `stream` 改了圖的邏輯**：`stream` 跟 `invoke` 跑的是同一張圖、同樣結果，只是輸出方式不同。圖的行為由節點與邊決定，與你用哪個方法執行無關。
- **LangSmith 沒追蹤到**：99% 是環境變數沒設好——`LANGSMITH_TRACING` 要是 `true`、`LANGSMITH_API_KEY` 要正確。它靠 env 自動啟用，不是靠 import 某個東西。
- **Capstone 的 `interrupt` 沒掛 checkpointer**：`interrupt` 需要 checkpointer 才能保存「暫停點」，否則無法 resume。compile 時一定要帶 `checkpointer=`，且 `stream` / `invoke` 都要帶含 `thread_id` 的 config。

## 小結 & 下一步

你完成了整套課程。回顧這張學習地圖：

- **第一冊（LangChain 框架）**：模型與訊息 → Prompt 與結構化輸出 → LCEL 管線 → 工具 → RAG → create_agent。你學會「把一個個組件用 `|` 疊成可組合的 Runnable」。
- **第二冊（LangGraph 編排）**：為什麼需要圖 → StateGraph → 狀態與 reducer → 條件路由 → 持久化記憶 → 人介入 → 多智能體 → 串流與可觀測。你學會「當流程有分支、循環、需要記憶與人介入時，用圖來編排」。

一句話總結：**Runnable 負責「一步怎麼算」，Graph 負責「多步怎麼走」，stream 與 LangSmith 負責「讓你看見它在走」。**

延伸方向（本課程之外）：
- **部署**：把圖包成 API 服務（FastAPI + `graph.astream`），或用 **LangGraph Platform / `langgraph-cli`** 一鍵部署、自帶持久化與排程。
- **持久化升級**：把 `InMemorySaver` 換成 `SqliteSaver` / Postgres checkpointer，記憶就能跨重啟存活。
- **評估（Evaluation）**：用 LangSmith 的 dataset + evaluator 對你的圖跑回歸測試，量化「改了 prompt 之後是變好還是變壞」。

恭喜——你已經具備從零搭建、編排、觀測一個 LLM 應用的完整能力。
