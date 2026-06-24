# M05 — Human-in-the-loop（人介入）

> 讓圖在「危險或不可逆的動作」前停下來，把決定權交回給人，等人點頭後再續跑。

## 學習目標
- 說清楚「為什麼」要人介入：送出 email、付款、刪除資料這類不可逆操作前先審核。
- 用 `interrupt(...)` 在 node 內把圖暫停，並把待審內容拋給外面的人。
- 用 `Command(resume=...)` 在人做完決定後從中斷點續跑（必須搭 checkpointer）。
- 掌握三種常見模式：核准/拒絕、編輯後再跑、補上缺漏資訊。
- 用 `get_state_history` 做「時光旅行」：回到歷史 checkpoint、改狀態、分支重跑。

## 為什麼需要這個？

到 M04 為止，我們的圖是「一口氣跑完」的：`invoke` 進去、結果出來，中間沒人插得上手。
這對「查資料、算數、整理文字」這種讀取型任務沒問題。

但只要圖會做**不可逆的動作**，全自動就很危險：

- Agent 決定「寄出這封 email」——寄錯了收不回來。
- Agent 決定「執行這筆退款 / 付款」——錢出去了。
- Agent 決定「刪掉這個檔案 / 這筆資料」——刪了就沒了。

模型再聰明也會出錯。真正穩健的系統是：**在動手之前先停下來，把「我打算這樣做」攤給人看，
等人核准才繼續。** 這就是 human-in-the-loop。

關鍵前提是：圖要能「停在半路、過一段時間（甚至換個程式重啟）再從原地接著跑」。
這正好就是 M04 學的 **checkpointer**——它把每一步的狀態存了下來，所以暫停與續跑才成立。
M05 就是疊在 checkpointer 之上的一層應用。

## 核心概念

### 1. `interrupt`：在 node 內把圖按下暫停鍵

`interrupt(payload)` 寫在 node 函式裡。執行到它時，LangGraph 會：

1. 把 `payload`（你要給人看的東西，例如「準備寄這封信，核准嗎？」）往外拋；
2. **停在這個 node**，把目前狀態交給 checkpointer 存起來；
3. `invoke` 就此返回，回傳值裡帶著 `__interrupt__`，告訴呼叫端「我卡在這、在等你」。

```python
from langgraph.types import interrupt

def human_review(state):
    decision = interrupt({"action": "send_email", "draft": state["draft"]})
    return {"decision": decision}
```

心智模型：`interrupt` 就像程式裡的「中斷點」，把控制權還給外面的人。
它和 `input()` 很像，差別是「等待」這段期間圖是被**持久化保存**的，不是卡在記憶體裡空轉。

### 2. `Command(resume=...)`：人做完決定，從原地續跑

人看完、做了決定（例如回 `"approve"`），你**不是重新 invoke 整張圖**，而是：

```python
from langgraph.types import Command

graph.invoke(Command(resume="approve"), config)
```

LangGraph 從 checkpointer 撈回剛剛暫停的狀態，把 `"approve"` 當成那個 `interrupt(...)` 的回傳值，
讓 `human_review` 從中斷的那一行**接著往下跑**。注意 `config` 要帶同一個 `thread_id`，
LangGraph 才知道要續哪一條對話。

| 組件 | 職責 |
|------|------|
| `interrupt(payload)` | 在 node 內暫停，把待審內容拋出去 |
| 第一次 `invoke(inputs, config)` | 跑到 interrupt 就停，回傳裡含 `__interrupt__` |
| `Command(resume=value)` | 把 `value` 餵回 interrupt，從中斷點續跑 |
| checkpointer（M04） | 暫停期間保存狀態，續跑時撈回——**沒它就不能 HITL** |

### 3. 三種常見模式

同一套 `interrupt` / `resume` 機制，靠「人回傳什麼值」就能撐起三種場景：

- **核准 / 拒絕**：人回 `"approve"` 或 `"reject"`，node 內用條件決定要送出還是中止。
- **編輯後再跑**：人回一段改過的內容（例如修正後的 email 草稿），node 拿它覆蓋狀態再繼續。
- **補上缺漏資訊**：Agent 缺一個必要參數（收件人 email？），`interrupt` 問人、人補、續跑。

差別只在 payload 設計與 resume 值的型別，骨架完全一樣。

### 4. 時光旅行：`get_state_history` + `update_state`

checkpointer 不只存「最新狀態」，而是存了**每一步的歷史**。於是你可以倒帶：

```python
# List every checkpoint, newest first.
for snapshot in graph.get_state_history(config):
    print(snapshot.config["configurable"]["checkpoint_id"], snapshot.next)

# Pick one past checkpoint, tweak its state, and re-run from there.
forked = graph.update_state(some_past_config, {"draft": "改寫後的內容"})
graph.invoke(None, forked)   # re-run from that forked point
```

`get_state_history(config)` 給你一串 `StateSnapshot`（含各自的 `config`、`values`、`next`）。
挑一個過去的 checkpoint，用 `update_state` 在那個點改狀態，會產生一條**新的分支**，
從那裡 `invoke(None, ...)` 就能跑出和原本不同的結果。這就是 debug、改寫、「重來一遍但這次不一樣」的基礎。

## 與前一模組的銜接

M04 教 checkpointer，讓圖「記得」跨次呼叫的狀態。
M05 直接站在它上面：**HITL 完全依賴 checkpointer**——暫停時靠它存狀態，續跑時靠它撈狀態，
時光旅行靠它列歷史。沒有 checkpointer，`interrupt` / `resume` 都不成立。

所以本模組的圖在 `compile` 時一定帶 `checkpointer=`，每次呼叫一定帶 `thread_id`。
新增的東西只有兩個 API：`interrupt`（暫停）與 `Command(resume=...)`（續跑），外加 `get_state_history` 做倒帶。

## 動手做
請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱
- **忘了 checkpointer**：`compile()` 沒帶 `checkpointer=`，`interrupt` 會直接報錯，圖根本沒地方存暫停狀態。
- **續跑時又傳完整 inputs**：續跑要傳 `Command(resume=...)`，不是再丟一次原始輸入；後者會被當成「新的一輪」而非接續。
- **`config` 的 `thread_id` 對不上**：暫停和續跑必須用同一個 `thread_id`，否則 LangGraph 找不到要接哪一條，等於重開。
- **以為 `interrupt` 像普通函式會「立刻回傳值」**：第一次跑到它時它讓整個 `invoke` 返回（圖暫停）；
  只有在 `resume` 那一次重跑該 node 時，`interrupt(...)` 才會「回傳」你 resume 的值。理解這個「重跑該 node」的語意，才不會被搞混。

## 小結 & 下一步
本模組把圖從「全自動」升級成「可在關鍵點交給人決定」：`interrupt` 暫停、`Command(resume=...)` 續跑、
`get_state_history` + `update_state` 做時光旅行，全都建立在 M04 的 checkpointer 上。

下一站 **M06 — 多智能體系統**：當任務複雜到一個 Agent 撐不住，我們會把多個 Agent 編排成一張圖，
讓它們分工、交棒、協作——而 HITL 正好是在這種系統裡放「人類監督點」的方式。
