# M06 — 多智能體系統

> 把前面學的 `create_agent`、條件路由與共享狀態組起來：當一個 agent 的工具太多、職責太雜時，拆成各司其職的專家，再用一個指揮者（supervisor）協調它們。

## 學習目標

- 判斷**何時該從單一 agent 升級成多 agent**：工具爆量、職責互相干擾、prompt 越寫越長時就是訊號。
- 把一個 `create_agent`（或一張子圖）當成大圖的**一個 node** 來用（subgraph 概念）。
- 實作 **supervisor 模式**：一個 router/supervisor 依任務把工作分派給專家 agent，再收斂結果。
- 用 **handoff**：以 `Command(goto=...)` 在 agent 之間「交棒」，靠共享的 `messages` 傳遞上下文。
- 認識 **`Command(graph=Command.PARENT)`**：從子圖跳回父圖的概念。

## 為什麼需要這個？

第一冊你學會用 `create_agent` 做出一個「會思考、會用工具」的 agent。很自然地，你會想把所有能力塞進同一個 agent：給它搜尋工具、給它寫作工具、給它計算工具、給它查資料庫工具……

問題是，**單一 agent 的能力不是越多越好**。當工具清單越來越長，你會踩到幾個真實的痛點：

- **選錯工具**：模型每一步都要從十幾個工具裡挑一個，挑錯的機率隨工具數量上升。
- **prompt 互相打架**：「你要嚴謹地查證」和「你要文筆流暢地寫稿」是兩種不同人格，硬塞進同一個 system prompt 會互相稀釋。
- **難以維護**：所有邏輯擠在一個 agent 裡，改 A 能力常常弄壞 B 能力。

解法和寫程式拆函式是同一個道理：**一個 agent 只做一件事，並把它做好。** 把「研究」交給一個只懂研究的 `research_agent`，把「寫作」交給一個只懂寫作的 `writing_agent`，再請一個 **supervisor** 負責「現在該誰上場」。這就是多智能體系統。

承接第二冊前面：你已經會用 `StateGraph` 畫圖（M01）、會用 `MessagesState` 與 reducer 累積訊息（M02）、會用條件路由與 `Command` 決定下一步並更新狀態（M03）。多 agent 不是新魔法——它就是**把每個 agent 當成 node，用你已經會的路由把它們串起來**。

## 核心概念

### 1. agent 即 node：subgraph

`create_agent` 回傳的東西本身就是一張可以 `invoke` 的 LangGraph 圖。既然它有跟一般 node 一樣的呼叫介面（吃 `{"messages": [...]}`、回 `{"messages": [...]}`），我們就能把它**包成大圖的一個 node**：

```python
research_agent = create_agent(model, tools=[web_search], system_prompt="你是研究員")

def research_node(state: MessagesState) -> dict:
    result = research_agent.invoke({"messages": state["messages"]})
    # Only return the last message back to the shared state.
    return {"messages": [result["messages"][-1]]}
```

這就是 **subgraph（子圖）** 的精神：一張小圖（agent）變成大圖裡的一格。父圖不需要知道子圖內部怎麼跑，只看它的輸入與輸出。

### 2. supervisor：誰來指揮？

光把 agent 變成 node 還不夠——**誰決定下一棒給誰？** 這就是 supervisor 的職責。supervisor 是一個普通 node，它讀共享狀態，判斷任務進度，然後決定路由：

| 角色 | 是什麼 | 職責 |
|------|--------|------|
| **supervisor** | 一個路由 node（可選用 LLM 判斷） | 看現在的進度，決定下一棒給哪個專家，或收工到 `END` |
| **專家 agent** | 包成 node 的 `create_agent` | 只做自己擅長的那件事，做完把結果寫回共享狀態 |
| **共享 state** | `MessagesState`（含 `add_messages` reducer） | 所有 agent 共用的「白板」，上下文靠它流動 |

典型的 supervisor 圖長這樣：

```text
              ┌──────────────┐
   START ───▶ │  supervisor  │ ◀─────────────┐
              └──────┬───────┘               │
                     │ 條件路由               │ 做完回報
          ┌──────────┼──────────┐            │
          ▼          ▼          ▼            │
     research     writing     (END)          │
          └──────────┴───────────────────────┘
```

每個專家做完都回到 supervisor，由 supervisor 再判斷下一步——這個「分派 → 回報 → 再分派」的迴圈，正是 M03 條件路由的直接應用。

### 3. handoff：用 Command 交棒

supervisor 怎麼「決定並跳轉」？最乾淨的寫法是 M03 教過的 `Command`：它能**同時更新狀態 + 指定下一個 node**，一步到位：

```python
from langgraph.types import Command

def supervisor(state: MessagesState) -> Command:
    next_agent = decide_next(state)          # "research" / "writing" / END
    return Command(goto=next_agent)          # handoff：把棒子交給 next_agent
```

「交棒（handoff）」的關鍵在於**上下文不會掉**：因為所有 agent 共用同一份 `messages`，research_agent 查到的資料會留在白板上，writing_agent 一上場就讀得到。你不需要手動傳參數——共享狀態就是上下文的載體。

### 4. Command.PARENT：從子圖跳回父圖（概念）

當你的子圖內部也想直接決定「跳回父圖的哪個 node」，可以用 `Command` 的 `graph` 參數指向父圖：

```python
from langgraph.types import Command

# Inside a subgraph node: jump back to a node in the PARENT graph.
return Command(goto="supervisor", graph=Command.PARENT)
```

預設 `Command(goto=...)` 只在**當前這張圖**裡跳；加上 `graph=Command.PARENT` 才會跳到外層父圖。本模組以 supervisor 在父圖層協調為主，這個機制先建立概念即可——當你的子圖需要「自己決定回父圖哪一步」時會用到。

## 與前一模組的銜接

本模組是把第二冊前面三塊積木疊起來，沒有全新的 API：

- **疊在 M01 之上**：把每個專家 agent 當成 `StateGraph` 的一個 `add_node`。
- **疊在 M02 之上**：用 `MessagesState` 的 `add_messages` reducer，讓多個 agent 的產出**累加**在同一份 `messages`，上下文才能流動。
- **疊在 M03 之上**：supervisor 的分派完全是條件路由 + `Command(goto=...)`。
- **疊在第一冊 `create_agent` 之上**：每個專家就是一個 `create_agent`，只是這次它不是終點，而是大圖裡的一格。

新增的只有一個**架構觀念**：把「一個大 agent」拆成「supervisor + 多個專家 agent」。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。lab 會：

1. 建兩個專家（`research_agent`、`writing_agent`），各自用 `create_agent` 包成 node。
2. 寫一個 supervisor node，用條件路由 / `Command` 決定下一棒給誰，完成後到 `END`。
3. 用共享的 `MessagesState` 讓上下文在 agent 間流動。
4. 印出 ASCII 圖，親眼看到多 agent 結構。
5. 🧪 練習：加入第三個專家（如校稿 agent），擴充 supervisor 的路由。

## 常見陷阱

- **node 把整包 `result` 寫回 state。** `agent.invoke(...)` 回的是**完整對話**，如果你 `return {"messages": result["messages"]}`，配上 `add_messages` reducer 會把整段歷史重複塞回去。慣例是只回傳**最後一則訊息** `result["messages"][-1]`，讓共享白板乾淨。
- **supervisor 沒有終止條件，無限繞圈。** 一定要有「任務完成 → `goto=END`」的判斷，否則 supervisor 會一直分派、永遠不收工。可加一個步數上限或明確的完成旗標。
- **混淆 `goto` 的作用域。** `Command(goto=...)` 預設只在**當前圖**裡跳；想跳回父圖要 `graph=Command.PARENT`。在父圖層協調時不需要 PARENT，別亂加。
- **以為多 agent 一定比單 agent 好。** 不是。多 agent 增加了協調成本與延遲。只有當單一 agent 的工具/職責真的多到互相干擾時才拆——先問「這個 agent 真的雜到需要拆嗎？」

## 小結 & 下一步

- 當一個 agent 工具太多、職責太雜，就拆成**各司其職的專家 agent**，每個只做一件事。
- 把 `create_agent` 包成 node 就是 **subgraph**；用 **supervisor** node 做分派、用 **`Command(goto=...)`** 做 handoff，靠**共享 `MessagesState`** 傳遞上下文。
- 這一切都是 M01（node）+ M02（reducer）+ M03（路由/Command）+ 第一冊（`create_agent`）的組合，不是新魔法。

**下一步：M07 — 串流、可觀測與整合專案**。我們會替這些圖加上串流輸出與觀測，並把第二冊學到的東西收斂成一個完整專案。
