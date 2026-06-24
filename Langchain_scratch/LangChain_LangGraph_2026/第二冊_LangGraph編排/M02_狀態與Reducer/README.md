# M02 — 狀態與 Reducer

> 在 M01 你會畫圖了；這一模組教你**讓 State 自己決定怎麼被更新**，從此不用在每個 node 手動 append。

## 學習目標
- 看懂 LangGraph 更新 state 的**預設行為**：node 回傳的鍵會**覆蓋**舊值。
- 知道何時「覆蓋」是錯的（多個 node 想往同一個 list 累加，卻互相蓋掉）。
- 用 `Annotated[type, reducer]` 改變某個鍵的更新方式（例如 `operator.add` 累加 list）。
- 用 `add_messages` reducer 與內建 `MessagesState`，做一個會累積對話、自動去重的圖。
- 建立 Linus 式直覺：**把 state 結構設計對，特殊情況就消失**，不再到處寫 append。

## 為什麼需要這個？

M01 我們學會了 `StateGraph`：定義一個 `TypedDict` 當 state，每個 node 回傳一個 dict，
LangGraph 把它合併回 state。當時的範例很簡單，所以沒人注意到一個關鍵問題——

**合併時，預設是「覆蓋」。**

假設 state 裡有一個 `logs: list`，你有兩個 node 都想往裡面記一筆。直覺上你以為最後會有兩筆，
但實際上第二個 node 回傳的 `{"logs": ["B"]}` 會**整個蓋掉**第一個 node 寫的 `["A"]`，最後只剩 `["B"]`。

你當然可以在每個 node 裡寫 `return {"logs": state["logs"] + ["B"]}` 來手動累加。但這就是 Linus 最討厭的東西：
**到處都是「先讀舊值再合併」的樣板**。一旦有十個 node，你就有十處要記得做這件事，漏一處就有 bug。

正確的解法不是「在每個 node 修補」，而是**改變 state 這個鍵的合併規則**——這就是 reducer。

## 核心概念

### 1. 預設行為：覆蓋（last-write-wins）

沒有 reducer 時，node 回傳的每個鍵都直接覆蓋 state 裡的同名鍵：

```python
class State(TypedDict):
    foo: str          # node 回傳 {"foo": "x"} -> 直接變成 "x"
    logs: list        # node 回傳 {"logs": ["B"]} -> 整個覆蓋，舊的不見了
```

對 `foo: str` 這種「最新值才對」的欄位，覆蓋正是你要的。
對 `logs: list` 這種「想累積」的欄位，覆蓋就是 bug。

### 2. Reducer：用 `Annotated[type, reducer]` 改變更新方式

reducer 是一個函式 `(舊值, 新值) -> 合併後的值`。你把它掛在型別上，LangGraph 合併這個鍵時就改用它：

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    logs: Annotated[list, add]    # 新值不再覆蓋，而是 舊 + 新（list 串接）
```

現在兩個 node 各回傳 `{"logs": ["A"]}`、`{"logs": ["B"]}`，最終 state 是 `["A", "B"]`。
**node 裡只要回傳「我這一步新增的部分」，累加由 reducer 統一處理。** 特殊情況消失了。

| 鍵的語意 | 該用的 reducer | 心智模型 |
|----------|----------------|----------|
| 「最新值才算數」 | 不加（預設覆蓋） | last-write-wins |
| 「要把每步結果累積起來」 | `operator.add`（list/數字） | 串接 / 求和 |
| 「對話訊息串」 | `add_messages` | append + 依 id 去重/更新 |

### 3. `add_messages` 與內建 `MessagesState`

聊天場景幾乎每個圖都有一個 `messages` 欄位要累積。LangGraph 內建了專門的 reducer `add_messages`，
它比 `operator.add` 聰明：

- 會把新訊息**接到後面**（append）。
- 會把 dict 形式的訊息（如 `{"role": "user", "content": "hi"}`）**自動轉成 Message 物件**。
- 會依**訊息 id 去重 / 更新**：同一個 id 的訊息再出現時是「取代」而非「重複加一筆」。

你可以自己寫：

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

但因為太常用，LangGraph 直接幫你準備好內建版本 `MessagesState`，裡面就是上面這一行。
要加別的欄位，繼承它再補就好：

```python
from langgraph.graph import MessagesState

class ChatState(MessagesState):     # 已含 messages: Annotated[list, add_messages]
    summary: str                    # 你自己加的其他欄位
```

### 4. 多鍵、不同 reducer、部分更新

一個 state 裡不同的鍵可以有不同 reducer，而且 node **只回傳它想改的鍵**即可，其餘鍵自動保持不變：

```python
class State(TypedDict):
    counter: Annotated[int, add]      # 累加
    logs: Annotated[list, add]        # 串接
    status: str                       # 覆蓋（預設）

def step(state):
    return {"counter": 1, "logs": ["did a thing"]}
    # 沒回傳 status -> status 維持原值
```

這就是 LangGraph state 的本質：**一張表，每個欄位各自決定怎麼被合併**。
你設計圖的第一件事，永遠是先把這張表的「欄位 + reducer」想清楚。

## 與前一模組的銜接

M01 你建立的 state 全部都是「預設覆蓋」——因為範例只有一個 node 寫一次，看不出問題。
M02 疊在 M01 之上，**只動一個地方**：把需要累積的鍵從 `list` 改成 `Annotated[list, reducer]`。
圖的結構（node / edge / `StateGraph` / `compile`）完全沿用 M01，沒有新東西。

換句話說：你不是學一套新 API，而是學會「**在型別上做標註**」這個小動作，
它讓你不必再寫一堆手動 append 的樣板。M03（條件路由與迴圈）會大量依賴累積型 state——
Agent 反覆思考時，`messages` 必須能持續累加，靠的正是這裡學的 `add_messages`。

## 動手做
請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱
- **以為 list 會自動累加。** 不會。沒掛 reducer 的 list 一律覆蓋。要累積就加 `Annotated[list, add]`。
- **在 node 裡手動 `state["logs"] + [...]` 又同時掛了 reducer。** 會「加兩次」（你手動加一次、reducer 再加一次）。
  掛了 reducer 後，node 只回傳「新增的部分」，不要自己先合併。
- **拿 `operator.add` 去合併 messages。** 能串接，但不會去重、不會把 dict 轉成 Message 物件。
  訊息一律用 `add_messages` 或直接用 `MessagesState`。
- **重新賦值整個 state。** node 回傳的是「這一步的更新（partial update）」，不是「新的完整 state」。
  只回傳你要改的鍵即可，別把不相干的鍵也塞回去。

## 小結 & 下一步
這一模組的核心只有一句話：**state 的每個鍵自己決定怎麼被合併，reducer 讓「累積」這個特殊情況消失。**
你學會了預設覆蓋、`operator.add` 累加、`add_messages` / `MessagesState` 累積對話，以及多鍵混搭。

下一站 **M03 — 條件路由與 Command**：到目前為止圖都是一條直線走到底。M03 讓圖會**分支、會迴圈**，
並用 `Command` 同時「更新 state + 決定下一步去哪」。屆時你會看到，正是因為 `messages` 有 `add_messages`，
Agent 才能在迴圈裡一輪一輪累積對話而不互相覆蓋。
