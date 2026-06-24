# M03 — 條件路由與 Command

> 把 M01／M02 那種「一條直線跑到底」的圖，升級成會**看狀況分岔、會迴圈重試**的圖。流程的 if/else 從業務函式裡搬到圖結構上。

## 學習目標
- 用 `add_conditional_edges` 加一條條件邊：靠一個「路由函式」讀 state、回傳下一個 node 的 key。
- 把邊接回前面的 node，做出「沒達標就重試」的迴圈，並用「上限」確保它一定會停。
- 用 `Command(update=..., goto=...)` 在 node 裡同時「改 state」加「決定去哪」，知道它什麼時候比條件邊更乾淨。
- 體會一個原則：把分支從 node 內部的 if/else 抽到圖的邊上，特殊情況就被「結構化」掉了。

## 為什麼需要這個？

到 M02 為止，你畫的圖都是**固定路線**：`START → A → B → END`，每次跑都走同一條路。
但真實流程幾乎都有「看情況」：分數夠了就放行、不夠就退回重寫；答案合格就結束、不合格就再問一次模型。

如果硬塞進直線圖，你只能把這些判斷寫進 node 內部，變成一坨 `if ... else ...`：

```python
def grade(state):
    score = score_it(state["answer"])
    if score >= 60:
        # ...放行的邏輯
    else:
        # ...重試的邏輯，還要自己記重試了幾次、自己控制要不要再跑
```

這就是 Linus 說的「特殊情況」：流程控制跟業務邏輯纏在一起，node 越長越難讀，重試的迴圈還得自己手刻。
LangGraph 給的解法是：**讓「往哪走」變成圖的一部分（邊），而不是 node 裡的程式碼**。
node 只管做事、回報結果；要往哪走，交給一條會讀 state 的「條件邊」決定。分支被搬到結構上，node 就乾淨了。

## 核心概念

### 條件邊 `add_conditional_edges`：用一個函式決定走向

普通邊 `add_edge("A", "B")` 是寫死的：A 跑完一定去 B。
條件邊則是：A 跑完，先呼叫一個**路由函式**，讓它讀現在的 state，回傳一個字串 key，圖再依這個 key 決定去哪。

```python
def route(state) -> str:
    return "pass" if state["score"] >= 60 else "retry"

builder.add_conditional_edges("grade", route, {"pass": "finalize", "retry": "answer"})
```

三個參數的職責很清楚：

| 參數 | 是什麼 | 職責 |
|------|--------|------|
| `"grade"` | 來源 node | 從哪個 node 跑完之後做這個判斷 |
| `route` | 路由函式 | 讀 state，回傳一個 key 字串（**只決定方向，不改 state**） |
| `{...}` | 對照表 | 把 key 字串對應到真正的目標 node 名稱（值可填 `END`） |

心智模型：路由函式是個**純粹的交通警察**——它只看路況（state）、舉牌指方向（回傳 key），自己不開車、不改貨物。
所有「往哪走」的特殊情況，現在集中在這一個函式裡，一眼看得完。

### 迴圈：邊可以指回前面的 node

條件邊的目標 node **可以是前面已經跑過的 node**。這就形成迴圈：

```
answer → grade → (score 不夠) → 退回 answer → grade → ...
                 (score 夠了) → finalize → END
```

這正是 Agent「反覆思考直到滿意」的骨架。但迴圈有個鐵律：**一定要有終止條件**，否則分數永遠不夠就會無限轉。
最實務的做法是在 state 裡放一個 `attempts` 計數器，路由函式同時檢查「達標了沒」和「次數到上限了沒」：

```python
def route(state) -> str:
    if state["score"] >= 60:
        return "pass"
    if state["attempts"] >= 3:     # safety cap: give up after 3 tries
        return "give_up"
    return "retry"
```

上限不是裝飾，是**保命機制**。任何會迴圈的圖都該先問自己：它在最壞情況下會停嗎？

### `Command`：在 node 裡同時「改 state」加「決定去哪」

條件邊把「做事」（node）和「決定方向」（路由函式）分成兩塊，大多數時候這樣最清楚。
但有時候「往哪走」這個決定，**本來就是這個 node 算出來的副產品**——硬要拆成 node 回傳 state、再讓另一個路由函式重讀一次 state 才能決定，反而繞。

這時用 `Command`：node 直接回傳一個 `Command`，裡面同時帶 `update`（要更新的 state）和 `goto`（下一個要去的 node）。

```python
from langgraph.types import Command

def grade(state) -> Command:
    score = score_it(state["answer"])
    target = "finalize" if score >= 60 else "answer"
    return Command(update={"score": score}, goto=target)   # 改 state + 跳轉，一次到位
```

用了 `Command` 之後，這個 node **不需要再接條件邊**——跳去哪它自己說了算。

| | 條件邊 `add_conditional_edges` | `Command(update=, goto=)` |
|---|---|---|
| 做事與決定方向 | 分在 node 和路由函式兩處 | 同一個 node 內 |
| 路由邏輯位置 | 獨立的路由函式（好測試、好重用） | 內嵌在 node 裡 |
| 適合時機 | 多個 node 共用同一套路由規則；方向判斷單純 | 方向是 node 計算的自然產物；想少寫一層 |
| 圖上要不要再加邊 | 要（用對照表接好） | 不用（goto 直接指定） |

兩者不是誰取代誰。記一個準則：**方向判斷如果獨立、會重用，就用條件邊；如果它跟 node 的計算綁死，就用 `Command` 收在一起。**

## 與前一模組的銜接

M01 你學會 `StateGraph` + `add_node` + `add_edge` 畫直線圖；M02 你學會用 `Annotated` reducer 控制 state 怎麼累加（例如 `add` 把列表接起來而非覆蓋）。
M03 不換零件，只多給圖兩種「控制流」能力：

- **分支**：`add_edge`（寫死）→ `add_conditional_edges`（看 state 決定）。
- **迴圈**：邊可以指回前面的 node，配合 M02 學的計數器 state（用 reducer 累加 `attempts`）做出有上限的重試。
- **Command**：把「更新 state（M02 的事）」和「決定走向（M03 的事）」合進同一個 node 回傳值。

換句話說，M03 是把 M02 的「state 設計」直接拿來當**流程控制的依據**——state 裡的 `score` 和 `attempts`，就是路由函式判斷的輸入。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱

- **路由函式回傳的 key 不在對照表裡**：`route` 回傳 `"redo"`，但對照表只寫了 `{"pass": ..., "retry": ...}`，圖會找不到目標而報錯。回傳值務必跟對照表的 key 一字不差（用 `Literal` 標註回傳型別可以提早抓到打錯字）。
- **迴圈沒有終止條件**：只檢查「達標了沒」、忘了檢查「次數到了沒」，遇到模型一直過不了關就會無限迴圈。**任何指回前面的邊，都要有一個次數上限或其他保證會停的條件。**
- **路由函式裡偷改 state**：路由函式只該「讀 state、回傳 key」，不該回傳 dict 去更新 state（它的回傳值是路由 key，不是 state 更新）。要在判斷的同時改 state，那是 `Command` 的活，別讓交通警察去搬貨。
- **用了 `Command` 又多接一條邊**：node 已經用 `Command(goto=...)` 指定下一步了，就不要再對它 `add_edge`／`add_conditional_edges`，否則路由意圖會打架、難以推理。`Command` 的 node 自己負責出口。

## 小結 & 下一步

你現在會讓圖**分岔**（`add_conditional_edges` + 路由函式）、會讓圖**迴圈**（邊指回前面 node + 計數器上限），
也知道何時該用 `Command(update=, goto=)` 把「改 state」和「決定走向」收進同一個 node。
最關鍵的觀念是：把 if/else 從 node 內部搬到圖的邊上，流程一眼可讀，特殊情況被結構消化掉。

下一個模組 **M04 — 持久化與記憶**：到目前為止每次 `invoke` 都是從零開始、跑完即忘。
M04 會用 `checkpointer` 加 `thread_id` 讓圖**記得上一輪的 state**，這樣才談得上多輪對話，也是 M05 中斷續跑的前提。
