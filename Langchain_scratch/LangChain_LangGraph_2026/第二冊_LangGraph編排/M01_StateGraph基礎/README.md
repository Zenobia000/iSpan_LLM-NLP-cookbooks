# M01 — StateGraph 基礎

> 把 M00 講的「圖 = 共享狀態 + 節點 + 邊」的心智模型，第一次用 `StateGraph` 真正寫出來、編譯、執行。

## 學習目標
- 用 `TypedDict` 定義 State：圖的「共享記憶體」，所有節點讀寫同一份。
- 寫一個 node：吃 `state`、回傳「只含要更新欄位」的 `dict`，其餘欄位不動。
- 用 edge 連接節點，認得 `START` 與 `END` 兩個特殊節點。
- 跑完整流程：`StateGraph(...)` → `add_node` → `add_edge` → `compile()` → `invoke(...)`。
- 用 `get_graph().draw_ascii()` 把圖結構印成文字，眼見為憑。

## 為什麼需要這個？

在第一冊，我們用 LCEL 的 `|` 把組件焊成一條直線管線：輸入從左流到右，一去不回頭。
這對「prompt → model → parser」這種固定直線流程很夠用。

但真實應用常常不是直線：你想先產大綱、再寫草稿、再修稿；中途可能要根據結果決定走哪條路、
或回頭重做。LCEL 的 `|` 沒有「狀態」概念——每個環節只看得到上一環的輸出，看不到「整個流程到目前為止累積了什麼」。
一旦要分支、迴圈、或讓多個步驟共享同一份逐步長大的資料，`|` 就不夠用了。

`StateGraph` 解決的就是這件事。它把流程拆成**節點（node）**，節點之間靠**邊（edge）**連接，
而所有節點共讀共寫同一份 **State**。State 就是這張圖的共享記憶體：資料不再「流過就消失」，
而是被一步步填進同一個結構裡，每個節點都看得到前面累積的成果。這就是邁向 Agent 的地基。

## 核心概念

### State：圖的共享記憶體（用 TypedDict 定義）

State 是一個 `TypedDict`，列出這張圖會用到的所有欄位。它不是某個節點私有的，而是**全圖共享**：
任何節點都能讀整份 state，也能回傳更新去改它。

```python
from typing_extensions import TypedDict

class State(TypedDict):
    topic: str      # input
    outline: str    # filled by first node
    draft: str      # filled by second node
```

心智模型：把 State 想成一張「逐步被填滿的表單」。一開始只有 `topic` 有值，
經過一個節點填上 `outline`，再經過一個節點填上 `draft`。流程跑完，整張表單就填滿了。

### node：吃 state、回傳「要更新的欄位」dict

一個 node 就是一個普通 Python 函式：參數是 `state`，回傳一個 `dict`。
**關鍵規則：回傳的 dict 只放你這一步要更新的鍵，不是整份 state。**

```python
def generate_outline(state: State) -> dict:
    # read what we need from the shared state
    topic = state["topic"]
    # return ONLY the field(s) this node updates
    return {"outline": f"關於「{topic}」的三段式大綱"}
```

LangGraph 會拿這個 dict 去更新共享 state——沒提到的欄位（如 `topic`、`draft`）原封不動。
你不需要自己複製整份 state、也不需要手動 merge，這正是「消除特殊情況」的好設計：
節點只專心宣告「我改了什麼」，合併交給框架。

### edge：連接節點，加上 START 與 END

node 是「做什麼」，edge 是「做完換誰做」。`add_edge(a, b)` 表示「a 跑完就跑 b」。

`START` 和 `END` 是兩個內建的特殊節點：
- `START`：圖的入口。`add_edge(START, "x")` 表示「一開始先跑 x」。
- `END`：圖的出口。`add_edge("x", END)` 表示「x 跑完整張圖就結束」。

把它們想成流程圖最上面的「開始」圈和最下面的「結束」圈。線性流程就是一條
`START → node1 → node2 → END` 的鏈。

### 完整流程：builder → add_node → add_edge → compile → invoke

| 步驟 | 程式碼 | 在做什麼 |
|------|--------|----------|
| 1. 建 builder | `builder = StateGraph(State)` | 開一張綁定該 State 結構的空白圖 |
| 2. 加節點 | `builder.add_node("name", fn)` | 把函式註冊成一個有名字的節點 |
| 3. 加邊 | `builder.add_edge(a, b)` | 宣告節點之間的執行順序 |
| 4. 編譯 | `graph = builder.compile()` | 把藍圖固化成可執行物件，做合法性檢查 |
| 5. 執行 | `graph.invoke({"topic": ...})` | 給初始 state，跑完回傳最終 state |

`builder` 是「藍圖」，`compile()` 之後才是「能跑的圖」。`invoke` 的回傳值是**整份最終 state**
（一個 dict），你可以從裡面把每個被填好的欄位挖出來看。

### 視覺化：draw_ascii()

```python
print(graph.get_graph().draw_ascii())
```

把圖結構印成文字版流程圖。剛開始學圖的拓樸時，這是最快確認「我接的邊對不對」的方法。

## 與前一模組的銜接

M00 用「為什麼直線管線不夠」帶出圖的三要素（State / node / edge）的心智模型，但還沒寫程式。
M01 就是把那個心智模型**第一次落地**：你會親手定義 `TypedDict` 的 State、寫成函式的 node、
用 `add_edge` 接成線性流程，然後 `compile` 再 `invoke`。

和第一冊的 LCEL `|` 對照著看：`|` 是「輸出直接餵下一個的輸入」，沒有共享記憶體；
`StateGraph` 則是「所有節點共讀共寫一份逐步長大的 State」。這份共享 State 正是後面所有進階能力
（reducer 累加、條件路由、迴圈、持久化、人類介入）能成立的根基。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱

- **node 回傳整份 state**：只回傳「要更新的欄位」，例如 `return {"draft": ...}`。
  回傳整份 state 不會壞，但會混淆心智模型，也容易誤覆蓋；養成「只宣告我改了什麼」的習慣。
- **忘了接 START 或 END**：沒有 `add_edge(START, ...)` 圖不知道從哪開始，
  沒有通往 `END` 的邊則可能不知道何時停。`compile()` 會幫你抓出沒接好的圖。
- **拿 builder 直接 invoke**：`builder` 只是藍圖，沒有 `invoke`。
  一定要先 `graph = builder.compile()`，對 `graph` 才能 `invoke`。
- **以為 invoke 回傳的是某個節點的輸出**：`invoke` 回傳的是**整份最終 state**（dict），
  不是最後一個節點的回傳值。要看某欄位就用 `result["draft"]` 這樣取。

## 小結 & 下一步

你現在會用 `TypedDict` 定義共享 State、把流程拆成回傳更新 dict 的 node、
用 `START`/`END` 與 `add_edge` 接成線性圖，並走完 `compile → invoke`，
還能用 `draw_ascii()` 把圖印出來檢查。這就是 LangGraph 的最小可運作骨架。

下一個模組 **M02 — 狀態與 Reducer**：目前一個欄位被更新時是「直接覆蓋」。
但有時你要的是「累加」——例如把訊息一則則疊進 `messages` 而不是覆蓋。
M02 會用 `Annotated` + reducer（如 `add`、`add_messages`）解決這個問題，
讓多個節點能安全地往同一欄位「追加」資料。
