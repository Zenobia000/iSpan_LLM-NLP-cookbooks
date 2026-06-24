# M06 — create_agent 與 Middleware

> 第一冊收尾：把 M04 手刻的「工具迴圈」交給 `create_agent` 一行搞定，順手認識 Middleware，並揭穿一個事實——Agent 底層就是一張 LangGraph 圖，自然把你帶進第二冊。

## 學習目標
- 從 M04 的「手動 think→act→observe 迴圈」進化到 `create_agent`：一行建好會自己跑迴圈的 Agent。
- 學會 `create_agent` 三件套：傳 `model` / `tools` / `system_prompt`，用 `invoke({"messages":[...]})` 呼叫，從 `result["messages"]` 讀出整段思考軌跡。
- 用 `response_format` 讓 Agent 的最終答案是一個 Pydantic 結構物件，而不是一段自由文字。
- 認識 Middleware：在 Agent 執行流程中插入行為（自動摘要壓縮長對話、危險操作前先問人）。
- 建立關鍵認知：`create_agent` 底層就是一張 LangGraph 圖——這就是第二冊的入口。

## 為什麼需要這個？

回想 M04 你做了什麼：模型回了 `tool_calls`，你**手動**把工具跑一遍，把結果包成 `ToolMessage` 塞回去，再呼叫一次模型，看它還要不要繼續呼叫工具……如果它又要呼叫，你得**再寫一圈迴圈**。

這個迴圈每個 Agent 都長一樣：

```
模型決定要不要用工具 → 用了就執行 → 把結果餵回去 → 再問模型 → 直到模型說「我講完了」
```

既然每次都一樣，手刻就是在重複造輪子，而且很容易漏掉邊界情況（工具報錯怎麼辦？呼叫了好幾個工具怎麼辦？無限迴圈怎麼辦？）。`create_agent` 把這個迴圈收進一個函式，你只要把「模型、工具、人設」交給它，迴圈它自己跑。

這正是好品味：**M04 的手刻是為了讓你看懂迴圈的本質，看懂之後就該把這個固定模式交給框架，消除重複。**

## 核心概念

### 1. `create_agent`：把 M04 的迴圈包成一行

```python
from langchain.agents import create_agent

agent = create_agent(model, tools=[add, multiply], system_prompt="你是計算助理")
result = agent.invoke({"messages": [{"role": "user", "content": "(3+5)*2 是多少?"}]})
print(result["messages"][-1].content)
```

注意輸入輸出的形狀：

| 項目 | 形狀 | 說明 |
|------|------|------|
| 輸入 | `{"messages": [...]}` | 一個 dict，`messages` 是訊息清單（可用 `{"role","content"}` 簡寫） |
| 輸出 | `{"messages": [...]}` | 同樣是 dict，但 `messages` 變長了——裡面是**整段對話軌跡** |

`result["messages"]` 不只有最後答案，而是**從你的問題到最終回答之間的每一步**：模型的工具呼叫（帶 `tool_calls` 的 `AIMessage`）、工具執行結果（`ToolMessage`）、再到收尾的 `AIMessage`。把它整段印出來，你會看到 M04 手刻的那個迴圈，現在自動跑完了。最終答案是 `result["messages"][-1].content`。

### 2. `response_format`：讓最終答案變成結構化物件

預設 Agent 的最後一則訊息是自由文字。但很多時候你要的是**能直接餵給下游程式的結構**（像 M02 學的 `with_structured_output`，只是這次套在整個 Agent 上）。

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    result: int = Field(description="最終計算結果")
    steps: str = Field(description="計算過程說明")

agent = create_agent(model, tools=[add, multiply], response_format=Answer)
```

之後 Agent 跑完，最終輸出會被整理成 `Answer` 物件。你也可以寫成 `response_format=ProviderStrategy(Answer)`，明確指定用「供應商原生結構化輸出」這個策略；不確定時直接傳 Pydantic 類別即可。

### 3. Middleware：在 Agent 流程裡插一隻手

`create_agent` 的迴圈是固定的，但有時你想在迴圈的某個環節**插入額外行為**，又不想自己重寫整個迴圈。Middleware 就是這個「插入點」。

| Middleware | 它做什麼 | 什麼時候要它 |
|------------|----------|--------------|
| `SummarizationMiddleware` | 對話太長時，自動把舊訊息摘要壓縮，避免爆 context | 多輪、長對話的 Agent |
| `HumanInTheLoopMiddleware` | 在執行特定（危險）工具前先暫停，等人核准 | 工具會刪資料、花錢、送出不可逆操作時 |

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model,
    tools=[add, multiply],
    middleware=[SummarizationMiddleware(model=model)],
)
```

心智模型：Middleware 是**疊在 Agent 迴圈外的一圈行為**。Agent 照常 think→act→observe，但每一圈經過 Middleware 設定的檢查點時，它有機會介入（壓縮歷史、攔下危險呼叫）。你不用碰迴圈本身，只要宣告「我要加這個行為」。

### 4. 底層就是一張 LangGraph 圖

這是本模組——也是整個第一冊——最重要的一句話：**`create_agent` 回傳的不是什麼黑盒子，它是一張編譯好的 LangGraph 圖。**

這解釋了前面所有形狀為什麼長這樣：

- 輸入輸出都是 `{"messages": [...]}`，因為那是 LangGraph 的**狀態（state）**。
- `result["messages"]` 會累積整段軌跡，因為 LangGraph 用 reducer 把每一步的新訊息**疊加**進狀態。
- Middleware 能插入行為，因為圖裡本來就有節點可以掛。

所以 `create_agent` 是「最常見的那張圖」的快捷方式。一旦你需要**它沒幫你做的事**——自訂迴圈邏輯、加記憶、人介入後續跑、多個 Agent 協作——你就要自己畫這張圖，那就是第二冊 LangGraph。

## 與前一模組的銜接

這個模組直接疊在 **M04（工具與工具呼叫）** 之上：

| M04 你學的 | M06 的進化 |
|------------|------------|
| `@tool` 定義工具 | 一樣用 `@tool`，工具定義不變 |
| `model.bind_tools([...])` 拿到 `tool_calls` | `create_agent` 內部幫你 bind |
| **手寫迴圈**：執行工具、包 `ToolMessage`、再 invoke | `create_agent` **自動跑這個迴圈** |
| 自由文字輸出 | `response_format` 給你結構化輸出（呼應 M02） |

同時它也整合了 M01（訊息）、M02（結構化輸出）、M03（組件可組合的心智）、M05（工具可以是檢索器）。M06 是第一冊的會合點，也是通往第二冊的橋。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱

- **以為 `create_agent` 收的是純字串**：它收的是 `{"messages": [...]}`。直接丟字串會出錯。輸出也是 dict，要取答案得寫 `result["messages"][-1].content`，不是 `result.content`。
- **沿用舊版 `AgentExecutor` / `initialize_agent` / `AgentType`**：這些是 0.x 的東西，v1 一律改用 `create_agent`。看到教學裡出現它們就是過時內容。
- **把 `response_format` 當成 prompt 在用**：它不是叫模型「請用 JSON 回答」那種 prompt 技巧，而是讓 Agent 的最終輸出被框架解析成 Pydantic 物件。別自己在 system_prompt 裡再手寫一套格式要求互相打架。
- **杜撰 Middleware 的複雜參數**：Middleware 的精確參數依版本而定。不確定時就用最簡單形式（如 `SummarizationMiddleware(model=model)`），把概念說清楚比硬塞參數重要。

## 小結 & 下一步

你已經：用 `create_agent` 一行取代 M04 的手刻迴圈、用 `response_format` 拿到結構化最終答案、用 Middleware 在 Agent 流程裡插入行為，並且知道了這一切底層就是一張 LangGraph 圖。

第一冊到此完成——你已經把每一塊積木（模型、prompt、parser、tool、retriever、agent）都摸過一遍。但 `create_agent` 是「別人幫你畫好的那張圖」，當你需要**完全控制迴圈、加記憶、人介入後續、多 Agent 協作**時，就得自己畫圖。

那就是 **第二冊 — LangGraph**。下一站，我們從 `StateGraph`、`START`、`END` 開始，親手畫出第一張圖——然後你會發現，`create_agent` 不過是其中一個特例。
