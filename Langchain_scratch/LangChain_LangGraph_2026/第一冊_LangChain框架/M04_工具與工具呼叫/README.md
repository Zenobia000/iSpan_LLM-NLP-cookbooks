# M04 — 工具與工具呼叫

> 讓模型不再只會「說」，而是能「叫外部能力幫忙做」：算數、查資料、執行動作。

## 學習目標
- 說清楚工具（tool）是什麼，以及為何 LLM 需要工具：突破知識截止、做精確計算、執行真實動作。
- 用 `@tool` 定義工具，並理解 docstring 與 type hint 為何是工具能不能被正確呼叫的關鍵。
- 用 `bind_tools` 把工具綁到模型，並讀懂 `ai.tool_calls` 這個結構。
- 手動跑完一整輪「工具迴圈」：模型產生 `tool_calls` → 執行工具 → 用 `ToolMessage` 回填 → 再 `invoke` 拿到最終答案。

## 為什麼需要這個？

到 M03 為止，你的模型很會「講話」，但它有三個先天限制：

1. **知識有截止日**：它不知道今天的天氣、不知道你資料庫裡的數字。
2. **不會算數**：語言模型是預測下一個 token，`13947 × 8231` 它常常猜錯。
3. **不能執行動作**：它沒辦法真的寄信、真的查 API、真的改檔案。

工具就是用來補這三個洞的。你把一段**真正的 Python 函式**交給模型，告訴它「你有這個能力可以用」。模型自己判斷：這個問題要不要用工具、要用哪個、參數該填什麼。注意一個常被誤解的重點——**模型不會執行你的函式**，它只會告訴你「我想呼叫 `add(a=3, b=5)`」；真正去跑那個函式、把結果送回去的，是**你的程式**。這一來一回就是本模組的核心。

## 核心概念

### 工具其實就是「有說明書的函式」

```python
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
```

`@tool` 把一個普通函式包裝成模型看得懂的工具。它會從三個地方抽取「說明書」交給模型：

| 來源 | 用途 | 模型靠它判斷 |
|------|------|--------------|
| 函式名稱 `add` | 工具的識別名 | 該叫哪個工具 |
| docstring `"Add two numbers."` | 工具的用途描述 | 什麼時候該用這個工具 |
| type hint `a: int, b: int` | 參數的名稱與型別 | 參數要填什麼、填什麼型別 |

所以 docstring 跟 type hint **不是給人看的裝飾，是給模型看的提示**。少了它們，模型就像拿到一個沒有標籤的按鈕——不知道何時按、也不知道該輸入什麼。

### `bind_tools`：把工具掛到模型上

```python
model_with_tools = model.bind_tools([add, get_weather])
```

`bind_tools` 不會改變模型本體，它回傳一個「已知道有哪些工具可用」的新模型。對它 `invoke`，模型若決定要用工具，回傳的 `AIMessage` 的 `.content` 通常是空的，真正的內容放在 `.tool_calls`：

```python
ai = model_with_tools.invoke("3 加 5 是多少?")
ai.tool_calls
# [{'name': 'add', 'args': {'a': 3, 'b': 5}, 'id': 'call_abc...', 'type': 'tool_call'}]
```

`tool_calls` 是一個 list（模型可能一次要求呼叫多個工具），每筆是一個 dict，含 `name`（叫哪個工具）、`args`（參數）、`id`（這次呼叫的識別碼，回填結果時要對得上）。

### `ToolMessage`：把工具執行結果還回去

這就接上了 M01 學過的訊息種類。M01 我們介紹過 `ToolMessage`，當時還沒派上用場，現在它的角色終於清楚了：執行完工具後，你要把結果包成 `ToolMessage` 加進對話歷史，**而且 `tool_call_id` 必須對上剛剛那筆 `tool_call` 的 `id`**，模型才知道「這個結果是哪次呼叫的回應」。

```python
from langchain.messages import ToolMessage
ToolMessage(content="8", tool_call_id="call_abc...")
```

### 一輪完整的工具迴圈

把上面串起來就是一輪對話：

```
1. messages = [HumanMessage("3 加 5?")]
2. ai = model_with_tools.invoke(messages)   → ai.tool_calls = [add(a=3,b=5)]
3. 自己執行 add(3,5)=8，包成 ToolMessage(tool_call_id=...)
4. messages += [ai, tool_message]
5. final = model_with_tools.invoke(messages) → "3 加 5 等於 8"
```

關鍵順序：第 4 步要**先把那個 `AIMessage`（帶 tool_calls 的）放回歷史，再放 `ToolMessage`**。模型看到的是「我要求呼叫 add → 這是 add 的結果」這條完整脈絡，才能產出最終答案。

## 與前一模組的銜接

這個模組疊在兩個已學組件之上：

- **疊在 M01 的訊息系統上**：工具迴圈全程在操作 `HumanMessage` / `AIMessage` / `ToolMessage` 這串對話歷史。M01 埋下的 `ToolMessage` 在這裡正式上場。
- **疊在 M03 的 `invoke` 心智模型上**：`bind_tools` 回傳的還是一個 Runnable，照樣 `.invoke()`。工具沒有改變呼叫方式，只是讓 `AIMessage` 多了 `tool_calls` 這個欄位。

新增的東西只有兩個：`@tool` 與 `bind_tools`，外加「讀 `tool_calls` → 執行 → 回填 `ToolMessage` → 再呼叫」這個迴圈手法。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱

- **以為模型會自己執行工具**：不會。`bind_tools` 後 `invoke` 只會拿到「想呼叫什麼」的意圖，真正執行的是你的程式碼。M06 的 `create_agent` 才會幫你自動執行。
- **docstring 或 type hint 漏寫**：模型看不懂工具用途或參數型別，會亂填或乾脆不呼叫。工具一定要寫清楚 docstring 和型別標註。
- **`ToolMessage` 的 `tool_call_id` 對不上**：忘了帶或填錯 id，模型無法把結果跟請求配對，可能報錯或答非所問。一定要用 `tool_call["id"]`。
- **回填順序錯誤**：只把 `ToolMessage` 加進去、卻漏了帶 `tool_calls` 的那個 `AIMessage`。歷史裡必須是「AIMessage(含 tool_calls) → ToolMessage」成對出現。
- **用了舊版 import**：`from langchain.agents import tool` 是舊寫法，本課程一律 `from langchain.tools import tool`。

## 小結 & 下一步

你已經學會工具呼叫的完整一輪：定義工具（`@tool`）、綁到模型（`bind_tools`）、讀 `tool_calls`、執行後用 `ToolMessage` 回填、再 `invoke` 取得最終答案。

但你也看到了——這個「手動工具迴圈」很繁瑣，而且只跑了一輪。真實情況下模型可能要連續呼叫好幾輪工具才能完成任務，你得寫一個 `while` 迴圈反覆判斷「還有 `tool_calls` 嗎？」。

下一個模組 **M06 `create_agent`** 做的就是把這整段迴圈自動化：你只要把工具丟給它，它會自己跑完「呼叫工具 → 回填 → 再判斷」直到產出最終答案。看完本模組你會明白，Agent 不是魔法，它只是幫你自動跑這個你已經手刻過一遍的迴圈。（中間的 M05 會先補上 RAG，讓 Agent 也能查外部知識。）
