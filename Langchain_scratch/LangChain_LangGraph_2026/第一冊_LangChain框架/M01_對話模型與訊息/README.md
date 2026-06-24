# M01 — 對話模型與訊息

> 整套課程的最小單位：學會用統一介面跟任何對話模型講話，並用「訊息物件」精準控制這段對話。

## 學習目標
- 掌握 chat model 的三個統一動詞：`invoke`（一次）、`stream`（逐塊）、`batch`（一次多筆）。
- 認識四種訊息物件：`SystemMessage` / `HumanMessage` / `AIMessage` / `ToolMessage`，以及 `dict` 簡寫形式。
- 理解 content blocks 與多模態（文字 + 圖片）的訊息結構，知道一則訊息不只是一段字串。
- 會用 `temperature`、`max_tokens` 調整模型行為，並知道用 `model.profile` 查模型能力。

## 為什麼需要這個？

M00 你已經拿到一個 `get_model()`，背後是 `init_chat_model`，所以**不管你用 OpenAI、Anthropic 還是本機 Ollama，拿到的物件操作方式都一樣**。這就是「供應商無關」的價值：換模型只改一個環境變數，程式碼一行都不用動。

但光有模型還不夠。真實對話有「角色」之分：誰在設定規則（系統）、誰在問問題（人類）、誰在回答（AI）、工具回了什麼結果（工具）。如果你只會丟一個字串給模型，你就失去了控制這些角色的能力——而後面所有東西（Prompt 模板、工具呼叫、Agent）都建立在「訊息是有角色的物件」這個前提上。

所以這個模組做兩件事：**把模型的統一介面摸熟**，再**把訊息這個資料結構搞清楚**。這兩件事是後面每一個模組的地基。

## 核心概念

### 1. 統一介面：invoke / stream / batch

一個 chat model 不管底層是誰，都提供同一組方法。記住這三個就夠你走很遠：

| 方法 | 做什麼 | 回傳 | 什麼時候用 |
|------|--------|------|-----------|
| `model.invoke(x)` | 送一次、等完整回答 | 一個 `AIMessage` | 大多數情況、最簡單 |
| `model.stream(x)` | 送一次、逐塊吐回 | 一串 `AIMessageChunk` | 要邊產生邊顯示（聊天 UI 打字機效果） |
| `model.batch([x1, x2, ...])` | 一次送多筆、平行處理 | 一個 `AIMessage` 的 list | 批次處理多筆獨立輸入 |

`invoke` 的輸入可以是一個字串，也可以是一個訊息 list。輸出永遠是 `AIMessage`，取文字用 `.content`：

```python
response = model.invoke("用一句話介紹你自己")
print(response.content)
```

`stream` 是把同一次回答切成很多塊，你把每塊的 `.content` 接起來就是完整答案：

```python
for chunk in model.stream("講個短笑話"):
    print(chunk.content, end="", flush=True)
```

### 2. 四種訊息物件

對話是由「帶角色的訊息」串成的 list。LangChain 用四個類別表示這些角色：

| 訊息類別 | 角色 | 用途 |
|----------|------|------|
| `SystemMessage` | system | 設定模型的人格、規則、語氣（整段對話的最高指導原則） |
| `HumanMessage` | user | 使用者說的話 |
| `AIMessage` | assistant | 模型的回覆（你也可以手動塞入，模擬對話歷史） |
| `ToolMessage` | tool | 工具執行後的結果（M04 工具呼叫時會大量用到） |

典型用法是 `SystemMessage` + `HumanMessage` 組一段對話：

```python
from langchain.messages import SystemMessage, HumanMessage
model.invoke([
    SystemMessage("你是嚴謹的物理老師，只用繁體中文回答"),
    HumanMessage("什麼是慣性？"),
])
```

**dict 簡寫**：每個訊息物件都等價於一個 `{"role": ..., "content": ...}` 的 dict。兩種寫法可混用，物件版較不易拼錯、IDE 也有提示：

```python
model.invoke([
    {"role": "system", "content": "你是物理老師"},
    {"role": "user", "content": "什麼是慣性？"},
])
```

### 3. content blocks 與多模態

最關鍵的觀念：**一則訊息的 `content` 不一定是字串，也可以是一個「內容區塊」的 list**。每個區塊是一個 dict，用 `type` 標明它是文字還是圖片。這就是多模態的基礎——同一則 `HumanMessage` 裡可以同時放文字和圖片：

```python
HumanMessage(content=[
    {"type": "text", "text": "這張圖裡有什麼？"},
    {"type": "image", "source_type": "url",
     "url": "https://example.com/cat.jpg"},
])
```

當 `content` 是純文字時，它只是「只有一個 text 區塊」的簡寫。理解這個資料結構，你才看得懂為什麼模型能「看圖」。（能不能真的看圖取決於模型本身，下一點教你怎麼查。）

### 4. 調參數與查能力

模型的行為可以調。最常用兩個參數，在 `get_model()` 時傳進去即可（會原封不動轉給 `init_chat_model`）：

- `temperature`：隨機性。`0` 最穩定、可重現；越高越有創意但越不可預測。要可重現就設 `0`。
- `max_tokens`：限制回答長度上限，控制成本與延遲。

```python
strict_model = get_model(temperature=0)      # deterministic
```

不確定某個模型支不支援工具呼叫、看圖、結構化輸出？查 `model.profile`，它回傳這顆模型的能力描述（dict），不用翻文件、也不用試錯。

## 與前一模組的銜接

M00 給了你 `get_model()`——一個供應商無關的模型工廠。本模組**疊在它之上**，做兩件事：

1. 把這顆模型的三個動詞（invoke / stream / batch）練熟。
2. 把餵給它的東西從「一個字串」升級成「一串帶角色的訊息物件」。

換句話說，M00 解決「拿到模型」，M01 解決「怎麼跟它好好說話」。下個模組 M02 會把這些訊息**模板化**，讓你不用每次手寫。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱

- **以為 `content` 一定是字串**：多模態訊息的 `content` 是區塊 list。直接對它做字串操作（如 `.upper()`）會炸。
- **直接 `ChatOpenAI(...)` 或散用供應商類別**：本課程一律走 `get_model()` / `init_chat_model`。直接綁死某供應商，等於放棄「換模型零成本」的好處。
- **舊版 import 路徑**：訊息一律從 `langchain.messages` 匯入（`from langchain.messages import HumanMessage`）。網路上很多舊教學用 `langchain.schema` 或 `langchain_core.messages`，在 v1 課程脈絡請以本課程速查為準。
- **`stream` 的每塊當成完整答案**：`stream` 吐的是 `AIMessageChunk`，每塊只是片段，要自己接起來；不要對單一 chunk 取整段內容。

## 小結 & 下一步

你學會了用統一介面（invoke / stream / batch）操作任何對話模型，並用四種訊息物件精準控制對話的角色；也理解了 content blocks 讓訊息能承載多模態，以及如何用 `temperature` 調行為、用 `model.profile` 查能力。

下一個模組 **M02：PromptTemplate 與結構化輸出**，會解決一個新痛點：每次都手寫整串訊息太累、又容易出錯。我們會用 `ChatPromptTemplate` 把訊息模板化，並用 `with_structured_output` 強迫模型回傳乾淨的結構化資料，而不是一坨需要你再去解析的文字。
