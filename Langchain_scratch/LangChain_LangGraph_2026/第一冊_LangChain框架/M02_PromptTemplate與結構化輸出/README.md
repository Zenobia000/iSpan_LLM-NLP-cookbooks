# M02 — PromptTemplate 與結構化輸出

> 把 M01 手寫的訊息升級成「可重用的模板」，並讓模型直接回傳乾淨的 Python 物件。

## 學習目標
- 說明為什麼要把 prompt 模板化，而不是每次手刻字串。
- 用 `ChatPromptTemplate.from_messages` 建立 system + human 模板，並用變數填充。
- 用 `MessagesPlaceholder` 把對話歷史注入模板，用 `partial` 先填好部分變數。
- 把 few-shot 範例放進模板，引導模型輸出格式。
- 用 Pydantic `BaseModel` + `with_structured_output` 做資訊抽取與分類，拿到結構化物件而非一段文字。

## 為什麼需要這個？

M01 我們是這樣呼叫模型的：

```python
model.invoke([SystemMessage("你是助理"), HumanMessage("幫我把這句翻成英文：你好")])
```

這在 demo 沒問題，但放進真實程式就會痛：

- **重複**：同一段 system 指令在十個地方各抄一份，改一次要改十處。
- **拼字串**：要把使用者輸入塞進 prompt，就得用 f-string 手動拼接，容易拼錯、容易把格式邏輯和業務內容攪在一起。
- **歷史難管理**：多輪對話時，得自己維護一個訊息 list 再手動接上新訊息。
- **輸出是一坨文字**：模型回的是自然語言，你還得自己寫正則或 `split` 去解析，模型多講一句話程式就壞了。

`ChatPromptTemplate` 解決前三個痛點（把 prompt 變成有變數的可重用樣板），`with_structured_output` 解決最後一個（讓模型直接回傳你定義好的資料結構）。

## 核心概念

### 1. ChatPromptTemplate：有洞的訊息樣板

心智模型：模板就是「**一串訊息，但裡面挖了幾個 `{變數}` 的洞**」。`invoke` 時把洞填上，就得到一串真正的訊息送進模型。

```python
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}，回答要精簡。"),
    ("human", "{question}"),
])
messages = prompt.invoke({"role": "歷史老師", "question": "誰是秦始皇？"})
```

`prompt.invoke(...)` 產出的就是 M01 那種訊息 list，只是現在由模板幫你組好。每個 tuple 的第一個元素是角色（`"system"` / `"human"` / `"ai"`），第二個是帶 `{變數}` 的字串。

### 2. partial：先填一部分變數

有些變數在程式啟動時就確定了（例如目前語言、品牌名），不必每次 invoke 都傳。`partial` 讓你先填好，之後只傳剩下的：

```python
zh_prompt = prompt.partial(role="歷史老師")
zh_prompt.invoke({"question": "誰是秦始皇？"})   # role 已固定
```

### 3. MessagesPlaceholder：留一個「插歷史」的洞

前兩個洞填的是字串，但對話歷史是一串訊息。`MessagesPlaceholder` 就是在模板裡預留一個位置，invoke 時塞進一個訊息 list：

```python
from langchain_core.prompts import MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是客服助理。"),
    MessagesPlaceholder("history"),     # 這裡會被一串訊息填滿
    ("human", "{question}"),
])
```

這正是 M01 訊息列表的延伸：歷史訊息（`HumanMessage` / `AIMessage`）原封不動放進去，模板把「固定指令」和「動態歷史」分開管理。

### 4. few-shot：把範例寫進模板

要模型照特定格式或風格回答，最有效的方法是給它幾個範例。最直接的做法就是把「範例問 / 範例答」當成額外的 human / ai 訊息放進模板：

```python
ChatPromptTemplate.from_messages([
    ("system", "把使用者的句子改寫成正式書面語。"),
    ("human", "這東西超讚"),
    ("ai", "此產品表現優異。"),          # 一組示範
    ("human", "{sentence}"),
])
```

模型看到前面的示範，就會模仿那個風格處理真正的輸入。

### 5. with_structured_output：要物件，不要文字

用 Pydantic 定義你想要的形狀，`with_structured_output` 包住模型後，`invoke` 回傳的就是那個物件實例：

```python
from pydantic import BaseModel, Field
class Person(BaseModel):
    name: str
    age: int = Field(description="年齡")

structured = model.with_structured_output(Person)
structured.invoke("小明今年 10 歲")   # -> Person(name='小明', age=10)
```

`Field(description=...)` 不只是註解，它會變成給模型的提示，幫助模型抓對欄位。分類任務則用 `Literal` 或 `Enum` 把答案限制在固定選項裡，模型就不會自由發揮回一個你沒預期的類別。

| 組件 | 職責 | 一句話 |
| --- | --- | --- |
| `ChatPromptTemplate` | 組訊息 | 有 `{變數}` 洞的訊息樣板 |
| `partial` | 預填變數 | 固定不變的洞先填好 |
| `MessagesPlaceholder` | 插歷史 | 留一個位置放整串訊息 |
| few-shot 範例 | 引導格式 | 用範例訊息示範想要的輸出 |
| `with_structured_output` | 解析輸出 | 回傳 Pydantic 物件而非文字 |

## 與前一模組的銜接

M01 你學會了手寫 `SystemMessage` / `HumanMessage` / `AIMessage` 並 `model.invoke([...])`。

本模組疊在這之上做兩件事：

1. **把手寫訊息升級成模板**。`("system", "...")` 對應 `SystemMessage`、`("human", "...")` 對應 `HumanMessage`，差別只是現在帶變數、可重用，而 `MessagesPlaceholder` 就是放 M01 那種訊息 list 的位置。
2. **把輸出從文字升級成物件**。M01 我們讀 `response.content`（一段字），本模組讓 `invoke` 直接回傳 `Person(...)` 這種結構化結果。

模型本身沒變，變的是「進去前怎麼組 prompt」與「出來後拿到什麼形狀」。

## 動手做
請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱
- **變數名稱對不上**：模板裡寫 `{question}`，invoke 卻傳 `{"questions": ...}`，會直接報缺變數的錯。模板裡有幾個 `{}` 就要傳幾個 key。
- **字串裡誤用大括號**：模板字串中真正想輸出 `{` 時要寫成 `{{`，否則會被當成變數洞。JSON 範例放進模板特別容易踩。
- **`MessagesPlaceholder` 傳錯型別**：它要的是「訊息 list」（`HumanMessage`/`AIMessage` 或 `{"role":..., "content":...}`），不是一個字串。傳字串進去不會自動變成歷史。
- **沿用舊版寫法**：不要用 `LLMChain`、`ConversationChain`、`ConversationBufferMemory` 這些舊 API（本課程一律不教）；結構化輸出也不要自己叫模型「請回 JSON」再手動 `json.loads`，直接用 `with_structured_output`。

## 小結 & 下一步
你現在能把 prompt 模板化、注入歷史、塞 few-shot 範例，並讓模型回傳乾淨的 Pydantic 物件。這讓「輸入怎麼組、輸出長什麼樣」都變成可控、可重用的零件。

下一個模組 **M03 — LCEL 與 Runnable 管線**，會把 `prompt`、`model`、輸出解析這些零件用 `|` 串成一條管線（`prompt | model | parser`），讓整個流程變成一個可組合、可並行的物件。
