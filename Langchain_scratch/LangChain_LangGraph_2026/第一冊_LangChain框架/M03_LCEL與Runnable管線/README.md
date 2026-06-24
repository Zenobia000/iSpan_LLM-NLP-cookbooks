# M03 — LCEL 與 Runnable 管線

> 把 M01 的對話模型、M02 的 Prompt 模板與結構化輸出，用一條 `|` 串成可組合、可並行、自動支援串流與批次的管線。

## 學習目標
- 看懂 LCEL 的 `|`：它在做什麼、為什麼能把不同組件串成一條鏈。
- 理解 Runnable 統一介面：任何環節都共用 `invoke` / `stream` / `batch`，整條鏈也是。
- 用 `StrOutputParser` 把模型回應的 `AIMessage` 收成純文字。
- 用 `RunnableParallel`、`RunnablePassthrough`、`RunnableLambda`、`@chain` 組出分支、傳遞與自訂步驟。
- 說得出串接的三大好處：可組合、可並行、整條鏈自動支援串流與批次。

## 為什麼需要這個？

在 M01 我們會 `model.invoke(...)`，在 M02 我們會 `prompt.invoke(...)` 產生訊息、再餵給模型。
但這兩步是**手動接線**的：你先呼叫 prompt，拿到結果，再傳給 model，最後從 `AIMessage` 裡挖 `.content`。
每多一個處理步驟，就多一段膠水程式碼，而且這段流程沒辦法整包重用、沒辦法整包串流、也沒辦法整包批次。

LCEL（LangChain Expression Language）解決的就是這個痛點。它讓你用一個 `|` 把組件「焊」在一起，
變成一個新的、同樣可呼叫的物件。接好之後，這條鏈本身就是一個 Runnable——
你對單一組件會的事（`invoke` / `stream` / `batch`），對整條鏈也照樣會。膠水程式碼消失了。

## 核心概念

### Runnable：所有組件共用的同一張臉

LangChain 把 prompt、model、parser、甚至你自己包的函式，全都做成 **Runnable**。
Runnable 是一個約定：只要是 Runnable，就一定有這幾個方法。

| 方法 | 用途 | 心智模型 |
|------|------|----------|
| `invoke(x)` | 餵一筆輸入、拿一筆輸出 | 同步、單筆 |
| `stream(x)` | 一塊一塊吐出輸出 | 邊產生邊拿，適合即時顯示 |
| `batch([x1, x2, ...])` | 一次餵多筆 | 平行處理一整批輸入 |

關鍵在於：**整條鏈也是一個 Runnable**。這就是為什麼後面你會看到，明明沒有特別寫串流邏輯，
`chain.stream(...)` 卻能動——因為串流是 Runnable 介面的一部分，串起來就自動繼承。

### `|` 管線運算子：把輸出接到下一個的輸入

`a | b` 的意思是「先跑 `a`，把 `a` 的輸出當成 `b` 的輸入」。所以：

```python
chain = prompt | model | StrOutputParser()
```

讀作：把輸入字典 → 丟進 `prompt` 產生訊息 → 丟進 `model` 得到 `AIMessage` → 丟進 parser 取出純文字字串。
這跟 Unix 的 `cat file | grep x | wc -l` 是同一個心智模型：每個環節只做一件事，靠管線把資料推下去。

### `StrOutputParser`：把 AIMessage 收成字串

`model` 回傳的是 `AIMessage`（M01 學過），要拿文字得寫 `.content`。
`StrOutputParser` 把這步標準化成一個 Runnable，接在 model 後面，鏈的輸出就直接是 `str`。
少一行 `.content`，鏈也更乾淨。

### 四個常用組合器

| 組件 | 做什麼 | 什麼時候用 |
|------|--------|-----------|
| `RunnableParallel` | 同一份輸入，**同時**跑多條鏈，回傳一個字典 | 一次要產生多個結果（標題＋摘要） |
| `RunnablePassthrough` | 原封不動把輸入往下傳 | 想保留原始輸入，又要在旁邊加工 |
| `RunnableLambda` | 把任何一般函式包成 Runnable | 在鏈中插入自訂處理步驟 |
| `@chain` | 裝飾器，把一個函式直接變成一條鏈 | 自訂邏輯較長、想當成獨立鏈重用時 |

`RunnableParallel` 常用「字典字面量」的簡寫：你在 `|` 旁邊寫一個 `dict`，LangChain 會自動把它當成
`RunnableParallel`。例如 `{"title": title_chain, "summary": summary_chain}` 會並行跑兩條鏈，
輸出 `{"title": ..., "summary": ...}`。

`RunnableLambda` 同理：你在 `|` 旁邊放一個普通函式，LangChain 會自動把它包成 `RunnableLambda`。
顯式寫 `RunnableLambda(fn)` 只是在你需要明確控制時用。

### 串接的三大好處

1. **可組合**：小組件 `|` 成大鏈，大鏈又能當小組件接進更大的鏈。
2. **可並行**：`RunnableParallel` 讓互不相依的分支同時跑，省時間。
3. **自動串流與批次**：因為鏈是 Runnable，`stream` / `batch` 整條免費取得，不用自己寫。

## 與前一模組的銜接

這個模組沒有引入新的「外部組件」，而是把 M01（`model`、`AIMessage`）和 M02（`ChatPromptTemplate`）
**串起來**。M02 你還是手動 `prompt.invoke(...)` 再 `model.invoke(...)`；
M03 用 `prompt | model | StrOutputParser()` 一行接好。新增的只有「接線的語法（`|`）」和「組合器
（`RunnableParallel` / `RunnablePassthrough` / `RunnableLambda` / `@chain`）」。
往後的 RAG（M05）與 Agent 周邊邏輯，都是這套管線思維的延伸。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱

- **`|` 的兩邊型別要對得上**：前一個環節的「輸出」必須是後一個環節能吃的「輸入」。
  `prompt` 吃字典、吐訊息；`model` 吃訊息、吐 `AIMessage`；parser 吃 `AIMessage`、吐字串。接錯就會型別錯誤。
- **忘了接 `StrOutputParser`**：沒接 parser，鏈的輸出是 `AIMessage` 不是 `str`，
  後面想做字串處理（如 `.upper()`）就會炸。要嘛接 parser，要嘛自己取 `.content`。
- **`RunnableParallel` 的分支共用同一份輸入**：每條分支拿到的都是「整包原始輸入」，不是上一條分支的輸出。
  並行分支之間是獨立的，別預期它們會互相傳值。
- **別退回舊版寫法**：不要用 `LLMChain`、`SimpleSequentialChain`、`SequentialChain`、`ConversationChain`。
  2026 的標準做法就是 LCEL 的 `|`，這些舊鏈類別已不在課程範圍內。

## 小結 & 下一步

你現在會用 `|` 把組件焊成一條 Runnable 管線，懂得用 `StrOutputParser` 收文字、
用 `RunnableParallel` 開並行分支、用 `RunnableLambda` / `@chain` 插自訂步驟，
也知道整條鏈自動繼承 `invoke` / `stream` / `batch`。這條「管線思維」是後面所有模組的骨架。

下一個模組 **M04 — 工具與工具呼叫**：讓模型不只會講話，還能呼叫你定義的函式（工具）去查資料、做計算。
那是邁向 Agent 的第一步。
