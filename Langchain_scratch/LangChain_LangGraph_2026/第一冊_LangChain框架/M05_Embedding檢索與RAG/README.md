# M05 — Embedding、檢索與 RAG

> 把「外部知識」接到模型上：先把文件變成可搜尋的向量，再把搜到的內容塞進 prompt，讓模型回答它沒被訓練過的私有／最新資料。

## 學習目標
- 說清楚為什麼需要 RAG，以及它和「直接問模型」差在哪。
- 建立 embedding / 向量 / 相似度的直覺：意思相近的句子，向量也相近。
- 用 `RecursiveCharacterTextSplitter` 把長文件切塊，並理解 `chunk_size` / `chunk_overlap` 的取捨。
- 用 `InMemoryVectorStore` 做 `add_texts`、`similarity_search`、`as_retriever`。
- 用 M03 學過的 LCEL，手刻一條完整 RAG 管線（不靠 `create_retrieval_chain` 黑盒）。

## 為什麼需要這個？

模型的知識凍結在它的訓練資料裡。你問它「我們公司上週的會議紀錄寫了什麼」，它不可能知道；問它今年最新的法規，它可能給你過時或乾脆編一個（幻覺）。

兩個看似可行但不夠用的做法：

- **塞進 prompt**：把整份文件貼進對話。文件一大就爆 context window，而且每次都把無關內容一起送進去，又貴又稀釋重點。
- **重新訓練模型**：成本高、週期長，知識一更新就得再來一次，私有資料也不該外流去訓練。

RAG（Retrieval-Augmented Generation，檢索增強生成）走第三條路：**先檢索、再生成**。把知識庫切成小塊存起來，每次提問時只撈出「跟這個問題最相關的幾塊」，連同問題一起交給模型。模型負責它擅長的「讀懂並組織答案」，知識則由你即時餵給它。知識更新只要更新向量庫，不用動模型。

## 核心概念

### 1. Embedding：把文字變成向量

embedding model 把一段文字壓成一個固定長度的數字陣列（向量），例如 1536 維。關鍵性質是：**語意相近的文字，向量在空間中也相近**。「貓在睡覺」和「小貓正在打盹」會很接近；「貓在睡覺」和「今天股市大跌」會很遠。

我們不再用「字面是否相同」來找資料，而是用「意思是否相近」。這就是 RAG 能跨越用詞差異找到正確段落的原因。

```python
emb = get_embeddings()
vec = emb.embed_query("貓在睡覺")   # -> list[float]，長度即向量維度
```

### 2. 相似度：怎麼比較兩個向量

把兩個向量「方向有多接近」量化成一個分數，最常見的是 cosine similarity（餘弦相似度），值越大代表越像。向量庫底層就是用這個分數，從成千上萬塊文件裡挑出最接近查詢的那幾塊。你通常不用自己算，但理解它能解釋「為什麼這篇被撈出來、那篇沒有」。

### 3. 切塊：RecursiveCharacterTextSplitter

文件不能整份丟進向量庫：一來太長、檢索回來塞爆 prompt，二來一份長文混了很多主題，向量會被「平均」掉而失焦。所以先切成小塊（chunk），每塊一個向量。

`RecursiveCharacterTextSplitter` 會優先沿著「段落 → 換行 → 句子 → 字」這種自然邊界切，盡量不把一句話從中間劈斷。

| 參數 | 意思 | 取捨 |
|------|------|------|
| `chunk_size` | 每塊的目標長度（字元數） | 太大：檢索不精準、塞太多無關內容；太小：句子被切碎、失去上下文 |
| `chunk_overlap` | 相鄰塊重疊的長度 | 留一點重疊，避免關鍵句剛好卡在切點被切斷；太大則內容重複、浪費 |

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text(long_text)   # -> list[str]
```

### 4. 向量庫與 retriever：InMemoryVectorStore

向量庫負責「存向量 + 依相似度搜尋」。`InMemoryVectorStore` 把東西放在記憶體，零安裝、適合教學與小型實驗（正式環境會換成 Chroma、PGVector 等持久化方案，介面類似）。

| 操作 | 用途 |
|------|------|
| `InMemoryVectorStore(emb)` | 建立向量庫，綁定一個 embedding model |
| `store.add_texts(chunks)` | 把每塊文字 embed 後存入 |
| `store.similarity_search(query, k=3)` | 直接回傳最相似的 k 個 `Document` |
| `store.as_retriever(search_kwargs={"k": 3})` | 包成一個 **Runnable**，可接進 LCEL |

`as_retriever()` 是關鍵：它把「搜尋」變成一個可以 `.invoke()` 的 Runnable，於是能像 M03 的任何組件一樣，用 `|` 串進管線。

### 5. 用 LCEL 手刻 RAG 管線

RAG 的資料流：**問題 → 檢索相關文件 → 把文件格式化成 context → 連同問題填進 prompt → 模型 → 解析文字**。

retriever 回傳的是 `Document` 物件清單，prompt 要的是純文字，所以中間需要一個小函式把它們接起來：

```python
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | model | StrOutputParser()
)
rag.invoke("使用者的問題")
```

這裡用了 M03 的兩個概念：字典寫法會**並行**準備 `context` 與 `question` 兩個 key；`RunnablePassthrough()` 把原始輸入（問題字串）原封不動傳到 `question`。同一個問題字串，一路送去 retriever 撈文件，也保留一份直接交給 prompt。

## 與前一模組的銜接

這個模組疊在 **M03（LCEL 與 Runnable 管線）** 之上。M03 你已經會用 `prompt | model | StrOutputParser()` 串一條鏈，也認識了 `RunnablePassthrough` 與「用字典平行準備多個輸入」的寫法。

M05 只是在這條鏈**前面多接一段檢索**：retriever 本身就是一個 Runnable，`format_docs` 用 `RunnableLambda` 的概念包成一步，於是整條 RAG 仍然只是「一串用 `|` 接起來的 Runnable」，沒有任何新魔法。Prompt 模板沿用 M02 的 `ChatPromptTemplate`，只是多了一個 `{context}` 變數。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。

## 常見陷阱
- **chunk 太大或太小**：太大會把無關內容一起撈進 prompt 稀釋重點，太小會把句子切碎失去上下文。先從 `chunk_size=300~500`、`chunk_overlap=50` 起步再調。
- **prompt 沒留 `{context}`，或忘了 `format_docs`**：retriever 給的是 `Document` 物件不是字串，直接塞進 prompt 會型別錯誤。一定要先 `format_docs` 攤平成文字。
- **以為 RAG 能保證正確**：模型只會根據撈回來的 context 回答。檢索撈錯、或知識庫根本沒這筆，它一樣可能瞎掰。可在 prompt 明確要求「context 沒提到就說不知道」。
- **沿用舊版黑盒鏈**：不要用 `create_retrieval_chain` / `RetrievalQA` / `load_qa_chain`，本課程一律用 LCEL 手刻，資料流才看得清楚。

## 小結 & 下一步

你現在能把任意一段外部知識變成可檢索的向量庫，並用 LCEL 手刻一條 RAG 問答管線：切塊 → embed → 存入向量庫 → 檢索 → 填進 prompt → 生成。核心心法是「retriever 也只是一個 Runnable」，所以 RAG 不過是 M03 那條鏈的延伸。

下一個模組 **M06（create_agent 與 Middleware）**：到目前為止流程都是我們寫死的固定管線。接下來讓模型自己決定「要不要呼叫工具、呼叫哪個、何時停」，把固定鏈升級成會自主決策的 Agent。
