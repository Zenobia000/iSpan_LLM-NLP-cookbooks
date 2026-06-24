# M00 — 為什麼需要 LangGraph

> 第二冊開場。把第一冊學到的「單向管線」心智模型，升級成「有狀態的圖」心智模型——這是後面所有 LangGraph 模組的地基。

## 學習目標

- 認清第一冊 LCEL chain（`prompt | model | parser`）的天花板：單向、無狀態，難做迴圈、分支、記憶與人介入。
- 建立 graph 的心智模型：**state**（共享狀態）+ **nodes**（做事的函式）+ **edges**（流程走向）。
- 學會判斷一個需求該用 **chain**、用 **create_agent**，還是該自己畫 **LangGraph**。
- 理解 LangGraph 與 LangChain 的關係：LangGraph 是「有狀態的編排層」，第一冊的 `create_agent` 其實就跑在它上面。

## 為什麼需要這個？

第一冊最後我們用 LCEL 把組件串成管線：

```python
chain = prompt | model | StrOutputParser()
chain.invoke({"question": "什麼是向量?"})
```

這個寫法很美，但它的本質是一條**單向輸送帶**：資料從左邊進、從右邊出，中間每個組件只被經過一次。對「一次性問答」這很夠用，但真實應用常常不是一次性的。

舉一個具體情境：**「寫草稿 → 批改 → 沒達標就重寫 → 再批改 …… 直到達標」**。

用純 LCEL 你會發現很彆扭：

- 「沒達標就重寫」是一個**迴圈**，但管線是單向的，沒有「往回走」這件事。
- 你想記住「這是第幾次重寫」「上一版被批評了什麼」，但管線**無狀態**，每次 `invoke` 都是乾淨的、彼此不認識。
- 你只好在外面手寫一個 `while` 迴圈、自己拿變數存中間結果、自己塞回去——這時候 LCEL 已經幫不上忙，邏輯散落在管線外。

換句話說：**只要流程需要「迴圈、分支、記憶、暫停等人」，單向無狀態的 chain 就到頂了。** 我們需要一個能描述「流程會繞、會分岔、狀態會留下來」的工具——這就是 LangGraph。

## 核心概念

LangGraph 把「一個流程」描述成一張**圖（graph）**。只要記住三個詞就掌握了八成：

| 組件 | 是什麼 | 一句話心智模型 |
|------|--------|----------------|
| **State** | 一個共享的資料結構（通常是 `TypedDict`） | 整張圖的「白板」，每個節點都能讀、能寫 |
| **Node** | 一個普通的 Python 函式 `f(state) -> dict` | 「做一件事」的工人，讀白板、做事、把結果寫回白板 |
| **Edge** | 節點之間的連線 | 「做完這步換做哪步」的路標；可固定，也可依狀態決定 |

再加兩個特殊節點：`START`（圖的入口）與 `END`（圖的出口）。

把前面那個「草稿→批改→重寫」翻成圖，心智圖大概長這樣：

```text
        ┌─────────────────────────────┐
        │   State（共享白板）         │
        │   draft:   目前草稿          │
        │   review:  上一輪批改意見     │
        │   passed:  是否達標 (bool)    │
        └─────────────────────────────┘

   START ──▶ write ──▶ review ──▶ (passed?) ──否──▶ write   ← 迴圈！往回走
                                      │
                                     是
                                      ▼
                                     END
```

- **state** 把「草稿、批改意見、是否達標」留下來——這就是 chain 缺的「記憶」。
- **node**（`write`、`review`）各做一件事，互相之間只透過白板溝通。
- **edge** 裡那條「否 → 回到 write」就是 chain 做不到的**迴圈**；「是 → END」則是**分支**。

關鍵體會（會貫穿整個第二冊）：**先設計對 State，流程就會變簡單。** 流程裡的特殊情況，多半是 State 設計沒到位才冒出來的。

### 那 `create_agent` 算什麼？

第一冊用過：

```python
from langchain.agents import create_agent
agent = create_agent(model, tools=[...], system_prompt="你是助理")
```

`create_agent` 給你一個**預先畫好的標準圖**——「模型思考 → 呼叫工具 → 把結果餵回模型 → 再思考……」這個迴圈，本身就是一張 LangGraph 圖，只是 LangChain 幫你封裝好了。所以三者的關係是一條光譜：

| 需求 | 該用什麼 | 為什麼 |
|------|----------|--------|
| 固定的一次性流程（問答、翻譯、抽取） | **chain（LCEL）** | 單向就夠，最簡單 |
| 標準的「思考＋用工具」迴圈 | **`create_agent`** | 圖已經畫好，省事 |
| 流程要客製：自訂分支、特殊迴圈、多 Agent、卡關等人 | **自己畫 LangGraph** | 你要完全掌控每一步 |

一句話：**LangGraph 是有狀態的編排層，`create_agent` 是跑在它上面的一張現成圖。** 本冊就是帶你掀開蓋子，自己畫圖。

## 與前一模組的銜接

本模組疊在**第一冊全部內容**之上，特別是 M06 的 `create_agent`：

- 第一冊：你會用 `model`、訊息物件、`@tool`、LCEL 管線、`create_agent`——這些是**積木**。
- 本模組：不教新 API（只給一個極簡 teaser），而是建立**心智轉換**——從「把積木串成單向管線」升級到「把積木組成有狀態的圖」。
- 新增的只有觀念：state / node / edge，以及「何時該離開 chain、改畫圖」的判斷力。

實際的 `StateGraph` API 從 **M01** 才正式開教；本模組只讓你「看一眼圖長什麼樣」。

## 動手做

請打開同資料夾的 `lab.ipynb`，逐格執行。lab 會：

1. 用「草稿→批改→重寫」情境，讓你親眼看到純 LCEL 卡在哪。
2. 用文字畫出 state / node / edge 心智圖。
3. 跑一個極簡的 LangGraph「hello world」當 teaser（先有感覺，細節留給 M01）。

## 常見陷阱

- **以為 LangGraph 要取代 LangChain。** 不是。LangGraph 是編排層，LangChain 的 model、tool、LCEL 全都還在圖裡用。chain 沒有被淘汰，只是不再扛迴圈與狀態。
- **什麼都想畫成圖。** 一次性問答用 chain 就好，硬畫成圖只是增加複雜度。先問：這流程真的需要迴圈、分支、記憶或人介入嗎？沒有就別畫。
- **用舊版記憶 API（如 `ConversationBufferMemory`）來補 chain 的無狀態。** 那是已淘汰的寫法。本課程的「記憶」一律改用 LangGraph 的 checkpointer（M04 教），別走回頭路。
- **node 函式回傳整個 state。** 慣例是**只回傳要更新的欄位**（一個 dict），LangGraph 會幫你合併進 state。回傳整包不只多餘，之後配 reducer（M02）還會出錯。

## 小結 & 下一步

- chain 是**單向、無狀態**的輸送帶；一旦需要迴圈、分支、記憶、人介入，它就到頂了。
- LangGraph 用 **state + node + edge** 描述流程，能自然表達迴圈與分支；先設計對 State，流程就簡單。
- chain → `create_agent` → 自畫 LangGraph 是一條「掌控度由低到高」的光譜，`create_agent` 本身就是一張現成的圖。

**下一步：M01 — StateGraph 基礎**。我們會正式拆解 `StateGraph`、`add_node`、`add_edge`、`START`/`END`，從零畫出你的第一張能跑的圖。
