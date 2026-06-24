# ihower LLM 應用開發工作坊 — 課程架構分析

> 教材來源：`ihower-LLM-workshop-v25-2025-05-28-課後訂正版.pdf`（392 頁 Keynote 簡報）
> 講師：張文鈿（ihower），AI Engineer 電子報作者
> 版本：v25（2025-05-28 課後訂正版）
> 分析日期：2026-06-25

---

## 一、課程定位與教學取向

| 面向 | 說明 |
| :--- | :--- |
| **視角** | 以**全端開發者（full-stack developer）**觀點切入，著重「應用開發」與「API 串接」 |
| **不涵蓋** | LLM 理論、Transformer 架構、機器學習/深度學習、模型訓練、數學線代統計、AI Coding |
| **主要模型** | OpenAI、Google Gemini、Anthropic Claude 系列 |
| **授課形式** | 投影片講述 + Google Colab 程式碼實作（範例採 OpenAI Python SDK，含 Structured Outputs） |
| **核心理念** | Prompting 是 AI 應用的根本；生成式 AI 是「機率軟體」，需以評估驅動開發 |

**三大主軸（Three Parts）對應三天課程：**

```
Part 1  Prompting 篇         → LLM API 基礎 + Prompt 設計與工程化評估
Part 2  工具串接 RAG 篇       → Workflow 編排 + Embeddings + RAG + 安全
Part 3  進階 Agents 篇        → Function Calling → Agent → Multi-Agent → Fine-Tuning
```

---

## 二、完整課程架構（Curriculum Tree）

### Part 1：Prompting 篇（Day 1）

#### 1-1　OpenAI API：把 LLM 當作 HTTP API
- LLM API 基本參數
- OpenAI models 認識與定位/價格比較
- Temperature 溫度
- Context Window（token 字數限制）
- Tokenizer 分詞器
- Completion API vs. Chat Completion API（原理、已過時的 Completion API）
- ChatGPT 是基於 LLM API 的應用 App
- 使用 LLM API 的注意事項、OpenAI Model Spec
- 🧪 Colab 實作

#### 1-2　Multi-Model 多模態
- 聽、說、讀、寫四種模態總覽
- 圖片辨識（Vision）→ Case Study：OCR 應用
- 語音辨識（Speech-to-Text）
- 語音合成（Text-to-Speech）
- 圖片生成（Image Generation）
- 影片多模態
- 🧪 Colab 實作

#### 1-3　Prompt Design 提示設計
- 原則 1：寫清晰且具體的指令
  - 設定 Persona 角色
  - 給範例（Few-shot example）
- 原則 2：給模型思考時間
  - Chain-of-Thought（CoT）
  - Few-shot vs. CoT 比較
- 原理補充：自回歸模型局限性、為何要給思考時間
- Hallucination 幻覺問題與緩解（提供 context、要求引用）
- 🧪 Colab 實作

#### 1-4　Reasoning Models 推理模型
- 什麼是 Reasoning Models（test-time compute）
- 運作方式與適用場景（planning / coding / 複雜推理）
- Prompting Best Practices：推理模型「給目標」vs. 非推理模型「給詳細指令」
- 適用場景比較、Overthinking 現象
- 觀點：API 串接應用仍以非推理型模型為主力
- 🧪 Colab 實作

#### 1-5　Prompt Engineering 與評估（Evaluation）★ 重點章節
- **Prompt Design ≠ Prompt Engineering**（不只設計，還要工程化）
- 什麼是評估 Evaluation？（以書籍分類為例計算準確率）
- **5-Level 評估成熟度等級（ihower 原創）：**
  - Level 0：外（不評估）
  - Level 1：入門
  - Level 2：專業（人工建 dataset、迭代 Prompt v1→v3；以 AI 產 Prompt、合成測試資料）
  - Level 3：自動化（自動化評估，braintrust 框架，有標準答案 / 無標準答案兩類）
  - Level 4：神乎其技（自動最佳化）
- 自動化評估類型：① 有標準答案（程式打分/Assertion）② 無標準答案（LLM-as-Judge）③ 有參考資料/答案（RAG）
- LLM 評估框架、LLMOps（langsmith、promptfoo）
- 自動最佳化：gpt-prompt-engineer、DSPy、TextGrad
- Dataset 需要幾筆？評估的數據輪次
- 🧪 Colab 實作

#### 1-6　LLM Ecosystem 生態系
- 有哪些 LLM 模型（含 Mistral AI 等）
- 開源 LLMs、參數量 vs. 訓練資料量、Open Source vs. Open Weights
- LLM 評測排行榜
- LLM 應用開發框架、LLM Ops、LLM APIs and Hosting
- 要不要自架 hosted LLM？閉源 API vs. 自架開源、本機部署方案
- 📌 Part 1 總結 + Day 1 Q&A

---

### Part 2：工具串接 RAG 篇（Day 2）

> 複習：Prompt 基本結構、Reasoning Prompt 基本結構

#### 2-1　Chaining Prompts（Agentic Workflow 基礎）
- 什麼是 Chaining Prompts、為何需要
- 案例：長文本摘要（搭配 Whisper 語音辨識）
- **五大 Workflow 模式（複習）：**
  - Prompt Chaining
  - Routing
  - Parallelization
  - Orchestrator-Workers
  - Evaluator-Optimizer
- 🧪 Colab 實作

#### 2-2　Embeddings
- 語意搜尋 vs. 關鍵字搜尋
- 能代表語意的 Embedding、Semantic Search
- 應用案例：AI 產生新聞內容
- Multimodal Embeddings
- 🧪 Colab 實作

#### 2-3　RAG（Retrieval Augmented Generation）
- 針對資料做 QA 問答（Chat with Your Data）
- 自行開發的向量檢索 vs. 內部資料檢索
- 為何做 RAG
- 🧪 Colab 實作

#### 2-4　Advanced RAG ★ 重點章節（13 個進階主題）
- Naive RAG 的問題
- 1. 資料載入與 Parsing
- 2. Chunking 切塊策略
- 3. 挑選 Embedding 模型
- 4. Multi-Model RAG
- 5. 向量搜尋資料庫（Vector Search 原理、Vector DB 關注點、是否需要專用 Vector DB）
- 6. Hybrid Search 混合檢索
- 7. Context Enrichment（small-to-big）
- 8. Multi-Vector 多重索引策略
- 9. Contextual Embeddings（Anthropic cookbook）
- 10. Query Optimization 查詢最佳化
- 11. 兩階段檢索 + Reranker 模型（合併排序）
- 12. Post-Processing
- 13. Agentic RAG（Deep Search 案例）
- 路線補充：GraphRAG、More than just RAG、OpenAI DevDay RAG 案例
- **RAG 評估：** Precision/Recall、Answer Relevance、Groundedness（幻覺）、Context Relevance、答案正確性 prompt 範例、Ragas 框架、合成有參考答案 dataset
- RAG 的未來：Long Context Window（還需要 RAG 嗎？）
- 🧪 Colab 實作

#### 2-5　LLM Security
- OWASP Top 10 for LLM Applications
- 1. Content Safety（自動越獄 prompting 策略）
- 2. Prompt Hacking（如何 hack 出 ChatGPT system prompt、防禦方式）
- 🧪 Colab 實作
- 📌 Part 2 總結 + Day 2 Q&A

---

### Part 3：進階 Agents 篇（Day 3）

#### 3-1　Function Calling 與 Agent
- Function Calling 串接外部工具（Step 1-5 流程）
- 背後原理：LLM 廠商 API 層的 tools 定義轉換
- 包成 loop → Agent 功能（`run_full_turn` 函式實作）
- Function Calling 最佳實務
- 🧪 Colab 實作

#### 3-2　Conversational Agent 對話型 Agent
- Streaming 改進使用體驗
- 處理過長的聊天記錄
- 打造 Single Agent 元件（支援 agent 參數的 `run_full_turn`）
- OpenAI / Anthropic 的 Agent 建議 talk
- Agents 函式庫與框架、撰寫 Agent Instructions
- 提升 Agent 呼叫工具能力（Pro tip：`think` tool）
- 知識庫文件多時的架構設計
- DAG orchestrators vs. Agent（The Bitter Lesson 痛苦的教訓）
- 課後補充：推理模型的 Function Calling
- 🧪 Colab 實作（OpenAI Agents SDK）

#### 3-3　Multi-Agents 與 Agentic Workflow ★ 重點章節
- 何時考慮 Multi-Agents
- **Handoff 交接機制**（`run_full_turn_v2`、雙層客服案例 → "OpenAI Swarm"）
- Multi-Agent 設計關鍵：
  - 協作方式：Supervisor / Agents as Tools
  - 資料共享與傳遞（chat history 共享）
- **Agent 之間對話（Autogen）：** Two-Agent Chat、Sequential Chats、Group Chat 與評論
- Agentic Workflow：代表框架、Building Effective Agents（Anthropic 2024/12）
- **何時該用 Agent？** 兩個硬需求；Form 型（一次性）vs. Chat 型（互動）應用
- 開放性任務：用推理模型生成 plan（SOP / 偽代碼）→ 陽春版 Agent 設計
- Work flow by Small / Focused Agents（個人觀點 2025/5）
- 案例：RFP Response Generation Workflow（Human-in-the-Loop）
- 補充：Google ADK 對照
- **進階 Agent 主題：**
  - Code Interpreter（開發方式）
  - Computer Use（GUI Agent）
  - **Model Context Protocol（MCP）：** 實作原理、server/client 開發、推薦 servers、組合性、問題、Roadmap（Registry / Server Discovery）
  - Computer Use vs. MCP 比較
  - CodeAgent（不靠 Function Calling 的另一種實現）
  - Agent 的評估方式
- 🧪 Colab 實作

#### 3-4　Fine-Tuning 微調
- 什麼是類神經網路、Neural network internals
- LLM 如何訓練：Base model → SFT 對話微調 → 強化學習 RL
- Andrej Karpathy 的比喻與演講推薦
- 如何 Fine-tuning：OpenAI 後台 SFT 示範、開源模型（Llama）
- **要不要 Fine-tune？** SFT vs. RAG 的選擇；成功案例（調整語氣風格行為）
- 🧪 Colab 實作
- 📌 Part 3 總結 + Day 3 Q&A

---

## 三、課程收尾觀點（最後總結）

1. **Prompting 是 AI 應用的根本（生命線）**
2. **生成式 AI 是機率軟體** — 準確率要多高才能上 production？需設計「AI 複合系統」
3. **新時代 AI Engineer 的崛起** — 技能差異比較

---

## 四、課程設計特色（系統分析觀察）

| 特色 | 說明 |
| :--- | :--- |
| **能力遞進清晰** | 單一 API 呼叫 → Prompt 設計 → 工作流編排 → RAG 檢索 → Agent → Multi-Agent，由淺入深 |
| **評估驅動為核心軸線** | 1-5「5-Level 評估成熟度」貫穿全課；RAG（2-4）與 Agent（3-3）皆回扣評估方法 |
| **工程化而非理論化** | 全程強調「可上 production」「可自動化測試」，刻意排除模型訓練理論 |
| **理論 + 實作並行** | 每個小節幾乎都對應一段 Colab 實作（全課約 18+ 次實作時間）|
| **跟進業界最新實踐** | 涵蓋 Reasoning Models、MCP、Computer Use、Agentic Workflow 等 2024-2025 最新主題 |
| **觀點鮮明** | 講師對 Agent vs. Workflow、Fine-tune vs. RAG、是否需專用 Vector DB 等皆給出實務判斷 |

---

## 五、學習路徑建議（依角色）

- **後端/全端工程師入門：** Part 1（1-1～1-3）→ 1-5 評估 → Part 3（3-1 Function Calling）
- **想做企業內部知識庫：** Part 1 基礎 → Part 2 全部（重點 2-4 Advanced RAG + RAG 評估）→ 2-5 Security
- **想做 AI Agent 產品：** 在掌握 Part 1/2 後，重點攻 Part 3（3-1 → 3-2 → 3-3），並補 MCP
- **想提升模型專屬能力：** 先判斷 3-4「SFT vs. RAG」決策，再決定走 Fine-Tuning 或 RAG 路線

---

## 六、Prompt 技巧的演化視角：為什麼初學仍需教傳統概念

> 教學設計依據：CoT、ToT 等傳統技巧並非「過時雜項」，而是模型演化的化石證據。學生看不懂今日模型，正是因為沒看過它的前世。

### 核心命題

隨著模型能力提升，CoT/ToT/ReAct 這類 prompt 技巧逐步被模型**內化成內建功能**。表面上它們像四個獨立技巧，本質卻是同一句話：

> 模型內部還不會「想」，所以我們把「想」的過程用 prompt 外掛到輸入裡。

一旦模型透過 RL／test-time compute 學會在內部自己想，這些外掛便塌縮成模型的一個能力。因此它們不是要淘汰的舊知識，而是**同一條演化線上的不同切片**——這正是「為什麼會有不同版本的模型」最具體的答案：

> **模型版本演進 = 把昨天還要你手動寫的 prompt 技巧，今天變成內建。**

### 過去技巧 → 今日內建能力的對照

| 過去的 Prompt 技巧（外掛思考） | 被內化成今日的什麼 | 課程裡的位置 |
| :--- | :--- | :--- |
| **CoT**「Let's think step by step」 | Reasoning Models 的內建推理鏈（o 系列 / Claude extended thinking） | **1-3 → 1-4 就是這條線** |
| **ToT / Self-Consistency** 多路徑探索後投票 | test-time compute 在內部分支、驗證、回溯 | 1-4 Reasoning Models |
| **Few-shot examples** | Instruction tuning 後 zero-shot 就夠好，範例需求大幅下降 | 1-3 原則 1 |
| **ReAct**（Reason+Act 交錯） | 原生 Function Calling / Agent loop | 3-1 Function Calling |
| **Self-Reflection / 自我批改** | 推理模型自己 verify、Evaluator-Optimizer workflow | 1-4、2-1 |
| **手動塞 context 進 prompt** | Long Context Window（部分吃掉 RAG） | 2-4 結尾「還需要 RAG 嗎？」 |

### 課程結構本身即一堂演化史

**1-3（CoT）緊接 1-4（Reasoning Models）這個排序，不需額外章節，本身就是「before → after」的最佳教案。** 學生先親手寫「請一步一步思考」，下一節立刻看到廠商用 RL 把這句話燒進模型——這就是模型版本演進最具體的演示。

### 教學主線（建議貫穿全課的一句話）

不要把這些技巧當「工具清單」教（會讓學生以為要全部背、全部用，反而製造混亂）。改用一條主線框住：

> 「Prompt engineering 的歷史，就是不斷把『人要手動補的思考鷹架』還給模型的歷史。你今天學的技巧，有一半明年會變成模型預設行為——但你得知道它原本長怎樣，才看得懂模型在幫你做什麼、什麼時候它還沒幫你做。」

如此一來，CoT/ToT 就從「過時知識」升級為**閱讀模型能力的座標系**：學生真正學到的不是單一技巧，而是「怎麼判斷一個新模型內建了哪些、還缺哪些」——這個判斷力比任何單一技巧都活得久。
