# M7　校準層：評估驅動開發（EDD）

> 控制權階梯第七層，課程靈魂章。對應 notebook：`07-calibration-eval/01~03`。
> 全篇以 2026 現行做法描述。

---

## 1. 開場：這一層要收回什麼控制權？

**把「對 / 錯」的裁決權，從你的感覺手裡收回到一份可重跑的評估資料集裡。**

前六層你做了很多事：寫了 spec（M2）、鎖了 schema（M3）、接了檢索（M4）、配了 harness（M5）、佈了跨模型互審（M6）。每一步你都「覺得」變好了。但「覺得」不是工程 —— 它不可重現、不可比較、不可累積。改了一行 system prompt，下次回答看起來順了，你就以為修好了；其實你只是換了一組沒被測到的失敗。

這一層要終結這種自我欺騙。**把「感覺有改善」變成「量測證明改善」**：建立一份固定的評估資料集，每次改動都跑同一份、看分數是升是降。你的工作從「不斷微調咒語、靠手感驗收」，升級為「定義判準、跑迴圈、用數字決定下一步」。

這呼應全課心法：**你可以外包你的思考，但不能外包你的理解。** 評估就是你「理解」的外顯形式 —— 你說得出什麼算對、抓得出哪裡錯、證明得了改進，才算真正掌控了這個系統。

---

## 2. 學習目標

1. 建立 **5-Level 評估成熟度**心智模型，定位自己團隊現在在哪一格、下一步往哪走
2. 用**二元判斷（對 / 錯）**取代模糊的 1–10 分，打造可重跑的 evals，並分清三類評估該用哪種裁判
3. 跑完整**回饋迴圈**：評分 → 失敗模式聚類 → 改進 → 同 dataset 再評估，並依正確優先序決定改哪裡
4. 判斷**何時該微調**，會用合成資料產 chat JSONL，並理解 SFT / RAG / SFT+RAG 的取捨

---

## 3. 核心概念脈絡

### 3.1 5-Level 評估成熟度 —— 開篇心智框架

評估不是「有沒有做」的開關，而是一條成熟度階梯。先定位自己在哪一格，才知道下一步往哪推。

| Level | 名稱 | 你怎麼判斷「變好了」 | 特徵 |
| :--- | :--- | :--- | :--- |
| **L0** | 外（無評估） | 自己看一眼、覺得順 | 不可重現，改 A 壞 B 也不知道 |
| **L1** | 入門 | 手動跑幾個 case 對照 | 有意識比對，但靠人眼、不成集 |
| **L2** | 專業 | 固定一份 dataset + 明確判準逐筆評分 | 有 ground truth 與二元裁判，可比較 |
| **L3** | 自動化 | CI 每次改動自動跑全套 evals | 評估進入流程，分數會擋下退步 |
| **L4** | 自動最佳化 | 系統依評估分數自動搜尋更好的 prompt / 參數 | 人定義判準，機器跑優化迴圈 |

> 心法：**每升一格，「判斷對錯」這件事就從人腦往系統移一步。** 本章帶你穩穩從 L0/L1 走到 L2/L3，並認識 L4 的樣貌（3.7）。多數團隊卡在 L1 不是因為難，而是因為一直用「感覺」代替「資料集」。

### 3.2 評估驅動開發（EDD）：放棄 1–10 分，改二元判斷

直覺上我們想給輸出打 1–10 分。**這是個陷阱。** 沒有人說得清 7 分和 8 分差在哪，連同一個人隔天打的分都不一樣 —— 這種尺度不可重現，累積不出可比較的數字。

EDD 的核心動作是把判準**收斂成二元**：這筆輸出，**對，還是錯？** 一旦只剩兩個格子，特殊情況就消失了：

- 你被逼著把「什麼算對」寫成一句說得清的判準（這正是 M2 spec 的 Success criteria 直接延伸過來）。
- 分數變成可加總的比率（通過率 = 對的筆數 / 總筆數），跨次數、跨版本都能比。
- 失敗一翻兩瞪眼，不會躲在「6.5 分還行吧」的模糊地帶。

> 與 M2 的接線：M2 你在 spec 裡寫的「Success criteria」當時是一句願望，到了 M7 它**變成可執行的二元裁判**。spec 寫得越具體，這裡的 eval 越好寫 —— 這就是為什麼 M2 一直叮嚀「Success criteria 要能直接拿去當 eval」。

### 3.3 三類評估：依「有沒有標準答案」選裁判

二元判斷由誰來下？取決於這個任務的答案性質。三類，對應三種裁判：

| 任務性質 | 裁判 | 怎麼判對錯 |
| :--- | :--- | :--- |
| **有標準答案** | 程式 / assertion | 直接 `==`、正則、schema 驗證、數值比對 —— 最便宜最可靠，能用就用 |
| **無標準答案** | LLM-as-judge | 用另一個模型依明確 rubric 判「對 / 錯」（如：是否具體、是否符合語氣） |
| **有參考資料** | RAG 專用指標 | 拿檢索到的 context 當基準，判答案有沒有忠於來源（見 3.4） |

> 順序很重要：**能用程式判就別請 LLM 當裁判。** LLM-as-judge 是給「對錯無法用規則窮舉」的開放式任務用的後備手段，不是預設。它本身也是個機率機器，rubric 要寫到二元、避免讓它自己打 1–10。

### 3.4 RAG 評估：給「有參考資料」的任務量身打造

RAG 系統（M4 的產物）最難評，因為它同時可能在兩個地方出錯：**檢索錯**（沒撈到對的 chunk）和**生成錯**（撈對了卻答歪）。EDD 在這裡分四步推進。

**① 合成評估資料 —— 讓 LLM 對 chunk 自己出題。** 你沒有現成的 Q&A 測試集，但你有文件。做法是把文件切成 chunk，請模型針對每個 chunk 產出 question / answer，並要它附上**依據的原文片段（reference）**。這份 (question, expected answer, reference) 就是你的評估資料集 —— 自動生成、可重跑、覆蓋全文。

**② No RAG vs RAG 對照。** 先建一條 baseline：不給 context、直接問模型（No RAG）。再跑開了檢索的版本。**沒有對照組，你證明不了 RAG 真的有貢獻** —— 也許模型本來就會答，檢索只是擺設。對照是「量測證明改善」最基本的形式。

**③ 用 RAGAS 四指標拆開診斷。** 借 Braintrust + autoevals 這組工具，RAGAS 把「答得好不好」拆成四個可分別歸因的維度：

| 指標 | 量什麼 | 低分代表 |
| :--- | :--- | :--- |
| **Context Recall** | 該撈的 context 撈齊了嗎 | 檢索漏東西 → 改檢索（chunking / top-k / reranker） |
| **Context Precision** | 撈進來的 context 乾不乾淨 | 撈太多雜訊 → 改 reranker / 收斂 top-k |
| **Faithfulness** | 答案有沒有忠於 context（不亂編） | 模型在幻覺 → 改生成 prompt、強制引用 |
| **Answer Correctness** | 最終答案對不對 | 端到端結果指標 |

> 洞察：**四指標的價值在「歸因」。** 單一通過率告訴你「壞了」，RAGAS 告訴你「壞在檢索還是生成」—— 這直接決定 3.5 回饋迴圈裡你該動哪一層。這也兌現了 M4 驗收標準裡那句「用 M7 指標證明帶 reranker 的 RAG 優於 vanilla 版」。

### 3.5 回饋迴圈：評分 → 聚類 → 改進 → 再評估

有了評估資料集，改進就不再是亂槍打鳥，而是一個可重複的迴圈：

```
跑評估 ──→ 撈出失敗案例 ──→ 聚類失敗模式 ──→ 針對性改進 ──→ 同 dataset 再跑
   ▲                                                              │
   └──────────────────  比較分數，升了才保留  ◀───────────────────┘
```

關鍵動作有兩個。

**失敗模式聚類，不是逐筆救火。** 把低分案例丟給模型，請它歸納出 1–3 個**共通失敗模式**（例如「回答太籠統，沒給實際數字」「沒引用來源」）。針對「模式」改一次，勝過針對「個案」補一百個 if。這正是 Linus 式的好品味 —— 消除類別，而不是堆特殊情況。

**改進優先序：先便宜後昂貴。** 同一個失敗，有很多種修法，成本天差地遠。永遠由便宜往貴試，每動一步就回去跑同一份 dataset 驗證有沒有升：

| 優先序 | 手段 | 成本 | 何時升級到下一步 |
| :--- | :--- | :--- | :--- |
| 1 | **改 system prompt** | 最低 | 改判準、加約束就能修的失敗 |
| 2 | **加 few-shot** | 低 | 格式 / 風格對不齊（接 M2 few-shot） |
| 3 | **改檢索** | 中 | RAGAS 顯示 context recall/precision 不足 |
| 4 | **微調** | 最高 | 前三項都到頂，仍有穩定、可大量舉例的系統性偏差 |

> 鐵律：**沒跑同一份 dataset 比較前後分數，就不准宣稱「改好了」。** 迴圈的靈魂是「同 dataset 再評估」這個閉環 —— 拿掉它，你又退回 L0 的感覺主義。

### 3.6 微調：什麼時候、怎麼做

微調是優先序的**最後一步**，不是第一招。它貴、慢、會把模型綁死在訓練分布上，只有當「前三項都試過、仍有系統性偏差、而且你能大量舉出正確示範」時才出手。

**合成資料 → chat JSONL。** 訓練資料的格式是每行一個 JSON 物件，內含一段 `messages`（system / user / assistant 三角色）。資料哪來？同樣用 3.4 的合成法：對文件 chunk 產生 (question, answer)，再包成 chat 格式：

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful chatbot about finance"}, {"role": "user", "content": "<question>"}, {"role": "assistant", "content": "<answer>"}]}
```

**SFT vs RAG vs SFT+RAG —— 通常 SFT+RAG 最佳。** 兩者解的是不同問題，不是二選一：

| 方案 | 擅長 | 限制 |
| :--- | :--- | :--- |
| **RAG** | 注入會變動的事實知識、可引用、可即時更新 | 教不會「風格 / 行為 / 領域語感」 |
| **SFT（微調）** | 內化穩定的風格、格式、行為偏好 | 知識會過期，更新要重訓 |
| **SFT+RAG** | 用 SFT 定行為、用 RAG 供事實 —— **多數場景最佳** | 工程複雜度最高 |

> 判準：**知識常變 → RAG；行為要穩 → SFT；兩者都要 → SFT+RAG。** 別把微調當成「讓模型變聰明」的萬靈丹，它是「讓模型穩定地照你的樣子做事」的工具。

### 3.7 降低人類審核成本：從 Markdown 到 HTML 互動介面

EDD 的盡頭仍需要人 —— LLM-as-judge 要校準、失敗模式要拍板、微調資料要抽查。但這裡有個 2026 的成本反轉：**當 AI 產出變得極便宜，人類「理解產出」的成本反而變成瓶頸。** 你一天能讓模型生成上千筆候選答案，卻讀不完上千段超長 Markdown。

解法是換載體：**放棄超長 Markdown 純文字審核，改用 HTML 互動式介面。** 把 AI 產出渲染成可操作的元件，讓人類的審核動作從「逐行讀」變成「快速裁決」：

- **卡片化**：一筆候選一張卡，approve / reject 用拖曳，幾秒一筆。
- **並排比對**：把 v1 vs v2、No RAG vs RAG 並列，差異一眼可見，不必腦中對照。

> 洞察：這是上下文工程「壓縮」手法用在**人**身上 —— 不是壓縮給模型看的 context，是壓縮給人看的審核負擔。介面設計本身就是評估效率的一部分。

### 3.8 自動最佳化（L4，理論視野）

迴圈跑熟了，下一個問題自然浮現：**連「改 prompt」這步能不能也交給系統？** 這就是成熟度 L4。人只負責定義評估判準，機器拿這個判準當目標函數，自動搜尋更好的 prompt 或參數：

- **gpt-prompt-engineer**：生成多版候選 prompt，用你的 evals 互相 PK，留分數最高的。
- **DSPy**：把 prompt 當「可編譯的程式」，依評估指標自動優化各模組的提示。
- **TextGrad**：把評估回饋當成「文字形式的梯度」，反向傳播去修改 prompt。

> 定位：這三者**本章只做概念介紹，不要求落地**。但它們揭示了評估的終極價值 —— **一份好的 eval，是自動最佳化的目標函數。** 你在 L2/L3 打磨的評估資料集，正是 L4 賴以運轉的燃料。沒有可靠的二元評估，自動最佳化只是在優化噪音。

---

## 4. 程式碼導讀

> 指向 `07-calibration-eval/` 各 notebook，僅列關鍵 pattern。
> **二元 evals 的判準寫法、HTML 審核介面為本章新增教材**（notebook 之外的補充段落 3.2 / 3.7）。

**`01-rag-evaluation.ipynb` —— 三類評估與 RAG 四指標（本章主軸）**
- 初始化：`wrap_openai(OpenAI(...))` + `init_logger(project=...)`，用 `@traced` 標注被評估函式
- 合成評估資料：Pydantic `QAPair`（含 `reference` / `question` / `answer`）+ `client.responses.parse(...)` 對每個 chunk 產 Q&A
- 對照組：`simple_qa()`（No RAG）vs `ask_with_rag()`（檢索 `collection.query()` 帶 context 生成）
- 跑評估：`Eval(name=..., experiment_name="No RAG"/"Naive RAG", data=eval_dataset, task=..., scores=[Factuality(...)])`
- RAGAS 四指標：`from autoevals import AnswerCorrectness, ContextRecall, ContextPrecision, Faithfulness`，用 `EvalAsync(...)` 一次跑四個 score，比對 No RAG / Naive RAG / Ragas 三個 experiment

**`02-feedback-loop.ipynb` —— 回饋迴圈的最小骨架**
- 撈失敗：從帶評分的 logs 篩出 `score <= 3` 的低分案例
- 聚類失敗模式：`analyze_failures()` 用 `responses.create(text={"format":{"type":"json_object"}})` 請模型歸納 `{"patterns":[{"name","fix"}]}`
- 針對性改進：把 `fix` 併入 `improved_system`，做出 `answer_v2()`
- 觀察重點：同一題「退貨要多久？」前後對照，驗證改進後更具體 —— 這就是「同 dataset 再評估」的縮影

**`03-fine-tuning-synthetic-data.ipynb` —— 合成資料與微調**
- 解析文件：`PdfReader(...).pages` 取每頁文字
- 合成 Q&A：對每頁 `responses.create(..., text={"format":{"type":"json_object"}})` 產 question/answer
- 轉 chat JSONL：包成 `{"messages":[{system},{user},{assistant}]}`，逐行寫入 `training_data.jsonl`
- 微調後比對：用微調 model id（`ft:...` 格式）對同一組問題作答，與基底模型對照差異

---

## 5. 練習與驗收

**練習**

1. 挑一個你前面模組做過的 prompt（客訴分類、報告摘要、RAG 問答皆可），把它的 spec「Success criteria」改寫成一條**二元判準**（這筆對 / 錯怎麼判）。
2. 建一份**至少 10 筆**的固定評估資料集（有標準答案的用程式判；開放式的寫 LLM-as-judge rubric；RAG 的用合成 Q&A）。跑第一輪，記下通過率。
3. 撈出失敗案例，請模型**聚類出 1–3 個失敗模式**。
4. 依優先序（system prompt → few-shot → 檢索 → 微調）挑**最便宜**能修的那層改一次。
5. 對**同一份** dataset 再跑一輪，比較前後通過率。

**驗收標準**

- 交出一份固定評估資料集 + 兩次評估結果（改進前 / 改進後），通過率**有明確、可重現的上升**。
- 說得出這次改進對應哪個失敗模式、為什麼選那一層下手（而非更貴的層）。
- 若做的是 RAG：附 No RAG vs RAG 對照，並用 RAGAS 至少一個指標說明改進歸因於「檢索」還是「生成」。

> 通過標準：你能用**數字**而非形容詞回答「它變好了嗎」。如果你的答案是「感覺比較順」，你還在 L0；如果是「通過率從 60% 升到 85%，主要修掉『回答太籠統』這個模式」，你才真正站上了校準層。

---

## 補充教材

> 以下兩份產物對應 3.2（二元判準）與 3.7（HTML 審核介面）的落地細節。它們難用 notebook 完整呈現，改以可直接複製取用的範本形式放在講義裡：拿走骨架、填上你自己的判準與欄位即可上線。

### A. 二元 rubric 設計指南

3.2 說「放棄 1–10 分、改二元判斷」是方法論原則；這裡給可操作的寫法。

**為什麼裁判不准自己打 1–10。** 把「給幾分」這個決定交給 LLM，等於把判準的詮釋權又還給了模型 —— 它今天覺得 7 分、明天覺得 8 分，你拿到的是一串不可重現的浮點數。**你的工作是把判準窮舉成二元，讓裁判只能回 pass / fail。** 模型不再「評價」，只負責「核對」你寫好的條件成立沒成立。

**把模糊判準收斂成二元的三步：**

| 步驟 | 動作 | 反例 → 正例 |
| :--- | :--- | :--- |
| 1 | 把形容詞拆成可觀察的條件 | 「回覆要有禮貌」→「結尾有致歉或感謝語句」 |
| 2 | 用 AND / OR 把多個條件接成一句可逐項核對的判準 | 「整體不錯」→「有正面回應 **AND** 沒承諾做不到的事」 |
| 3 | 為每個條件想一個會踩雷的負例，確認判準擋得住 | 「我幫您全額退款」（無權限承諾）應判 fail |

> 鐵律：**rubric 裡每一個條件都要能被一句 yes/no 回答。** 只要出現「夠不夠專業」「是否足夠詳細」這種需要打分的詞，就還沒收斂完 —— 回到步驟 1 把它拆開。

**二元評分 prompt 骨架。** 裁判模型的輸出鎖成 `{pass, reason}`：`pass` 是布林、`reason` 一句話交代依據（用來日後抽查校準裁判本身）。

```python
JUDGE_PROMPT = """你是嚴格的評分裁判。只依照下列判準逐項核對，不要自行加碼或寬鬆放行。

[判準]（全部成立才算 pass）
{rubric}

[待評輸出]
{candidate}

逐項核對每一條判準。只要有任何一條不成立，pass 即為 false。
僅輸出 JSON：{{"pass": <true|false>, "reason": "<一句話，指出哪條判準成立或哪條不成立>"}}
"""

# 裁判呼叫：輸出強制為 JSON 物件，便於程式聚總通過率
resp = client.responses.create(
    model="<judge-model>",
    input=[{"role": "user",
            "content": JUDGE_PROMPT.format(rubric=rubric, candidate=candidate)}],
    text={"format": {"type": "json_object"}},
)
verdict = json.loads(resp.output_text)   # {"pass": true/false, "reason": "..."}
```

**填好的具體範例 —— 評「客服回覆」。** 判準：這則客服回覆必須**同時**滿足「有正面回應顧客問題」**AND**「沒承諾做不到的事」。

```python
rubric = """
1. 有正面回應：回覆直接針對顧客的問題給了答覆或下一步，而非顧此言他、轉移話題。
2. 沒承諾做不到的事：未對退款金額、到貨時間、權限範圍做出客服無權保證的承諾
   （例如「保證明天到」「一定全額退」屬越權承諾）。
兩條皆成立才 pass；任一不成立即 fail。
"""

candidate = "您好，關於您的退貨，我已為您建立退貨單，物流到件後 3–5 個工作天會完成退款；實際入帳時間以發卡行為準。"
# 預期裁判輸出：
# {"pass": true, "reason": "正面回應退貨流程，且退款時間以發卡行為準、未越權保證確切入帳日"}

candidate_bad = "這個我不太確定，您再等等看吧，應該很快就會退給您，放心啦一定會退全額。"
# 預期裁判輸出：
# {"pass": false, "reason": "未正面回應（要顧客自己等），且『一定全額退』為越權承諾"}
```

> 把通過率算出來就是一行：`pass_rate = sum(v["pass"] for v in verdicts) / len(verdicts)`。這個比率才是你跨版本、跨次數比較的那個數字 —— 它能加總、能重跑，1–10 分做不到。

### B. HTML 審核介面

3.7 指出 2026 的成本反轉：AI 產出極便宜，**人類「理解產出」反而是瓶頸**。一天上千筆候選你讀不完純文字，但你能在卡片上幾秒裁決一筆。這裡給最小可用的審核卡片範本。

**審核卡片要顯示哪些欄位。** 一張卡片承載一筆候選的所有裁決依據，讓人不必跳檔、不必腦中對照：

| 欄位 | 放什麼 | 為什麼要它 |
| :--- | :--- | :--- |
| **輸入（input）** | 餵給模型的原始問題 / 資料 | 沒有輸入就無從判斷輸出對不對 |
| **AI 輸出（output）** | 模型這次產出的答案 | 審核主體 |
| **信心分數（confidence）** | 模型或評估管線給的分數 | 低分優先看，引導注意力分配 |
| **grounding 來源（source）** | 答案所依據的檢索片段 / 引用（接 M4） | 一眼核對有沒有亂編、有沒有忠於來源 |
| **通過 / 退回按鈕** | approve / reject 兩個動作 | 把審核動作收斂成二元裁決，呼應 A 段精神 |

> 設計原則：**卡片只放「做出 pass / reject 決定所需的最小資訊」。** 多一個欄位就是多一份人類認知負擔 —— 這是把上下文工程的「壓縮」手法用在人身上（見 3.7）。

**最小審核卡片 HTML 骨架。** 單檔、無相依套件，可直接用瀏覽器開啟；把候選資料填進卡片、按鈕掛上送出邏輯即可。

```html
<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <style>
    .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px;
            max-width: 640px; margin: 12px auto; font-family: system-ui; }
    .field { margin: 8px 0; }
    .label { font-size: 12px; color: #888; }
    .conf { font-weight: 600; }
    .conf.low { color: #c0392b; }            /* 低分標紅，引導優先審 */
    .source { font-size: 13px; color: #555; border-left: 3px solid #ccc;
              padding-left: 8px; }
    .actions button { padding: 8px 16px; margin-right: 8px; cursor: pointer; }
    .approve { background: #2ecc71; color: #fff; border: 0; }
    .reject  { background: #e74c3c; color: #fff; border: 0; }
  </style>
</head>
<body>
  <div class="card" data-id="case-001">
    <div class="field">
      <div class="label">輸入</div>
      <div class="input">退貨要多久才會退款？</div>
    </div>
    <div class="field">
      <div class="label">AI 輸出</div>
      <div class="output">物流到件後 3–5 個工作天完成退款，入帳時間以發卡行為準。</div>
    </div>
    <div class="field">
      <div class="label">信心分數</div>
      <div class="conf low">0.42</div>
    </div>
    <div class="field">
      <div class="label">grounding 來源</div>
      <div class="source">退貨政策 §4：商品檢驗通過後 3–5 個工作天退款。</div>
    </div>
    <div class="field actions">
      <button class="approve" onclick="review('case-001', true)">通過</button>
      <button class="reject"  onclick="review('case-001', false)">退回</button>
    </div>
  </div>

  <script>
    function review(id, passed) {
      // 蒐集人工裁決：可 POST 回後端，或累積成 review 結果集
      console.log({ id, passed, at: new Date().toISOString() });
      document.querySelector(`[data-id="${id}"]`).style.opacity = 0.4;
    }
  </script>
</body>
</html>
```

> 信心分數低於門檻就標紅（`conf low`），讓人把有限注意力先投在最可能出錯的卡片上 —— 介面設計本身就是審核效率的一部分。批次審核時，把多張卡片同頁渲染，人就能用滑鼠連續裁決，把「讀一千段 Markdown」壓成「點一千次按鈕」。
