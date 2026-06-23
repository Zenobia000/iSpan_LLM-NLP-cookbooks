# 《AI 可控性工程:從使用者到指揮官》課程設計文件

- **日期**:2026-06-23
- **狀態**:設計核可,待撰寫實作計畫
- **適用目錄**:`prompt-engineering/`
- **分支**:`dev`

---

## 1. 背景與動機

現有 `prompt-engineering/` 教材有兩個層級的過時:

- **表層**:約 85 處仍以 `gpt-3.5-turbo` 為預設;最新內容(`claude-3-7-sonnet-20250219`、`gpt-4.1-2025-04-14`)約停在 2025 年 4 月,落後當前 14 個月。
- **深層(真問題)**:教學「典範」停在 2023–2024 的「prompt → RAG → agent」能力疊加觀。2xx 章節仍教學員手刻 CoT/ToT scaffolding,而在 reasoning model 時代這已是模型內建能力;`960` Swarm、`720` Assistants API 已被取代;完全缺席 2026 核心(Harness Engineering、MCP/Skills、Spec-driven、Feedback loop、Cross-model review)。

本課程**不打掉重練**,而是以現有 notebook 為素材庫,圍繞新主軸重新編排,並補上 2026 典範。

## 2. 目標受眾(TA)

- 具基礎 Python(會用 dict/list/function、能跑 notebook)
- 用過 ChatGPT/Claude 類工具,但**把 AI 定位為使用者**
- 不理解底層原理與能力邊界
- **對 AI 輸出的控制力不足**

兩個核心缺口:(1) 不懂原理與邊界、(2) 輸出控制力不足。

## 3. 設計決策(brainstorming 結論)

| 維度 | 決策 |
|---|---|
| 課程容器 | 一學期 16 週,**8 個模組**,每模組 2 週(模組化) |
| SDK 主軸 | **OpenAI 為主軸,Claude 為對照**(沿用現有慣性、改寫成本低) |
| 與現有 notebook 關係 | **現有 notebook 當素材庫**,重新編排,標註 沿用/改寫/刪除/新增 |
| 主軸貫穿線 | **可控性**:從「理解 AI 為何不可控」→「逐層奪回控制權」,原理是達成控制的手段 |
| 編排取向 | **取向 ①:控制權階梯(Control Ladder)** |
| 每章體例 | **完整教科書體例(C)**,含失敗模式/真實案例/延伸閱讀 |

## 4. 課程定位與畢業能力

- **課程名稱(暫定)**:《AI 可控性工程:從使用者到指揮官》
- **主軸**:沿控制權階梯,逐層把 AI 的不確定性收斂成可控、可驗證的產出。
- **先修**:基礎 Python、用過 ChatGPT/Claude 類工具。
- **畢業能力**:學生結業時能「寫規格而非長提示詞、為任務挑對的控制機制、設計驗證迴圈、判讀 AI 的失敗模式」——從「AI 使用者」變成「AI 系統的指揮官」。

## 5. 每章固定體例(完整版)

每個模組統一包含:

1. 學習目標(Learning Objectives)
2. 核心原理(Why / 邊界在哪)
3. 動手實作(notebook,標註 OpenAI 主 / Claude 對照)
4. **可控性檢核點**(本章控制機制 + 自我驗證清單)— 主軸所在
5. 常見失敗模式(該層特有的翻車情境)
6. 真實案例
7. 本章小結
8. 練習題(概念題 / 實作題 / 控制力挑戰題 三級)

## 6. 16 週對應與評量

| 週 | 模組 | 里程碑 |
|---|---|---|
| W1–2 | M1 不可控的根源 | |
| W3–4 | M2 意圖收斂:Prompt → Spec | |
| W5–6 | M3 結構收斂:JSON / Function Calling | |
| W7–8 | M4 知識收斂:RAG | **期中:小型可控問答系統** |
| W9–10 | M5 行為收斂:Agent Harness | |
| W11–12 | M6 協作收斂:多 Agent 與失敗模式 | |
| W13–14 | M7 校準層:驗證與評估 | |
| W15–16 | M8 整合專題 | **期末 Capstone** |

**評量配比(建議)**:平時練習題 40% ｜ 期中可控問答系統 25% ｜ 期末 Capstone 35%。

**資產標記**:🟢 沿用 ｜ 🟡 改寫 ｜ 🔴 刪除/降級 ｜ ⭐ 新增。

---

## 7. 模組詳細設計

### M1 不可控的根源(W1–2｜原理地基)

- **學習目標**:說清楚「AI 為何不聽話」——能指出任一段輸出的不確定性來自哪裡。
- **核心原理**:Tokenization、機率取樣(temperature / top_p)、Context window 限制、知識截止、幻覺來源、模型能力邊界(「泡泡」概念)。
- **動手實作**:🟢 `101` 基礎 API 呼叫 ｜ ⭐ temperature/top_p 視覺化 demo(同一 prompt 跑 10 次看分歧) ｜ 🟡 `130` thinking/reasoning model(示範模型「內部已會 CoT」,埋下 M2「不必手刻 CoT」伏筆)。
- **可控性檢核點**:給一段輸出,能歸因出三個不確定性來源並各提一個收斂手段。
- **常見失敗模式**:把機率性輸出當確定事實;溫度設定誤用。
- **真實案例**:律師引用 ChatGPT 捏造判例遭法院制裁。
- **延伸**:`110` 多模態、`115` Gemini 作為「不同模型不同邊界」對照。

### M2 意圖收斂:Prompt → Spec(W3–4)

- **學習目標**:從「寫提示詞技巧」升級到「寫規格(Spec)」;掌握 2026 減法哲學;能區分 business rule 與 model rule。
- **核心原理**:為何長提示詞在強模型上反而失效;意圖精確度(Intention)。
- **動手實作**:🟡 `102` 提示詞基礎(保留 5W1H/角色,套上「減法」框架) ｜ 🟡 `201` CoT(改成「reasoning model 時代 CoT 何時用、為何不必手刻」) ｜ 🟢 `207` chaining(當 spec 分解示範) ｜ ⭐ Spec 模板與撰寫練習(目標/受眾/驗證標準/回滾/禁區)。
- **可控性檢核點**:把一個模糊任務寫成完整 Spec。
- **常見失敗模式**:保母型提示詞、Spec drift(長對話偏離)。
- **🔴 降級**:`208` ToT 手刻——移到延伸閱讀,明講「為何在 reasoning model 時代已過時」。

### M3 結構收斂:JSON / Function Calling(W5–6)

- **學習目標**:把自由文本收斂成「程式可驗證」的結構化輸出;理解 function calling 的本質是「讓輸出可被驗證」。
- **核心原理**:結構化 = 可驗證 = 可控。
- **動手實作**:🟢 `103` JSON mode ｜ 🟢 `702` function calling 基礎 ｜ 🟢 `706` 結構化抽取 ｜ 🟡 `402` Gradio 分類(改成結構化輸出範例)。
- **可控性檢核點**:用 schema 強制格式,並在程式端做驗證與重試。
- **常見失敗模式**:schema 不符;silent failure(格式對但值錯)。

### M4 知識收斂:RAG(W7–8｜期中專題)

- **學習目標**:用檢索 grounding 控制幻覺,讓 AI「只根據給定知識回答」;理解檢索本身的邊界。
- **核心原理**:embedding / 相似度 / grounding;檢索不是萬靈丹。
- **動手實作**:🟢 `601` embedding、`602` vanilla RAG、`604` vector DB、`605` 相似度、`606` dynamic few-shot ｜ 🟡 `607` advanced RAG(query expansion / HyDE / rerank / small-to-big)、`612` PDF 解析。
- **可控性檢核點**:讓模型「只用檢索內容回答、找不到就說不知道、附來源」。
- **常見失敗模式**:檢索失敗仍硬答;context 塞太多稀釋重點。
- **⭐ 期中專題**:對一份文件做「grounded QA + 引用 + 拒答」的小型可控問答系統。

### M5 行為收斂:Agent Harness(W9–10)

- **學習目標**:理解 `Agent = LLM + Harness`;掌握五大架構(Tools / Context / Memory / Guardrails / Feedback loop);內化「人類掌舵、Agent 執行」。
- **核心原理**:Harness Engineering;`agent.md` 作為單一真實來源;MCP 與 Skills 的角色。
- **動手實作**:🟢 `701` LangChain agents、`703` FC agents、`705` FC+RAG ｜ 🟡 `711` ReAct(當 agent 核心迴圈)、`712` shop(業務規則 guardrails) ｜ 🟡 `202` prompt injection(guardrails:意圖被劫持) ｜ ⭐ 寫一份 `agent.md`、⭐ MCP / Skills 概念與 demo。
- **可控性檢核點**:為一個 agent 完整回答「五大問題」(能做/能看/記得/不能做/怎麼知道對)。
- **常見失敗模式**:guardrails 缺失導致越權;把 model rule 誤寫成 business rule。
- **🔴 改寫**:`720` Assistants API → 改用 Responses API(或降級為「已淘汰 API」說明)。

### M6 協作收斂:多 Agent 與失敗模式(W11–12)

- **學習目標**:掌握多 agent 設計模式(Handoffs、Agents-as-Tools、Orchestrator-Workers、Parallelization);會判讀 Cascade failure;建立 Cross-Model Review 觀念。
- **核心原理**:何時該用多 agent、何時是過度設計(YAGNI);協作的脆弱性。
- **動手實作**:🟡 `970` OpenAI Agents SDK(handoffs / agents-as-tools / deterministic / orchestrator / self-reflection / human-in-the-loop,幾乎可整段沿用) ｜ 🟡 `721` deep search(single→multi 對照) ｜ ⭐ Cross-model review(OpenAI 產出 + Claude 審查)。
- **可控性檢核點**:設計一個 orchestrator-workers 流程,並加入 human-in-the-loop 確認點。
- **常見失敗模式**:Cascade failure、Sycophancy、為用而用的多 agent。
- **🔴 刪除**:`960` Swarm(已被 Agents SDK 取代)。

### M7 校準層:驗證與評估(W13–14)

- **學習目標**:設計 feedback loop 讓 agent 自我驗證;評估 RAG / agent 品質;理解「能力泡泡會擴張、要持續移動交接點」。
- **核心原理**:Verifiability 的大循環;讓 AI 在封閉環境瘋狂試錯。
- **動手實作**:🟢 `610` RAG evaluation ｜ 🟡 `721` 加上驗證迴圈 ｜ ⭐ 自動化 feedback loop(跑測試 → 讀 log → 自我修正) ｜ 🟡 `810` 合成資料微調(校準的延伸選讀)。
- **可控性檢核點**:為自己的專題設計一個自動驗證迴圈。
- **常見失敗模式**:自我審核過度自信;沒有 ground truth 的假評估。

### M8 整合專題 Capstone(W15–16)

- **學習目標**:整合控制權階梯每一層,從 0 打造一個「可控」的 AI 系統。
- **交付物**:Spec + 系統實作 + 驗證迴圈 + 失敗模式分析 + Demo。
- **評量 rubric**:對應五層控制機制 + 可驗證性,每項計分。
- **題庫範例**:可控客服 agent ｜ 文件 grounded 問答 ｜ 研究助理 deep search ｜ 結構化抽取 pipeline。

---

## 8. 現有 notebook 處置總表

### 🔴 刪除 / 降級
- `960` Swarm — 刪除(被 Agents SDK 取代)
- `208` ToT 手刻 — 降級至 M2 延伸閱讀
- `720` Assistants API — 改寫為 Responses API

### ⭐ 新增(無現成 notebook)
- M1:temperature/top_p sampling 視覺化 demo
- M2:Spec 撰寫模板與練習
- M5:`agent.md` 撰寫、MCP / Skills 概念 demo
- M6:Cross-model review
- M7:自動化 feedback loop

### 🟢 大量沿用(未被典範淘汰)
- RAG 系列(6xx)、function calling 系列(7xx)、embedding。

### 未編入主線(待決,避免無聲遺漏)
以下 notebook 目前未進主線,需後續決定保留為選讀、併入模組或汰除:
- `11`、`test`(疑似草稿/測試)
- `205` plugin-tools、`206` prompt-integration-usecase
- `401` chatbot
- `501` / `502` Whisper 摘要(語音應用,主線未涵蓋)
- `601--embedding` 與 `605` 之外的 `610__`(檔名疑似重複,需確認 `610--` vs `610__`)
- `810` 合成資料微調(目前列為 M7 選讀)

## 9. 後續步驟

1. 本設計文件經使用者複審。
2. 進入 writing-plans,產出逐模組的實作計畫(notebook 改寫工單:沿用驗證、改寫範圍、新增內容草稿)。
3. 建議實作順序:先做 L1 機械更新(model 字串升級、刪 deprecated)建立乾淨基線,再按模組改寫/新增。
