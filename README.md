# iSpan LLM-NLP 生成式 AI 系統化教學專案

## 🎯 專案概述

本專案為 **iSpan** 學院 LLM（大型語言模型）與 NLP（自然語言處理）的系統化教學課程程式碼庫，採用**金字塔結構**與 **MECE原則**（Mutually Exclusive, Collectively Exhaustive）設計，提供從基礎到進階的完整學習路徑。

## 📊 專案架構（金字塔結構）

```
                      🎯 生成式 AI 應用開發
                     /                    \
              🤗 HuggingFace 生態系統    🦜 LangChain 框架系統
             /         |         \      /         |         \
      基礎組件    進階任務    模型優化   框架基礎   🤖多智能體   📝長文寫作
        |          |          |        |    RAG     |         |
    Pipeline   微調優化    量化訓練   Chain   🗺️旅行   STORM   提示工程
    Tokenizer  問答系統   分散式訓練  基礎   規劃系統   框架    Function
     Model     聊天機器人   PEFT             CrewAI           Calling
```

## ⭐ 課程核心亮點

### 🤖 多智能體框架教學
> **創新特色：** 完整的多智能體系統設計與實作，從基礎協作到複雜階層式任務分配

- **CrewAI 完整生態**：從入門模板到複雜旅行規劃系統
- **階層式協作架構**：Manager-Worker 模式的深度實作
- **工具整合生態**：計算器、搜索引擎、翻譯工具的無縫整合
- **實際應用場景**：旅行規劃、軟體開發、內容創作等多領域應用

### 📝 長文本寫作專案
> **技術突破：** 基於 STORM 框架的結構化長文寫作系統，結合 RAG 技術實現資料驅動創作

- **STORM 寫作框架**：多層次大綱生成與內容組織
- **Globe Explorer 主題展開**：創新的主題分解與展開模式
- **RAG 整合寫作**：向量檢索支援的智能內容生成
- **多智能體協作寫作**：不同角色 Agent 的專業分工合作

---

## 🗂️ 目錄結構與學習路徑

### 🔥 第一層：核心技術領域

#### 1️⃣ [HuggingFace 生態系統](./HuggingFace_scratch/)
> **學習目標：** 掌握 HuggingFace Transformers 生態系統的完整應用

#### 2️⃣ [LangChain 框架系統](./Langchain_scratch/)
> **學習目標：** 建構企業級 LLM 應用與多智能體系統

#### 3️⃣ [專案開發實務](./Python_project_sample/)
> **學習目標：** Python 專案架構與最佳實踐

---

### 🎓 第二層：技能模組矩陣

#### 🤗 HuggingFace 技能樹

| 模組 | 內容 | 學習階段 | 核心技能 |
|------|------|----------|----------|
| **[基礎組件](./HuggingFace_scratch/01-Component/)** | Pipeline, Tokenizer, Model, Datasets, Evaluate, Trainer | 🟢 初級 | Transformers 基礎操作 |
| **[進階任務](./HuggingFace_scratch/02-Adv-tasks/)** | 微調優化、序列標註、問答系統、相似度計算、檢索聊天機器人、語言模型、文本摘要、生成式聊天機器人 | 🟡 中級 | 實際應用場景 |
| **[參數高效微調](./HuggingFace_scratch/03-PEFT/)** | LoRA, IA3 | 🟡 中級 | 高效模型適配 |
| **[量化訓練](./HuggingFace_scratch/04-kbits-tuning/)** | 16-bit, 8-bit, 4-bit 量化 | 🔴 高級 | 模型壓縮與優化 |
| **[分散式訓練](./HuggingFace_scratch/05-Distributed%20Training/)** | 遠程SSH、資料並行 | 🔴 高級 | 大規模訓練 |

#### 🦜 LangChain 技能樹

| 模組 | 內容 | 學習階段 | 核心技能 |
|------|------|----------|----------|
| **[框架基礎](./Langchain_scratch/langchain_framework/)** | 安裝配置、LLM整合、Chain基礎、RAG實現 | 🟢 初級 | LangChain 核心概念 |
| **[提示工程](./Langchain_scratch/prompt-engineering/)** | CoT、ToT、Function Calling、RAG評估、Agent深度搜索 | 🟡 中級 | 提示設計與優化 |
| **[應用開發](./Langchain_scratch/streamlit_resource/)** | Streamlit UI 開發 | 🟡 中級 | Web 應用介面 |
| **[🤖 多智能體框架](./Langchain_scratch/Multi-agent-system/)** | CrewAI、MetaGPT、協作式AI系統 | 🔴 高級 | 多智能體協作設計 |
| **[📝 長文本寫作專案](./Langchain_scratch/Multi-agent-system/project/long_context_writing/)** | STORM寫作框架、Globe Explorer主題展開 | 🔴 高級 | 長文本生成與結構化寫作 |

---

### 🎯 第三層：具體實作技術

#### 📚 HuggingFace 詳細技術棧

<details>
<summary><strong>01-基礎組件學習</strong></summary>

- **Pipeline 系統**：`01pipeline/01.pipeline.ipynb`
  - 預訓練模型快速使用
  - 任務管道配置與自定義

- **Tokenizer 深度學習**：`02tokenizer/02.tokenizer.ipynb_`
  - 文本預處理與編碼
  - 中文分詞與特殊符號處理

- **Model 架構與應用**：`03Model/`
  - 模型載入與配置
  - 分類任務實戰：`03 Model classification_demo.ipynb`
  - 中文情感分析：`dataset/ChnSentiCorp_htl_all.ipynb`

- **Datasets 資料處理**：`04Datasets/`
  - 資料集載入與預處理
  - 自定義資料集腳本：`load_script.py`

- **Evaluate 評估框架**：`05evaluate/`
  - 模型性能評估
  - 多指標評估系統
  - 推文情感分析評估

- **Trainer 訓練框架**：`06Trainer/`
  - 訓練流程設計
  - 超參數調優

- **Demo 實作展示**：`demo/demo.ipynb`
  - Gradio 界面開發
  - 文本分類與問答系統
</details>

<details>
<summary><strong>02-進階任務實戰</strong></summary>

- **微調優化**：`01-finetune_optimize/`
- **序列標註**：`02-token_classification/` - NER 實作
- **問答系統**：`03-question_answering/` - 閱讀理解與CMRC評估
- **語義相似度**：`04-sentence_similarity/` - Cross & Dual Model
- **檢索聊天機器人**：`05-retrieval_chatbot/`
- **語言模型**：`06-language_model/` - Causal & Masked LM
- **文本摘要**：`07-text_summarization/` - GLM與通用摘要
- **生成式聊天機器人**：`08-generative_chatbot/`
</details>

#### 🔗 LangChain 詳細技術棧

<details>
<summary><strong>框架核心技術</strong></summary>

- **基礎設施**：`langchain_framework/Course/`
  - Module1: 框架概覽、安裝設置、LLM整合、Chain基礎
  - Module2: Agent概念、工具使用、自動化示例
  - Module3: RAG基礎、評估指標、樣本創建

- **RAG 系統實作**：`3_1_rag_basics.py`
  - 文檔管理與向量存儲
  - 檢索增強生成
  - 系統架構設計

- **工具生態**：`tools/`
  - LLM API 集成
  - 網頁爬蟲與搜索引擎
  - 螢幕截圖工具
</details>

<details>
<summary><strong>🤖 多智能體框架系統</strong></summary>

### CrewAI 框架深度教學
- **🚀 起始模板**：`starter_template/`
  - 基礎 Agent 架構設計
  - 任務分配與協作機制
  - 工具整合與使用

- **🗺️ 旅行規劃多智能體系統**：
  - `trip_planner_from_scratch/` - 基礎版本
  - `trip_planner_from_scratch_sequential_tasks/` - 序列任務版本  
  - `trip_planner_from_scratch_sequential_hierarchical_tasks/` - 階層式任務版本

### 多智能體協作模式
- **序列協作**：Agent 按順序執行任務
- **階層協作**：Manager-Worker 架構設計
- **工具共享**：計算器、搜索、語言翻譯工具整合

### MetaGPT 框架
- **軟體開發多智能體**：`metaGPT/`
- **角色分工**：產品經理、架構師、程式設計師、測試工程師
</details>

<details>
<summary><strong>📝 長文本寫作專案系統</strong></summary>

### STORM 寫作框架
- **Level 1 - 基礎寫作**：`level1/`
  - `W1_&_W2_STORM_長文寫作簡易.ipynb` - 基礎框架
  - `W1_&_W2_STORM_長文寫作簡易版（授課版）.ipynb` - 教學版本
  - `共學練習 - Globe Explorer 的主題大綱展開模式.ipynb` - 主題展開實作

- **Level 2 - 進階寫作**：`level2/`
  - LangChain 整合：`1_1_installation_setup.py`
  - 框架概覽：`1_2_framework_overview.ipynb`
  - RAG 寫作支援：`LangChain_RAG.ipynb`
  - 向量存儲與檢索：`LangChain_Vector_Store_&_Retriever.ipynb`

### 長文寫作核心技術
- **多層次大綱生成**：主題分解與層次化組織
- **資料驅動寫作**：RAG 技術支援內容生成
- **多智能體協作寫作**：不同角色的 Agent 分工合作
- **結構化輸出**：格式化與風格一致性控制

### 應用場景
- **學術論文寫作**：研究報告與學術文章
- **商業報告生成**：市場分析與業務文件
- **新聞稿件創作**：結構化新聞寫作系統
- **教學內容開發**：課程大綱與教材編寫
</details>

<details>
<summary><strong>🏗️ 系統架構與開發文件</strong></summary>

- **系統設計**：`system_design.md` - 多智能體系統架構設計
- **運算思維框架**：`基於運算思維的新聞寫作多智能體系統軟體開發通用流程框架.md`
- **實戰案例**：`article_claude_1.md` - Claude 智能體應用案例
- **詞彙管理**：`word_set.docm` - 專業詞彙與術語管理
</details>

<details>
<summary><strong>提示工程技術</strong></summary>

**基礎技術**：
- OpenAI API 使用：`101-start-openai.ipynb`
- 提示工程：`102-prompt-engineering.ipynb`
- JSON 模式：`103-json-mode.ipynb`

**進階提示技術**：
- 思維鏈（CoT）：`201-CoT-prompt.ipynb`
- 思維樹（ToT）：`208-ToT-prompt.ipynb`
- 提示整合：`206-prompt-integration-usecase.ipynb`
- 鏈式提示：`207-chaining-prompt.ipynb`

**RAG 與檢索**：
- 向量嵌入：`601--LLM-workshop-embedding.ipynb`
- 基礎 RAG：`602-varnilla-RAG.ipynb`
- 向量資料庫：`604-vector-db-RAG.ipynb`
- 相似度與相關性：`605--LLM-workshop-similarity-and-relevance.ipynb`
- 動態少樣本學習：`606--LLM-workshop-dynamic-few-shot.ipynb`
- MedPrompt 進階擴展：`606-LLM-Medprompt_Extending.py`
- MedPrompt 線性擴展：`606-LLM-Medprompt_Extending_Linear.py`
- 進階 RAG：`607-advance-RAG.ipynb`
- RAG 評估：`610-RAG-evaluation-ragas.ipynb`
- RAG 評估進階：`610__LLM_workshop_RAG_evaluation.ipynb`
- PDF 解析：`612--LLM-workshop-pdf-parsing-v2.ipynb`

**Agent 與 Function Calling**：
- 基礎 Agent：`701-langchain-agents.ipynb`
- Function Calling：`702-function-calling-basic.ipynb`
- Agent 框架：`703-function-calling-agents.ipynb`
- Function Calling RAG：`705-function-calling-rag.ipynb`
- 資料抽取：`706-function-calling-extract.ipynb`
- ReAct 框架：`711--LLM-workshop-react.ipynb`
- 購物助手：`712-function-calling-shop.ipynb`
- Assistants API：`720-assistants-api.ipynb`
- Agent 深度搜索：`721--LLM-workshop-agent-deep-search.ipynb`
- Swarm 多智能體：`960--LLM-workshop-swarm.ipynb`
- OpenAI Agents SDK：`970--LLM-workshop-openai-agents-sdk.ipynb`

**語音與微調**：
- Whisper 語音轉文字：`501-whisper-summarization.ipynb`
- 合成資料微調：`810-fine-tune-with-synthetic-data.ipynb`
</details>

---

## 🛠️ 技術棧與環境配置

### 核心依賴

```python
# LangChain 生態系統
langchain>=0.3.0
langchain-community>=0.0.16
langchain-openai
langchain-text-splitters

# HuggingFace 生態系統  
transformers
datasets
tokenizers
accelerate
peft

# 向量資料庫與檢索
chromadb>=0.4.22
sentence-transformers
hnswlib
rank-bm25

# Web 應用開發
streamlit
gradio
fastapi

# 資料處理與分析
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### 環境設置

```bash
# 安裝依賴
pip install -r requirements.txt

# 設置 API Key
export OPENAI_API_KEY="your-api-key"
export HUGGINGFACE_API_TOKEN="your-token"
```

---

## 🎯 學習路徑建議

### 🟢 初學者路徑（4-6週）
1. **Week 1-2**: HuggingFace 基礎組件
2. **Week 3-4**: LangChain 框架基礎與 RAG
3. **Week 5-6**: 提示工程與簡單應用開發

### 🟡 進階路徑（6-8週）
1. **Week 1-3**: HuggingFace 進階任務與 PEFT
2. **Week 4-5**: **🤖 多智能體框架系統**（CrewAI、MetaGPT）
3. **Week 6-7**: **📝 長文本寫作專案**（STORM 框架）
4. **Week 8**: 企業級應用整合開發

### 🔴 專家路徑（8-12週）
1. **Week 1-4**: 量化訓練與分散式系統
2. **Week 5-6**: **🤖 複雜多智能體系統設計**（階層式協作）
3. **Week 7-8**: **📝 進階長文寫作系統**（RAG整合寫作）
4. **Week 9-10**: 多智能體與長文寫作系統整合
5. **Week 11-12**: 大規模生產環境部署與優化

---

## 📖 重要資源

### 📊 課程投影片
- `Slides/LangChain - Deep Dive.pdf` - LangChain 深度剖析
- `Slides/Vector Databases.pdf` - 向量資料庫原理
- `Slides/Project - Question-Answering.pdf` - 問答系統項目
- `Slides/Project - Summarization.pdf` - 摘要系統項目

### 📚 參考文檔
- `HuggingFace_scratch/Transformers_hugging_face.pdf` - Transformers 官方文檔
- 多智能體系統設計文檔
- 運算思維開發框架

---

## 🤝 貢獻指南

1. **Fork** 本專案
2. 創建特性分支 (`git checkout -b feature/新功能`)
3. 提交更改 (`git commit -am '新增某功能'`)
4. 推送分支 (`git push origin feature/新功能`)
5. 創建 **Pull Request**

---

## 📄 授權聲明

本專案採用 [LICENSE](./LICENSE) 授權。

---

## 🏷️ 標籤

`#LLM` `#NLP` `#HuggingFace` `#LangChain` `#RAG` `#Agent` `#教學` `#Python` `#AI`

---

**⭐ 如果這個專案對你有幫助，請給我們一個星星！**
