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
        |          |          |        |         |         |
    Pipeline   微調優化    量化訓練   Chain      CrewAI    STORM
    Tokenizer  問答系統   分散式訓練  Agent     MetaGPT   框架
     Model     聊天機器人   PEFT      RAG      協作模式   主題展開
```

## 📂 專案目錄結構

```
iSpan_LLM-NLP-cookbooks/
├── 📄 .env                              # 環境變數配置
├── 📄 .gitignore                        # Git 忽略文件
├── 📄 LICENSE                           # 專案授權
├── 📄 README.md                         # 專案說明文件
│
├── 🤗 HuggingFace_scratch/              # HuggingFace 完整學習路徑
│   ├── 01-Component/                    # 基礎組件學習
│   │   ├── 01pipeline/                  # Pipeline 系統
│   │   ├── 02tokenizer/                 # Tokenizer 深度學習
│   │   ├── 03Model/                     # Model 架構與應用
│   │   ├── 04Datasets/                  # Datasets 資料處理
│   │   ├── 05evaluate/                  # Evaluate 評估框架
│   │   ├── 06Trainer/                   # Trainer 訓練框架
│   │   └── demo/                        # 實作展示
│   │
│   ├── 02-Adv-tasks/                    # 進階 NLP 任務
│   │   ├── 01-finetune_optimize/        # 微調優化
│   │   ├── 02-token_classification/     # 序列標註 (NER)
│   │   ├── 03-question_answering/       # 問答系統
│   │   ├── 04-sentence_similarity/      # 語義相似度
│   │   ├── 05-retrieval_chatbot/        # 檢索聊天機器人
│   │   ├── 06-language_model/           # 語言模型
│   │   ├── 07-text_summarization/       # 文本摘要
│   │   └── 08-generative_chatbot/       # 生成式聊天機器人
│   │
│   ├── 03-PEFT/                         # 參數高效微調
│   │   ├── 01-LoRA/                     # LoRA 方法
│   │   └── 02-IA3/                      # IA3 方法
│   │
│   ├── 04-kbits-tuning/                 # 量化訓練
│   │   ├── 01-llm_download/             # 模型下載
│   │   ├── 02-16bits_training/          # 16-bit 訓練
│   │   ├── 03-8bits_training/           # 8-bit 訓練
│   │   ├── 04-4bits_training/           # 4-bit 訓練 (QLoRA)
│   │   └── LLaMA2-prompt-tuning/        # LLaMA2 提示調優
│   │
│   └── 05-Distributed Training/         # 分散式訓練
│       ├── 01-remote ssh/               # 遠程 SSH 訓練
│       └── 02-data parallel/            # 資料並行訓練
│
├── 🦜 Langchain_scratch/                # LangChain 框架系統
│   ├── langchain_framework/             # 核心框架學習
│   │   ├── Course/                      # 課程模組
│   │   │   ├── Module1/                 # 基礎概念
│   │   │   ├── Module2/                 # Agent 與工具
│   │   │   └── Module3/                 # RAG 系統
│   │   ├── project/                     # 實戰專案
│   │   │   ├── 01-Project - Building a Custom ChatGPT App/
│   │   │   ├── 02-Project - QA on Private Documents/
│   │   │   ├── 03-Project - Summarization/
│   │   │   ├── 04-Project - Streamlit Custom ChatGPT/
│   │   │   └── 05-Project - Streamlit Front-End for QA/
│   │   └── tools/                       # 實用工具庫
│   │
│   ├── Multi-agent-system/              # 🤖 多智能體系統
│   │   ├── framework/                   # 框架實作
│   │   │   └── CrewAI/                  # CrewAI 框架
│   │   │       ├── 01_basic_structure/  # 基礎結構
│   │   │       ├── 02_task_dependency/  # 任務依賴
│   │   │       └── 03_manager_pattern/  # 管理者模式
│   │   │
│   │   └── 應用專案-多智能體長文寫作/      # 📝 長文寫作專案
│   │       ├── 01_概念理解/              # 基礎概念
│   │       ├── 02_手動實作/              # 手動版實現
│   │       └── 03_框架實作/              # 框架版實現
│   │
│   └── streamlit_resource/              # Streamlit UI 開發
│
├── 📝 prompt-engineering/               # 提示工程專題
│   ├── 基礎技術 (101-103)              # OpenAI 基礎
│   ├── 進階提示 (201-208)              # CoT, ToT 等
│   ├── 應用開發 (401-402)              # 聊天機器人
│   ├── 語音處理 (501-502)              # Whisper 整合
│   ├── RAG 系統 (601-607, 610, 612)    # 檢索增強生成
│   ├── Agent 框架 (701-721)            # Function Calling
│   ├── 模型微調 (810)                  # 合成資料微調
│   └── 多智能體 (960, 970)             # Swarm, SDK
│
└── 📊 Slides/                           # 課程投影片
```

## 📋 專案統計

- **總文件數量**: 201+ 個重要文件
- **Python 程式**: 47 個 .py 文件
- **Jupyter Notebooks**: 128 個 .ipynb 文件
- **文檔資料**: 15 個 .md 文件
- **資料文件**: 8 個 .txt 文件

## ⭐ 核心特色

### 🤖 多智能體框架教學
- **CrewAI 完整生態**: 從入門模板到複雜階層式任務分配
- **協作模式設計**: Sequential、Hierarchical、Manager-Worker 模式
- **工具整合生態**: 計算器、搜索引擎、翻譯工具的無縫整合
- **實際應用場景**: 旅行規劃、軟體開發、內容創作等多領域

### 📝 長文本寫作專案
- **STORM 寫作框架**: 多層次大綱生成與內容組織
- **主題展開技術**: Globe Explorer 創新展開模式
- **RAG 整合寫作**: 向量檢索支援的智能內容生成
- **多智能體協作**: 研究員、作家、編輯的專業分工

### 🚀 進階技術亮點
- **量化訓練**: 支援 16/8/4-bit 訓練，包含 QLoRA
- **分散式訓練**: 資料並行與遠程訓練配置
- **PEFT 技術**: LoRA、IA3 等參數高效微調
- **Function Calling**: 完整的工具調用與 Agent 整合

## 🛠️ 技術棧

### 核心框架
- **HuggingFace Transformers**: 模型訓練與推理
- **LangChain**: LLM 應用開發框架
- **CrewAI**: 多智能體協作框架
- **OpenAI API**: GPT 系列模型整合

### 支援工具
- **向量資料庫**: ChromaDB, Pinecone
- **Web 框架**: Streamlit, Gradio, FastAPI
- **資料處理**: Pandas, NumPy, Scikit-learn
- **深度學習**: PyTorch, Accelerate, PEFT

## 🎯 學習路徑建議

### 🟢 初學者路徑（4-6週）
1. HuggingFace 基礎組件 → LangChain 框架基礎
2. 提示工程基礎 → 簡單 RAG 系統
3. 基礎聊天機器人開發

### 🟡 進階路徑（6-8週）
1. PEFT 與模型微調 → Function Calling
2. 多智能體框架 (CrewAI) → 協作模式設計
3. STORM 長文寫作系統 → RAG 整合應用

### 🔴 專家路徑（8-12週）
1. 量化訓練與分散式系統
2. 複雜多智能體系統架構
3. 生產環境部署與優化

## 🚀 快速開始

```bash
# 1. 克隆專案
git clone https://github.com/Zenobia000/iSpan_LLM-NLP-cookbooks.git

# 2. 進入專案目錄
cd iSpan_LLM-NLP-cookbooks

# 3. 安裝依賴（根據子專案選擇）
pip install -r requirements.txt

# 4. 設置環境變數
cp .env.example .env
# 編輯 .env 文件，添加您的 API keys

# 5. 開始學習
jupyter notebook
```

## 📖 推薦學習順序

1. **基礎入門**: 從 `HuggingFace_scratch/01-Component/` 開始
2. **框架學習**: 進入 `Langchain_scratch/langchain_framework/Course/`
3. **實戰專案**: 選擇感興趣的 project 目錄
4. **進階技術**: 探索多智能體系統或長文寫作專案

## 🤝 貢獻指南

歡迎貢獻程式碼、文檔或建議！請遵循以下步驟：

1. Fork 本專案
2. 創建特性分支 (`git checkout -b feature/新功能`)
3. 提交更改 (`git commit -am '新增某功能'`)
4. 推送分支 (`git push origin feature/新功能`)
5. 創建 Pull Request

## 📄 授權聲明

本專案採用 [LICENSE](./LICENSE) 授權。

## 🏷️ 標籤

`#LLM` `#NLP` `#HuggingFace` `#LangChain` `#RAG` `#Agent` `#MultiAgent` `#PEFT` `#QuantizedTraining` `#教學` `#Python` `#AI`

---

**⭐ 如果這個專案對你有幫助，請給我們一個星星！**

## 📮 聯絡資訊

- **問題回報**: 請在 [Issues](https://github.com/Zenobia000/iSpan_LLM-NLP-cookbooks/issues) 頁面提出
- **專案維護**: iSpan 資訊教育中心

## 🔄 最後更新

**日期**: 2026-02-04

**內容**:
- 完整掃描專案結構並更新 README.md
- 新增詳細目錄樹狀結構
- 更新專案統計資訊
- 優化學習路徑建議