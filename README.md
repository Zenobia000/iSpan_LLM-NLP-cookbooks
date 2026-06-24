# iSpan LLM-NLP 生成式 AI 系統化教學專案

## 🎯 專案概述

本專案為 **iSpan** 學院 LLM（大型語言模型）與 NLP（自然語言處理）的系統化教學課程程式碼庫，採用**金字塔結構**與 **MECE原則**（Mutually Exclusive, Collectively Exhaustive）設計，提供從基礎到進階的完整學習路徑。

> 📐 **課程重構中**:本庫正依《AI 可控性工程:從使用者到指揮官》重新編排,主軸為「可控性」。設計文件見 [`docs/superpowers/specs/2026-06-23-ai-controllability-engineering-curriculum-design.md`](docs/superpowers/specs/2026-06-23-ai-controllability-engineering-curriculum-design.md)。部分非主軸內容已移至 [`_archive/`](_archive/)。

## 📊 專案架構（金字塔結構）

```
                      🎯 生成式 AI 應用開發
                     /                    \
              🤗 HuggingFace 生態系統    🦜 LangChain 框架系統
             /         |         \      /         |         \
      基礎組件    進階任務    模型優化   框架基礎   🤖多智能體   📝長文寫作
        |          |          |        |         |         |
    Pipeline   微調優化    量化訓練   Chain    LangGraph  STORM
    Tokenizer  問答系統   分散式訓練  Agent   Deep Research 長文研究
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
│
├── 🦜 Langchain_scratch/                # LangChain 框架系統
│   ├── langchain_framework/             # 核心框架學習
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
│   │   │   └── LangGraph_2026/          # LangGraph deep research 主線
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
├── 📊 Slides/                           # 課程投影片
│
├── 📝 prompt-engineering/               # AI 可控性工程主課程（控制權階梯）
│   ├── 01-uncontrollability/            # 不可控的根源
│   ├── 02-intent-convergence/           # 意圖收斂：prompt → spec
│   ├── 03-structured-output/            # 結構收斂：JSON / function calling
│   ├── 04-knowledge-rag/                # 知識收斂：RAG
│   ├── 05-agent-harness/                # 行為收斂：agent harness
│   ├── 06-multi-agent/                  # 協作收斂：多 agent
│   ├── 07-calibration-eval/             # 校準層：驗證與評估
│   └── 08-capstone/                     # 整合專題
│
└── 🗄️ _archive/                         # 已封存(非主軸,保留歷史)
    ├── HuggingFace_scratch/05-Distributed Training/
    ├── Langchain_scratch/Slides/
    └── Langchain_scratch/langchain_framework/Course/
```

## 📋 專案統計

- **總文件數量**: 201+ 個重要文件
- **Python 程式**: 47 個 .py 文件
- **Jupyter Notebooks**: 128 個 .ipynb 文件
- **文檔資料**: 15 個 .md 文件
- **資料文件**: 8 個 .txt 文件

## ⭐ 核心特色

### 🤖 多智能體框架教學
- **LangGraph 2026 主線**: 用 StateGraph、Send、reducer 建立可控多智能體流程
- **Deep Research 架構**: 參考 open_deep_research，示範研究規劃、並行研究與報告生成
- **協作模式設計**: Planner、Researcher、Writer、Reviewer 的明確節點分工
- **Legacy 對照**: CrewAI 範例保留為歷史素材，不再作為新課程主線

### 📝 長文本寫作專案
- **STORM 寫作框架**: 多視角研究、大綱生成、初稿與修訂
- **LangChain LCEL 教學**: 把研究、大綱、寫作串成可讀的 Runnable 管線
- **LangGraph 寫作流程**: 用 state 管理 research_notes、outline、draft、review
- **Deep Research 延伸**: 將研究節點升級為 open_deep_research 風格的並行研究流程

### 🚀 進階技術亮點
- **量化訓練**: 支援 16/8/4-bit 訓練，包含 QLoRA
- **分散式訓練**: 資料並行與遠程訓練配置
- **PEFT 技術**: LoRA、IA3 等參數高效微調
- **Function Calling**: 完整的工具調用與 Agent 整合

## 🛠️ 技術棧

### 核心框架
- **HuggingFace Transformers**: 模型訓練與推理
- **LangChain**: LLM 應用開發框架
- **LangGraph**: 多步驟、多智能體、有狀態編排
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
2. 多智能體框架 (LangGraph) → 協作模式設計
3. STORM 長文寫作系統 → Deep Research 整合應用

### 🔴 專家路徑（8-12週）
1. 量化訓練與分散式系統
2. 複雜多智能體系統架構
3. 生產環境部署與優化

## 🚀 快速開始

```bash
# 1. 克隆專案
git clone https://github.com/Zenobia000/iSpan_LLM-NLP-cookbooks.git
cd iSpan_LLM-NLP-cookbooks
```

## 🤖 Agent 導入

本專案已加入可讓 coding agent 實際採用的導入層：

- `AGENT.md`：專案層 agent 規則、驗證方式與安全邊界。
- `.mcp.example.json`：本地課程 MCP server 與 GitHub read-only server 設定範例，不包含真實 token。
- `prompt-engineering/mcp/`：自己寫的最小 MCP server、教學 client、工具選擇情境。
- `prompt-engineering/agent-skills/`：課程維護用 Skills。
- `docs/agent-integration.md`：AGENT.md / MCP / Skills 的啟用方式與權限建議。

使用 agent 維護教材前，先讓 agent host 讀取根目錄 `AGENT.md`；若需要 MCP，依 host 格式套用 `.mcp.example.json`，並以環境變數提供 token。

### 主課程 `prompt-engineering/`（以 uv 管理，2026 定版）

```bash
cd prompt-engineering
uv sync                      # 安裝 uv.lock 鎖定的依賴
cp .env.example .env         # 填入 OPENAI/ANTHROPIC/GEMINI 等金鑰
uv run jupyter lab           # 啟動
```

> 詳見 [`prompt-engineering/README.md`](prompt-engineering/README.md)。其餘子專案（HuggingFace_scratch、Langchain_scratch）仍依各自說明以 `pip install` 安裝。

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

**日期**: 2026-06-24

**內容**:
- 啟動《AI 可控性工程》課程重構(spec 見 docs/superpowers/specs/)
- 升級全 notebook model 字串、移除淘汰範例(960 Swarm、610 重複檔)
- 封存非主軸內容至 _archive/
- prompt-engineering 依控制權階梯分層為 8 個英文模組資料夾、notebook 全面英文化命名
