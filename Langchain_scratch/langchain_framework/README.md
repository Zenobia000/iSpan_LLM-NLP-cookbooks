# LangChain v1.0+ 教學框架

## 📁 目錄結構

```
langchain_framework/
├── Course/                 # 核心教學課程
│   ├── Module1/           # LangChain 基礎
│   │   ├── 1_1_framework_overview.py      # 框架概覽
│   │   ├── 1_2_installation_setup.py      # 安裝設置
│   │   ├── 1_3_llm_integration.py         # LLM 整合
│   │   └── 1_4_chains_basics.py           # Chains 基礎
│   ├── Module2/           # Agents 與工具
│   │   ├── 2_1_agent_concepts.py          # Agent 概念
│   │   ├── 2_2_tools_usage.py             # 工具使用
│   │   ├── 2_3_agent_automation_demo1.py  # 自動化範例 1
│   │   └── 2_3_agent_automation_demo2.py  # 自動化範例 2
│   └── Module3/           # RAG 與進階功能
│       ├── 3_0_create_samples.py          # 建立範例資料
│       ├── 3_1_rag_basics.py              # RAG 基礎
│       ├── 3_3_document_loaders.py        # 文件載入器
│       ├── 3_4_text_splitters.py          # 文本分割
│       ├── 3_5_1_custom_llm_agent_template.py # 自定義 Agent 模板
│       ├── 3_5_2_custom_llm_agent.py      # 自定義 Agent 實作
│       ├── 3_5_3_custom_embedding.py      # 自定義 Embedding
│       └── 3_5_4_custom_chain.py          # 自定義 Chain
├── project/               # 實作專案
│   ├── 04-Project - Streamlit Custom ChatGPT App/
│   └── 05-Project - Streamlit Front-End for Question-Answering App/
├── tools/                 # 實用工具
│   ├── llm_api.py         # LLM API 工具
│   ├── screenshot_utils.py # 截圖工具
│   ├── search_engine.py   # 搜尋引擎
│   └── web_scraper.py     # 網頁爬蟲
├── requirements.txt       # 依賴套件
└── .cursorrules          # 開發規則
```

## 🚀 快速開始

### 1. 環境設置

```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 2. 環境變數設置

建立 `.env` 檔案：
```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # 可選
```

### 3. 執行範例

```bash
# 執行框架概覽範例
python Course/Module1/1_1_framework_overview.py

# 執行 Agent 概念範例
python Course/Module2/2_1_agent_concepts.py

# 執行 RAG 基礎範例
python Course/Module3/3_1_rag_basics.py
```

## 📚 學習路徑

### 🎯 初學者路徑
1. **Module1**: 學習 LangChain 基礎概念
2. **Module2**: 理解 Agent 與工具使用
3. **Module3**: 掌握 RAG 系統

### 🎯 進階路徑
1. **自定義組件**: Module3 的 3_5_x 系列
2. **實作專案**: project 目錄的完整應用
3. **工具開發**: tools 目錄的實用工具

## ⚡ 版本特色

- ✅ **LangChain v1.0+**: 最新穩定版本
- ✅ **新 Agent API**: 使用 `create_agent` 替代舊版 `AgentExecutor`
- ✅ **標準化 Import**: 符合 v1.0+ 路徑結構
- ✅ **完整測試**: 所有範例經過驗證
- ✅ **中文註解**: 完整的中文教學說明

## 🔧 技術要求

- Python 3.10+
- LangChain ≥1.0.0
- OpenAI API Key (必需)
- Anthropic API Key (可選)

## 📖 教學重點

### Module1: 基礎建構
- LangChain 架構理解
- LCEL (LangChain Expression Language)
- 基本 Chain 操作

### Module2: Agent 系統
- Agent 概念與 ReAct 框架
- 工具整合與使用
- 自動化工作流程

### Module3: 進階應用
- RAG 系統設計與實作
- 文件處理與向量化
- 自定義組件開發

## 🎨 專案特色

- **模組化設計**: 每個功能獨立，易於學習
- **漸進式學習**: 從簡單到複雜的學習曲線
- **實用導向**: 每個範例都有實際應用價值
- **最佳實務**: 遵循 LangChain v1.0+ 最佳實務

---

*LangChain v1.0+ 教學框架 - 打造您的 AI 應用開發技能*