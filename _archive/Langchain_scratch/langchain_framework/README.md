# LangChain v1.0+ 教學框架 🚀

> 🎯 **全面更新至 LangChain v1.0+**
> 采用最新的 `create_agent` API 和 LCEL (LangChain Expression Language)

## 🔥 v1.0+ 重大更新特色

- ✅ **新版 Agent API**: 使用 `create_agent` 取代舊版 `AgentExecutor`
- ✅ **LCEL 語法**: LangChain Expression Language 簡化 Chain 建構
- ✅ **Middleware 支援**: 人機互動、PII 脫敏、監控等功能
- ✅ **Python 3.10+**: 符合 v1.0+ 最低系統要求
- ✅ **最新模型**: gpt-4o-mini、claude-3-5-sonnet 等新模型
- ✅ **完整中文化**: 繁體中文教學與範例

## 📁 專案結構

```
langchain_framework/
├── Course/                          # 🎓 核心教學課程
│   ├── 01-langchain-intro.ipynb    # 📖 v1.0+ 快速入門 (新)
│   ├── requirements-v1.txt         # 📦 v1.0+ 依賴清單 (新)
│   ├── Module1/                    # 🏗️ LangChain 基礎
│   │   ├── 1_1_framework_overview.py      # 框架概覽
│   │   ├── 1_2_installation_setup.py      # 🔄 v1.0+ 環境設置 (更新)
│   │   ├── 1_3_llm_integration.py         # 🔄 v1.0+ LLM 整合 (更新)
│   │   ├── 1_4_chains_basics.py           # 🔄 LCEL Chains (更新)
│   │   └── test.ipynb                     # 測試 Notebook
│   ├── Module2/                    # 🤖 Agent 系統 (v1.0+)
│   │   ├── 2_1_agent_concepts.py          # 🔄 create_agent API (更新)
│   │   ├── 2_2_tools_usage.py             # 工具使用
│   │   ├── 2_3_agent_automation_demo1.py  # 自動化範例 1
│   │   └── 2_3_agent_automation_demo2.py  # 自動化範例 2
│   └── Module3/                    # 🔍 RAG 與進階功能
│       ├── 3_0_create_samples.py          # 建立範例資料
│       ├── 3_2_1_vectorstores_comparison.ipynb  # 向量庫比較
│       ├── 3_2_2_embedding_models.ipynb   # Embedding 模型
│       ├── 3_2_3_similarity_search.ipynb  # 相似度搜尋
│       ├── 3_3_document_loaders.py        # 文件載入器
│       ├── 3_5_3_custom_embedding.py      # 自定義 Embedding
│       ├── 3_5_4_custom_chain.py          # 舊版自定義 Chain
│       ├── 3_5_4_custom_chain_v1.py       # 🆕 v1.0+ LCEL Chain (新)
│       └── samples/                       # 範例數據
├── project/               # 🎨 實作專案
│   ├── 04-Project - Streamlit Custom ChatGPT App/
│   │   └── project_streamlit_custom_chatgpt.py
│   └── 05-Project - Streamlit Front-End for Question-Answering App/
│       └── chat_with_documents.py     # RAG 文件問答系統
├── tools/                 # 🛠️ 實用工具
│   ├── llm_api.py         # LLM API 工具
│   ├── screenshot_utils.py # 截圖工具
│   ├── search_engine.py   # 搜尋引擎
│   └── web_scraper.py     # 網頁爬蟲
├── requirements.txt       # 舊版依賴套件
└── .cursorrules          # 開發規則
```

## 🚀 快速開始

### 🔧 系統需求 (v1.0+ 要求)

- **Python**: 3.10 或更高版本 ⚠️ (v1.0+ 不再支援 Python 3.9)
- **記憶體**: 建議 8GB 以上
- **API 金鑰**: OpenAI API Key (必須)

### 1. 環境設置

```bash
# 檢查 Python 版本 (必須 3.10+)
python --version

# 建立虛擬環境
python -m venv venv-langchain-v1
source venv-langchain-v1/bin/activate  # Linux/Mac
# 或 venv-langchain-v1\Scripts\activate  # Windows

# 🔥 使用 v1.0+ 專用依賴清單
pip install -r Course/requirements-v1.txt

# 或手動安裝核心套件
pip install langchain>=1.0.0 langchain-openai>=0.2.0 langchain-core>=0.3.0
```

### 2. 環境變數設置

建立 `.env` 檔案：
```bash
# 🔥 v1.0+ 環境變數
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key  # 可選
LANGCHAIN_API_KEY=your_langsmith_key      # 監控用 (可選)
LANGCHAIN_TRACING_V2=true                 # 啟用追蹤 (可選)
```

### 3. v1.0+ 快速測試

```bash
# 🚀 推薦：從 Jupyter 開始
cd Course/
jupyter notebook 01-langchain-intro.ipynb

# 或執行 Python 範例
python Module1/1_2_installation_setup.py   # 環境檢查
python Module1/1_3_llm_integration.py      # LLM 整合測試
python Module2/2_1_agent_concepts.py       # 新版 Agent API
```

### 4. 驗證 v1.0+ 安裝

```python
# 快速驗證腳本
from langchain.agents import create_agent  # ✅ v1.0+ import
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

print("🎉 LangChain v1.0+ 安裝成功！")
```

## 📚 v1.0+ 學習路徑

### 🎯 初學者路徑 (2-3 週)
1. **📖 入門導覽**: `01-langchain-intro.ipynb` - v1.0+ 快速概覽
2. **🏗️ Module1**: 基礎建構與 LCEL 語法
   - `1_2_installation_setup.py` - 環境設置檢查
   - `1_3_llm_integration.py` - 新版 LLM 整合
   - `1_4_chains_basics.py` - LCEL Chain 基礎
3. **🤖 Module2**: 新版 Agent 系統
   - `2_1_agent_concepts.py` - create_agent API
   - `2_2_tools_usage.py` - 工具整合實務

### 🎯 進階路徑 (3-4 週)
1. **🔍 Module3**: RAG 與進階功能
   - Jupyter Notebooks: 向量化、Embedding、搜尋
   - `3_5_4_custom_chain_v1.py` - v1.0+ LCEL 自定義 Chain
2. **🎨 實作專案**: project 目錄完整應用
3. **🛠️ 工具開發**: tools 目錄實用工具

### 🎯 專家路徑 (進階自定義)
1. **Middleware 開發**: 人機互動、PII 脫敏
2. **LangGraph 整合**: 複雜工作流程
3. **LLMOps 部署**: 監控、優化、維護

## 🔥 v1.0+ 核心變化解析

### 🚨 重大變更 (Breaking Changes)
| 功能 | v0.3.x (舊版) | v1.0+ (新版) |
|------|---------------|--------------|
| **Agent 建立** | `AgentExecutor.from_agent_and_tools()` | `create_agent(llm, tools, prompt)` |
| **Import 路徑** | `from langchain import create_agent` | `from langchain.agents import create_agent` |
| **工具裝飾器** | `from langchain.agents import tool` | `from langchain_core.tools import tool` |
| **Python 版本** | 3.8+ | 3.10+ |
| **Chain 建構** | 舊版 Chain 類別 | LCEL Pipeline |

### ✨ 新增功能
- **🔧 Middleware 系統**: 攔截和修改 Agent 行為
- **📊 LangSmith 整合**: 內建監控和除錯
- **⚡ 效能優化**: 更快的推理和執行速度
- **🔄 Streaming 改良**: 更好的流式回應處理

## 🔧 技術需求與依賴

### 核心需求
```bash
Python >= 3.10.0           # ⚠️ v1.0+ 最低要求
langchain >= 1.0.0         # 核心框架
langchain-openai >= 0.2.0  # OpenAI 整合
langchain-core >= 0.3.0    # 核心組件
```

### 完整依賴清單
詳見 `Course/requirements-v1.txt`

## 📖 教學重點與目標

### 🏗️ Module1: v1.0+ 基礎架構
- **LCEL 語法掌握**: `chain = prompt | llm | parser`
- **新版 Import**: 正確的模組路徑
- **環境設置**: Python 3.10+ 相容性

### 🤖 Module2: create_agent 精通
- **新版 API**: 告別 `AgentExecutor`
- **工具整合**: `@tool` 裝飾器正確用法
- **Middleware**: 人機互動控制

### 🔍 Module3: RAG 與自定義
- **向量化處理**: Embedding 模型選擇
- **LCEL Chain**: 自定義處理流程
- **效能優化**: 批次處理與並行

## 🎨 專案特色與優勢

### 🎯 v1.0+ 專屬特色
- **🔄 完全更新**: 100% v1.0+ 相容代碼
- **📚 詳細註解**: 每個變更都有說明
- **🧪 實戰測試**: 所有範例可執行
- **🌍 繁體中文**: 完整本地化內容

### 📈 學習效益
- **快速上手**: 1 週掌握基礎 API 變更
- **實用技能**: 企業級 AI 應用開發
- **前瞻性**: 掌握最新 LangChain 技術
- **社群支援**: 完整的學習資源

## 🚀 開始您的 v1.0+ 學習之旅

1. **🔧 環境準備**: 確保 Python 3.10+
2. **📦 安裝依賴**: 使用 `requirements-v1.txt`
3. **🎓 開始學習**: 從 `01-langchain-intro.ipynb` 開始
4. **🤝 社群交流**: 加入 LangChain 中文社群

---

*🚀 LangChain v1.0+ 教學框架 - 迎接 AI 應用開發的新時代！*