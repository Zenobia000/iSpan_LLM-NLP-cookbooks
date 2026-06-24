# 多智能體長文寫作系統 - LangChain / LangGraph 2026 版

這條教學線保留「長文寫作」主題，但框架實作已從 CrewAI 改為 **LangChain v1 + LangGraph**。

核心方向：

- 先用 STORM 建立寫作心智模型：多視角研究 → 大綱 → 初稿 → 修訂。
- 再用 LangChain LCEL 把研究、大綱、寫作做成可讀的 Runnable 管線。
- 最後用 LangGraph 把長文寫作升級為有狀態、可分支、可審核的流程。
- 延伸應用到 `open_deep_research` 風格的 deep research：先做深度研究，再生成長文。

## 課程架構

```text
長文寫作系統開發課程
├── 01_概念理解
│   ├── 01_什麼是智能體.md
│   ├── 01_多智能體系統概念.md
│   ├── 02_STORM寫作法簡介.md
│   └── 02_STORM_長文寫作原理.md
│
├── 02_手動實作
│   ├── 01_基礎環境設定.ipynb
│   ├── 01_環境設定.ipynb
│   ├── 02_建立研究智能體.ipynb
│   ├── 02_手動版_研究智能體.ipynb
│   ├── 03_手動版_寫作智能體.ipynb
│   └── 04_手動版_完整流程.ipynb
│
└── 03_框架實作
    ├── 01_LangChain_LCEL_長文寫作基礎.ipynb
    ├── 02_LangGraph_STORM_長文寫作系統.ipynb
    └── 03_延伸_OpenDeepResearch_長文研究.ipynb
```

## 學習路徑

### 基礎路徑

1. 讀 `01_概念理解/02_STORM寫作法簡介.md`
2. 做 `02_手動實作/02_建立研究智能體.ipynb`
3. 做 `03_框架實作/01_LangChain_LCEL_長文寫作基礎.ipynb`

完成後，你會理解長文寫作不是「直接叫模型寫文章」，而是把研究、大綱、寫作拆成可控步驟。

### 進階路徑

1. 做 `03_框架實作/02_LangGraph_STORM_長文寫作系統.ipynb`
2. 做 `03_框架實作/03_延伸_OpenDeepResearch_長文研究.ipynb`
3. 回到 `../framework/LangGraph_2026/` 看可 import、可 CLI 執行的 deep research 實作

完成後，你會知道如何把長文寫作擴展成真正的 deep research workflow。

## 安裝

建議使用 2026 主線依賴：

```bash
pip install langchain langchain-core langgraph langchain-openai python-dotenv pydantic
```

`.env` 範例：

```bash
COURSE_MODEL=openai:gpt-4o-mini
OPENAI_API_KEY=sk-...

# 選用：deep research 真實搜尋
TAVILY_API_KEY=tvly-...
```

## 為什麼改成 LangChain / LangGraph？

CrewAI 適合快速展示「角色 + 任務 + 團隊」概念，但對 2026 課程主線有三個問題：

- 它把流程藏在框架內，學生不容易看到 state 如何流動。
- 它不是 LangChain/LangGraph 的原生編排模型，後續接 LangGraph 記憶、串流、human-in-the-loop 會繞路。
- 它會讓課程焦點從「如何設計可控的 AI workflow」偏到「如何使用某個多智能體框架」。

LangGraph 的優點是每個節點、狀態、分支、重試與審核都明確可見。對 notebook 教學尤其適合，因為學生可以逐格執行、檢查 state、替換節點。

## 與 Deep Research 的銜接

長文寫作與 deep research 的差別在於研究深度：

```text
一般長文寫作:
research notes → outline → draft → review → final article

Deep research 長文寫作:
research plan → parallel researchers → findings → final report → outline/draft/review
```

本資料夾的 `03_延伸_OpenDeepResearch_長文研究.ipynb` 會示範如何呼叫 `framework/LangGraph_2026/deep_research`，把 deep research 的 `final_report` 當作長文寫作素材。

## 學習成果

完成這條線後，你應該能：

- 解釋 STORM 為什麼能提升長文品質。
- 用 LCEL 建立可組合的寫作管線。
- 用 LangGraph 建立有狀態的長文寫作 workflow。
- 把 research node 升級成 deep research graph。
- 知道何時該用簡單 LCEL，何時該升級到 LangGraph。
