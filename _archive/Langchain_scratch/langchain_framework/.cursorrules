# Instructions

During you interaction with the user, if you find anything reusable in this project (e.g. version of a library, model name), especially about a fix to a mistake you made or a correction you received, you should take note in the `Lessons` section in the `.cursorrules` file so you will not make the same mistake again.

You should also use the `.cursorrules` file as a scratchpad to organize your thoughts. Especially when you receive a new task, you should first review the content of the scratchpad, clear old different task if necessary, first explain the task, and plan the steps you need to take to complete the task. You can use todo markers to indicate the progress, e.g.
[X] Task 1
[ ] Task 2

Also update the progress of the task in the Scratchpad when you finish a subtask.
Especially when you finished a milestone, it will help to improve your depth of task accomplishment to use the scratchpad to reflect and plan.
The goal is to help you maintain a big picture as well as the progress of the task. Always refer to the Scratchpad when you plan the next step.

# Tools

Note all the tools are in python. So in the case you need to do batch processing, you can always consult the python files and write your own script.

## Screenshot Verification
The screenshot verification workflow allows you to capture screenshots of web pages and verify their appearance using LLMs. The following tools are available:

1. Screenshot Capture:
```bash
python tools/screenshot_utils.py URL [--output OUTPUT] [--width WIDTH] [--height HEIGHT]
```

2. LLM Verification with Images:
```bash
python tools/llm_api.py --prompt "Your verification question" --provider {openai|anthropic} --image path/to/screenshot.png
```

Example workflow:
```python
from screenshot_utils import take_screenshot_sync
from llm_api import query_llm

# Take a screenshot
screenshot_path = take_screenshot_sync('https://example.com', 'screenshot.png')

# Verify with LLM
response = query_llm(
    "What is the background color and title of this webpage?",
    provider="openai",  # or "anthropic"
    image_path=screenshot_path
)
print(response)
```

## LLM

You always have an LLM at your side to help you with the task. For simple tasks, you could invoke the LLM by running the following command:
```
python ./tools/llm_api.py --prompt "What is the capital of France?" --provider "anthropic"
```

The LLM API supports multiple providers:
- OpenAI (default, model: gpt-4o)
- Azure OpenAI (model: configured via AZURE_OPENAI_MODEL_DEPLOYMENT in .env file, defaults to gpt-4o-ms)
- DeepSeek (model: deepseek-chat)
- Anthropic (model: claude-3-sonnet-20240229)
- Gemini (model: gemini-pro)
- Local LLM (model: Qwen/Qwen2.5-32B-Instruct-AWQ)

But usually it's a better idea to check the content of the file and use the APIs in the `tools/llm_api.py` file to invoke the LLM if needed.

## Web browser

You could use the `tools/web_scraper.py` file to scrape the web.
```
python ./tools/web_scraper.py --max-concurrent 3 URL1 URL2 URL3
```
This will output the content of the web pages.

## Search engine

You could use the `tools/search_engine.py` file to search the web.
```
python ./tools/search_engine.py "your search keywords"
```
This will output the search results in the following format:
```
URL: https://example.com
Title: This is the title of the search result
Snippet: This is a snippet of the search result
```
If needed, you can further use the `web_scraper.py` file to scrape the web page content.

# Lessons

## User Specified Lessons

- You have a python venv in ./venv. Use it.
- Include info useful for debugging in the program output.
- Read the file before you try to edit it.
- Due to Cursor's limit, when you use `git` and `gh` and need to submit a multiline commit message, first write the message in a file, and then use `git commit -F <filename>` or similar command to commit. And then remove the file. Include "[Cursor] " in the commit message and PR title.

## Cursor learned

- For search results, ensure proper handling of different character encodings (UTF-8) for international queries
- Add debug information to stderr while keeping the main output clean in stdout for better pipeline integration
- When using seaborn styles in matplotlib, use 'seaborn-v0_8' instead of 'seaborn' as the style name due to recent seaborn version changes
- Use 'gpt-4o' as the model name for OpenAI's GPT-4 with vision capabilities


# Scratchpad

## Task: 建立 LangChain 課程範例程式

### 檔案結構規劃
Course/
├── Module1/
│   ├── 1_1_framework_overview.py      ✓
│   ├── 1_2_installation_setup.py      ✓
│   ├── 1_3_llm_integration.py         ✓
│   └── 1_4_chains_basics.py           ✓
├── Module2/
│   ├── 2_1_agent_concepts.py          ✓
│   ├── 2_2_tools_usage.py             ✓
│   └── 2_3_agent_automation.py        ✓
├── Module3/
│   ├── 3_1_rag_basics.py              ✓
│   ├── 3_2_vector_retrieval.py        ✓
│   │   ├── 3_2_1_vectorstores_comparison.py  ✓
│   │   ├── 3_2_2_embedding_models.py         ✓
│   │   └── 3_2_3_similarity_search.py        ✓
│   ├── 3_3_document_loaders.py        ✓
│   ├── 3_4_text_splitters.py          ✓
│   └── 3_5_custom_models.py           △
│       ├── 3_5_1_custom_llm_agent_template.py    
│       ├── 3_5_2_custom_llm_agent.py    
│       ├── 3_5_3_custom_embedding.py  
│       └── 3_5_4_custom_chain.py      
├── Module4/
│   ├── 4_1_monitoring.py
│   ├── 4_2_optimization.py
│   └── 4_3_deployment.py
└── Module5/
    ├── 5_1_ai_assistant.py
    ├── 5_2_document_analysis.py
    ├── 5_3_customer_service.py
    └── 5_4_advanced_features.py

### Progress
[X] 規劃檔案結構
[X] 建立 Module1 範例
    [X] 1.1 框架概覽
    [X] 1.2 安裝設定
    [X] 1.3 LLM 整合
    [X] 1.4 Chains 基礎
[X] 建立 Module2 範例
    [X] 2.1 Agent 概念
    [X] 2.2 工具使用
    [X] 2.3 Agent 自動化
[X] 建立 Module3 範例
    [X] 3.1 RAG 基礎
    [X] 3.2 向量檢索
        [X] 3.2.1 向量資料庫比較
        [X] 3.2.2 Embedding 模型比較
        [X] 3.2.3 相似度搜尋方法比較
    [X] 3.3 文件載入器
    [X] 3.4 文本分割
    [ ] 3.5 自定義模型
        [X] 3.5.1 自定義 LLM Agent 模板
        [X] 3.5.2 自定義 LLM Agent 實作
        [ ] 3.5.3 自定義 Embedding
        [ ] 3.5.4 自定義 Chain
[ ] 建立 Module4 範例
    [ ] 4.1 監控與日誌
        [ ] 4.1.1 系統監控
        [ ] 4.1.2 性能追蹤
        [ ] 4.1.3 錯誤處理
    [ ] 4.2 效能優化
        [ ] 4.2.1 批次處理
        [ ] 4.2.2 快取機制
        [ ] 4.2.3 並行處理
    [ ] 4.3 部署實務
        [ ] 4.3.1 容器化部署
        [ ] 4.3.2 雲端服務整合
        [ ] 4.3.3 負載平衡
[ ] 建立 Module5 範例
    [ ] 5.1 AI 助理開發
        [ ] 5.1.1 對話管理
        [ ] 5.1.2 上下文處理
        [ ] 5.1.3 記憶機制
    [ ] 5.2 文件分析系統
        [ ] 5.2.1 文件處理流程
        [ ] 5.2.2 資訊擷取
        [ ] 5.2.3 報告生成
    [ ] 5.3 客服自動化
        [ ] 5.3.1 意圖識別
        [ ] 5.3.2 回應生成
        [ ] 5.3.3 工單處理
    [ ] 5.4 進階功能整合
        [ ] 5.4.1 多模態處理
        [ ] 5.4.2 知識圖譜
        [ ] 5.4.3 評估與優化

### Notes
- Module 3.5 (自定義模型) 將是下一個開發重點
- 需要規劃具體的自定義模型範例，包括：
  - 自定義 LLM 模型封裝
  - 自定義 Embedding 模型實作
  - 自定義 Chain 元件開發
- Module 4 和 5 的細項任務已完整規劃
- 建議先完成 Module 3.5 再開始 Module 4

### Lessons Learned
- 使用 langchain 0.3+ 的新 API 結構
- 使用 LCEL (LangChain Expression Language) 建立處理鏈
- ChatOpenAI 替代舊版的 OpenAI
- 使用 logging 模組進行錯誤處理
- Anthropic Claude 模型名稱更新為 claude-3-5-sonnet-20240620
- 使用 ChatAnthropic 替代舊版的 Anthropic
- 需要安裝 pandas-stubs 以解決類型提示問題
- 使用 pydantic v2 (>=2.0.0) 而不是 langchain_core.pydantic_v1
- BaseRetriever.get_relevant_documents 已棄用，應使用 invoke 方法
- 需要安裝 scikit-learn 以使用 cosine_similarity
- 文本分割器設計經驗：
  - 不需要抽象基類，直接實現具體分割器更簡單
  - 統一使用 chunk_size 和 chunk_overlap 屬性
  - 重疊比例計算應統一處理
  - 分割器參數需要根據實際文本長度調整
  - chunk_size 建議設為 200 以上，確保語義完整性
  - chunk_overlap 建議為 chunk_size 的 5-10%

# RAG 相關經驗
- 不同檢索方法的分數需要正規化到相同範圍 (0-100)
- 相似度計算方法：
  - 統一使用向量的餘弦相似度作為基準
  - 預先計算所有文檔的向量表示以提高效率
  - MMR 使用同一向量空間計算相似度和冗餘度
  - BM25 分數需要通過向量相似度重新校準
- 需要記錄搜尋時間以評估性能
- 原始分數和正規化分數都應該保留以便分析
- 使用 perf_counter 而不是 time.time 以獲得更精確的時間測量
- 向量計算應使用 numpy 提高效率
- 相似度計算應考慮數值穩定性
- 避免重複定義方法描述字典，統一使用一個字典來管理
- 確保方法描述的一致性和可維護性
- 使用外部字典便於集中管理和更新方法描述
- DataFrame 操作注意事項：
  - 避免使用中文作為欄位名稱
  - 檢查 DataFrame 是否為空
  - 確保分組欄位存在
  - 使用英文命名提高相容性
- HNSW 參數調優：
  - ef_construction: 建構時的精確度，小數據集可設為 200
  - M: 每個節點的最大連接數，平衡搜尋速度和精確度
  - ef: 搜尋時的精確度，可設為數據集大小
  - 小數據集優先考慮精確度而非速度
  - 使用 cosine 空間並正規化分數
- 索引處理：
  - HNSW 返回的索引需要轉換為整數類型
  - 使用 astype(int) 進行類型轉換
  - 在使用索引前進行類型檢查
  - 確保 metadata 中的索引也是整數類型
- 測試資料設計：
  - 包含不同長度和複雜度的文檔
  - 跨領域的主題分布
  - 相關度分級（0-2分）而非二元
  - 每個查詢都有明確的最佳答案
- 評估指標計算：
  - 多級相關度（0-2分）需要正規化到 0-1
  - Precision 考慮最高可能分數
  - Recall 使用相關度總分作為分母
  - MRR 根據相關度級別調整排名分數
  - NDCG 在對數計算前正規化相關度

# Chroma 相關經驗
- 從 langchain_community.vectorstores 改為使用 langchain_chroma
- Chroma Client API 改用 PersistentClient 替代舊版 Client
- collection_name 命名規則：
  - 長度在 3-63 字符之間
  - 只能包含字母、數字、下劃線和連字符
  - 開頭和結尾必須是字母或數字
  - 不能包含連續的句點
  - 不能是有效的 IPv4 地址
- persist_directory 用於指定向量資料庫的存儲位置
- 需要確保存儲目錄存在
- 使用 os.makedirs 建立目錄結構
- 使用 exist_ok=True 避免目錄已存在的錯誤
- 新版需要明確設定 client_settings
- 建議關閉匿名遙測功能
- 測試完成後清理資料庫文件

# 文件載入器經驗
- 文件載入器應從 langchain_community.document_loaders 導入
- 自定義載入器應繼承 langchain_core.document_loaders.BaseLoader
- Pydantic 模型的字串欄位不接受 Path 物件，需要使用 str() 轉換
- 使用 Path 物件處理檔案路徑時，傳給 loader 前需要轉換為字串
- 在 Windows 系統上處理含中文的 CSV 檔案時，需要明確指定 UTF-8 編碼
- JSONLoader 的 metadata_func 需要接受兩個參數：metadata 和 additional_fields

# 資料庫管理經驗
- 在初始化前清理舊的資料庫文件
- 使用 cosine 相似度空間配置
- 資料庫路徑需要一致性管理
- 建議實作資料庫清理和重建功能

# Embedding 評估經驗
# 性能測試標準化
- 測試環境標準化：
  - 硬體規格記錄（CPU、RAM、GPU）
  - Python 版本與相依套件版本固定
  - 系統負載監控

- 測試數據標準化：
  - 固定測試集大小（如 1000 筆文本）
  - 文本長度分佈一致（短、中、長文本各佔比）
  - 多語言測試集（中、英、日等比例分配）
  - 特殊字符與數字的分佈控制

- 評估指標標準化：
  - 向量維度與品質
    - 向量維度記錄
    - 向量正規化方法統一
    - 零向量檢查
  - 性能指標
    - 批次處理大小統一（如 32）
    - 預熱回合（warm-up）次數固定
    - 多次重複測試取平均值（如 5 次）
    - 計時精確度（使用 time.perf_counter）
  - 資源使用
    - CPU 使用率基線校正
    - GPU 記憶體監控標準化
    - RAM 使用量基線校正
  - 成本計算
    - 標準化每千字符成本
    - API 呼叫計費單位統一

- 測試流程標準化：
  - 測試前系統狀態重置
  - 緩存清理標準化
  - 錯誤處理與重試機制
  - 結果記錄格式統一
  - 測試日誌完整性

- 報告生成標準化：
  - 測試環境完整記錄
  - 原始數據保存
  - 統計指標一致性
  - 視覺化圖表標準化
  - 異常值處理規則

# 視覺化注意事項
- 向量資料需要轉換為 numpy array
- t-SNE perplexity 參數需小於樣本數
- 圖表需要適當的大小和格式化
- 使用 tight_layout() 避免標籤被截斷
- 添加完整的錯誤處理

# 文本分割經驗
- 分割策略選擇：
  - 使用 RecursiveCharacterTextSplitter 處理巢狀結構
  - 使用 MarkdownHeaderTextSplitter 處理 Markdown 文件
  - 使用 TokenTextSplitter 處理需要精確 token 控制的文本
- 分割參數調優：
  - chunk_size: 根據 LLM 的上下文窗口大小設定
  - chunk_overlap: 通常設為 chunk_size 的 10-20%
  - separators: 按優先順序設定分隔符
- 中文文本處理：
  - 使用 jieba 進行中文分詞
  - 保留標點符號作為分割參考
  - 避免破壞完整句子結構
- 分割品質控制：
  - 檢查分割後的文本完整性
  - 確保關鍵信息不被切分
  - 保持段落語意的連貫性
- 效能優化：
  - 批次處理大型文檔
  - 使用多進程加速分割
  - 緩存分割結果

