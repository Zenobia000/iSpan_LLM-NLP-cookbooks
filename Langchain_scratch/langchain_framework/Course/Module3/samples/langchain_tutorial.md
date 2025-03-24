# LangChain 開發指南

## 1. 基礎概念
LangChain 是一個強大的框架，用於開發 LLM 驅動的應用程序。

### 1.1 核心組件
- LLMs
- Prompt Templates
- Memory
- Chains
- Agents

## 2. 安裝設定
使用 pip 安裝 LangChain：
```bash
pip install langchain
```

### 2.1 環境配置
設置必要的環境變量：
- OPENAI_API_KEY
- SERPAPI_API_KEY

## 3. 基本用法
以下是一個簡單的示例：
```python
from langchain.llms import OpenAI
llm = OpenAI()
```

### 3.1 創建 Chain
Chains 可以將多個組件連接起來：
1. LLM Chain
2. Sequential Chain
3. Router Chain