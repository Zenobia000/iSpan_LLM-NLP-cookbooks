"""
建立 LangChain 課程的範例資料
"""

import os
from pathlib import Path
import json
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_files():
    """建立範例檔案"""
    # 取得當前目錄
    current_dir = Path(__file__).parent
    sample_dir = current_dir / "samples"
    
    try:
        # 確保目錄存在
        sample_dir.mkdir(exist_ok=True)
        logger.info(f"建立/確認範例目錄: {sample_dir}")
        
        # 建立 AI 介紹文章
        text_content = """人工智能（AI）是計算機科學的一個分支，它致力於研究和開發能夠模擬、延伸和擴展人類智能的理論、
方法、技術及應用系統。人工智能的研究包括機器學習、自然語言處理、計算機視覺、專家系統等領域。

近年來，深度學習技術的突破使得人工智能在圖像識別、語音識別、自然語言處理等任務上取得了顯著進展。
人工智能正在改變著我們的生活方式，從智能手機助手到自動駕駛汽車，從醫療診斷到金融預測，
人工智能技術的應用範圍正在不斷擴大。

然而，人工智能的發展也帶來了一些挑戰，例如數據隱私、算法偏見、就業影響等問題需要社會各界共同關注和解決。"""
        
        with open(sample_dir / "ai_introduction.txt", "w", encoding="utf-8") as f:
            f.write(text_content)
        logger.info("已建立 ai_introduction.txt")
        
        # 建立模型比較表
        csv_content = """Model,Accuracy,Training Time,Memory Usage,Description
BERT-base,0.92,12.5,5.2,"基礎BERT模型，適合一般NLP任務"
RoBERTa-large,0.95,24.3,10.8,"改進版BERT，性能更優但資源消耗較大"
GPT-3,0.98,168.0,45.6,"大型語言模型，功能強大但成本高"
T5-small,0.89,8.2,3.4,"輕量級T5模型，適合資源受限場景"
BART,0.93,15.7,7.1,"序列到序列模型，適合生成任務\""""
        
        with open(sample_dir / "model_comparison.csv", "w", encoding="utf-8") as f:
            f.write(csv_content)
        logger.info("已建立 model_comparison.csv")
        
        # 建立 AI 助手配置
        json_content = {
            "project": "AI Assistant",
            "version": "1.0.0",
            "models": [
                {
                    "name": "GPT-4",
                    "type": "Language Model",
                    "parameters": {
                        "temperature": 0.7,
                        "max_tokens": 2000,
                        "top_p": 0.95
                    },
                    "capabilities": [
                        "Text Generation",
                        "Code Completion",
                        "Question Answering"
                    ]
                },
                {
                    "name": "DALL-E 3",
                    "type": "Image Model",
                    "parameters": {
                        "quality": "high",
                        "style": "natural",
                        "size": "1024x1024"
                    },
                    "capabilities": [
                        "Image Generation",
                        "Image Editing",
                        "Style Transfer"
                    ]
                }
            ],
            "api_endpoints": [
                "/chat",
                "/complete",
                "/generate-image"
            ]
        }
        
        with open(sample_dir / "ai_assistant_config.json", "w", encoding="utf-8") as f:
            json.dump(json_content, f, ensure_ascii=False, indent=2)
        logger.info("已建立 ai_assistant_config.json")
        
        # 建立技術新聞
        news_content = """AI突破|2024-03-15|科技記者王小明|OpenAI發布GPT-5模型，在多項基準測試中超越人類表現。新模型在邏輯推理、創意寫作和程式設計等領域都展現出驚人的能力。
半導體產業|2024-03-14|財經記者李小華|台積電宣布在日本熊本建立第二座晶圓廠，預計2024年底完工。此舉將強化台積電在日本的布局，並為日本半導體產業注入新動力。
科技創新|2024-03-13|科技記者張小芳|Google DeepMind推出新一代AlphaFold模型，可準確預測蛋白質結構，為藥物研發帶來重大突破。
產業趨勢|2024-03-12|產業記者陳小強|全球AI晶片市場快速成長，預計2025年市場規模將達1000億美元，台灣廠商積極布局搶占商機。
研究發展|2024-03-11|科學記者林小雨|中研院發表重大研究成果，運用AI技術成功預測新冠病毒變異株的傳播趨勢，為防疫工作提供重要參考。"""
        
        with open(sample_dir / "tech_news.txt", "w", encoding="utf-8") as f:
            f.write(news_content)
        logger.info("已建立 tech_news.txt")
        
        # 建立開發指南
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>AI Development Guide</title>
    <meta charset="UTF-8">
    <meta name="description" content="A comprehensive guide to AI development">
</head>
<body>
    <h1>人工智能開發指南</h1>
    
    <div class="section">
        <h2>基礎要求</h2>
        <ul>
            <li>Python 3.8+</li>
            <li>TensorFlow 或 PyTorch</li>
            <li>CUDA 支援（可選）</li>
            <li>基礎數學和統計知識</li>
        </ul>
    </div>

    <div class="section">
        <h2>開發環境設置</h2>
        <ol>
            <li>安裝 Python</li>
            <li>設置虛擬環境</li>
            <li>安裝依賴套件</li>
            <li>配置 GPU 環境</li>
        </ol>
    </div>

    <div class="section">
        <h2>推薦學習路徑</h2>
        <p>建議按照以下順序學習：</p>
        <ol>
            <li>Python 基礎程式設計</li>
            <li>機器學習基礎理論</li>
            <li>深度學習框架使用</li>
            <li>專案實戰練習</li>
        </ol>
    </div>
</body>
</html>"""
        
        with open(sample_dir / "dev_guide.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info("已建立 dev_guide.html")
        
        # 建立 LangChain 教程
        md_content = """# LangChain 開發指南

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
3. Router Chain"""
        
        with open(sample_dir / "langchain_tutorial.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("已建立 langchain_tutorial.md")
        
        return sample_dir
        
    except Exception as e:
        logger.error(f"建立範例檔案時發生錯誤: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    create_sample_files() 