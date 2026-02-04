"""
LangChain v1.0+ 自定義 Chain 範例

展示如何使用 v1.0+ 的 LCEL (LangChain Expression Language) 建立自定義處理流程：
1. 基礎 LCEL Chain
2. 多階段處理 Chain
3. 條件分支 Chain
4. 並行處理 Chain
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
    RunnableBranch
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import json

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

def create_basic_chain():
    """建立基礎 v1.0+ LCEL Chain"""

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_template("""
    你是一位專業的文案寫手。
    請根據以下要求撰寫內容：{request}

    風格要求：{style}
    字數限制：{word_limit} 字以內

    請提供高品質的內容。
    """)

    output_parser = StrOutputParser()

    # v1.0+ LCEL 語法
    chain = prompt | model | output_parser

    return chain

def create_multi_stage_chain():
    """建立多階段處理 Chain"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 第一階段：分析內容
    analysis_prompt = ChatPromptTemplate.from_template("""
    請分析以下文本的主要特點：{text}

    請以 JSON 格式回答，包含：
    - topic: 主題
    - tone: 語調
    - key_points: 重點列表
    """)

    # 第二階段：改寫內容
    rewrite_prompt = ChatPromptTemplate.from_template("""
    根據分析結果：{analysis}

    請將原文：{text}

    改寫成更專業的版本，保持原意但提升品質。
    """)

    def parse_analysis(text: str) -> dict:
        """解析分析結果"""
        try:
            return json.loads(text)
        except:
            return {"topic": "未知", "tone": "中性", "key_points": []}

    # 建立多階段 Chain
    chain = (
        # 輸入處理
        {"text": RunnablePassthrough()}
        # 第一階段：分析
        | RunnableParallel({
            "text": lambda x: x["text"],
            "analysis": analysis_prompt | model | StrOutputParser() | RunnableLambda(parse_analysis)
        })
        # 第二階段：改寫
        | rewrite_prompt | model | StrOutputParser()
    )

    return chain

def create_branching_chain():
    """建立條件分支 Chain"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 判斷語言的函數
    def detect_language(text: str) -> str:
        """簡單的語言檢測"""
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "chinese"
        else:
            return "english"

    # 中文處理 Chain
    chinese_prompt = ChatPromptTemplate.from_template("""
    請將以下中文內容總結成重點：{text}

    要求：條列式，每點不超過20字
    """)

    # 英文處理 Chain
    english_prompt = ChatPromptTemplate.from_template("""
    Please summarize the following English text into key points: {text}

    Requirements: Bullet points, max 20 words per point
    """)

    chinese_chain = chinese_prompt | model | StrOutputParser()
    english_chain = english_prompt | model | StrOutputParser()

    # v1.0+ 條件分支語法
    branching_chain = RunnableBranch(
        # 條件 1: 中文
        (lambda x: detect_language(x["text"]) == "chinese", chinese_chain),
        # 條件 2: 英文
        (lambda x: detect_language(x["text"]) == "english", english_chain),
        # 預設
        chinese_chain
    )

    return branching_chain

def create_parallel_chain():
    """建立並行處理 Chain"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    # 不同的處理任務
    summary_prompt = ChatPromptTemplate.from_template("請總結：{text}")
    sentiment_prompt = ChatPromptTemplate.from_template("請分析情感傾向：{text}")
    keywords_prompt = ChatPromptTemplate.from_template("請提取關鍵詞：{text}")

    # 並行處理 Chain
    parallel_chain = RunnableParallel({
        "summary": summary_prompt | model | StrOutputParser(),
        "sentiment": sentiment_prompt | model | StrOutputParser(),
        "keywords": keywords_prompt | model | StrOutputParser(),
        "original": RunnablePassthrough()
    })

    return parallel_chain

async def create_async_chain():
    """建立異步處理 Chain"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("請分析：{text}")

    # v1.0+ 異步 Chain
    async_chain = prompt | model | StrOutputParser()

    return async_chain

def main():
    """展示 v1.0+ 自定義 Chain 的使用"""

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return

    print("🔗 LangChain v1.0+ 自定義 Chain 範例")
    print("✨ 使用最新的 LCEL (LangChain Expression Language)")

    # 測試輸入
    test_inputs = {
        "basic": {
            "request": "寫一篇關於AI的介紹",
            "style": "專業且易懂",
            "word_limit": "200"
        },
        "multi_stage": {
            "text": "人工智慧正在改變我們的生活，從智慧手機到自動駕駛汽車。"
        },
        "branching": {
            "text": "Artificial intelligence is transforming our daily lives."
        },
        "parallel": {
            "text": "今天天氣很好，心情愉快，準備去公園散步。"
        }
    }

    try:
        # 1. 測試基礎 Chain
        logger.info("🔸 測試基礎 LCEL Chain...")
        basic_chain = create_basic_chain()
        basic_result = basic_chain.invoke(test_inputs["basic"])
        print(f"\n基礎 Chain 結果：\n{basic_result[:100]}...")

        # 2. 測試多階段 Chain
        logger.info("🔸 測試多階段 Chain...")
        multi_stage_chain = create_multi_stage_chain()
        multi_stage_result = multi_stage_chain.invoke(test_inputs["multi_stage"])
        print(f"\n多階段 Chain 結果：\n{multi_stage_result[:100]}...")

        # 3. 測試分支 Chain
        logger.info("🔸 測試條件分支 Chain...")
        branching_chain = create_branching_chain()
        branch_result = branching_chain.invoke(test_inputs["branching"])
        print(f"\n分支 Chain 結果：\n{branch_result}")

        # 4. 測試並行 Chain
        logger.info("🔸 測試並行 Chain...")
        parallel_chain = create_parallel_chain()
        parallel_result = parallel_chain.invoke(test_inputs["parallel"])
        print(f"\n並行 Chain 結果：")
        for key, value in parallel_result.items():
            if key != "original":
                print(f"  {key}: {value[:50]}...")

        logger.info("✅ 所有 v1.0+ Chain 測試完成！")

    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()