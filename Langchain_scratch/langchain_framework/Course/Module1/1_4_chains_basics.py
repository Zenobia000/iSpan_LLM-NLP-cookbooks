"""
LangChain 0.3+ Chains 基礎教學
展示不同類型的 Chains 使用方式，包含 Sequential Chain、Router Chain 等

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- langchain-core>=0.1.0
- python-dotenv>=0.19.0
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import Dict, Any, List
from dotenv import load_dotenv
import logging
import os
import json

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()


def create_translation_chain():
    """
    建立翻譯鏈：將輸入文本翻譯成指定語言
    """
    # 建立 LLM
    llm = ChatOpenAI(temperature=0.1)

    # 建立翻譯提示詞
    translation_prompt = ChatPromptTemplate.from_template("""
    請將以下文本翻譯成{target_language}：

    文本：{text}

    只需要提供翻譯結果，不需要其他說明。
    """)

    # 建立翻譯鏈
    translation_chain = (
        translation_prompt
        | llm
        | StrOutputParser()
    )

    return translation_chain


def create_analysis_chain():
    """
    建立分析鏈：分析文本的主要觀點和情感
    """
    llm = ChatOpenAI(temperature=0.3)

    analysis_prompt = ChatPromptTemplate.from_template("""
    請分析以下文本的主要觀點和情感：

    文本：{text}

    請以 JSON 格式回答，包含以下欄位：
    - main_points: 主要觀點列表
    - sentiment: 情感傾向 (positive/neutral/negative)
    - confidence: 信心分數 (0-1)
    """)

    def parse_json_response(text: str) -> Dict:
        """解析 JSON 格式的回應"""
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失敗: {str(e)}")
            return {
                "main_points": ["解析失敗"],
                "sentiment": "neutral",
                "confidence": 0
            }

    # 建立分析鏈
    analysis_chain = (
        analysis_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(parse_json_response)
    )

    return analysis_chain


def create_router_chain():
    """
    建立路由鏈：根據輸入內容選擇適當的處理鏈
    """
    llm = ChatOpenAI(temperature=0)

    router_prompt = ChatPromptTemplate.from_template("""
    請判斷以下文本需要什麼類型的處理：

    文本：{text}

    只需回答以下其中之一：
    - translation：如果文本需要翻譯
    - analysis：如果文本需要分析
    """)

    def route_chain(inputs: Dict[str, Any]):
        """根據路由結果選擇處理鏈"""
        text = inputs["text"]
        chain_type = inputs["chain_type"].strip().lower()

        if chain_type == "translation":
            chain = create_translation_chain()
            result = chain.invoke({
                "text": text,
                "target_language": "英文"
            })
            return result
        elif chain_type == "analysis":
            chain = create_analysis_chain()
            result = chain.invoke({"text": text})
            return result
        else:
            return f"不支援的處理類型: {chain_type}"

    # 建立路由鏈
    router_chain = (
        {"text": RunnablePassthrough(), "chain_type": router_prompt | llm | StrOutputParser()}
        | RunnableLambda(route_chain)
    )

    return router_chain


def demonstrate_chains():
    """
    展示不同類型的 Chains 使用方式
    """
    test_cases = [
        {
            "name": "翻譯鏈",
            "chain": create_translation_chain(),
            "input": {
                "text": "人工智慧正在改變我們的生活方式",
                "target_language": "英文"
            }
        },
        {
            "name": "分析鏈",
            "chain": create_analysis_chain(),
            "input": {
                "text": "這個新產品的設計非常創新，但價格偏高，可能會影響銷量。"
            }
        },
        {
            "name": "路由鏈",
            "chain": create_router_chain(),
            "input": "Please translate this text to Chinese"
        }
    ]

    for case in test_cases:
        print(f"\n=== 測試 {case['name']} ===")
        try:
            result = case["chain"].invoke(case["input"])
            print(f"輸入: {case['input']}")
            print(f"輸出: {result}")
        except Exception as e:
            logger.error(f"{case['name']}執行失敗: {str(e)}")


def main():
    """
    主程式：展示 Chains 的基本使用方式
    """
    print("=== LangChain 0.3+ Chains 基礎展示 ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return

    try:
        demonstrate_chains()
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")


if __name__ == "__main__":
    main()


