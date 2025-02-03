"""
LangChain 0.3+ Agent 概念與架構
展示 Agent 的基本概念、ReAct 框架和 OpenAI Function Calling

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- langchain-core>=0.1.0
- langchain-community>=0.0.1
- python-dotenv>=0.19.0
"""

from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
import os
import json
import datetime

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()


@tool
def get_current_time() -> str:
    """獲取當前時間"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate_bmi(height: float, weight: float) -> str:
    """
    計算 BMI 指數

    Args:
        height: 身高(公尺)
        weight: 體重(公斤)
    """
    try:
        bmi = weight / (height ** 2)

        # 判斷 BMI 範圍
        if bmi < 18.5:
            category = "體重過輕"
        elif 18.5 <= bmi < 24:
            category = "體重正常"
        elif 24 <= bmi < 27:
            category = "體重過重"
        else:
            category = "肥胖"

        return f"BMI: {bmi:.1f}, 狀態: {category}"
    except Exception as e:
        return f"計算失敗: {str(e)}"


@tool
def analyze_text_sentiment(text: str) -> str:
    """
    分析文本情感

    Args:
        text: 要分析的文本
    """
    # 這裡使用簡單的關鍵詞判斷，實際應用中可以使用更複雜的情感分析模型
    positive_words = {"喜歡", "開心", "讚", "優秀", "棒", "好"}
    negative_words = {"討厭", "生氣", "爛", "差", "糟", "壞"}

    positive_count = sum(1 for word in positive_words if word in text)
    negative_count = sum(1 for word in negative_words if word in text)

    if positive_count > negative_count:
        return "正面情感"
    elif negative_count > positive_count:
        return "負面情感"
    else:
        return "中性情感"


def create_agent() -> AgentExecutor:
    """
    建立 Agent 執行器
    """
    # 建立 LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    # 定義系統提示詞
    system_prompt = """你是一位專業的 AI 助理，可以使用各種工具來協助用戶。

    你可以：
    1. 查詢當前時間
    2. 計算 BMI 指數
    3. 分析文本情感

    請根據用戶的需求，選擇合適的工具來提供協助。
    回答時請使用繁體中文，並保持專業、友善的態度。
    """

    # 建立提示詞模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name='{chat_history}'),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name='{agent_scratchpad}')
    ])

    # 建立工具列表
    tools = [
        get_current_time,
        calculate_bmi,
        analyze_text_sentiment
    ]

    # 建立 Agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # 建立 Agent 執行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return agent_executor


def demonstrate_agent():
    """
    展示 Agent 的使用方式
    """
    agent_executor = create_agent()

    # 測試案例
    test_cases = [
        "現在幾點了？",
        "我身高 1.75 公尺，體重 70 公斤，請幫我計算 BMI",
        "這部電影真的很棒，我非常喜歡！",
        "請幫我分析以下評論的情感：這家餐廳的服務態度很差，食物也不好吃",
        "我想知道現在的時間，順便幫我計算 BMI，我身高 1.8 公尺，體重 75 公斤"
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n=== 測試案例 {i} ===")
        print(f"問題: {question}")
        try:
            response = agent_executor.invoke({"input": question})
            print(f"回答: {response['output']}")
        except Exception as e:
            logger.error(f"執行失敗: {str(e)}")


def main():
    """
    主程式：展示 Agent 的基本概念
    """
    print("=== LangChain 0.3+ Agent 概念展示 ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return

    try:
        demonstrate_agent()
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")


if __name__ == "__main__":
    main()
