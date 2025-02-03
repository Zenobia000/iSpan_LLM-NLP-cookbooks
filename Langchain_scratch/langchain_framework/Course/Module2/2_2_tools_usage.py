"""
LangChain 0.3+ 工具使用教學
展示如何整合和使用各種實用工具，包含搜尋、計算、API 調用等

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- langchain-community>=0.0.1
- python-dotenv>=0.19.0
- wikipedia-api>=0.6.0
- python-arxiv>=0.7.1
- duckduckgo-search>=4.1.1
"""

from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ArxivQueryRun
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
import os
import json
import arxiv
import wikipediaapi

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()


@tool
def calculator(expression: str) -> str:
    """
    安全的數學計算工具

    Args:
        expression: 數學表達式，如 '1 + 2 * 3'
    """
    try:
        # 使用 eval 前先檢查表達式是否安全
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "錯誤：表達式包含不允許的字符"

        result = eval(expression)
        return f"計算結果: {result}"
    except Exception as e:
        return f"計算失敗: {str(e)}"


def setup_wikipedia_tool() -> WikipediaQueryRun:
    """設定 Wikipedia 搜尋工具"""
    wiki_client = wikipediaapi.Wikipedia(
        language="zh",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="langchain_tutorial/1.0"
    )

    wikipedia = WikipediaAPIWrapper(
        wiki_client=wiki_client,
        lang="zh",
        top_k_results=3
    )
    return WikipediaQueryRun(api_wrapper=wikipedia)


def setup_search_tools() -> List[Any]:
    """設定搜尋相關工具"""
    # DuckDuckGo 搜尋
    search = DuckDuckGoSearchRun()

    # arXiv 論文搜尋
    arxiv_tool = ArxivQueryRun()

    return [
        search,
        arxiv_tool,
        setup_wikipedia_tool()
    ]


def create_agent_with_tools() -> AgentExecutor:
    """
    建立具備多種工具的 Agent
    """
    # 建立 LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    # 定義系統提示詞
    system_prompt = """你是一位專業的研究助理，可以使用各種工具來協助用戶。

    你可以：
    1. 使用計算器進行數學計算
    2. 搜尋維基百科獲取知識
    3. 使用 DuckDuckGo 搜尋網路資訊
    4. 搜尋 arXiv 查找學術論文

    請根據用戶的需求，選擇最合適的工具來提供協助。
    回答時請使用繁體中文，並保持專業、友善的態度。

    如果是搜尋結果，請：
    1. 摘要重要資訊
    2. 標註資訊來源
    3. 確保資訊的準確性
    """

    # 建立提示詞模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 建立工具列表
    tools = [calculator] + setup_search_tools()

    # 建立 Agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # 建立 Agent 執行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return agent_executor


def demonstrate_tools_usage():
    """
    展示各種工具的使用方式
    """
    agent_executor = create_agent_with_tools()
    chat_history = []  # 初始化對話歷史

    # 測試案例
    test_cases = [
        "計算 123 * 456 - 789",
        "請幫我查詢人工智慧的基本概念",
        "搜尋最新的 GPT-4 相關新聞",
        "找找看最近有什麼關於大型語言模型的論文",
        "我想了解量子計算機的原理，並找找相關的研究論文"
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n=== 測試案例 {i} ===")
        print(f"問題: {question}")
        try:
            response = agent_executor.invoke({
                "input": question,
                "chat_history": chat_history
            })
            print(f"回答: {response['output']}")

            # 更新對話歷史
            chat_history.extend([
                ("human", question),
                ("assistant", response["output"])
            ])
        except Exception as e:
            logger.error(f"執行失敗: {str(e)}")


def main():
    """
    主程式：展示工具使用方式
    """
    print("=== LangChain 0.3+ 工具使用展示 ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return

    try:
        demonstrate_tools_usage()
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")


if __name__ == "__main__":
    main()
