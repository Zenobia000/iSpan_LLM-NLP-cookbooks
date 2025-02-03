"""
LangChain 0.3+ 框架概覽與核心概念
此範例展示 LangChain 的基本架構和核心組件

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- python-dotenv>=0.19.0
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# 載入環境變數
load_dotenv()


def demonstrate_core_components():
    """
    展示 LangChain 0.3+ 的核心組件：
    1. Model: 語言模型 (使用 ChatOpenAI)
    2. PromptTemplate: 提示詞模板
    3. RunnablePassthrough: 數據流處理
    4. OutputParser: 輸出解析
    """

    # 1. 初始化 LLM
    model = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo"
    )

    # 2. 建立提示詞模板
    prompt = PromptTemplate.from_template("""
    你是一位專業的 AI 助理。
    請用簡潔的方式回答以下問題：
    問題: {question}
    """)

    # 3. 建立輸出解析器
    output_parser = StrOutputParser()

    # 4. 建立 Chain (使用 LCEL - LangChain Expression Language)
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )

    return chain


def main():
    """
    主程式：展示如何使用 LangChain 0.3+ 進行基本對話
    """
    print("=== LangChain 0.3+ 框架展示 ===")

    # 檢查環境變數
    if not os.getenv("OPENAI_API_KEY"):
        print("請先設定 OPENAI_API_KEY 環境變數！")
        return

    try:
        # 建立對話 chain
        chain = demonstrate_core_components()

        # 執行對話
        questions = [
            "什麼是 LangChain 0.3+？",
            "LangChain 0.3+ 有哪些主要更新？"
        ]

        for question in questions:
            print(f"\n問題: {question}")
            response = chain.invoke(question)
            print(f"回答: {response}")

    except Exception as e:
        print(f"執行過程發生錯誤: {str(e)}")


if __name__ == "__main__":
    main()
