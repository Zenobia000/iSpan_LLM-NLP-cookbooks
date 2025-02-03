"""
LangChain 0.3+ Agent 自動化範例
展示如何建立自動化工作流程，包含資料處理、API 調用和結果輸出

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- langchain-community>=0.0.1
- python-dotenv>=0.19.0
- pandas>=2.0.0
"""

from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import logging
import os
import json
import csv
from datetime import datetime

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()


@tool
def read_csv_data(file_path: str) -> str:
    """
    讀取 CSV 檔案並返回摘要資訊

    Args:
        file_path: CSV 檔案路徑
    """
    try:
        df = pd.read_csv(file_path)
        summary = {
            "欄位": list(df.columns),
            "資料筆數": len(df),
            "數值欄位統計": df.describe().to_dict()
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"讀取失敗: {str(e)}"


@tool
def analyze_sales_data(file_path: str, column: str) -> str:
    """
    分析銷售數據的特定欄位

    Args:
        file_path: CSV 檔案路徑
        column: 要分析的欄位名稱
    """
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return f"找不到欄位: {column}"

        analysis = {
            "平均值": float(df[column].mean()),
            "最大值": float(df[column].max()),
            "最小值": float(df[column].min()),
            "總和": float(df[column].sum())
        }
        return json.dumps(analysis, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"分析失敗: {str(e)}"


@tool
def generate_report(data: str, report_type: str) -> str:
    """
    生成分析報告

    Args:
        data: 要包含在報告中的數據（JSON 格式）
        report_type: 報告類型（"銷售", "庫存", "客戶"）
    """
    try:
        # 解析輸入數據
        content = json.loads(data)

        # 根據報告類型生成對應的報告
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = f"reports/report_{report_type}_{timestamp}.json"

        # 確保報告目錄存在
        os.makedirs("reports", exist_ok=True)

        # 寫入報告
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

        return f"報告已生成: {report_path}"
    except Exception as e:
        return f"報告生成失敗: {str(e)}"


def create_automation_agent() -> AgentExecutor:
    """
    建立自動化處理的 Agent
    """
    # 建立 LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    # 定義系統提示詞
    system_prompt = """你是一位數據分析助理，可以協助進行自動化的數據處理與分析。

    你可以：
    1. 讀取 CSV 檔案數據
    2. 分析銷售數據
    3. 生成分析報告

    請根據用戶的需求，選擇適當的工具來完成任務。
    回答時請使用繁體中文，並保持專業、友善的態度。
    """

    # 建立提示詞模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # 建立工具列表
    tools = [
        read_csv_data,
        analyze_sales_data,
        generate_report
    ]

    # 建立 Agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # 建立 Agent 執行器
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )


def demonstrate_automation():
    """
    展示自動化處理流程
    """
    agent_executor = create_automation_agent()
    chat_history = []

    # 準備測試數據
    test_file = "test_data/sales_data.csv"
    os.makedirs("test_data", exist_ok=True)

    # 如果測試檔案不存在，創建一個範例檔案
    if not os.path.exists(test_file):
        sample_data = {
            "日期": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "產品": ["A", "B", "C"],
            "銷售量": [100, 150, 200],
            "單價": [10, 20, 15],
            "總金額": [1000, 3000, 3000]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(test_file, index=False, encoding="utf-8")

    # 測試案例
    test_cases = [
        f"請幫我讀取 {test_file} 的內容並告訴我基本資訊",
        f"分析 {test_file} 中的銷售量數據",
        "請根據分析結果生成一份銷售報告"
    ]

    for i, question in enumerate(test_cases, 1):
        print(f"\n=== 測試案例 {i} ===")
        print(f"任務: {question}")
        try:
            response = agent_executor.invoke({
                "input": question,
                "chat_history": chat_history
            })
            print(f"結果: {response['output']}")

            # 更新對話歷史
            chat_history.extend([
                ("human", question),
                ("assistant", response["output"])
            ])
        except Exception as e:
            logger.error(f"執行失敗: {str(e)}")


def main():
    """
    主程式：展示 Agent 自動化流程
    """
    print("=== LangChain 0.3+ Agent 自動化展示 ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return

    try:
        demonstrate_automation()
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")


if __name__ == "__main__":
    main()
