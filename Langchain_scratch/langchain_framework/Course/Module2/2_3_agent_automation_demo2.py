"""
LangChain 0.3+ Agent 自動化範例
展示如何建立自動化工作流程，包含資料處理、API 調用和結果輸出

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- langchain-community>=0.0.1
- python-dotenv>=0.19.0
- pydantic>=2.0.0
"""

from langchain_openai import ChatOpenAI
from langchain.agents import tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
import logging
import os
import json
from datetime import datetime

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()


class TaskPlan(BaseModel):
    """任務規劃結構"""
    steps: List[str] = Field(description="執行步驟列表")
    tools_needed: List[str] = Field(description="需要使用的工具列表")
    estimated_time: str = Field(description="預估完成時間")


class TaskResult(BaseModel):
    """任務結果結構"""
    success: bool = Field(description="是否成功完成")
    result: str = Field(description="執行結果")
    error: str = Field(default="", description="錯誤訊息")


@tool
def save_to_file(content: str, filename: str) -> str:
    """
    將內容保存到檔案
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"成功保存到檔案: {filename}"
    except Exception as e:
        return f"保存失敗: {str(e)}"


@tool
def read_from_file(filename: str) -> str:
    """
    從檔案讀取內容
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"讀取失敗: {str(e)}"


@tool
def process_task(task_description: str) -> str:
    """
    處理任務（示例：模擬耗時操作）
    """
    return f"已完成任務: {task_description}"


def create_automation_agent() -> AgentExecutor:
    """
    建立自動化 Agent
    """
    # 建立 LLM
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )

    # 建立輸出解析器
    plan_parser = PydanticOutputParser(pydantic_object=TaskPlan)
    result_parser = PydanticOutputParser(pydantic_object=TaskResult)

    # 定義系統提示詞
    system_prompt = """你是一位自動化助理，負責規劃和執行任務。

    你可以：
    1. 保存內容到檔案
    2. 從檔案讀取內容
    3. 處理指定的任務

    執行任務時，請：
    1. 先規劃任務步驟，並以以下 JSON 格式輸出：
    {{
        "steps": ["步驟1", "步驟2", ...],
        "tools_needed": ["工具1", "工具2", ...],
        "estimated_time": "預估時間"
    }}

    2. 按步驟執行任務

    3. 最後以以下 JSON 格式輸出結果：
    {{
        "success": true/false,
        "result": "執行結果描述",
        "error": "如果有錯誤，請描述錯誤信息"
    }}

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
        save_to_file,
        read_from_file,
        process_task
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
    展示自動化任務處理
    """
    agent_executor = create_automation_agent()
    chat_history = []

    # 建立輸出解析器
    plan_parser = PydanticOutputParser(pydantic_object=TaskPlan)
    result_parser = PydanticOutputParser(pydantic_object=TaskResult)

    # 測試案例
    test_cases = [
        "請將'這是測試內容'保存到 test.txt，然後讀取並確認內容",
        "請執行一個任務：整理當前目錄下的檔案",
        "請規劃一個多步驟任務：1.創建檔案 2.寫入內容 3.讀取確認 4.輸出結果"
    ]

    for i, task in enumerate(test_cases, 1):
        print(f"\n=== 測試案例 {i} ===")
        print(f"任務: {task}")
        try:
            # 執行任務
            response = agent_executor.invoke({
                "input": task,
                "chat_history": chat_history
            })

            try:
                # 嘗試解析回應中的 JSON
                response_text = response["output"]

                # 檢查是否包含任務計劃
                if "steps" in response_text.lower():
                    plan_json = json.loads(response_text)
                    task_plan = TaskPlan(**plan_json)
                    print("\n任務計劃:")
                    print(f"步驟: {task_plan.steps}")
                    print(f"需要工具: {task_plan.tools_needed}")
                    print(f"預估時間: {task_plan.estimated_time}")

                # 檢查是否包含執行結果
                if "success" in response_text.lower():
                    result_json = json.loads(response_text)
                    task_result = TaskResult(**result_json)
                    print("\n執行結果:")
                    print(f"成功: {task_result.success}")
                    print(f"結果: {task_result.result}")
                    if task_result.error:
                        print(f"錯誤: {task_result.error}")

            except Exception as parse_error:
                logger.warning(f"解析結果失敗: {str(parse_error)}")
                print(f"原始回應: {response_text}")

            # 更新對話歷史
            chat_history.extend([
                ("human", task),
                ("assistant", response["output"])
            ])

        except Exception as e:
            logger.error(f"執行失敗: {str(e)}")


def main():
    """
    主程式：展示 Agent 自動化功能
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
