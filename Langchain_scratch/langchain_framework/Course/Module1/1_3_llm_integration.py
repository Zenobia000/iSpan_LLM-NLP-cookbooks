"""
LangChain 0.3+ LLM 整合示例
展示如何整合不同的 LLM 提供者（OpenAI、Anthropic、Local LLM）

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- langchain-anthropic>=0.0.1
- python-dotenv>=0.19.0
"""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()


def create_llm(provider: str = "openai", **kwargs) -> Any:
    """
    根據提供者建立 LLM 實例

    Args:
        provider: LLM 提供者 ("openai", "anthropic")
        **kwargs: 其他參數設定
    """
    default_params = {
        "temperature": 0.7,
        "streaming": True
    }

    # 合併預設參數和自定義參數
    params = {**default_params, **kwargs}

    try:
        if provider == "openai":
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                **params
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model_name="claude-3-5-sonnet-20240620",
                **params
            )
        else:
            raise ValueError(f"不支援的 LLM 提供者: {provider}")

    except Exception as e:
        logger.error(f"建立 LLM 實例失敗: {str(e)}")
        raise


def create_prompt_template() -> ChatPromptTemplate:
    """
    建立通用的提示詞模板
    """
    template = """你是一位專業的 AI 助理。

    請根據以下指示回答問題：
    指示: {instruction}
    問題: {question}

    請提供清晰且有條理的回答。
    """

    return ChatPromptTemplate.from_template(template)


def demonstrate_llm_usage(
    provider: str,
    instruction: str,
    question: str,
    **llm_params
) -> str:
    """
    展示 LLM 的基本使用方式

    Args:
        provider: LLM 提供者
        instruction: 特定指示
        question: 問題內容
        **llm_params: LLM 參數設定
    """
    try:
        # 建立 LLM
        llm = create_llm(provider, **llm_params)

        # 建立提示詞模板
        prompt = create_prompt_template()

        # 建立輸出解析器
        output_parser = StrOutputParser()

        # 建立完整的處理鏈
        chain = prompt | llm | output_parser

        # 執行查詢
        response = chain.invoke({
            "instruction": instruction,
            "question": question
        })

        return response

    except Exception as e:
        logger.error(f"執行 LLM 查詢失敗: {str(e)}")
        raise


def main():
    """
    主程式：展示不同 LLM 提供者的使用方式
    """
    print("=== LangChain 0.3+ LLM 整合展示 ===\n")

    # 檢查環境變數
    required_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY"
    }

    test_cases = [
        {
            "provider": "openai",
            "instruction": "請用簡潔的方式回答",
            "question": "什麼是機器學習？",
            "params": {"temperature": 0.5}
        },
        {
            "provider": "anthropic",
            "instruction": "請用深入淺出的方式解釋",
            "question": "什麼是深度學習？",
            "params": {"temperature": 0.7}
        }
    ]

    for case in test_cases:
        provider = case["provider"]
        key_name = required_keys[provider]

        if not os.getenv(key_name):
            logger.warning(f"缺少 {provider} 的 API 金鑰，跳過測試")
            continue

        try:
            print(f"\n=== 使用 {provider} ===")
            print(f"問題: {case['question']}")

            response = demonstrate_llm_usage(
                provider=provider,
                instruction=case["instruction"],
                question=case["question"],
                **case["params"]
            )

            print(f"回答: {response}\n")

        except Exception as e:
            logger.error(f"{provider} 測試失敗: {str(e)}")


if __name__ == "__main__":
    main()
