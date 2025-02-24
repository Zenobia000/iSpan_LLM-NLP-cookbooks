"""
LangChain 自定義 Chain 範例

展示如何建立自定義的 Chain，包括：
1. 基礎自定義 Chain
2. 多階段處理 Chain
3. 條件分支 Chain
4. 並行處理 Chain
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

class CustomChain(RunnablePassthrough):
    """基礎自定義 Chain"""
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        prompt: Optional[ChatPromptTemplate] = None,
        output_parser: Optional[StrOutputParser] = None,
        **kwargs
    ):
        """初始化 Chain
        
        Args:
            llm: 語言模型
            prompt: 提示詞模板
            output_parser: 輸出解析器
        """
        super().__init__()
        self.llm = llm or ChatOpenAI()
        self.prompt = prompt or ChatPromptTemplate.from_template("{input}")
        self.output_parser = output_parser or StrOutputParser()
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """執行 Chain"""
        try:
            # 1. 處理輸入
            formatted_prompt = self.prompt.format(**input)
            
            # 2. 呼叫 LLM
            response = self.llm.invoke(formatted_prompt)
            
            # 3. 解析輸出
            result = self.output_parser.invoke(response)
            
            return {"output": result}
            
        except Exception as e:
            logger.error(f"Chain 執行失敗: {str(e)}")
            raise

class MultiStageChain(RunnablePassthrough):
    """多階段處理 Chain"""
    
    def __init__(
        self,
        stages: List[CustomChain],
        **kwargs
    ):
        """初始化多階段 Chain
        
        Args:
            stages: Chain 階段列表
        """
        super().__init__()
        self.stages = stages
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """依序執行多個階段"""
        current_input = input
        results = []
        
        try:
            for i, stage in enumerate(self.stages, 1):
                logger.info(f"執行第 {i} 階段...")
                result = stage.invoke(current_input)
                results.append(result)
                current_input = {**current_input, **result}
                
            return {
                "stage_results": results,
                "final_output": results[-1]["output"]
            }
            
        except Exception as e:
            logger.error(f"多階段處理失敗: {str(e)}")
            raise

class BranchingChain(RunnablePassthrough):
    """條件分支 Chain"""
    
    def __init__(
        self,
        condition_func: callable,
        true_branch: CustomChain,
        false_branch: CustomChain,
        **kwargs
    ):
        """初始化分支 Chain
        
        Args:
            condition_func: 條件判斷函數
            true_branch: 條件為真時執行的 Chain
            false_branch: 條件為假時執行的 Chain
        """
        super().__init__()
        self.condition_func = condition_func
        self.true_branch = true_branch
        self.false_branch = false_branch
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """根據條件執行不同分支"""
        try:
            should_take_true_branch = self.condition_func(input)
            
            if should_take_true_branch:
                logger.info("執行 True 分支...")
                return self.true_branch.invoke(input)
            else:
                logger.info("執行 False 分支...")
                return self.false_branch.invoke(input)
                
        except Exception as e:
            logger.error(f"分支處理失敗: {str(e)}")
            raise

class ParallelChain(RunnablePassthrough):
    """並行處理 Chain"""
    
    def __init__(
        self,
        chains: List[CustomChain],
        max_workers: int = 3,
        **kwargs
    ):
        """初始化並行 Chain
        
        Args:
            chains: 要並行執行的 Chain 列表
            max_workers: 最大執行緒數
        """
        super().__init__()
        self.chains = chains
        self.max_workers = max_workers
        
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """並行執行多個 Chain"""
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(chain.invoke, input)
                    for chain in self.chains
                ]
                results = [future.result() for future in futures]
                
            return {
                "parallel_results": results,
                "combined_output": [r["output"] for r in results]
            }
            
        except Exception as e:
            logger.error(f"並行處理失敗: {str(e)}")
            raise

def main():
    """展示自定義 Chain 的使用"""
    
    # 建立基本元件
    llm = ChatOpenAI()
    
    # 1. 測試基礎 Chain
    basic_prompt = ChatPromptTemplate.from_template("總結以下內容：{input}")
    basic_chain = CustomChain(llm=llm, prompt=basic_prompt)
    
    # 2. 測試多階段 Chain
    summary_chain = CustomChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template("總結以下內容：{input}")
    )
    translation_chain = CustomChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template("將以下內容翻譯成英文：{input}")
    )
    multi_stage_chain = MultiStageChain(stages=[summary_chain, translation_chain])
    
    # 3. 測試分支 Chain
    def is_chinese(text: str) -> bool:
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    chinese_chain = CustomChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template("將以下中文內容翻譯成英文：{input}")
    )
    english_chain = CustomChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template("將以下英文內容翻譯成中文：{input}")
    )
    branching_chain = BranchingChain(is_chinese, chinese_chain, english_chain)
    
    # 4. 測試並行 Chain
    parallel_chain = ParallelChain([
        summary_chain,
        translation_chain,
        branching_chain
    ])
    
    # 測試輸入
    test_input = {
        "input": "人工智能正在快速發展，改變著我們的生活方式。"
    }
    
    try:
        # 執行測試
        logger.info("測試基礎 Chain...")
        basic_result = basic_chain.invoke(test_input)
        print(f"\n基礎 Chain 結果: {basic_result['output']}")
        
        logger.info("\n測試多階段 Chain...")
        multi_stage_result = multi_stage_chain.invoke(test_input)
        print(f"\n多階段 Chain 結果: {multi_stage_result['final_output']}")
        
        logger.info("\n測試分支 Chain...")
        branch_result = branching_chain.invoke(test_input)
        print(f"\n分支 Chain 結果: {branch_result['output']}")
        
        logger.info("\n測試並行 Chain...")
        parallel_result = parallel_chain.invoke(test_input)
        print("\n並行 Chain 結果:")
        for i, result in enumerate(parallel_result["combined_output"], 1):
            print(f"Chain {i}: {result}")
            
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 