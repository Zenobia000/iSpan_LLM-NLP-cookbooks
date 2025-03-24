"""
LangChain 工具適配器 - 使 LangChain 工具與 CrewAI 兼容
注：此模組為未來擴展準備，目前不使用
"""
from typing import Any, Callable, Optional
from crewai.tools import BaseTool

class LangChainToolAdapter(BaseTool):
    """將 LangChain 工具轉換為 CrewAI 工具的適配器"""
    
    def __init__(self, func: Callable, name: str, description: str):
        """
        初始化適配器
        
        Args:
            func: LangChain 工具的功能函數
            name: 工具名稱
            description: 工具描述
        """
        self._func = func
        self.tool_name = name
        self.tool_description = description
        super().__init__()
    
    @property
    def name(self) -> str:
        return self.tool_name
    
    @property
    def description(self) -> str:
        return self.tool_description
    
    def _run(self, input_str: str) -> Any:
        """執行 LangChain 工具功能"""
        return self._func(input_str)

def convert_langchain_tool(lc_tool) -> BaseTool:
    """
    將 LangChain 工具轉換為 CrewAI 工具
    
    Args:
        lc_tool: LangChain 工具實例
    
    Returns:
        CrewAI 兼容的工具
    """
    return LangChainToolAdapter(
        func=lc_tool.run,
        name=lc_tool.name,
        description=lc_tool.description
    ) 