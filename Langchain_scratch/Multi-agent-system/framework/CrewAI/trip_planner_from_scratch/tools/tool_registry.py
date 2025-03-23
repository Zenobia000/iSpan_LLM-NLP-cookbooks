"""
工具註冊系統 - 統一管理 CrewAI 使用的各種工具
"""
from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool

class ToolRegistry:
    """集中管理所有可用於 CrewAI 的工具"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool, key: Optional[str] = None) -> None:
        """註冊一個工具到系統"""
        tool_key = key or tool.name
        self._tools[tool_key] = tool
        return self
    
    def get(self, key: str) -> BaseTool:
        """獲取指定的工具"""
        if key not in self._tools:
            raise KeyError(f"工具 '{key}' 未註冊")
        return self._tools[key]
    
    def list_tools(self) -> List[str]:
        """列出所有已註冊的工具名稱"""
        return list(self._tools.keys())
    
    def get_all(self) -> List[BaseTool]:
        """獲取所有工具的列表"""
        return list(self._tools.values())
    
    @property
    def tools(self) -> Dict[str, BaseTool]:
        """獲取所有工具的字典"""
        return self._tools.copy() 