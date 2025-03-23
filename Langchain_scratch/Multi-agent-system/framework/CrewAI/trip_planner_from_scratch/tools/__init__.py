"""
工具系統初始化模組
"""
from .tool_registry import ToolRegistry
from .search_tools import create_search_tools
from .calculator_tools import create_calculator_tools

def initialize_tools() -> ToolRegistry:
    """初始化並註冊所有可用工具"""
    registry = ToolRegistry()
    
    # 註冊搜索工具
    for key, tool in create_search_tools().items():
        registry.register(tool, key)
    
    # 註冊計算工具
    for key, tool in create_calculator_tools().items():
        registry.register(tool, key)
    
    # 未來可在此處註冊更多工具...
    
    return registry