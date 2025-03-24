"""
工具系統初始化模組
"""
from .tool_registry import ToolRegistry
from .search_tools import create_search_tools
from .calculator_tools import create_calculator_tools
from .language_tools import create_language_tools

def initialize_tools() -> ToolRegistry:
    """初始化並註冊所有可用工具"""
    registry = ToolRegistry()
    
    # 註冊搜索工具
    for key, tool in create_search_tools().items():
        registry.register(
            tool=tool,
            key=key,
            category="search",
            metadata={
                "type": "search",
                "description": "用於網絡搜索的工具",
                "requires_api_key": True
            }
        )
    
    # 註冊計算工具
    for key, tool in create_calculator_tools().items():
        registry.register(
            tool=tool,
            key=key,
            category="calculator",
            metadata={
                "type": "calculator",
                "description": "用於數學計算的工具",
                "requires_api_key": False
            }
        )
    
    # 註冊中文翻譯工具 - 直接註冊單個工具實例
    for key, tool in create_language_tools().items():
        registry.register(
            tool=tool,
            key=key,
            category="language",
            metadata={
                "type": "language",
                "description": "用於將文本轉換為繁體中文",
                "requires_api_key": False
            }
        )
    
    # 未來可在此處註冊更多工具...
    
    return registry


