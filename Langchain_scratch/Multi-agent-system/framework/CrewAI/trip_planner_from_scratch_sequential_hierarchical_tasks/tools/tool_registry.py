"""
工具註冊系統 - 統一管理 CrewAI 使用的各種工具
"""
from typing import Dict, Any, List, Optional, Callable
from crewai.tools import BaseTool

class ToolRegistry:
    """集中管理所有可用於 CrewAI 的工具"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, tool: BaseTool, key: Optional[str] = None, 
                category: Optional[str] = None, metadata: Optional[Dict] = None) -> 'ToolRegistry':
        """
        註冊一個工具到系統
        
        Args:
            tool: 要註冊的工具
            key: 工具的唯一標識符
            category: 工具類別
            metadata: 工具的元數據
        """
        tool_key = key or tool.name
        
        # 註冊工具
        self._tools[tool_key] = tool
        
        # 註冊類別
        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool_key)
        
        # 註冊元數據
        if metadata:
            self._metadata[tool_key] = metadata
        
        return self
    
    def get(self, key: str) -> BaseTool:
        """獲取指定的工具"""
        if key not in self._tools:
            raise KeyError(f"工具 '{key}' 未註冊")
        return self._tools[key]
    
    def get_by_category(self, category: str) -> List[BaseTool]:
        """獲取特定類別的所有工具"""
        if category not in self._categories:
            return []
        return [self._tools[key] for key in self._categories[category]]
    
    def list_tools(self) -> List[str]:
        """列出所有已註冊的工具名稱"""
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """列出所有工具類別"""
        return list(self._categories.keys())
    
    def get_all(self) -> List[BaseTool]:
        """獲取所有工具的列表"""
        return list(self._tools.values())
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """獲取工具的元數據"""
        return self._metadata.get(key, {})
    
    def remove(self, key: str) -> None:
        """移除指定的工具"""
        if key in self._tools:
            # 從工具字典中移除
            del self._tools[key]
            
            # 從類別中移除
            for category in self._categories.values():
                if key in category:
                    category.remove(key)
            
            # 從元數據中移除
            if key in self._metadata:
                del self._metadata[key]
    
    def clear(self) -> None:
        """清空所有註冊的工具"""
        self._tools.clear()
        self._categories.clear()
        self._metadata.clear()
    
    @property
    def tools(self) -> Dict[str, BaseTool]:
        """獲取所有工具的字典"""
        return self._tools.copy()
    
    def __len__(self) -> int:
        """獲取已註冊工具的數量"""
        return len(self._tools)
    
    def __contains__(self, key: str) -> bool:
        """檢查工具是否已註冊"""
        return key in self._tools 