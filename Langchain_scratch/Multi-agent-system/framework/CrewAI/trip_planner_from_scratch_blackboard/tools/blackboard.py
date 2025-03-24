"""
黑板系統 - 用於管理代理之間的中間狀態
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json

class BlackboardManager:
    """黑板系統，用於管理代理之間的狀態共享"""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._history: Dict[str, list] = {}
        self._metadata: Dict[str, Dict] = {}
    
    def write(self, key: str, value: Any, agent_id: str) -> None:
        """
        寫入數據到黑板
        
        Args:
            key: 數據的鍵名
            value: 要存儲的數據
            agent_id: 寫入數據的代理ID
        """
        # 保存歷史版本
        if key not in self._history:
            self._history[key] = []
        
        # 如果已存在數據，將其添加到歷史記錄
        if key in self._data:
            self._history[key].append({
                'value': self._data[key],
                'metadata': self._metadata.get(key, {}),
                'timestamp': datetime.now().isoformat()
            })
        
        # 更新數據和元數據
        self._data[key] = value
        self._metadata[key] = {
            'last_modified_by': agent_id,
            'last_modified_at': datetime.now().isoformat(),
            'version': len(self._history[key]) + 1
        }
    
    def read(self, key: str) -> Optional[Any]:
        """從黑板讀取數據"""
        return self._data.get(key)
    
    def get_history(self, key: str) -> list:
        """獲取特定鍵的歷史記錄"""
        return self._history.get(key, [])
    
    def get_metadata(self, key: str) -> Dict:
        """獲取特定鍵的元數據"""
        return self._metadata.get(key, {})
    
    def list_keys(self) -> list:
        """列出所有可用的鍵"""
        return list(self._data.keys())
    
    def export_state(self) -> Dict:
        """導出當前狀態"""
        return {
            'data': self._data,
            'metadata': self._metadata,
            'history': self._history
        }
    
    def import_state(self, state: Dict) -> None:
        """導入狀態"""
        self._data = state.get('data', {})
        self._metadata = state.get('metadata', {})
        self._history = state.get('history', {})
    
    def clear(self) -> None:
        """清空黑板"""
        self._data.clear()
        self._metadata.clear()
        self._history.clear() 