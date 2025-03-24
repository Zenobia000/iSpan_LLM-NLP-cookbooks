"""
網絡搜索工具模組 - 提供網路搜索功能
"""
import os
import json
import requests
from crewai.tools import BaseTool
from typing import Optional, Any

class SearchInternetTool(BaseTool):
    """搜索網絡的工具"""
    name: str = "Search Internet"
    description: str = "搜尋網路獲取最新資訊。輸入查詢字串，獲取相關網頁內容。"
    
    def _run(self, query: str) -> str:
        """執行網絡搜索"""
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.environ.get('SERPER_API_KEY'),
            'content-type': 'application/json'
        }
        
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()  # 檢查HTTP錯誤
            
            # 檢查是否存在有機結果
            if 'organic' not in response.json():
                return "抱歉，找不到相關資訊，可能是搜索 API 配置有誤。"
            
            results = response.json()['organic']
            string = []
            for result in results[:top_result_to_return]:
                try:
                    string.append('\n'.join([
                        f"標題: {result['title']}", 
                        f"連結: {result['link']}",
                        f"摘要: {result['snippet']}", 
                        "\n----------------"
                    ]))
                except KeyError:
                    pass
            
            return '\n'.join(string)
        except Exception as e:
            return f"搜索過程中發生錯誤: {str(e)}"

def create_search_tools():
    """創建並返回所有搜索相關工具"""
    return {
        "search_internet": SearchInternetTool()
    }
