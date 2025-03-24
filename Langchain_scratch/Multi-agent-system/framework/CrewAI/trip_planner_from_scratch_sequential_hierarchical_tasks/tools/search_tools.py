"""
網絡搜索工具模組 - 提供網路搜索功能，並正確處理中文字符
"""
import os
import json
import requests
from crewai.tools import BaseTool
from typing import Optional, Any

class SearchInternetTool(BaseTool):
    """搜索網絡的工具，支持中文查詢"""
    name: str = "Search Internet"
    description: str = "搜尋網路獲取最新資訊。輸入查詢字串，獲取相關網頁內容。支持中文查詢。"
    
    def _run(self, query: str) -> str:
        """執行網絡搜索，正確處理中文字符"""
        # 如果是 JSON 格式，嘗試解析並提取查詢
        if query.startswith('{') and query.endswith('}'):
            try:
                query_obj = json.loads(query)
                if 'query' in query_obj:
                    query = query_obj['query']
            except:
                pass
        
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        
        # 只需添加 ensure_ascii=False 來確保中文正確處理
        payload = json.dumps({"q": query}, ensure_ascii=False)
        
        headers = {
            'X-API-KEY': os.environ.get('SERPER_API_KEY'),
            'Content-Type': 'application/json; charset=utf-8'
        }
        
        try:
            response = requests.request("POST", url, headers=headers, data=payload.encode('utf-8'))
            response.raise_for_status()
            
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
