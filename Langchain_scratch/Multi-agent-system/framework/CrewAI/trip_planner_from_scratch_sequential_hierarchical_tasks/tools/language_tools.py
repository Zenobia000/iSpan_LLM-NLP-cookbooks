"""
語言處理工具模組 - 提供語言轉換和格式化功能
"""
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional

class ChineseTranslationToolSchema(BaseModel):
    """中文翻譯工具輸入模式"""
    input_text: str = Field(description="要翻譯的文本內容")

class ChineseTranslationTool(BaseTool):
    """中文翻譯與轉換工具"""
    name: str = "Chinese Translation"
    description: str = "將輸入文本轉換為繁體中文。使用格式：{'input_text': '要翻譯的文本'}"
    schema: Optional[BaseModel] = ChineseTranslationToolSchema
    
    def _run(self, input_text: str) -> str:
        """將文本轉換為繁體中文"""
        return f"""請將以下文本完全轉換為繁體中文，同時嚴格遵循以下規則：
        1. 保持所有原始格式，包括所有 Markdown 標記（#、-、*、數字列表等）
        2. 保留所有段落分隔和換行
        3. 保持標題層級結構不變（####、###等）
        4. 列表項目格式必須保持一致
        5. 確保所有文本僅翻譯內容，不修改格式標記

原始文本：
{input_text}

轉換結果（保持完整 Markdown 格式）：
"""

def create_language_tools():
    """創建並返回所有語言處理相關工具"""
    return {
        "chinese_translation": ChineseTranslationTool(),
        # 未來可以在這裡添加更多語言工具
    }

# 然後在 tools/__init__.py 註冊這個工具並配置給相關代理使用 