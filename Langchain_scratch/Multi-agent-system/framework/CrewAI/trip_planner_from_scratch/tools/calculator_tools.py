"""
計算工具模組 - 提供各類數學運算功能
"""
from crewai.tools import BaseTool
from typing import Union, Any

class BasicCalculateTool(BaseTool):
    """基本計算工具"""
    name: str = "Basic Calculate"
    description: str = "執行基本數學運算。輸入數學表達式，例如：2+2、(3*4)/2 等。"
    
    def _run(self, operation: str) -> Union[str, float]:
        """執行基本數學運算"""
        try:
            return eval(operation)
        except SyntaxError:
            return "錯誤：數學表達式語法無效"
        except Exception as e:
            return f"計算錯誤：{str(e)}"

class AdvancedCalculateTool(BaseTool):
    """進階計算工具，支持因數倍增"""
    name: str = "Advanced Calculate"
    description: str = "執行複雜的數學計算並應用因數。格式：'數學表達式 | 因數'，例如：'5+5 | 2'"
    
    def _run(self, input_str: str) -> str:
        """執行高級計算，支持因數倍增"""
        try:
            parts = input_str.split("|")
            operation = parts[0].strip()
            factor = float(parts[1].strip()) if len(parts) > 1 else 1.0
            
            # 安全地評估數學表達式
            result = self._safe_eval(operation) * factor
            
            # 返回結果
            return f"'{operation}' 乘以 {factor} 的結果是 {result}。"
        except Exception as e:
            return f"高級計算錯誤：{str(e)}"
    
    def _safe_eval(self, expression: str) -> float:
        """安全地評估數學表達式，只允許基本算術運算和數字"""
        # 檢查表達式是否只包含允許的字符
        allowed = set("0123456789.+-*/() ")
        if not all(c in allowed for c in expression):
            raise ValueError("表達式包含不允許的字符")
        
        # 評估表達式
        return eval(expression)

def create_calculator_tools():
    """創建並返回所有計算相關工具"""
    return {
        "calculate": BasicCalculateTool(),
        "advanced_calculate": AdvancedCalculateTool()
    }



    