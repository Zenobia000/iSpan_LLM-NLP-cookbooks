"""
輸出攔截器模組 - 攔截控制台輸出並轉換各種編碼為可讀中文
"""
import sys
import re
import json

class ChineseOutputInterceptor:
    """攔截並解碼控制台輸出中的各類編碼"""
    
    def __init__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
    
    def write(self, text):
        # 第一步：處理 Unicode 編碼 \uXXXX
        decoded_text = re.sub(
            r'\\u([0-9a-fA-F]{4})',
            lambda match: chr(int(match.group(1), 16)),
            text
        )
        
        # 第二步：處理直接轉義的中文字符，如 \台 \灣
        decoded_text = re.sub(
            r'\\([\u4e00-\u9fff])',  # 匹配 \ 後面跟著中文字符
            r'\1',  # 只保留中文字符
            decoded_text
        )
        
        # 第三步：嘗試檢測並處理 JSON 字符串
        if '{"query":' in decoded_text:
            try:
                # 找到 JSON 結構的起止位置
                start_idx = decoded_text.find('{"query":')
                end_idx = decoded_text.find('}"', start_idx) + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = decoded_text[start_idx:end_idx]
                    # 嘗試解析並重新格式化 JSON
                    try:
                        parsed_json = json.loads(json_str)
                        formatted_json = json.dumps(parsed_json, ensure_ascii=False)
                        # 替換原始文本中的 JSON 字符串
                        decoded_text = decoded_text[:start_idx] + formatted_json + decoded_text[end_idx:]
                    except json.JSONDecodeError:
                        # JSON 解析失敗，繼續使用處理後的文本
                        pass
            except Exception:
                # 發生任何錯誤，使用已處理的文本即可
                pass
        
        self.original_stdout.write(decoded_text)
    
    def flush(self):
        self.original_stdout.flush()
    
    @staticmethod
    def setup():
        """設置輸出攔截器"""
        return ChineseOutputInterceptor() 