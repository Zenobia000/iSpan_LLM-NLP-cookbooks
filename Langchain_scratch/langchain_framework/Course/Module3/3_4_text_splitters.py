"""
LangChain 文本分割器示例

本模組展示了 LangChain 中不同的文本分割策略，包括：
1. 基於字符的分割 (CharacterTextSplitter)
2. 基於 Token 的分割 (TokenTextSplitter)
3. 遞迴字符分割 (RecursiveCharacterTextSplitter)
4. Markdown 標題分割 (MarkdownHeaderTextSplitter)
5. 中文文本分割 (ChineseTextSplitter)

同時展示了分割參數調優和效能優化策略。
"""

import jieba
import logging as jieba_logging
jieba_logging.getLogger('jieba').setLevel(jieba_logging.ERROR)  # 關閉 jieba 的日誌輸出

import os
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
import json
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import re

from langchain.text_splitter import (
    CharacterTextSplitter,
    TokenTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_core.documents import Document

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SplitMetrics:
    """分割結果評估指標"""
    splitter_name: str
    chunk_count: int
    avg_chunk_size: float
    max_chunk_size: int
    min_chunk_size: int
    processing_time: float
    overlap_ratio: float
    metadata_preserved: bool

class FileProcessor:
    """文件處理類"""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.sample_dir = base_dir / "samples"
        self.output_dir = base_dir / "samples_split_docs"
        self.output_dir.mkdir(exist_ok=True)
    
    def read_file(self, file_path: Path) -> str:
        """讀取文件內容"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def get_file_type(self, file_path: Path) -> str:
        """根據文件擴展名判斷類型"""
        return file_path.suffix.lower()

class SplitterEvaluator:
    """分割器評估類"""
    def __init__(self, file_processor: FileProcessor):
        self.file_processor = file_processor
    
    def _calculate_metrics(
        self,
        chunks: List[str],
        processing_time: float,
        splitter: Any
    ) -> Dict:
        """計算評估指標"""
        chunk_sizes = [len(chunk) for chunk in chunks]
        
        # 直接從分割器獲取重疊比例
        overlap_ratio = (
            splitter.chunk_overlap / splitter.chunk_size 
            if hasattr(splitter, "chunk_size") and hasattr(splitter, "chunk_overlap")
            else 0.0
        )
        
        return {
            "chunk_count": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks) if chunks else 0,
            "max_chunk_size": max(chunk_sizes) if chunks else 0,
            "min_chunk_size": min(chunk_sizes) if chunks else 0,
            "processing_time": processing_time,
            "overlap_ratio": overlap_ratio
        }
    
    def evaluate_splitter(
        self,
        splitter: Any,  # 改為 Any 類型
        text: str,
        doc_type: str
    ) -> Dict:
        """評估分割器效果"""
        start_time = time.perf_counter()
        
        # 根據分割器類型處理
        if isinstance(splitter, MarkdownHeaderTextSplitter):
            chunks = [doc.page_content for doc in splitter.split_text(text)]
        else:
            chunks = splitter.split_text(text)
            
        processing_time = time.perf_counter() - start_time
        
        return {
            "document_type": doc_type,
            "splitter": splitter.__class__.__name__,
            "metrics": self._calculate_metrics(chunks, processing_time, splitter),
            "chunks": self._format_chunks(chunks)
        }
    
    def _format_chunks(self, chunks: List[str]) -> List[Dict]:
        """格式化分塊結果"""
        return [
            {
                "index": i,
                "content": chunk,
                "length": len(chunk)
            } for i, chunk in enumerate(chunks)
        ]

class ResultSaver:
    """結果保存類"""
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
    
    def save_result(
        self,
        result: Dict,
        file_name: str,
        splitter_name: str
    ) -> None:
        """保存評估結果"""
        output_file = self.output_dir / f"{file_name}_{splitter_name.lower()}_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存分割結果: {output_file}")

class HTMLTextSplitter:  # 移除 BaseTextSplitter 繼承
    """HTML 文本分割器"""
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", ".", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_text(self, html_content: str) -> List[str]:
        """分割 HTML 文本，保留結構信息"""
        # 解析 HTML
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 提取結構化文本
        structured_text = []
        
        # 處理標題和內容
        for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div"]):
            # 跳過空標籤
            if not tag.get_text().strip():
                continue
                
            # 根據標籤類型添加前綴
            if tag.name.startswith('h'):
                level = tag.name[1]  # 獲取標題級別
                prefix = '#' * int(level) + ' '
            else:
                prefix = ''
            
            # 添加結構化文本
            text = tag.get_text().strip()
            if text:
                structured_text.append(f"{prefix}{text}")
        
        # 合併為單一文本
        full_text = '\n\n'.join(structured_text)
        
        # 使用遞歸分割器進行分割
        chunks = self.recursive_splitter.split_text(full_text)
        
        # 後處理：確保每個分塊的語義完整性
        processed_chunks = []
        for chunk in chunks:
            # 移除孤立的標題
            if chunk.strip().startswith('#') and len(chunk.strip().split('\n')) == 1:
                continue
            # 確保分塊不以句子中間結束
            if not chunk.strip().endswith(('.', '!', '?', '。', '！', '？')):
                # 找到最後一個完整句子的位置
                last_sentence = max(
                    chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'),
                    chunk.rfind('。'), chunk.rfind('！'), chunk.rfind('？')
                )
                if last_sentence > 0:
                    chunk = chunk[:last_sentence+1]
            processed_chunks.append(chunk)
        
        return processed_chunks

class ChineseTextSplitter:  # 移除 BaseTextSplitter 繼承
    """中文文本分割器"""
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """使用 jieba 分詞進行中文文本分割"""
        # 使用 jieba 分詞
        sentences = []
        current_chunk = []
        current_length = 0
        
        # 按標點符號分句
        for sentence in re.split(r'([。！？；])', text):
            if sentence:
                current_chunk.append(sentence)
                current_length += len(sentence)
                
                # 如果達到分塊大小，保存當前分塊
                if current_length >= self.chunk_size:
                    sentences.append(''.join(current_chunk))
                    # 保留重疊部分
                    overlap_size = len(''.join(current_chunk[-2:])) if len(current_chunk) > 1 else 0
                    if overlap_size > self.chunk_overlap:
                        current_chunk = current_chunk[-1:]
                        current_length = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_length = 0
        
        # 處理剩餘文本
        if current_chunk:
            sentences.append(''.join(current_chunk))
        
        return sentences

def main():
    """主程序"""
    try:
        # 初始化組件
        file_processor = FileProcessor(Path(__file__).parent)
        evaluator = SplitterEvaluator(file_processor)
        result_saver = ResultSaver(file_processor.output_dir)
        
        # 定義分割器
        splitters = {
            "純文本": ("ai_introduction.txt", CharacterTextSplitter(
                separator="\n\n",
                chunk_size=200,
                chunk_overlap=10
            )),
            "結構化數據": ("model_comparison.csv", TokenTextSplitter(
                chunk_size=200,
                chunk_overlap=10
            )),
            "技術文檔_Recursive": ("dev_guide.html", RecursiveCharacterTextSplitter(
                separators=["\n\n", ".", " ", "，", "；", "！", "？"],
                chunk_size=200,
                chunk_overlap=10
            )),
            "技術文檔_HTML": ("dev_guide.html", HTMLTextSplitter(
                chunk_size=200,
                chunk_overlap=10
            )),
            "中文新聞": ("tech_news.txt", ChineseTextSplitter(
                chunk_size=200,
                chunk_overlap=10
            )),
            "Markdown教程": ("langchain_tutorial.md", MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "h1"),
                    ("##", "h2"),
                    ("###", "h3")
                ]
            ))
        }
        
        # 評估每種分割器
        for doc_type, (file_name, splitter) in splitters.items():
            file_path = file_processor.sample_dir / file_name
            if not file_path.exists():
                logger.warning(f"找不到文件: {file_name}")
                continue
            
            # 讀取和評估
            text = file_processor.read_file(file_path)
            result = evaluator.evaluate_splitter(splitter, text, doc_type)
            
            # 保存結果
            result_saver.save_result(
                result,
                Path(file_name).stem,
                splitter.__class__.__name__
            )
        
        logger.info("\n=== 評估完成 ===")
        
    except Exception as e:
        logger.error(f"評估過程發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 