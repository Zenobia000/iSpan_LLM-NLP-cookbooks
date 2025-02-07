"""
LangChain 0.3+ Embedding 模型比較
比較不同 Embedding 模型的性能與特性

需求套件:
- langchain>=0.3.0
- langchain-community>=0.0.1
- sentence-transformers>=2.2.2
- openai>=1.1.0
- cohere>=4.37
- pandas>=2.0.0
- numpy>=1.24.0
- python-dotenv>=0.19.0
"""

import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from dotenv import load_dotenv
import torch

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    CohereEmbeddings,
    HuggingFaceBgeEmbeddings
)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

@dataclass
class EmbeddingMetrics:
    """Embedding 模型評估指標"""
    name: str
    embedding_time: float
    vector_dimension: int
    memory_usage: float
    gpu_usage: float
    cost_per_1k: float
    local_deployment: bool
    batch_support: bool
    multilingual: bool

class EmbeddingEvaluator:
    """Embedding 模型評估器"""
    def __init__(self):
        self.metrics: List[EmbeddingMetrics] = []
        
    def prepare_test_data(self, n_samples: int = 100) -> List[str]:
        """準備測試數據"""
        texts = [
            f"This is test document {i} containing specific information about topic {i%10}"
            for i in range(n_samples)
        ]
        texts.extend([
            "這是中文測試文件，用於評估多語言支援",
            "これは日本語のテストドキュメントです",
            "이것은 한국어 테스트 문서입니다"
        ])
        return texts

    def evaluate_openai(self, texts: List[str]) -> EmbeddingMetrics:
        """評估 OpenAI Embeddings"""
        try:
            embeddings = OpenAIEmbeddings()
            
            # 測量嵌入時間
            start_time = time.time()
            vectors = embeddings.embed_documents(texts)
            embedding_time = time.time() - start_time
            
            # 計算記憶體使用
            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            
            return EmbeddingMetrics(
                name="OpenAI Ada 002",
                embedding_time=embedding_time,
                vector_dimension=1536,
                memory_usage=memory_usage,
                gpu_usage=0,  # 雲端 API
                cost_per_1k=0.0001,  # USD per 1K tokens
                local_deployment=False,
                batch_support=True,
                multilingual=True
            )
        except Exception as e:
            logger.error(f"OpenAI Embeddings 評估失敗: {str(e)}")
            raise

    def evaluate_huggingface(self, texts: List[str]) -> EmbeddingMetrics:
        """評估 HuggingFace Embeddings (all-MiniLM-L6-v2)"""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # 測量嵌入時間
            start_time = time.time()
            vectors = embeddings.embed_documents(texts)
            embedding_time = time.time() - start_time
            
            # 計算 GPU 使用
            gpu_usage = 0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024
            
            # 計算記憶體使用
            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            
            return EmbeddingMetrics(
                name="HF MiniLM-L6",
                embedding_time=embedding_time,
                vector_dimension=384,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                cost_per_1k=0,  # 免費
                local_deployment=True,
                batch_support=True,
                multilingual=False
            )
        except Exception as e:
            logger.error(f"HuggingFace Embeddings 評估失敗: {str(e)}")
            raise

    def evaluate_bge(self, texts: List[str]) -> EmbeddingMetrics:
        """評估 BGE Embeddings"""
        try:
            embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-large-en-v1.5"
            )
            
            # 測量嵌入時間
            start_time = time.time()
            vectors = embeddings.embed_documents(texts)
            embedding_time = time.time() - start_time
            
            # 計算 GPU 使用
            gpu_usage = 0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024
            
            # 計算記憶體使用
            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            
            return EmbeddingMetrics(
                name="BGE Large",
                embedding_time=embedding_time,
                vector_dimension=1024,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                cost_per_1k=0,  # 免費
                local_deployment=True,
                batch_support=True,
                multilingual=True
            )
        except Exception as e:
            logger.error(f"BGE Embeddings 評估失敗: {str(e)}")
            raise

    def evaluate_cohere(self, texts: List[str]) -> EmbeddingMetrics:
        """評估 Cohere Embeddings"""
        try:
            embeddings = CohereEmbeddings(
                cohere_api_key=os.getenv("COHERE_API_KEY")
            )
            
            # 測量嵌入時間
            start_time = time.time()
            vectors = embeddings.embed_documents(texts)
            embedding_time = time.time() - start_time
            
            # 計算記憶體使用
            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            
            return EmbeddingMetrics(
                name="Cohere Embed",
                embedding_time=embedding_time,
                vector_dimension=4096,
                memory_usage=memory_usage,
                gpu_usage=0,  # 雲端 API
                cost_per_1k=0.0001,  # USD per 1K tokens
                local_deployment=False,
                batch_support=True,
                multilingual=True
            )
        except Exception as e:
            logger.error(f"Cohere Embeddings 評估失敗: {str(e)}")
            raise

    def run_evaluation(self, n_samples: int = 100):
        """執行評估"""
        texts = self.prepare_test_data(n_samples)
        
        # 評估各個 Embedding 模型
        evaluations = [
            self.evaluate_openai(texts),
            self.evaluate_huggingface(texts),
            self.evaluate_bge(texts),
            self.evaluate_cohere(texts)
        ]
        
        # 轉換為 DataFrame
        df = pd.DataFrame([
            {
                "Embedding 模型": e.name,
                "嵌入時間 (秒)": round(e.embedding_time, 3),
                "向量維度": e.vector_dimension,
                "記憶體使用 (MB)": round(e.memory_usage, 2),
                "GPU 使用 (MB)": round(e.gpu_usage, 2),
                "每千字成本 (USD)": e.cost_per_1k,
                "本地部署": "是" if e.local_deployment else "否",
                "批次處理": "是" if e.batch_support else "否",
                "多語言支援": "是" if e.multilingual else "否"
            }
            for e in evaluations
        ])
        
        # 設定顯示格式
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print("\n=== Embedding 模型比較結果 ===")
        print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"測試數據量: {n_samples} 筆")
        print("\n" + str(df))
        
        # 輸出建議
        print("\n=== 使用建議 ===")
        print("1. 預算充足時建議使用 OpenAI：性能穩定，多語言支援好")
        print("2. 本地部署建議使用 BGE：效能與準確度均衡")
        print("3. 輕量化場景建議使用 MiniLM：速度快，資源消耗低")
        print("4. 企業應用建議使用 Cohere：向量維度大，適合複雜場景")
        
        return df

def main():
    """主程式"""
    print("=== LangChain 0.3+ Embedding 模型比較 ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return
    
    try:
        evaluator = EmbeddingEvaluator()
        evaluator.run_evaluation(n_samples=100)
    except Exception as e:
        logger.error(f"評估過程發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 