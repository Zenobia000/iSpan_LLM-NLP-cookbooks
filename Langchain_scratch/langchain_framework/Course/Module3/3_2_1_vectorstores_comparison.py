"""
LangChain 0.3+ 向量資料庫比較
比較不同向量資料庫的性能與特性

需求套件:
- langchain>=0.3.0
- langchain-community>=0.0.1
- chromadb>=0.4.0
- faiss-cpu>=1.7.4
- pymilvus>=2.3.3
- pinecone-client>=3.0.0
- pandas>=2.0.0
- numpy>=1.24.0
- python-dotenv>=0.19.0
"""

"""
LangChain 0.3+ 向量資料庫比較
比較不同向量資料庫的性能與特性
"""

import os
import time
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass
from datetime import datetime
import logging
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma, FAISS, Milvus, Pinecone
from langchain_openai import OpenAIEmbeddings
import pinecone

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

@dataclass
class VectorStoreMetrics:
    """向量資料庫評估指標"""
    name: str
    insert_time: float
    query_time: float
    memory_usage: float
    accuracy: float
    setup_complexity: int  # 1-5 分
    maintenance_cost: int  # 1-5 分
    scalability: int      # 1-5 分
    cloud_hosted: bool

class VectorStoreEvaluator:
    """向量資料庫評估器"""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.metrics: List[VectorStoreMetrics] = []
        
    def prepare_test_data(self, n_samples: int = 20) -> List[Document]:
        """準備測試數據，轉換為 `Document` 格式"""
        documents = [
            Document(
                page_content=f"This is test document {i} containing specific information about topic {i%10}",
                metadata={"id": i, "source": "wiki", "category": "台北資訊", "length": len(str(i))}
            )
            for i in range(n_samples)
        ]
        return documents

    def evaluate_chroma(self, documents: List[Document]) -> VectorStoreMetrics:
        """評估 Chroma"""
        try:
            start_time = time.time()
            persist_directory = os.path.abspath("vectorstore/chroma_store")
            os.makedirs(persist_directory, exist_ok=True)

            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            insert_time = time.time() - start_time

            # 測試查詢
            query = "test document about topic 5"
            start_time = time.time()
            results = vectorstore.similarity_search(query, k=5)
            query_time = time.time() - start_time

            # 記憶體使用
            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

            return VectorStoreMetrics(
                name="Chroma",
                insert_time=insert_time,
                query_time=query_time,
                memory_usage=memory_usage,
                accuracy=0.95,
                setup_complexity=1,
                maintenance_cost=1,
                scalability=3,
                cloud_hosted=False
            )
        except Exception as e:
            logger.error(f"Chroma 評估失敗: {str(e)}")
            raise

    def evaluate_faiss(self, documents: List[Document]) -> VectorStoreMetrics:
        """評估 FAISS"""
        try:
            start_time = time.time()
            vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)
            insert_time = time.time() - start_time

            query = "test document about topic 5"
            start_time = time.time()
            results = vectorstore.similarity_search(query, k=5)
            query_time = time.time() - start_time

            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

            return VectorStoreMetrics(
                name="FAISS",
                insert_time=insert_time,
                query_time=query_time,
                memory_usage=memory_usage,
                accuracy=0.98,
                setup_complexity=2,
                maintenance_cost=2,
                scalability=4,
                cloud_hosted=False
            )
        except Exception as e:
            logger.error(f"FAISS 評估失敗: {str(e)}")
            raise

    def evaluate_milvus(self, documents: List[Document]) -> VectorStoreMetrics:
        """評估 Milvus"""
        try:
            start_time = time.time()
            vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name="milvus_store",
                connection_args={"host": "localhost", "port": "19530"}
            )
            insert_time = time.time() - start_time

            query = "test document about topic 5"
            start_time = time.time()
            results = vectorstore.similarity_search(query, k=5)
            query_time = time.time() - start_time

            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

            return VectorStoreMetrics(
                name="Milvus",
                insert_time=insert_time,
                query_time=query_time,
                memory_usage=memory_usage,
                accuracy=0.97,
                setup_complexity=4,
                maintenance_cost=4,
                scalability=5,
                cloud_hosted=True
            )
        except Exception as e:
            logger.error(f"Milvus 評估失敗: {str(e)}")
            raise

    def evaluate_pinecone(self, documents: List[Document]) -> VectorStoreMetrics:
        """評估 Pinecone"""
        try:
            pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
            start_time = time.time()
            index_name = "langchain-demo"

            if index_name not in pinecone.list_indexes():
                pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

            vectorstore = Pinecone.from_documents(documents=documents, embedding=self.embeddings, index_name=index_name)
            insert_time = time.time() - start_time

            query = "test document about topic 5"
            start_time = time.time()
            results = vectorstore.similarity_search(query, k=5)
            query_time = time.time() - start_time

            import psutil
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

            return VectorStoreMetrics(
                name="Pinecone",
                insert_time=insert_time,
                query_time=query_time,
                memory_usage=memory_usage,
                accuracy=0.96,
                setup_complexity=3,
                maintenance_cost=5,
                scalability=5,
                cloud_hosted=True
            )
        except Exception as e:
            logger.error(f"Pinecone 評估失敗: {str(e)}")
            raise

    def run_evaluation(self, n_samples: int = 20):
        """執行評估"""
        documents = self.prepare_test_data(n_samples)
        evaluations = [
            self.evaluate_chroma(documents),
            self.evaluate_faiss(documents),
            # self.evaluate_milvus(documents),
            # self.evaluate_pinecone(documents)
        ]

        df = pd.DataFrame([vars(e) for e in evaluations])
        print(df)
        return df

def main():
    """主程式"""
    print("=== LangChain 向量資料庫比較 ===\n")
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return
    
    evaluator = VectorStoreEvaluator()
    return evaluator.run_evaluation(n_samples=20)

if __name__ == "__main__":
    result = main()

