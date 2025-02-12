"""
LangChain 向量檢索進階範例
展示不同的向量檢索方法與評估

主要功能：
1. 多種檢索方法比較 (MMR, Similarity, BM25)
2. 檢索參數調優 (fetch_k, k, lambda_mult, b, k1)
3. 檢索結果評估
4. 向量資料庫進階操作
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Set, Optional
from collections import Counter
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import jieba  # 導入結巴分詞
from time import perf_counter
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
import shutil
from pathlib import Path

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

@dataclass
class RetrievalMetrics:
    """檢索評估指標"""
    search_time: float  # 搜尋時間（秒）
    normalized_score: float  # 正規化分數 (0-100)
    raw_score: float  # 原始分數

class RetrievalResult(BaseModel):
    """檢索結果模型"""
    content: str = Field(description="文件內容")
    metadata: Dict[str, Any] = Field(description="文件元數據")
    metrics: RetrievalMetrics = Field(description="檢索評估指標")
    method: str = Field(description="檢索方法")

class BM25:
    """BM25 檢索算法實現"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        初始化 BM25
        
        Args:
            k1: 詞頻飽和參數 (通常在 1.2 ~ 2.0 之間)
            b: 文檔長度正規化參數 (通常為 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.doc_freqs = Counter()  # 文檔頻率
        self.idf = {}  # 逆文檔頻率
        self.doc_len = []  # 文檔長度
        self.avgdl = 0  # 平均文檔長度
        self._tokenizer = jieba.cut  # 使用結巴分詞
        
    def _tokenize(self, text: str) -> List[str]:
        """使用結巴分詞進行分詞"""
        return [token for token in self._tokenizer(text) if token.strip()]
    
    def fit(self, documents: List[str]):
        """訓練 BM25 模型"""
        self.corpus_size = len(documents)
        
        # 計算文檔頻率和長度
        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_len.append(len(tokens))
            
            # 計算每個詞在文檔中的出現次數
            word_set = set(tokens)
            for word in word_set:
                self.doc_freqs[word] += 1
        
        # 計算平均文檔長度
        self.avgdl = sum(self.doc_len) / self.corpus_size
        
        # 計算 IDF
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
    
    def get_scores(self, query: str, documents: List[str]) -> List[float]:
        """計算查詢與所有文檔的 BM25 分數"""
        scores = []
        query_tokens = self._tokenize(query)
        
        for idx, doc in enumerate(documents):
            score = 0
            doc_tokens = self._tokenize(doc)
            doc_len = len(doc_tokens)
            doc_freqs = Counter(doc_tokens)
            
            for word in query_tokens:
                if word not in self.idf:
                    continue
                    
                # 計算 TF
                freq = doc_freqs.get(word, 0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                
                # 計算該詞的 BM25 分數
                score += self.idf[word] * numerator / denominator
            
            scores.append(score)
        
        return scores

class VectorRetriever:
    """向量檢索器"""
    
    def __init__(self, documents: List[Document], collection_name: str = "taipei_advanced"):
        """
        初始化向量檢索器
        
        Args:
            documents: 文檔列表
            collection_name: 集合名稱（只能包含字母、數字、下劃線和連字符）
        """
        self.documents = documents
        self.collection_name = collection_name
        # 建立向量存儲目錄
        self.persist_directory = os.path.join("vectorstore", self.collection_name)
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.embeddings = OpenAIEmbeddings()
        
        # 預先計算所有文檔的向量表示
        self.doc_vectors = self.embeddings.embed_documents(
            [doc.page_content for doc in documents]
        )
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        
        # 初始化 BM25
        self.bm25 = BM25()
        texts = [doc.page_content for doc in documents]
        self.bm25.fit(texts)
        
        logger.info("向量檢索器初始化完成")

    
    def _compute_similarity(self, query_vector: List[float], doc_vector: List[float]) -> float:
        """計算向量相似度（統一使用餘弦相似度）"""
        query_array = np.array(query_vector).reshape(1, -1)
        doc_array = np.array(doc_vector).reshape(1, -1)
        return cosine_similarity(query_array, doc_array)[0][0]
    
    def _get_query_vector(self, query: str) -> List[float]:
        """獲取查詢的向量表示"""
        return self.embeddings.embed_query(query)
    
    def similarity_search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        """相似度搜尋"""
        start_time = perf_counter()
        
        # 計算查詢向量
        query_vector = self._get_query_vector(query)
        
        # 計算所有文檔的相似度
        similarities = [
            self._compute_similarity(query_vector, doc_vector)
            for doc_vector in self.doc_vectors
        ]
        
        # 獲取前 k 個最相似的文檔
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        search_time = perf_counter() - start_time
        
        return [
            RetrievalResult(
                content=self.documents[idx].page_content,
                metadata=self.documents[idx].metadata,
                metrics=RetrievalMetrics(
                    search_time=search_time,
                    normalized_score=similarities[idx] * 100,  # 直接轉換為百分比
                    raw_score=similarities[idx]
                ),
                method="similarity"
            )
            for idx in top_k_idx
        ]
    
    def mmr_search(self, query: str, k: int = 3, lambda_mult: float = 0.7) -> List[RetrievalResult]:
        """最大邊際相關性搜尋"""
        start_time = perf_counter()
        
        # 計算查詢向量
        query_vector = self._get_query_vector(query)
        
        # 使用 MMR 算法選擇文檔
        selected_indices = []
        candidate_indices = list(range(len(self.documents)))
        
        # 先計算所有文檔與查詢的相似度
        similarities = [
            self._compute_similarity(query_vector, self.doc_vectors[idx])
            for idx in range(len(self.documents))
        ]
        
        for _ in range(k):
            if not candidate_indices:
                break
                
            # 計算候選文檔與查詢的相似度
            candidate_similarities = [
                similarities[idx] for idx in candidate_indices
            ]
            
            if not selected_indices:
                # 第一個文檔：選擇最相似的
                best_idx = candidate_indices[np.argmax(candidate_similarities)]
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
            else:
                # 後續文檔：使用 MMR 公式
                mmr_scores = []
                for cand_idx, sim in zip(candidate_indices, candidate_similarities):
                    # 計算與已選文檔的最大相似度
                    redundancy = max(
                        self._compute_similarity(
                            self.doc_vectors[cand_idx],
                            self.doc_vectors[sel_idx]
                        )
                        for sel_idx in selected_indices
                    )
                    mmr_score = lambda_mult * sim - (1 - lambda_mult) * redundancy
                    mmr_scores.append(mmr_score)
                
                best_idx = candidate_indices[np.argmax(mmr_scores)]
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
        
        search_time = perf_counter() - start_time
        
        return [
            RetrievalResult(
                content=self.documents[idx].page_content,
                metadata=self.documents[idx].metadata,
                metrics=RetrievalMetrics(
                    search_time=search_time,
                    normalized_score=similarities[idx] * 100,  # 使用預先計算的相似度
                    raw_score=similarities[idx]
                ),
                method="mmr"
            )
            for idx in selected_indices
        ]
    
    def bm25_search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        """BM25 搜尋，最終以餘弦相似度衡量相似度"""
        start_time = perf_counter()
        
        texts = [doc.page_content for doc in self.documents]
        
        # BM25 計算原始分數
        bm25_scores = self.bm25.get_scores(query, texts)

        # 取得前 k 個文檔索引
        top_k_idx = np.argsort(bm25_scores)[-k:][::-1]

        # 計算查詢向量
        query_vector = self._get_query_vector(query)
        
        search_time = perf_counter() - start_time

        return [
            RetrievalResult(
                content=self.documents[idx].page_content,
                metadata=self.documents[idx].metadata,
                metrics=RetrievalMetrics(
                    search_time=search_time,
                    normalized_score=self._compute_similarity(query_vector, self.doc_vectors[idx]) * 100,  # 轉為百分比
                    raw_score=bm25_scores[idx]  # 保留原始 BM25 分數
                ),
                method="bm25"
            )
            for idx in top_k_idx
        ]



def create_demo_documents() -> List[Document]:
    """建立示範文件"""
    # 使用更多樣化的台北相關文本
    texts = [
        "台北市位於台灣北部，是台灣的首都及最大的都市，人口約250萬。",
        "台北101高度達509.2公尺，是台北市最著名的地標建築。",
        "台北的捷運系統包含多條路線，連接大台北地區。",
        "台北夜市文化豐富，士林夜市是規模最大的夜市。",
        "台北故宮博物院收藏了眾多中國古代文物。",
        "台北市分為12個行政區，每個區都有其特色。",
        "台北車站是重要的交通樞紐，整合多種運輸方式。",
        "信義區是台北的商業中心，有許多百貨公司。",
        "台北的陽明山國家公園擁有豐富的自然資源。",
        "松山文創園區是台北重要的文創基地。"
    ]

    
    documents = []
    for i, text in enumerate(texts):
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "id": i,
                    "source": "wiki",
                    "category": "台北資訊",
                    "length": len(text)
                }
            )
        )
    return documents

def compare_retrieval_methods(query: str):
    """比較不同檢索方法"""
    documents = create_demo_documents()
    retriever = VectorRetriever(documents)
    
    print(f"\n查詢: {query}")
    print("=" * 50)
    
    methods = {
        "相似度搜尋": retriever.similarity_search,
        "MMR 搜尋": retriever.mmr_search,
        "BM25 搜尋": retriever.bm25_search
    }
    
    for method_name, search_func in methods.items():
        print(f"\n【{method_name}結果】")
        results = search_func(query)
        for i, result in enumerate(results, 1):
            metrics = result.metrics
            print(f"\n文件 {i}")
            print(f"相似度: {metrics.normalized_score:.1f}% (原始分數: {metrics.raw_score:.4f})")
            print(f"搜尋時間: {metrics.search_time*1000:.2f}ms")
            print(f"內容: {result.content}")
            print(f"元數據: {result.metadata}")
            print("-" * 40)

def main():
    """主程式：展示向量檢索進階功能"""
    print("=== LangChain 向量檢索進階展示 ===\n")
    print("本範例展示三種檢索方法的比較：")
    print("1. 向量相似度搜尋：基於文本嵌入的餘弦相似度")
    print("2. MMR 搜尋：在相關性和多樣性之間取得平衡")
    print("3. BM25 搜尋：經典的詞頻-逆文檔頻率算法")
    print("\n" + "=" * 50 + "\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return
    
    try:
        # 測試查詢
        queries = [
            "台北的交通系統",
            # "台北的文化景點",
            # "台北的地標建築",
            # "台北的自然景觀"
        ]
        
        for query in queries:
            compare_retrieval_methods(query)
            print("\n" + "=" * 80 + "\n")
            
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 