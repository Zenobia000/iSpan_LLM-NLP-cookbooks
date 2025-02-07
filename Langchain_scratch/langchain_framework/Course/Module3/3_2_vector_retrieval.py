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
from typing import List, Dict, Any
from collections import Counter
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

class RetrievalResult(BaseModel):
    """檢索結果模型"""
    content: str = Field(description="文件內容")
    metadata: Dict[str, Any] = Field(description="文件元數據")
    score: float = Field(description="相似度分數")
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
        
    def fit(self, documents: List[str]):
        """訓練 BM25 模型"""
        self.corpus_size = len(documents)
        
        # 計算文檔頻率和長度
        for doc in documents:
            words = doc.split()
            self.doc_len.append(len(words))
            
            # 計算每個詞在文檔中的出現次數
            word_set = set(words)
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
        query_words = query.split()
        
        for idx, doc in enumerate(documents):
            score = 0
            doc_words = doc.split()
            doc_len = len(doc_words)
            word_freqs = Counter(doc_words)
            
            for word in query_words:
                if word not in self.idf:
                    continue
                    
                # 計算 BM25 分數
                freq = word_freqs[word]
                numerator = self.idf[word] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
            
            scores.append(score)
        
        return scores

class VectorRetriever:
    """向量檢索器"""
    
    def __init__(self, documents: List[Document]):
        """初始化向量檢索器"""
        self.documents = documents
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="taipei_advanced"
        )
        
        # 初始化 BM25
        self.bm25 = BM25()
        texts = [doc.page_content for doc in documents]
        self.bm25.fit(texts)
        
        logger.info("向量檢索器初始化完成")
    
    def similarity_search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        """相似度搜尋"""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            RetrievalResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=(1 - score) * 100,  # 轉換為百分比
                method="similarity"
            )
            for doc, score in results
        ]
    
    def mmr_search(
        self, query: str, k: int = 3, fetch_k: int = 6, lambda_mult: float = 0.7
    ) -> List[RetrievalResult]:
        """最大邊際相關性搜尋 (Maximal Marginal Relevance)"""
        # 先將查詢轉換為向量
        query_embedding = self.embeddings.embed_query(query)
        
        # 使用 MMR 搜尋
        results = self.vectorstore.max_marginal_relevance_search_by_vector(
            embedding=query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        
        # 計算相似度分數
        scores = []
        for doc in results:
            doc_embedding = self.embeddings.embed_documents([doc.page_content])[0]
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            scores.append(similarity)
        
        # 組合結果
        return [
            RetrievalResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score * 100,  # 轉換為百分比
                method="mmr"
            )
            for doc, score in zip(results, scores)
        ]
    
    def bm25_search(self, query: str, k: int = 3) -> List[RetrievalResult]:
        """BM25 搜尋"""
        texts = [doc.page_content for doc in self.documents]
        scores = self.bm25.get_scores(query, texts)
        
        # 獲取前 k 個最高分數的索引
        top_k_idx = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            # 將分數正規化到 0-100 範圍
            normalized_score = min(100, scores[idx] * 10)  # 調整係數以獲得合理的分數範圍
            results.append(
                RetrievalResult(
                    content=self.documents[idx].page_content,
                    metadata=self.documents[idx].metadata,
                    score=normalized_score,
                    method="bm25"
                )
            )
        return results

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
    
    # 相似度搜尋
    print("\n【相似度搜尋結果】")
    similarity_results = retriever.similarity_search(query)
    for i, result in enumerate(similarity_results, 1):
        print(f"\n文件 {i} (相似度: {result.score:.1f}%)")
        print(f"內容: {result.content}")
        print(f"元數據: {result.metadata}")
        print("-" * 40)
    
    # MMR 搜尋
    print("\n【MMR 搜尋結果】")
    mmr_results = retriever.mmr_search(query)
    for i, result in enumerate(mmr_results, 1):
        print(f"\n文件 {i} (相似度: {result.score:.1f}%)")
        print(f"內容: {result.content}")
        print(f"元數據: {result.metadata}")
        print("-" * 40)
    
    # BM25 搜尋
    print("\n【BM25 搜尋結果】")
    bm25_results = retriever.bm25_search(query)
    for i, result in enumerate(bm25_results, 1):
        print(f"\n文件 {i} (相關度: {result.score:.1f}%)")
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
            "台北的文化景點",
            "台北的地標建築",
            "台北的自然景觀"
        ]
        
        for query in queries:
            compare_retrieval_methods(query)
            print("\n" + "=" * 80 + "\n")
            
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 