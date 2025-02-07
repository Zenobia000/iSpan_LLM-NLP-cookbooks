"""
LangChain 0.3+ 相似度搜尋方法比較
比較不同相似度搜尋方法的性能與特性

需求套件:
- langchain>=0.3.0
- langchain-community>=0.0.1
- rank_bm25>=0.2.2
- scikit-learn>=1.3.0
- pandas>=2.0.0
- numpy>=1.24.0
- python-dotenv>=0.19.0
"""

import os
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances
)
from sklearn.preprocessing import normalize

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import (
    BM25Retriever,
    EnsembleRetriever
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
class SearchMetrics:
    """搜尋方法評估指標"""
    name: str
    search_time: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    diversity: float
    explanation: str

class SearchMethodEvaluator:
    """相似度搜尋方法評估器"""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.metrics: List[SearchMetrics] = []
        
    def prepare_test_data(self) -> Tuple[List[str], List[str], List[List[int]]]:
        """準備測試數據"""
        # 準備文檔集合
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision focuses on helping machines interpret visual information.",
            "Reinforcement learning is learning through interaction with an environment.",
            "Supervised learning uses labeled training data.",
            "Unsupervised learning finds patterns in unlabeled data.",
            "Neural networks are inspired by biological brain structures.",
            "Convolutional neural networks are commonly used in image processing.",
            "Recurrent neural networks are good at processing sequential data."
        ]
        
        # 準備查詢
        queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain deep learning concepts",
            "Tell me about computer vision",
            "What is natural language processing?"
        ]
        
        # 準備相關性標記（1表示相關，0表示不相關）
        relevance = [
            [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 對於查詢1
            [0, 1, 0, 0, 0, 0, 0, 1, 1, 1],  # 對於查詢2
            [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],  # 對於查詢3
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # 對於查詢4
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]   # 對於查詢5
        ]
        
        return documents, queries, relevance

    def calculate_metrics(
        self,
        results: List[int],
        relevance: List[int],
        k: int = 5
    ) -> Tuple[float, float, float, float]:
        """計算評估指標"""
        # Precision@K
        precision = sum(relevance[i] for i in results[:k]) / k
        
        # Recall@K
        total_relevant = sum(relevance)
        recall = sum(relevance[i] for i in results[:k]) / total_relevant if total_relevant > 0 else 0
        
        # MRR (Mean Reciprocal Rank)
        for i, doc_id in enumerate(results, 1):
            if relevance[doc_id] == 1:
                mrr = 1.0 / i
                break
        else:
            mrr = 0
        
        # NDCG@K
        dcg = sum(relevance[i] / np.log2(rank + 2) for rank, i in enumerate(results[:k]))
        ideal_results = sorted(range(len(relevance)), key=lambda i: relevance[i], reverse=True)
        idcg = sum(relevance[i] / np.log2(rank + 2) for rank, i in enumerate(ideal_results[:k]))
        ndcg = dcg / idcg if idcg > 0 else 0
        
        return precision, recall, mrr, ndcg

    def calculate_diversity(self, results: List[int], embeddings: List[np.ndarray]) -> float:
        """計算結果多樣性"""
        if len(results) < 2:
            return 0.0
        
        # 獲取結果文檔的嵌入向量
        result_embeddings = [embeddings[i] for i in results]
        
        # 計算結果之間的平均餘弦相似度
        similarities = cosine_similarity(result_embeddings)
        
        # 計算多樣性分數 (1 - 平均相似度)
        n = len(results)
        diversity = 1.0 - (np.sum(similarities) - n) / (n * (n - 1))
        
        return diversity

    def evaluate_cosine(
        self,
        documents: List[str],
        queries: List[str],
        relevance: List[List[int]]
    ) -> SearchMetrics:
        """評估餘弦相似度搜尋"""
        try:
            # 獲取嵌入向量
            doc_embeddings = self.embeddings.embed_documents(documents)
            doc_embeddings = np.array(doc_embeddings)
            
            start_time = time.time()
            metrics_list = []
            
            for query, rel in zip(queries, relevance):
                # 獲取查詢嵌入
                query_embedding = self.embeddings.embed_query(query)
                query_embedding = np.array(query_embedding).reshape(1, -1)
                
                # 計算相似度
                similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
                results = np.argsort(-similarities)[:5].tolist()
                
                # 計算評估指標
                precision, recall, mrr, ndcg = self.calculate_metrics(results, rel)
                metrics_list.append((precision, recall, mrr, ndcg))
            
            search_time = time.time() - start_time
            
            # 計算平均指標
            avg_metrics = np.mean(metrics_list, axis=0)
            diversity = self.calculate_diversity(results, doc_embeddings)
            
            return SearchMetrics(
                name="Cosine Similarity",
                search_time=search_time,
                precision_at_k=avg_metrics[0],
                recall_at_k=avg_metrics[1],
                mrr=avg_metrics[2],
                ndcg=avg_metrics[3],
                diversity=diversity,
                explanation="基於向量空間中的夾角計算相似度"
            )
        except Exception as e:
            logger.error(f"餘弦相似度評估失敗: {str(e)}")
            raise

    def evaluate_mmr(
        self,
        documents: List[str],
        queries: List[str],
        relevance: List[List[int]],
        lambda_param: float = 0.5
    ) -> SearchMetrics:
        """評估 MMR (Maximal Marginal Relevance) 搜尋"""
        try:
            # 獲取嵌入向量
            doc_embeddings = self.embeddings.embed_documents(documents)
            doc_embeddings = np.array(doc_embeddings)
            
            start_time = time.time()
            metrics_list = []
            
            for query, rel in zip(queries, relevance):
                query_embedding = self.embeddings.embed_query(query)
                query_embedding = np.array(query_embedding).reshape(1, -1)
                
                # MMR 實現
                remaining_docs = list(range(len(documents)))
                selected = []
                
                while len(selected) < 5 and remaining_docs:
                    # 計算相關性分數
                    relevance_scores = cosine_similarity(
                        query_embedding,
                        doc_embeddings[remaining_docs]
                    )[0]
                    
                    if not selected:
                        # 第一個文檔：選擇最相關的
                        idx = np.argmax(relevance_scores)
                        selected.append(remaining_docs[idx])
                        del remaining_docs[idx]
                    else:
                        # 後續文檔：考慮相關性和多樣性
                        diversity_scores = np.max(cosine_similarity(
                            doc_embeddings[remaining_docs],
                            doc_embeddings[selected]
                        ), axis=1)
                        
                        mmr_scores = lambda_param * relevance_scores - \
                                   (1 - lambda_param) * diversity_scores
                        
                        idx = np.argmax(mmr_scores)
                        selected.append(remaining_docs[idx])
                        del remaining_docs[idx]
                
                # 計算評估指標
                precision, recall, mrr, ndcg = self.calculate_metrics(selected, rel)
                metrics_list.append((precision, recall, mrr, ndcg))
            
            search_time = time.time() - start_time
            
            # 計算平均指標
            avg_metrics = np.mean(metrics_list, axis=0)
            diversity = self.calculate_diversity(selected, doc_embeddings)
            
            return SearchMetrics(
                name="MMR",
                search_time=search_time,
                precision_at_k=avg_metrics[0],
                recall_at_k=avg_metrics[1],
                mrr=avg_metrics[2],
                ndcg=avg_metrics[3],
                diversity=diversity,
                explanation="平衡相關性與多樣性的檢索方法"
            )
        except Exception as e:
            logger.error(f"MMR 評估失敗: {str(e)}")
            raise

    def evaluate_bm25(
        self,
        documents: List[str],
        queries: List[str],
        relevance: List[List[int]]
    ) -> SearchMetrics:
        """評估 BM25 搜尋"""
        try:
            # 初始化 BM25
            tokenized_docs = [doc.lower().split() for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)
            
            start_time = time.time()
            metrics_list = []
            
            for query, rel in zip(queries, relevance):
                # 進行搜尋
                tokenized_query = query.lower().split()
                scores = bm25.get_scores(tokenized_query)
                results = np.argsort(-scores)[:5].tolist()
                
                # 計算評估指標
                precision, recall, mrr, ndcg = self.calculate_metrics(results, rel)
                metrics_list.append((precision, recall, mrr, ndcg))
            
            search_time = time.time() - start_time
            
            # 計算平均指標
            avg_metrics = np.mean(metrics_list, axis=0)
            
            # 為了計算多樣性，需要獲取文檔的嵌入向量
            doc_embeddings = self.embeddings.embed_documents(documents)
            diversity = self.calculate_diversity(results, doc_embeddings)
            
            return SearchMetrics(
                name="BM25",
                search_time=search_time,
                precision_at_k=avg_metrics[0],
                recall_at_k=avg_metrics[1],
                mrr=avg_metrics[2],
                ndcg=avg_metrics[3],
                diversity=diversity,
                explanation="基於詞頻與文檔長度的概率檢索模型"
            )
        except Exception as e:
            logger.error(f"BM25 評估失敗: {str(e)}")
            raise

    def run_evaluation(self):
        """執行評估"""
        documents, queries, relevance = self.prepare_test_data()
        
        # 評估各個搜尋方法
        evaluations = [
            self.evaluate_cosine(documents, queries, relevance),
            self.evaluate_mmr(documents, queries, relevance),
            self.evaluate_bm25(documents, queries, relevance)
        ]
        
        # 轉換為 DataFrame
        df = pd.DataFrame([
            {
                "搜尋方法": e.name,
                "搜尋時間 (秒)": round(e.search_time, 3),
                "Precision@5": round(e.precision_at_k, 3),
                "Recall@5": round(e.recall_at_k, 3),
                "MRR": round(e.mrr, 3),
                "NDCG@5": round(e.ndcg, 3),
                "多樣性": round(e.diversity, 3),
                "說明": e.explanation
            }
            for e in evaluations
        ])
        
        # 設定顯示格式
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print("\n=== 相似度搜尋方法比較結果 ===")
        print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"測試文檔數: {len(documents)}")
        print(f"測試查詢數: {len(queries)}")
        print("\n" + str(df))
        
        # 輸出建議
        print("\n=== 使用建議 ===")
        print("1. 一般場景建議使用餘弦相似度：簡單有效，實現容易")
        print("2. 需要結果多樣性時建議使用 MMR：可以避免結果過於相似")
        print("3. 文本搜尋場景建議使用 BM25：特別適合關鍵詞匹配")
        
        return df

def main():
    """主程式"""
    print("=== LangChain 0.3+ 相似度搜尋方法比較 ===\n")
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return
    
    try:
        evaluator = SearchMethodEvaluator()
        evaluator.run_evaluation()
    except Exception as e:
        logger.error(f"評估過程發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 