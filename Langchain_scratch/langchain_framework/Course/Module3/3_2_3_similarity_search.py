import os
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
import hnswlib

from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever


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

@dataclass
class SearchResult:
    """搜尋結果"""
    content: str
    score: float
    metadata: Dict[str, Any]
    method: str
    search_time: float

def prepare_test_data() -> Tuple[List[str], List[str], List[List[int]]]:
    """準備測試數據，包含台灣相關的文本內容"""
    documents = [
        "台北101是台灣最高的摩天大樓，也是世界知名地標。",
        "台灣小吃如珍珠奶茶、滷肉飯和蚵仔煎廣受歡迎。",
        "阿里山的日出與雲海是台灣最著名的自然景觀之一。",
        "台灣擁有豐富的夜市文化，如士林夜市和逢甲夜市。",
        "故宮博物院收藏了大量中國歷史文物，是亞洲重要的博物館。",
        "台灣的半導體產業全球領先，台積電是最具代表性的公司。",
        "陽明山國家公園是台北近郊的熱門旅遊景點，擁有溫泉與登山步道。",
        "高雄港是台灣最大港口，對亞洲貿易至關重要。",
        "台灣的高鐵系統連接台北到高雄，提供便捷的交通選擇。",
        "台灣擁有多家知名大學，如台灣大學與清華大學。"
    ]
    
    queries = [
        "台北最高的大樓是什麼？",
        "哪裡可以吃到台灣的特色小吃？",
        "台灣哪裡可以欣賞壯觀的日出？",
        "台灣的半導體產業領導者是哪家公司？",
        "台灣最重要的歷史博物館是哪裡？"
    ]
    
    relevance = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 查詢1 - 台北101
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 查詢2 - 台灣小吃 & 夜市
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # 查詢3 - 阿里山 & 陽明山
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 查詢4 - 台積電
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]   # 查詢5 - 故宮博物院
    ]
    
    return documents, queries, relevance

class SearchMethodEvaluator:
    """相似度搜尋方法評估器"""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def calculate_precision(self, results: List[int], relevance: List[int], k: int) -> float:
        return sum(relevance[i] for i in results[:k]) / k

    def calculate_recall(self, results: List[int], relevance: List[int], k: int) -> float:
        total_relevant = sum(relevance)
        return sum(relevance[i] for i in results[:k]) / total_relevant if total_relevant > 0 else 0

    def calculate_mrr(self, results: List[int], relevance: List[int]) -> float:
        return next((1.0 / (i + 1) for i, doc_id in enumerate(results) if relevance[doc_id] == 1), 0)

    def calculate_ndcg(self, results: List[int], relevance: List[int], k: int) -> float:
        dcg = sum(relevance[i] / np.log2(rank + 2) for rank, i in enumerate(results[:k]))
        ideal_results = sorted(range(len(relevance)), key=lambda i: relevance[i], reverse=True)
        idcg = sum(relevance[i] / np.log2(rank + 2) for rank, i in enumerate(ideal_results[:k]))
        return dcg / idcg if idcg > 0 else 0

    def calculate_diversity(self, results: List[int], documents: List[str]) -> float:
        """計算搜尋結果的多樣性
        
        使用 Jaccard 距離計算文檔間的差異度
        返回值範圍: 0-1，越大表示結果越多樣化
        """
        if len(results) <= 1:
            return 0.0
        
        def jaccard_distance(doc1: str, doc2: str) -> float:
            """計算兩個文檔的 Jaccard 距離"""
            tokens1 = set(doc1.split())
            tokens2 = set(doc2.split())
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            return 1 - (intersection / union if union > 0 else 0)
        
        # 計算所有結果對之間的距離
        distances = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                doc1 = documents[results[i]]
                doc2 = documents[results[j]]
                distances.append(jaccard_distance(doc1, doc2))
        
        # 返回平均距離
        return np.mean(distances) if distances else 0.0

    def evaluate_search_method(self, method_list: Dict[str, str], method_name: str, search_results: List[int], 
                             relevance: List[int], search_time: float, documents: List[str]) -> SearchMetrics:
        """評估搜尋方法的效果"""
        k = 5  # 評估前 k 個結果
        precision = self.calculate_precision(search_results, relevance, k)
        recall = self.calculate_recall(search_results, relevance, k)
        mrr = self.calculate_mrr(search_results, relevance)
        ndcg = self.calculate_ndcg(search_results, relevance, k)
        diversity = self.calculate_diversity(search_results, documents)
        
        method_list = {
            "BM25": "基於詞頻與文檔長度的檢索模型",
            "FAISS": "基於向量近似最近鄰的快速檢索",
            "Annoy": "基於隨機投影樹的輕量級檢索",
            "Hybrid": "結合 BM25 與向量檢索的混合模型"
        }
        
        return SearchMetrics(
            name=method_name,
            search_time=search_time,
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg=ndcg,
            diversity=diversity,
            explanation=method_list.get(method_name, "未知搜尋方法")
        )

    def evaluate_all_methods(self, documents: List[str], queries: List[str], 
                           relevance: List[List[int]]) -> pd.DataFrame:
        """評估所有搜尋方法"""
        vector_evaluator = VectorSearchEvaluator(documents)
        all_metrics = []
        
        for query, rel in zip(queries, relevance):
            # 評估 FAISS
            faiss_results = vector_evaluator.evaluate_faiss(query)
            faiss_indices = [r.metadata["index"] for r in faiss_results]
            faiss_metrics = self.evaluate_search_method(
                "FAISS", faiss_indices, rel, faiss_results[0].search_time, documents
            )
            all_metrics.append(faiss_metrics)
            
            # 評估 Annoy
            annoy_results = vector_evaluator.evaluate_annoy(query)
            annoy_indices = [r.metadata["index"] for r in annoy_results]
            annoy_metrics = self.evaluate_search_method(
                "Annoy", annoy_indices, rel, annoy_results[0].search_time, documents
            )
            all_metrics.append(annoy_metrics)
            
            # 評估 Hybrid
            hybrid_results = vector_evaluator.evaluate_hybrid(query)
            hybrid_indices = [r.metadata["index"] for r in hybrid_results]
            hybrid_metrics = self.evaluate_search_method(
                "Hybrid", hybrid_indices, rel, hybrid_results[0].search_time, documents
            )
            all_metrics.append(hybrid_metrics)
        
        # 轉換為 DataFrame
        df = pd.DataFrame([
            {k: round(v, 3) if isinstance(v, float) else v 
             for k, v in m.__dict__.items()}
            for m in all_metrics
        ])
        
        # 計算每種方法的平均指標
        return df.groupby('name').mean().reset_index()

class VectorSearchEvaluator:
    """向量搜尋評估器"""
    
    def __init__(self, documents: List[str], embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        # self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.embeddings = OpenAIEmbeddings()
        # 預計算所有文檔的向量表示
        self.doc_vectors = self._compute_embeddings(documents)
        self.vector_dim = len(self.doc_vectors[0])
        
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """計算文本的向量表示"""
        vectors = self.embeddings.embed_documents(texts)
        return normalize(np.array(vectors))  # 正規化向量
    
    def evaluate_faiss(self, query: str, k: int = 3) -> List[SearchResult]:
        """評估 FAISS 搜尋"""
        start_time = time.perf_counter()
        
        # 建立 FAISS 索引
        index = faiss.IndexFlatIP(self.vector_dim)  # 使用內積相似度
        index.add(self.doc_vectors)
        
        # 搜尋
        query_vector = self._compute_embeddings([query])[0]
        scores, indices = index.search(query_vector.reshape(1, -1), k)
        
        search_time = time.perf_counter() - start_time
        
        return [
            SearchResult(
                content=self.documents[idx],
                score=score,
                metadata={"index": idx},
                method="FAISS",
                search_time=search_time
            )
            for score, idx in zip(scores[0], indices[0])
        ]
    
    def evaluate_annoy(self, query: str, k: int = 3) -> List[SearchResult]:
        """評估 Annoy 搜尋"""
        start_time = time.perf_counter()
        
        # 建立 Annoy 索引
        index = AnnoyIndex(self.vector_dim, 'angular')  # 使用角度距離
        for i, vector in enumerate(self.doc_vectors):
            index.add_item(i, vector)
        index.build(10)  # 建立 10 棵樹
        
        # 搜尋
        query_vector = self._compute_embeddings([query])[0]
        indices, distances = index.get_nns_by_vector(
            query_vector, k, 
            include_distances=True
        )
        
        search_time = time.perf_counter() - start_time
        
        # 將距離轉換為相似度分數
        scores = 1 - np.array(distances) / 2  # 角度距離轉換為相似度
        
        return [
            SearchResult(
                content=self.documents[idx],
                score=score,
                metadata={"index": idx},
                method="Annoy",
                search_time=search_time
            )
            for score, idx in zip(scores, indices)
        ]
    
    def evaluate_hybrid(self, query: str, k: int = 3) -> List[SearchResult]:
        """評估混合搜尋 (BM25 + 向量)"""
        start_time = time.perf_counter()
        
        # BM25 搜尋
        bm25 = BM25Okapi([doc.split() for doc in self.documents])
        bm25_scores = bm25.get_scores(query.split())
        
        # 向量搜尋
        query_vector = self._compute_embeddings([query])[0]
        vector_scores = np.dot(self.doc_vectors, query_vector)
        
        # 結合分數 (簡單加權平均)
        combined_scores = 0.3 * normalize(bm25_scores.reshape(1, -1))[0] + \
                         0.7 * vector_scores
        
        # 取得前 k 個結果
        top_k_idx = np.argsort(combined_scores)[-k:][::-1]
        top_k_scores = combined_scores[top_k_idx]
        
        search_time = time.perf_counter() - start_time
        
        return [
            SearchResult(
                content=self.documents[idx],
                score=score,
                metadata={
                    "index": idx,
                    "bm25_score": bm25_scores[idx],
                    "vector_score": vector_scores[idx]
                },
                method="Hybrid",
                search_time=search_time
            )
            for score, idx in zip(top_k_scores, top_k_idx)
        ]

    def evaluate_hnsw(self, query: str, k: int = 3) -> List[SearchResult]:
        """評估 HNSW 搜尋"""
        start_time = time.perf_counter()
        
        # 建立 HNSW 索引
        index = hnswlib.Index(space='cosine', dim=self.vector_dim)
        index.init_index(
            max_elements=len(self.documents),
            ef_construction=200,  # 建構時的搜尋深度
            M=16  # 每個節點的最大鄰居數
        )
        
        # 添加向量
        index.add_items(self.doc_vectors)
        
        # 設定搜尋參數
        index.set_ef(50)  # 搜尋時的深度
        
        # 搜尋
        query_vector = self._compute_embeddings([query])[0]
        scores, indices = index.knn_query(query_vector, k=k)
        
        search_time = time.perf_counter() - start_time
        
        return [
            SearchResult(
                content=self.documents[idx],
                score=1 - score/2,  # 將距離轉換為相似度
                metadata={"index": idx},
                method="HNSW",
                search_time=search_time
            )
            for score, idx in zip(scores[0], indices[0])
        ]
    
    def evaluate_bm25(self, query: str, k: int = 3) -> List[SearchResult]:
        """評估純 BM25 搜尋"""
        start_time = time.perf_counter()
        
        # 建立 BM25 模型
        tokenized_docs = [doc.split() for doc in self.documents]
        bm25 = BM25Okapi(tokenized_docs)
        
        # 搜尋
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        
        # 取得前 k 個結果
        top_k_idx = np.argsort(-scores)[:k]
        top_k_scores = scores[top_k_idx]
        
        search_time = time.perf_counter() - start_time
        
        return [
            SearchResult(
                content=self.documents[idx],
                score=score,
                metadata={"index": idx},
                method="BM25",
                search_time=search_time
            )
            for score, idx in zip(top_k_scores, top_k_idx)
        ]

def compare_search_methods(documents: List[str], queries: List[str], 
                         relevance: List[List[int]]) -> pd.DataFrame:
    """比較不同搜尋方法的效果"""
    logger.info("開始評估搜尋方法")
    evaluator = SearchMethodEvaluator()
    vector_evaluator = VectorSearchEvaluator(documents)
    
    # 定義要評估的方法
    search_methods = {
        # "FAISS": vector_evaluator.evaluate_faiss,
        "HNSW": vector_evaluator.evaluate_hnsw,
        "BM25": vector_evaluator.evaluate_bm25,
        # "Hybrid": vector_evaluator.evaluate_hybrid
    }
    
    method_descriptions = {
        "FAISS": "基於向量近似最近鄰的快速檢索",
        "HNSW": "基於分層導航小世界圖的向量檢索",
        "BM25": "基於詞頻與文檔長度的檢索模型",
        "Hybrid": "結合 BM25 與向量檢索的混合模型"
    }
    
    # 初始化結果列表
    evaluation_results = []
    
    # 對每個查詢評估所有方法
    for query_idx, (query, rel) in enumerate(zip(queries, relevance), 1):
        logger.info(f"\n評估查詢 {query_idx}/{len(queries)}: {query}")
        
        for method_name, search_func in search_methods.items():
            logger.info(f"使用 {method_name} 方法搜尋")
            
            # 執行搜尋
            try:
                results = search_func(query)
                result_indices = [r.metadata["index"] for r in results]
                
                # 計算評估指標
                metrics = evaluator.evaluate_search_method(
                    method_name=method_name,
                    search_results=result_indices,
                    relevance=rel,
                    search_time=results[0].search_time,
                    documents=documents
                )
                
                logger.info(
                    f"{method_name} 搜尋結果:\n"
                    f"- 查詢時間: {metrics.search_time*1000:.2f}ms\n"
                    f"- 準確率@K: {metrics.precision_at_k:.3f}\n"
                    f"- 召回率@K: {metrics.recall_at_k:.3f}\n"
                    f"- MRR: {metrics.mrr:.3f}\n"
                    f"- NDCG: {metrics.ndcg:.3f}\n"
                    f"- 多樣性: {metrics.diversity:.3f}"
                )
                
                # 添加到結果列表
                evaluation_results.append({
                    "方法": metrics.name,
                    "查詢時間(ms)": round(metrics.search_time * 1000, 2),
                    "準確率@K": round(metrics.precision_at_k, 3),
                    "召回率@K": round(metrics.recall_at_k, 3),
                    "MRR": round(metrics.mrr, 3),
                    "NDCG": round(metrics.ndcg, 3),
                    "多樣性": round(metrics.diversity, 3),
                    "說明": method_descriptions.get(method_name, "未知搜尋方法")
                })
                
            except Exception as e:
                logger.error(f"{method_name} 評估失敗: {str(e)}", exc_info=True)
                continue
    
    # 轉換為 DataFrame 並計算平均值
    results_df = pd.DataFrame(evaluation_results)
    average_metrics = results_df.groupby("方法").agg({
        "查詢時間(ms)": "mean",
        "準確率@K": "mean",
        "召回率@K": "mean",
        "MRR": "mean",
        "NDCG": "mean",
        "多樣性": "mean"
    }).round(3)
    
    logger.info("\n評估完成，平均指標:")
    logger.info("\n" + str(average_metrics))
    
    return average_metrics

def main():
    """主程式：執行搜尋方法評估"""
    logger.info("開始執行搜尋方法評估")
    try:
        documents, queries, relevance = prepare_test_data()
        logger.info(f"載入測試數據: {len(documents)} 文檔, {len(queries)} 查詢")
        
        # 評估所有方法
        print("\n=== 搜尋方法評估結果 ===")
        average_metrics = compare_search_methods(documents, queries, relevance)
        print("\n平均評估指標:")
        print(average_metrics)
        
        logger.info("評估完成")
        
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
