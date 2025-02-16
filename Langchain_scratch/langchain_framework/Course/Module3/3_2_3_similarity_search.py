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
    """準備測試資料集
    
    Returns:
        Tuple[List[str], List[str], List[List[int]]]: 
            (文檔列表, 查詢列表, 相關度矩陣)
    """
    # 文檔集合：不同長度、主題和複雜度
    documents = [
        # 科技類 (0-7)
        "台積電是全球最大的晶圓代工企業，總部位於新竹科學園區。公司專注於先進製程技術研發，在3奈米、5奈米等製程居於領先地位。",
        "蘋果公司新款iPhone採用台積電的4奈米製程晶片，具備更強大的運算能力和更低的功耗。預計年底在台灣量產。",
        "聯發科發表最新5G晶片，採用先進製程，整合AI運算單元，主打中高階手機市場。預計在第三季開始出貨。",
        "華碩推出新一代電競筆電，搭載最新的NVIDIA RTX顯示卡，採用獨特的散熱設計，螢幕更新率達360Hz。",
        "大立光是全球最大的手機鏡頭供應商，為iPhone等高階手機提供光學元件。公司持續投資研發，布局AR/VR領域。",
        "群創光電發表全球首款摺疊式OLED面板，可向內外摺疊，解析度達4K，將用於下一代摺疊手機。",
        "研華科技推出工業物聯網平台，整合邊緣運算和雲端服務，協助製造業實現智慧化轉型。",
        "緯創資通在台中設立新廠，主要生產伺服器和網路設備，預計創造上千個就業機會。",
        
        # 旅遊類 (8-15)
        "台北101是台灣最高的摩天大樓，高達509.2公尺。大樓內有觀景台、購物中心和米其林餐廳。每年跨年煙火是重要景點。",
        "日月潭是台灣最大的淡水湖泊，環湖步道景色優美。春季可以看櫻花，夏季可以遊湖，秋季可以賞楓葉，冬季可以觀星。",
        "阿里山森林遊樂區以日出、雲海、森林鐵路、巨木和晚霞聞名。每年三月賞櫻季是最熱門的旅遊時節。",
        "太魯閣國家公園以峽谷地形著稱，清水斷崖高聳入雲，砂卡礑步道沿著溪流蜿蜒，是台灣最受歡迎的國家公園之一。",
        "九份老街保留完整的日治時期建築，茶樓、紅燈籠與石階街道，充滿懷舊氛圍。夜景尤其迷人，常吸引大量觀光客。",
        "墾丁國家公園擁有潔白的沙灘和湛藍的海水，是台灣最南端的度假勝地。每年春天的音樂節吸引許多年輕人參加。",
        "七星潭是花蓮最著名的景點之一，彎月形的海灣佈滿黑色卵石，清晨可以欣賞日出，傍晚可以漫步海岸。",
        "陽明山國家公園以溫泉、花季和火山地形聞名，大屯山、七星山等火山群環繞，春天的杜鵑花季最為壯觀。",
        
        # 美食類 (16-23)
        "鼎泰豐的小籠包在米其林指南獲得推薦，以18摺的完美褶皺聞名，湯汁豐富，外皮有嚼勁。",
        "臭豆腐是台灣夜市必吃小吃，外酥內嫩，搭配泡菜和蒜蓉辣醬最對味。以大腸麵線和珍珠奶茶聞名。",
        "牛肉麵的湯頭以紅燒和清燉為主，牛肉軟嫩，麵條Q彈。台北市舉辦的牛肉麵節是年度美食盛事。",
        "度小月擔仔麵是台南百年老店，以乾麵為主，配上新鮮蝦仁，湯頭鮮美，是台南必吃小吃。",
        "阿宗麵線是台北西門町的人氣小吃，以大腸麵線聞名，湯頭濃郁，加上香菜和黑醋更添風味。",
        "雪王冰淇淋是台南老字號，以水果冰淇淋聞名，芒果、荔枝等口味都使用新鮮水果製作。",
        "高雄六合夜市的海鮮粥使用新鮮海產，香菇、蝦仁、魚肉豐富，是高雄必吃美食。",
        "基隆廟口夜市的天婦羅、藥燉排骨、鹹酥雞都是招牌小吃，在地人從小吃到大。",
        
        # 文化類 (24-31)
        "故宮博物院收藏大量中華文物，包括翠玉白菜、毛公鼎等國寶。每年吸引數百萬遊客參觀。定期舉辦特展。",
        "台北當代藝術館位於舊市政廳，定期展出現代藝術作品，推廣台灣當代藝術，舉辦藝術教育活動。",
        "林家花園是板橋知名古蹟，建於清朝，是台灣最完整的園林建築，體現傳統閩南建築特色。",
        "十鼓仁糖文創園區由廢棄糖廠改建，結合擊鼓表演與工業遺址，展現台灣文創轉型的成功案例。",
        "鶯歌陶瓷博物館展示台灣陶瓷發展史，提供陶藝DIY體驗，是認識台灣陶瓷文化的重要場所。",
        "台南孔廟是台灣最古老的孔廟，建於1665年，每年舉行祭孔大典，展現傳統文化儀式。",
        "蘭陽博物館以獨特的建築設計聞名，展示宜蘭的人文歷史，是台灣新型態博物館的代表。",
        "台灣文學館位於台南古蹟建築內，收藏豐富的台灣文學史料，定期舉辦文學講座和展覽。"
    ]
    
    # 查詢及其預期相關文檔
    queries = [
        "台灣最先進的半導體製程技術",  # 科技類查詢
        "最新的顯示器技術發展",        # 科技類查詢
        "台北最受歡迎的觀光景點",      # 旅遊類查詢
        "台灣最美的自然風景區",        # 旅遊類查詢
        "台灣最有名的小吃有哪些",      # 美食類查詢
        "台南最具特色的美食",          # 美食類查詢
        "哪裡可以看到珍貴文物",        # 文化類查詢
        "台灣最具代表性的古蹟"         # 文化類查詢
    ]
    
    # 相關度矩陣（0-2分，2分最相關）
    relevance = [
        # 半導體查詢的相關度 "台灣最先進的半導體製程技術"
        [2, 2, 1, 0, 1, 0, 0, 0,  # 科技類 (0-7)
         0, 0, 0, 0, 0, 0, 0, 0,  # 旅遊類 (8-15)
         0, 0, 0, 0, 0, 0, 0, 0,  # 美食類 (16-23)
         0, 0, 0, 0, 0, 0, 0, 0], # 文化類 (24-31)
        
        # 顯示器查詢的相關度 "最新的顯示器技術發展"
        [0, 0, 0, 1, 0, 2, 0, 0,  # 科技類
         0, 0, 0, 0, 0, 0, 0, 0,  # 旅遊類
         0, 0, 0, 0, 0, 0, 0, 0,  # 美食類
         0, 0, 0, 0, 0, 0, 0, 0], # 文化類
        
        # 台北景點查詢的相關度 "台北最受歡迎的觀光景點"
        [0, 0, 0, 0, 0, 0, 0, 0,  # 科技類
         2, 0, 0, 0, 1, 0, 0, 2,  # 旅遊類
         0, 0, 0, 0, 0, 0, 0, 0,  # 美食類
         0, 0, 0, 0, 0, 0, 0, 0], # 文化類
        
        # 自然風景查詢的相關度 "台灣最美的自然風景區"
        [0, 0, 0, 0, 0, 0, 0, 0,  # 科技類
         0, 2, 2, 2, 0, 1, 1, 1,  # 旅遊類
         0, 0, 0, 0, 0, 0, 0, 0,  # 美食類
         0, 0, 0, 0, 0, 0, 0, 0], # 文化類
        
        # 台灣小吃查詢的相關度 "台灣最有名的小吃有哪些"
        [0, 0, 0, 0, 0, 0, 0, 0,  # 科技類
         0, 0, 0, 0, 0, 0, 0, 0,  # 旅遊類
         2, 2, 1, 1, 2, 1, 1, 2,  # 美食類
         0, 0, 0, 0, 0, 0, 0, 0], # 文化類
        
        # 台南美食查詢的相關度 "台南最具特色的美食"
        [0, 0, 0, 0, 0, 0, 0, 0,  # 科技類
         0, 0, 0, 0, 0, 0, 0, 0,  # 旅遊類
         0, 0, 0, 2, 0, 2, 0, 0,  # 美食類
         0, 0, 0, 0, 0, 0, 0, 0], # 文化類
        
        # 文物查詢的相關度 "哪裡可以看到珍貴文物"
        [0, 0, 0, 0, 0, 0, 0, 0,  # 科技類
         0, 0, 0, 0, 0, 0, 0, 0,  # 旅遊類
         0, 0, 0, 0, 0, 0, 0, 0,  # 美食類
         2, 1, 1, 0, 1, 1, 1, 0], # 文化類
        
        # 古蹟查詢的相關度 "台灣最具代表性的古蹟"
        [0, 0, 0, 0, 0, 0, 0, 0,  # 科技類
         0, 0, 0, 0, 0, 0, 0, 0,  # 旅遊類
         0, 0, 0, 0, 0, 0, 0, 0,  # 美食類
         1, 0, 2, 1, 0, 2, 0, 1]  # 文化類
    ]
    
    return documents, queries, relevance

class SearchMethodEvaluator:
    """相似度搜尋方法評估器"""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # 將 method_descriptions 移到類別內部
        self.method_descriptions = {
            "FAISS": "基於向量近似最近鄰的快速檢索",
            "HNSW": "基於分層導航小世界圖的向量檢索",
            "BM25": "基於詞頻與文檔長度的檢索模型",
            "Hybrid": "結合 BM25 與向量檢索的混合模型"
        }

    def calculate_precision(self, results: List[int], relevance: List[int], k: int) -> float:
        """計算準確率，考慮不同相關度級別
        
        Args:
            results: 搜尋結果的索引列表
            relevance: 相關度列表（0-2分）
            k: 評估的結果數量
            
        Returns:
            float: 正規化的準確率（0-1）
        """
        max_score = 2.0  # 最高相關度分數
        actual_score = sum(relevance[i] for i in results[:k])
        max_possible = k * max_score  # k個結果的最高可能分數
        return actual_score / max_possible if max_possible > 0 else 0.0

    def calculate_recall(self, results: List[int], relevance: List[int], k: int) -> float:
        """計算召回率，考慮不同相關度級別
        
        Args:
            results: 搜尋結果的索引列表
            relevance: 相關度列表（0-2分）
            k: 評估的結果數量
            
        Returns:
            float: 正規化的召回率（0-1）
        """
        total_relevant_score = sum(relevance)  # 所有文檔的相關度總分
        if total_relevant_score == 0:
            return 0.0
        
        retrieved_score = sum(relevance[i] for i in results[:k])
        return retrieved_score / total_relevant_score

    def calculate_mrr(self, results: List[int], relevance: List[int]) -> float:
        """計算 MRR，考慮不同相關度級別
        
        Args:
            results: 搜尋結果的索引列表
            relevance: 相關度列表（0-2分）
            
        Returns:
            float: MRR 分數（0-1）
        """
        # 找到第一個相關（分數>0）的文檔位置
        for rank, doc_id in enumerate(results):
            if relevance[doc_id] > 0:
                # 根據相關度調整 MRR
                rel_score = relevance[doc_id]
                return (rel_score / 2.0) * (1.0 / (rank + 1))
        return 0.0

    def calculate_ndcg(self, results: List[int], relevance: List[int], k: int) -> float:
        """計算 NDCG，考慮不同相關度級別
        
        Args:
            results: 搜尋結果的索引列表
            relevance: 相關度列表（0-2分）
            k: 評估的結果數量
            
        Returns:
            float: NDCG 分數（0-1）
        """
        # 計算 DCG
        dcg = sum(
            (relevance[i] / 2.0) / np.log2(rank + 2)  # 正規化相關度到 0-1
            for rank, i in enumerate(results[:k])
        )
        
        # 計算理想 DCG（將相關度排序）
        ideal_results = sorted(range(len(relevance)), 
                             key=lambda i: relevance[i], 
                             reverse=True)
        idcg = sum(
            (relevance[i] / 2.0) / np.log2(rank + 2)
            for rank, i in enumerate(ideal_results[:k])
        )
        
        return dcg / idcg if idcg > 0 else 0.0

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

    def evaluate_search_method(self, method_name: str, search_results: List[int], 
                             relevance: List[int], search_time: float, documents: List[str]) -> SearchMetrics:
        """評估搜尋方法的效果"""
        k = 5  # 評估前 k 個結果
        precision = self.calculate_precision(search_results, relevance, k)
        recall = self.calculate_recall(search_results, relevance, k)
        mrr = self.calculate_mrr(search_results, relevance)
        ndcg = self.calculate_ndcg(search_results, relevance, k)
        diversity = self.calculate_diversity(search_results, documents)
        
        return SearchMetrics(
            name=method_name,
            search_time=search_time,
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg=ndcg,
            diversity=diversity,
            explanation=self.method_descriptions.get(method_name, "未知搜尋方法")
        )

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
        try:
            # 初始化 HNSW 索引
            dim = self.vector_dim
            num_elements = len(self.documents)
            
            # 調整 HNSW 參數
            index = hnswlib.Index(space='cosine', dim=dim)
            index.init_index(
                max_elements=num_elements,
                ef_construction=200,  # 增加建構時的精確度
                M=16,                # 每個節點的最大連接數
            )
            
            # 添加文檔向量
            index.add_items(self.doc_vectors)
            
            # 設定搜尋參數
            index.set_ef(num_elements)  # 增加搜尋時的精確度
            
            # 搜尋
            query_vector = self._compute_embeddings([query])[0]
            scores, indices = index.knn_query(query_vector.reshape(1, -1), k=k)
            
            search_time = time.perf_counter() - start_time
            
            # 正規化分數到 0-1 範圍
            normalized_scores = 1 - scores[0]  # 因為使用 cosine 距離，轉換為相似度
            
            # 將索引轉換為整數類型
            indices = indices[0].astype(int)
            
            return [
                SearchResult(
                    content=self.documents[idx],
                    score=score,
                    metadata={"index": int(idx)},  # 確保索引是整數
                    method="HNSW",
                    search_time=search_time
                )
                for score, idx in zip(normalized_scores, indices)
            ]
        except Exception as e:
            logger.error(f"HNSW 搜尋失敗: {str(e)}", exc_info=True)
            raise
    
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
        "FAISS": vector_evaluator.evaluate_faiss,
        "HNSW": vector_evaluator.evaluate_hnsw,
        "BM25": vector_evaluator.evaluate_bm25,
        "Hybrid": vector_evaluator.evaluate_hybrid
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
                
                # logger.info(
                #     f"{method_name} 搜尋結果:\n"
                #     f"- 查詢時間: {metrics.search_time*1000:.2f}ms\n"
                #     f"- 準確率@K: {metrics.precision_at_k:.3f}\n"
                #     f"- 召回率@K: {metrics.recall_at_k:.3f}\n"
                #     f"- MRR: {metrics.mrr:.3f}\n"
                #     f"- NDCG: {metrics.ndcg:.3f}\n"
                #     f"- 多樣性: {metrics.diversity:.3f}"
                # )
                
                # 添加到結果列表
                evaluation_results.append({
                    "Method": metrics.name,  # 改用英文欄位名
                    "QueryTime": round(metrics.search_time * 1000, 2),
                    "Precision": round(metrics.precision_at_k, 3),
                    "Recall": round(metrics.recall_at_k, 3),
                    "MRR": round(metrics.mrr, 3),
                    "NDCG": round(metrics.ndcg, 3),
                    "Diversity": round(metrics.diversity, 3),
                    "Description": metrics.explanation
                })
                
            except Exception as e:
                logger.error(f"{method_name} 評估失敗: {str(e)}", exc_info=True)
                continue
    
    # 轉換為 DataFrame 並計算平均值
    results_df = pd.DataFrame(evaluation_results)
    if not results_df.empty:
        average_metrics = results_df.groupby("Method").agg({
            "QueryTime": "mean",
            "Precision": "mean",
            "Recall": "mean",
            "MRR": "mean",
            "NDCG": "mean",
            "Diversity": "mean"
        }).round(3)
        
        logger.info("\n評估完成，平均指標:")
        # logger.info("\n" + str(average_metrics))
        
        return average_metrics
    else:
        logger.warning("沒有成功的評估結果")
        return pd.DataFrame()

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
