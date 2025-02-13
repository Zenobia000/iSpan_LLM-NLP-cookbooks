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
import torch
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings, HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


from sklearn.manifold import TSNE
from sklearn.metrics import ndcg_score

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    avg_similarity: float
    ndcg_score: float

class EmbeddingEvaluator:
    """Embedding 模型評估器"""
    def __init__(self, model_list: dict):
        self.models = model_list
        self.metrics = []

    def prepare_test_data(self) -> tuple:
        """準備測試數據，包含查詢與標準答案 (ground truth)"""
        queries = [
            "What is the capital of France?",
            "Who discovered gravity?",
            "How does blockchain work?",
            "Explain quantum mechanics in simple terms.",
            "What are the symptoms of COVID-19?"
        ]
        ground_truths = [
            ["Paris is the capital of France.", "Paris is located in Europe.", "The capital of France is Paris."],
            ["Isaac Newton discovered gravity.", "Gravity was formulated by Newton.", "Newton introduced the law of gravitation."],
            ["Blockchain is a decentralized ledger technology.", "Blockchain ensures secure transactions.", "Cryptocurrency uses blockchain for security."],
            ["Quantum mechanics describes the behavior of subatomic particles.", "Quantum mechanics is fundamental to physics.", "Electrons follow quantum mechanical principles."],
            ["Common symptoms include fever, cough, and difficulty breathing.", "COVID-19 symptoms vary but often include respiratory issues.", "Fever and shortness of breath are signs of COVID-19."]
        ]

        return queries, ground_truths

    def evaluate_ndcg(self, model, queries, ground_truths):
        """計算 NDCG 分數"""
        scores = []
        for query, truth in zip(queries, ground_truths):
            retrieved_docs = model.embed_documents([query] + truth)  # 查詢 + 多個 ground truth
            if len(truth) < 2:
                logger.warning(f"NDCG 計算被跳過，因為 {query} 只有 {len(truth)} 個 ground truth.")
                continue
            sim_scores = np.inner(retrieved_docs[0], retrieved_docs[1:]) / (
                np.linalg.norm(retrieved_docs[0]) * np.linalg.norm(retrieved_docs[1:], axis=1)
            )
            ndcg = ndcg_score([np.ones(len(truth))], [sim_scores])
            scores.append(ndcg)
        return np.mean(scores) if scores else 0.0


    def evaluate_model(self, model_name: str, model, queries, ground_truths) -> EmbeddingMetrics:
        """統一評估不同嵌入模型"""
        try:
            start_time = time.time()
            vectors = model.embed_documents(queries + [x[0] for x in ground_truths])
            embedding_time = time.time() - start_time

            # 計算 GPU & 記憶體使用
            gpu_usage = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

            # 計算語意一致性 (Cosine Similarity 平均值)
            sim_matrix = np.inner(vectors, vectors) / (np.linalg.norm(vectors, axis=1)[:, None] * np.linalg.norm(vectors, axis=1))
            avg_similarity = np.mean(sim_matrix)

            # 計算檢索效能 NDCG
            ndcg_score_value = self.evaluate_ndcg(model, queries, ground_truths)

            # 模型規格
            vector_dimension = len(max(vectors, key=len))
            cost_per_1k = 0.0001 if "OpenAI" in model_name or "Cohere" in model_name else 0
            local_deployment = "HF" in model_name or "BGE" in model_name

            return EmbeddingMetrics(
                name=model_name,
                embedding_time=embedding_time,
                vector_dimension=vector_dimension,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                cost_per_1k=cost_per_1k,
                local_deployment=local_deployment,
                batch_support=True,
                multilingual="BGE" in model_name or "Cohere" in model_name,
                avg_similarity=avg_similarity,
                ndcg_score=ndcg_score_value
            )
        except Exception as e:
            logger.error(f"{model_name} 評估失敗: {str(e)}")
            raise

    def run_experiments(self):
        """執行所有模型評估"""
        queries, ground_truths = self.prepare_test_data()
        for name, model in self.models.items():
            self.metrics.append(self.evaluate_model(name, model, queries, ground_truths))

        df = pd.DataFrame([vars(e) for e in self.metrics])
        print("\n=== Embedding 模型比較結果 ===")
        print(df)

        self.visualize_embeddings()
        return df

    def visualize_embeddings(self):
        """🔹 t-SNE, Cosine Similarity Heatmap, Latency vs Accuracy"""
        try:
            queries, _ = self.prepare_test_data()
            vectors = {}
            for e in self.metrics:
                vecs = self.models[e.name].embed_documents(queries)
                vectors[e.name] = np.array(vecs)

            # 1️⃣ t-SNE 可視化
            plt.figure(figsize=(8, 6))
            for model_name, vecs in vectors.items():
                reduced_vecs = TSNE(n_components=2, random_state=42, perplexity=min(len(queries)-1, 5)).fit_transform(vecs)
                plt.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], label=model_name, alpha=0.6)
            plt.legend()
            plt.title("t-SNE Embedding Visualization")
            plt.show(block=True)

            # 2️⃣ Cosine Similarity Heatmap for all models
            fig, axes = plt.subplots(1, len(vectors), figsize=(12, 4))
            for ax, (model_name, vecs) in zip(axes, vectors.items()):
                sim_matrix = np.inner(vecs, vecs) / (np.linalg.norm(vecs, axis=1)[:, None] * np.linalg.norm(vecs, axis=1))
                sns.heatmap(sim_matrix, ax=ax, cmap="coolwarm", cbar=False)
                ax.set_title(f"Cosine Similarity ({model_name})")
            plt.show(block=True)

            # 3️⃣ Latency vs Accuracy
            plt.figure(figsize=(8, 6))
            latencies = [e.embedding_time for e in self.metrics]
            accuracies = [e.avg_similarity for e in self.metrics]
            names = [e.name for e in self.metrics]
            plt.scatter(latencies, accuracies)
            for i, name in enumerate(names):
                plt.annotate(name, (latencies[i], accuracies[i]))
            plt.xlabel("Latency (sec)")
            plt.ylabel("Semantic Similarity Score")
            plt.title("Latency vs. Accuracy Trade-off")
            plt.show(block=True)

        except Exception as e:
            logger.error(f"視覺化過程發生錯誤: {str(e)}")
            raise

def main():
    """主程式"""
    print("=== LangChain Embedding 模型比較 ===\n")
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return
    models = {
        "OpenAI Ada 002": OpenAIEmbeddings(),
        "HF MiniLM-L6": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        "BGE Large": HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    }
    evaluator = EmbeddingEvaluator(model_list=models)
    return evaluator.run_experiments()

if __name__ == "__main__":
    result = main()
