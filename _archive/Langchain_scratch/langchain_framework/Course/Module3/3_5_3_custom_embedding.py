"""
LangChain 自定義 Embedding 模型範例

展示如何建立自定義的 Embedding 模型，包括：
1. 基於 CLIP 的多模態 Embedding
2. 文本與圖像的相似度計算
"""

import os
import logging
from typing import List
import numpy as np
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 確保環境變數正確載入
load_dotenv()

class MultiModalEmbedding(Embeddings):
    """使用 CLIP 進行文本與圖像的對齊學習"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        logger.info(f"已載入 CLIP 模型 {model_name} 到 {device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量處理文本"""
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            return text_features.cpu().numpy().tolist()

    def embed_query(self, text: str) -> List[float]:
        """處理單個查詢文本"""
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            return text_features[0].cpu().numpy().tolist()

    def embed_image(self, image_path: str) -> List[float]:
        """處理圖像"""
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
                image_features = self.model.get_image_features(**inputs)
                return image_features[0].cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"圖像 Embedding 生成失敗: {str(e)}")
            return []

    def compute_similarity(self, text_embedding: List[float], image_embedding: List[float]) -> float:
        """計算文本與圖像的相似度"""
        return float(cosine_similarity([text_embedding], [image_embedding])[0][0])

def main():
    query = "A cup of coffee"  # 使用英文以增強匹配效果
    image_path = "Course/Module3/images/coffee.jpg"  # 生活化範例
    
    try:
        multimodal_embedding = MultiModalEmbedding()
        
        logger.info("測試文本與圖像 Embedding...")
        query_vector = multimodal_embedding.embed_query(query)
        print(f"文本 Embedding 維度: {len(query_vector)}")
        
        if os.path.exists(image_path):
            image_vector = multimodal_embedding.embed_image(image_path)
            if image_vector:
                print(f"圖像 Embedding 維度: {len(image_vector)}")
                similarity_score = multimodal_embedding.compute_similarity(query_vector, image_vector)
                print(f"文本與圖像的相似度: {similarity_score:.4f}")
            else:
                print("無法獲取圖像 Embedding")
    
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main()
