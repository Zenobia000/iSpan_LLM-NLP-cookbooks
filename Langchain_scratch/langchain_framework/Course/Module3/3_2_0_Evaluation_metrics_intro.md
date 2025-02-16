# 語義準確性與排名評估指標於 LLM RAG 檢索技術

## 1. 引言
隨著 **大規模語言模型（Large Language Models, LLM）** 的應用擴展，檢索增強生成（Retrieval-Augmented Generation, RAG）技術成為提升語言模型準確性與知識能力的關鍵方法。為了評估 RAG 系統的檢索效能，常見的衡量指標包括：
- **語義準確性（Semantic Accuracy）**
- **平均排名倒數（Mean Reciprocal Rank, MRR）**
- **正規化累積折減增益（Normalized Discounted Cumulative Gain, NDCG）**
- **精度（Precision@K）**
- **召回率（Recall@K）**

本章節將以 **理論與實務範例** 說明這些指標的計算方式與在 LLM RAG 檢索技術中的應用。

---

## 2. 語義準確性（Semantic Accuracy）
語義準確性評估檢索結果與查詢語義的匹配程度，主要依賴：
- **查詢與文件的相似性**（基於向量表示）
- **排名結果的相關性**（基於人為標註或自動標註）

### **2.1 語義準確性的衡量方式**
常見的語義準確性評估方法：
1. **Cosine Similarity（餘弦相似度）**  
   - 計算查詢與文件的向量餘弦相似度：
     $$
     \text{Cosine Similarity} = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \|\mathbf{d}\|}
     $$
     其中，\( \mathbf{q} \) 為查詢向量，\( \mathbf{d} \) 為文件向量。

2. **BM25（Best Matching 25）**  
   - 一種加權檢索模型，用於評估文件與查詢關聯性：
     $$
     \text{BM25}(d, q) = \sum_{t \in q} IDF(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
     $$
     其中：
     - \( IDF(t) \) 為逆文件頻率
     - \( f(t, d) \) 為詞頻
     - \( k_1, b \) 為調節參數

3. **Embedding Similarity（向量嵌入相似性）**  
   - 使用 Transformer-based LLM 生成查詢與文件嵌入，計算內積或 L2 距離。

---

## 3. 平均排名倒數（Mean Reciprocal Rank, MRR）
### **3.1 MRR 指標定義**
MRR 用於衡量檢索系統是否能在較高排名提供相關結果：
$$
MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{rank_i}
$$
其中：
- \( N \) 為測試查詢總數
- \( rank_i \) 為第 \( i \) 個查詢的第一個相關結果的排名

### **3.2 MRR 計算範例**
假設我們有一個搜尋系統，輸入查詢後，它會返回一組文件列表。我們假設有 3 個查詢（queries），每個查詢的 Top-5 檢索結果如下：

| Query ID | 檢索結果（文件 ID） | 相關性分數（Ground Truth） |
|------|------------------|------------------|
| Q1   | D3, D2, D5, D4, D1 | [0, 1, 0, 1, 0] |
| Q2   | D1, D4, D2, D3, D5 | [1, 0, 0, 0, 1] |
| Q3   | D2, D3, D4, D5, D1 | [0, 0, 1, 0, 0] |

MRR 計算：

| Query ID | 第一個相關文件的排名 | Reciprocal Rank |
|------|------------------|------------------|
| Q1   | 2               | 1/2 = 0.5        |
| Q2   | 1               | 1/1 = 1.0        |
| Q3   | 3               | 1/3 ≈ 0.333      |

$$
MRR = \frac{1}{3} \left( \frac{1}{2} + \frac{1}{1} + \frac{1}{3} \right) = \frac{1.833}{3} = 0.611
$$

---

## 4. 正規化累積折減增益（NDCG@K）
### **4.1 NDCG 計算範例**
計算 DCG（Discounted Cumulative Gain）：
$$
DCG@K = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i+1)}
$$

計算 IDCG（Ideal DCG）：
$$
IDCG@K = \sum_{i=1}^{K} \frac{rel_i^{ideal}}{\log_2(i+1)}
$$

計算 NDCG：
$$
NDCG@K = \frac{DCG@K}{IDCG@K}
$$

計算結果示例如下：

| Query | DCG@5 | IDCG@5 | NDCG@5 |
|------|-------|--------|--------|
| Q1   | 1.5   | 1.63   | 0.920  |
| Q2   | 1.43  | 1.63   | 0.877  |
| Q3   | 0.5   | 1.0    | 0.500  |

$$
\text{Mean NDCG@5} = \frac{0.920 + 0.877 + 0.500}{3} = 0.766
$$

---

## 5. Precision@K
### **5.1 Precision@K 定義**
Precision@K 衡量前 K 個檢索結果中相關文件的比例：
$$
Precision@K = \frac{\text{Relevant Documents in Top-K}}{K}
$$

### **5.2 計算範例**
假設我們的檢索系統返回的結果如下：

| Query ID | 檢索結果（Top-5） | 相關性分數 |
|----------|------------------|------------|
| Q1       | D3, D2, D5, D4, D1 | [0, 1, 0, 1, 0] |
| Q2       | D1, D4, D2, D3, D5 | [1, 0, 0, 0, 1] |
| Q3       | D2, D3, D4, D5, D1 | [0, 0, 1, 0, 0] |

計算 Precision@3：

| Query ID | 相關文件數 (Top-3) | Precision@3 |
|----------|------------------|-------------|
| Q1       | 1               | 1/3 = 0.333  |
| Q2       | 1               | 1/3 = 0.333  |
| Q3       | 1               | 1/3 = 0.333  |

$$
\text{Mean Precision@3} = \frac{0.333 + 0.333 + 0.333}{3} = 0.333
$$

---

## 6. Recall@K
### **6.1 Recall@K 定義**
Recall@K 衡量前 K 個檢索結果覆蓋到的相關文件比例：
$$
Recall@K = \frac{\text{Relevant Documents in Top-K}}{\text{Total Relevant Documents}}
$$

### **6.2 計算範例**
使用與 Precision@K 相同的檢索結果：

| Query ID | 總相關文件數 | 相關文件數 (Top-3) | Recall@3 |
|----------|------------|------------------|----------|
| Q1       | 2          | 1                | 1/2 = 0.5  |
| Q2       | 2          | 1                | 1/2 = 0.5  |
| Q3       | 1          | 1                | 1/1 = 1.0  |

$$
\text{Mean Recall@3} = \frac{0.5 + 0.5 + 1.0}{3} = 0.667
$$



