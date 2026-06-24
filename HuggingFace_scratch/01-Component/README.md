# 01-Component：HuggingFace 六大核心元件

> 模組定位：本模組是整套 cookbook 的地基。學完之後，你會掌握 HuggingFace 生態系裡「從文字到模型輸出」這條管線上的六個可替換零件，並理解它們各自負責什麼、彼此如何銜接。後續所有進階任務（02-Adv-tasks）、PEFT 微調（03-PEFT）、量化微調（04-kbits-tuning）乃至多模態（05-Multimodal）都是這六個零件的重新組合。

相關文件：

- 上層導覽：[../README.md](../README.md)
- 環境建置（先做）：[../00-Setup-and-Foundations/00-environment-setup.md](../00-Setup-and-Foundations/00-environment-setup.md)
- 2026 慣例聖經（強烈建議搭配閱讀）：[../00-Setup-and-Foundations/01-2026-conventions.md](../00-Setup-and-Foundations/01-2026-conventions.md)
- 術語與架構速查：[../00-Setup-and-Foundations/02-glossary-and-architectures.md](../00-Setup-and-Foundations/02-glossary-and-architectures.md)
- 下一站：[../02-Adv-tasks/README.md](../02-Adv-tasks/README.md)

---

## 一、為什麼要「拆元件」來學

剛接觸 HuggingFace 的人常常只記得一行 `pipeline("text-classification")` 就能跑出結果，但一旦遇到「我想換模型」「我想算 F1」「我想自己微調」就卡住，因為他不知道 `pipeline` 內部到底發生了什麼。

`pipeline` 不是魔法，它只是把四件事串起來的便利封裝：

```
原始文字
  → tokenizer  把字串切成 token、轉成 input_ids / attention_mask
  → model      前向傳播（forward），輸出 logits
  → 後處理     softmax / argmax，把 logits 變成人看得懂的標籤與分數
  → 結果
```

當你訓練模型時，這條鏈會多兩個零件：

```
datasets   負責把資料載入、切分、批次化（取代手刻 Dataset / collate_fn）
evaluate   負責把模型輸出與標準答案比對，算出 accuracy / F1 等指標
Trainer    把「前向 → 算 loss → 反向 → 更新權重 → 存檔 → 評估」整條訓練迴圈封裝起來
```

理解這個拆解，是本模組唯一真正重要的事。記住：**所有元件都是可替換的**——把 `tokenizer` 換成 `AutoProcessor`、把 `input_ids` 換成 `pixel_values`，同一條管線就從文字延伸到影像（見 [05-Multimodal](../05-Multimodal/README.md)）。

---

## 二、學習順序與 notebook 清單

請依照 01 → 06 的順序學習。前三個（pipeline / tokenizer / model）是「推論側」的理解，後三個（datasets / evaluate / trainer）是「訓練側」的工具。

| 順序 | 子目錄 | Notebook | 學什麼（一句話） |
| :--- | :--- | :--- | :--- |
| 01 | `01pipeline/` | [01.pipeline.ipynb](01pipeline/01.pipeline.ipynb) | `pipeline` 高階抽象：任務選擇、裝置管理、推論流程；首次接觸 `AutoProcessor`（多模態前置） |
| 02 | `02tokenizer/` | `02.tokenizer.ipynb_` | tokenizer 三件套（`input_ids` / `attention_mask` / `token_type_ids`）、特殊 token、`return_tensors` |
| 03 | `03Model/` | [03.Model.ipynb](03Model/03.Model.ipynb) | `AutoModel` / `AutoConfig` 載入、`last_hidden_state` vs `pooler_output`、注意力視覺化 |
| 03 | `03Model/` | [03 Model classification_demo.ipynb](03Model/03%20Model%20classification_demo.ipynb) | 用 BERT 做中文情感二分類（ChnSentiCorp 飯店評論）—— 現代化後改用 `Trainer` |
| 03 | `03Model/dataset/` | [ChnSentiCorp_htl_all.ipynb](03Model/dataset/ChnSentiCorp_htl_all.ipynb) | 資料集載入與分層切分（`stratify_by_column`）、類別不平衡處理 |
| 04 | `04Datasets/` | [04 Datasets.ipynb](04Datasets/04%20Datasets.ipynb) | `load_dataset` + `map(batched=True)` + `DataCollatorWithPadding` 動態 padding |
| 04 | `04Datasets/` | [04 Datasets classification_demo.ipynb](04Datasets/04%20Datasets%20classification_demo.ipynb) | 把 datasets 3.x 管線接上分類微調 |
| 05 | `05evaluate/` | [05 evaluate.ipynb](05evaluate/05%20evaluate.ipynb) | `evaluate.load` / `combine` / `compute`，指標計算與視覺化 |
| 05 | `05evaluate/` | [05 evaluate classification_demo.ipynb](05evaluate/05%20evaluate%20classification_demo.ipynb) | 把 `compute_metrics` 接進訓練流程（二分類） |
| 05 | `05evaluate/` | `05 evaluate classification_demo_tweets.ipynb.ipynb` | 航空推文三分類，補 precision/recall/F1 與混淆矩陣 |
| 06 | `06Trainer/` | [06 Trainer classification_demo.ipynb](06Trainer/06%20Trainer%20classification_demo.ipynb) | `Trainer` + 完整 `TrainingArguments` 的標準微調範式 |
| — | `demo/` | [demo.ipynb](demo/demo.ipynb) | Gradio + `pipeline` 的快速體驗 demo（非主線） |

> 註：`02.tokenizer.ipynb_` 與 tweets 的雙副檔名 `.ipynb.ipynb` 是 repo 既有命名，這裡如實標示。

---

## 三、六大元件逐一拆解（WHY 重於 HOW）

### 元件 1：pipeline —— 為什麼存在

`pipeline` 的價值是「把推論的樣板程式碼一次到位」。但教學上你必須知道它**隱藏**了什麼：模型載入到哪個裝置、輸出的 logits 怎麼變成機率、`device=0` 是什麼意思。

`device` 的整數慣例（`device=0` 是 GPU、`device=-1` 是 CPU）是 PyTorch 早期遺留，語意不直觀。2026 一律改用 `device_map='auto'`，由 accelerate 自動決定權重放 GPU、CPU 還是 disk offload：

```python
# Before (2024)：整數 device，語意晦澀，無多 GPU / offload 概念
pipe = pipeline("text-classification", model=model_id, device=0)

# After (2026)：device_map='auto' + 明確 dtype
pipe = pipeline(
    "text-classification",
    model=model_id,
    device_map="auto",
    torch_dtype="auto",  # 或 torch.bfloat16
)
```

### 元件 2：tokenizer —— 三件套與注意力遮罩

tokenizer 把字串轉成模型吃得下的數字。輸出通常是三個張量：

- `input_ids`：每個 token 的詞表索引。
- `attention_mask`：標記哪些位置是真實 token（1）、哪些是 padding（0）。變長序列要能批次處理，全靠它。
- `token_type_ids`：只在「句對任務」（如 NSP、句相似度）才有意義，標記 token 屬於第一句還是第二句。單句分類用不到。

教學重點：**為什麼需要 padding？** 因為一個 batch 裡的序列要等長才能堆成張量。但固定 padding 到 128 是浪費——短句被塞一堆無用的 `[PAD]`。正解是動態 padding（見元件 4）。

### 元件 3：model —— 載入慣例與輸出語意

`AutoModel.from_pretrained` 是萬用入口。2026 的載入慣例固定為三件事：`device_map='auto'`、`torch_dtype=torch.bfloat16`、`use_safetensors=True`。

```python
# Before (2024)：手動 .cuda()、無 dtype、預設 pickle (.bin)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model = model.cuda()

# After (2026)：裝置/精度/格式一次到位
import torch
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
```

為什麼這樣改：

- **bf16 優於 fp16/fp32**：bf16 的指數位數與 fp32 相同，動態範圍大、訓練穩定，又只佔一半記憶體。fp16 容易在大值時溢位。
- **`device_map='auto'`**：自動把放不下 GPU 的層 offload 到 CPU 甚至 disk，大模型不再一載入就 OOM。
- **safetensors 取代 pickle**：`.safetensors` 載入更快、且不會像 pickle (`pytorch_model.bin`) 那樣執行任意程式碼，安全性更高。儲存時用 `save_pretrained(safe_serialization=True)`。

輸出語意是常見誤區：`last_hidden_state` 是 **token 級**的表示（每個 token 一個向量），不是句向量；`pooler_output` 是 BERT 特有的 `[CLS]` 經過一層 tanh，**不等於**好的句向量（句相似度應改用 mean pooling 或 sentence-transformers，見 [../02-Adv-tasks/04-sentence_similarity](../02-Adv-tasks/04-sentence_similarity)）。

### 元件 4：datasets —— 為什麼 `batched=True` 快 3-5 倍

手刻 `torch.utils.data.Dataset` + `collate_fn` 是 2024 的寫法，2026 一律用 `datasets` 函式庫：

```python
# Before (2024)：pandas + 自製 Dataset + 固定 padding
df = pd.read_csv(path)
# ... 手刻 Dataset 類別、手刻 collate_fn 把每筆 pad 到 128 ...

# After (2026)：load_dataset + map(batched=True) + 動態 padding
from datasets import load_dataset
from transformers import DataCollatorWithPadding

ds = load_dataset("csv", data_files=path)
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True)  # 不在這裡 padding
ds = ds.map(tokenize_fn, batched=True, num_proc=4)
collator = DataCollatorWithPadding(tokenizer)  # 在 collate 時才 pad 到該 batch 的最長
```

`batched=True` 為什麼快：tokenizer 的 Rust 後端能向量化處理整批字串，比逐筆呼叫快 3-5 倍。`num_proc` 再用多核並行。

切分一律帶 `stratify_by_column` 與 `seed=42`：分層抽樣保留各類別比例（ChnSentiCorp 正負比約 2.5:1），固定 seed 確保可重現。

### 元件 5：evaluate —— 不要只看 accuracy

`evaluate.load("accuracy")` 載入指標，`add_batch` 累積、`compute` 算結果。但在類別不平衡時 accuracy 會騙人——全猜多數類也能有高 accuracy。務必補上 precision / recall / F1，並畫混淆矩陣做錯誤分析：

```python
import evaluate
import numpy as np

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=preds, references=labels)
```

注意 `evaluate.combine()` 在不同任務該配對的指標不同：分類用上述四件套；**回歸任務（如句相似度）要用 MSE / Pearson，不能把回歸輸出硬套 threshold 當分類**——這是 02 模組要修的常見錯誤。

### 元件 6：Trainer —— 別再手刻訓練迴圈

手刻 epoch/batch 迴圈、`torch.optim.Adam`、每個 batch 手動 `.cuda()` 是最大的反模式。`Trainer` 把整條訓練迴圈封裝起來，自動處理混合精度、梯度累積、checkpoint、分散式訓練、進度條。

```python
# Before (2024)：50+ 行手刻迴圈
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
for epoch in range(3):
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()

# After (2026)：Trainer + 完整 TrainingArguments
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

args = TrainingArguments(
    output_dir="./out",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,      # 有效 batch = 32 × 2 = 64
    warmup_ratio=0.1,                   # 暖身 10%，避免初期大步長震盪
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,                  # 梯度裁剪，穩定訓練
    bf16=True,
    optim="adamw_torch_fused",          # AdamW，非 Adam
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    save_safetensors=True,
    seed=42,
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["test"],
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()
```

關鍵觀念：

- **AdamW，不是 Adam**：AdamW 把 weight decay 與梯度更新解耦，這是 transformer 微調的標準。
- **為何 warmup**：訓練初期權重隨機，大學習率會把模型推歪；先用小步長暖身再升到目標 lr。
- **有效 batch size** = `per_device_train_batch_size` × `gradient_accumulation_steps` × GPU 數。記憶體不夠時用梯度累積換取等效大 batch。

> 進階提醒：當任務是**指令微調 / chat**（而非純監督分類），就不該再用 `Trainer`，而是改用 `trl.SFTTrainer` + `apply_chat_template`，由它自動套模板、自動做 response-only 標籤遮罩。詳見 [../03-PEFT/README.md](../03-PEFT/README.md) 與 [2026 慣例](../00-Setup-and-Foundations/01-2026-conventions.md)。

---

## 四、資料 → 模型 → 評測 管線圖

```
                            ┌─────────────┐
   原始 CSV / Hub dataset ─>│  datasets   │  load_dataset + map(batched=True)
                            │             │  train_test_split(stratify, seed=42)
                            └──────┬──────┘
                                   │ input_ids / attention_mask
                          (DataCollatorWithPadding 動態 padding)
                                   v
        ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
        │  tokenizer   │──>│    model    │──>│   logits     │
        │ (→Processor) │   │ device_map  │   │              │
        └──────────────┘   │ bf16/safet. │   └──────┬───────┘
                           └─────────────┘          │
                                   ^                 v
                                   │          ┌─────────────┐
                              ┌────┴────┐     │  evaluate   │ compute_metrics
                              │ Trainer │<────│ acc/F1/...  │ (混淆矩陣)
                              │ 訓練迴圈 │     └─────────────┘
                              └─────────┘
```

從 `tokenizer` 升級到 `AutoProcessor`、從 `input_ids` 升級到 `pixel_values`，這張圖原封不動就能套用到影像分類（[../05-Multimodal/01-image_classification](../05-Multimodal/01-image_classification)）。這就是「多模態是單模態的平滑延伸」的具體含義。

---

## 五、常見陷阱（修這些就少踩八成的雷）

| 陷阱 | 為什麼錯 | 正解 |
| :--- | :--- | :--- |
| 固定 padding 到 128 | 短句被塞滿 `[PAD]`，浪費算力 | `DataCollatorWithPadding` 動態 padding 到該 batch 最長 |
| `Adam` 當優化器 | 缺解耦的 weight decay | 用 `AdamW`（`optim="adamw_torch_fused"`） |
| `device=0` 整數寫法 | 語意晦澀、無 offload | `device_map="auto"` |
| 只報 accuracy | 類別不平衡時失真 | 補 precision/recall/F1 + 混淆矩陣 |
| 用 `pooler_output` 當句向量 | BERT 特有、品質差 | mean pooling 或 sentence-transformers |
| `token_type_ids` 用在單句分類 | 單句任務無此語意 | 只在句對任務使用 |
| 把回歸輸出套 threshold 當分類 | 任務型態判斷錯誤 | 回歸用 MSE/Pearson 評測 |
| 沒設 `seed` | 結果不可重現 | `TrainingArguments(seed=42)` + `set_seed(42)` |
| 變數拼字 `f1_metirc` | 既有 notebook bug | 修正為 `f1_metric` |
| 硬路徑（Google Drive / `d:/...`） | 無法跨機器重現 | HF Hub model id 或 `pathlib` + 環境變數 |

---

## 六、2026 環境與版本

本模組所有 notebook 假設以下鎖定版本（完整安裝步驟見 [../00-Setup-and-Foundations/00-environment-setup.md](../00-Setup-and-Foundations/00-environment-setup.md)）：

```text
torch >= 2.4
transformers >= 4.46
datasets >= 3.0
evaluate >= 0.4
accelerate >= 1.0
safetensors >= 0.4
```

四大共用慣例（細節見 [2026 慣例聖經](../00-Setup-and-Foundations/01-2026-conventions.md)）：

1. 載入：`device_map='auto'` + `torch_dtype=torch.bfloat16` + `use_safetensors=True`
2. 儲存：`save_pretrained(safe_serialization=True)`
3. 訓練：純監督任務用 `Trainer`；chat/指令微調用 `trl.SFTTrainer` + `apply_chat_template`
4. 可重現：`set_seed(42)`、移除硬路徑、訓練產物 `push_to_hub` 並附最小 model card（保存 `id2label` / `label2id`）

---

## 七、學完之後

你現在能讀懂任何一段 HuggingFace 推論/訓練程式碼，並知道每一行屬於哪個元件。下一步前往 [02-Adv-tasks](../02-Adv-tasks/README.md)，把這六個零件重新組合成 NER、問答、句相似度、檢索、語言模型、摘要與生成式 chatbot 等真實任務。
