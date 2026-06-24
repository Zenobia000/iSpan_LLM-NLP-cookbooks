# 02 - 進階 NLP 任務（Advanced NLP Tasks）

> 模組定位：把 `01-Component` 學到的「pipeline → tokenizer → model → datasets → evaluate → Trainer」六件套，套用到八個真實的下游 NLP 任務上。本模組是整套 cookbook 從「會用元件」走向「會解任務」的轉折點，也是通往 `03-PEFT` / `04-kbits-tuning` 的最後一哩路。

本文件先說明**為什麼**這八個任務值得各自獨立教，再用一張對照表把每個 notebook 對映到它教的能力、用的 model head、評測指標與常見錯誤，最後給出 2026 版本鎖定與現代化說明。

---

## 0. 前置知識與環境

- 前置模組：[`../01-Component/README.md`](../01-Component/README.md)（必修）
- 共用慣例：[`../00-Setup-and-Foundations/01-2026-conventions.md`](../00-Setup-and-Foundations/01-2026-conventions.md)
- 術語速查：[`../00-Setup-and-Foundations/02-glossary-and-architectures.md`](../00-Setup-and-Foundations/02-glossary-and-architectures.md)
- 環境鎖版本：見 [`../00-Setup-and-Foundations/00-environment-setup.md`](../00-Setup-and-Foundations/00-environment-setup.md)
  - `transformers>=4.46`、`datasets>=3.0`、`trl>=0.12`、`peft>=0.13`、`accelerate>=1.0`、`evaluate>=0.4`、`safetensors>=0.4`、`torch>=2.4`

本模組所有 notebook 共用四大慣例，後續不再每段重述：

1. **載入** 一律 `from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True)`。
2. **儲存** 一律 `save_pretrained(..., safe_serialization=True)`（產出 `.safetensors`，不是 `pytorch_model.bin`）。
3. **chat / 指令** 一律 `tokenizer.apply_chat_template(...)`，不手寫 `Human:/Assistant:`。
4. **可重現性** 一律 `set_seed(42)`，移除硬路徑（不用 `d:/...` 或 Google Drive mount）。

---

## 1. 為什麼要分八個任務教？（任務地圖）

NLP 下游任務的差異**不在於程式碼長相**（它們都走同一條 datasets → Trainer → evaluate 管線），而在於三個維度：

- **架構選型**：任務的本質決定該用 encoder-only、encoder-decoder、還是 decoder-only。
- **model head**：同一個 backbone，換不同的 `AutoModelFor*` 等於換一顆輸出頭。
- **評測指標**：分類用 F1，序列標註用 seqeval，回歸用 Pearson，生成用 ROUGE，檢索用 Recall@k。用錯指標等於沒評測。

把這三個維度想清楚，八個任務其實只是同一個心智模型的八種參數化。這就是本模組要建立的「好品味」——消除特殊情況，讓每個任務看起來都一樣。

```
                       輸入是什麼？要產出什麼？
                                │
         ┌──────────────────────┼───────────────────────┐
         │                      │                        │
   理解 / 標註            轉換 / 重寫              生成 / 對話
  (encoder-only)        (encoder-decoder)        (decoder-only)
         │                      │                        │
  ┌──────┴──────┐         ┌─────┴─────┐            ┌─────┴─────┐
  NER 標註       句相似度   摘要 (T5/GLM)            CLM 語言模型
  抽取式 QA      檢索 rerank                         生成式 chatbot
  分類微調                                          (指令微調入門)
```

---

## 2. 架構選型：三型 Transformer 對應三類任務

> 詳細理論見 [`../00-Setup-and-Foundations/02-glossary-and-architectures.md`](../00-Setup-and-Foundations/02-glossary-and-architectures.md)。這裡只講「為什麼這個任務該選這型」。

| 架構 | 注意力方向 | 適合的任務 | 本模組對應 |
| :--- | :--- | :--- | :--- |
| **encoder-only**（BERT/RoBERTa/MacBERT） | 雙向，看得到整句 | 理解、分類、標註、抽取 | 微調分類、NER、抽取式 QA、句相似度、檢索 |
| **encoder-decoder**（T5/GLM/BART） | encoder 雙向 + decoder 自回歸 | 把一段文字「轉換」成另一段 | 摘要 |
| **decoder-only**（BLOOM/Llama/Qwen） | 單向，只看左邊 | 自由生成、續寫、對話 | 因果語言模型、生成式 chatbot |

**為什麼抽取式 QA 用 encoder-only 而非生成模型？** 因為抽取式 QA 的答案保證是「原文的一個片段」，模型只要預測 start/end 兩個位置（span prediction），雙向注意力能同時看到問題與全文脈絡，比讓 decoder 自由生成更準、更快、更不會幻覺。

**為什麼摘要用 encoder-decoder？** 摘要是典型的「輸入序列 → 輸出序列」轉換，encoder 先把全文編成表示，decoder 用 cross-attention 一邊看全文一邊生成，這正是 seq2seq 的設計初衷。

---

## 3. Notebook 對照表：每個任務教什麼

| # | Notebook | 任務 | 架構 / model head | 推薦 model（2026） | 評測指標 |
| :- | :--- | :--- | :--- | :--- | :--- |
| 01 | [`01-finetune_optimize/01 train opti classification_demo.ipynb`](01-finetune_optimize/) | 微調與記憶體優化 | encoder-only / `AutoModelForSequenceClassification` | `hfl/rbt3` | accuracy / F1 |
| 02 | [`02-token_classification/ner.ipynb`](02-token_classification/ner.ipynb) | 中文命名實體辨識 NER | encoder-only / `AutoModelForTokenClassification` | `hfl/chinese-macbert-base` | **seqeval**（precision/recall/F1，IOB2） |
| 03 | [`03-question_answering/mrc_simple_version.ipynb`](03-question_answering/mrc_simple_version.ipynb) | 抽取式機器閱讀理解 | encoder-only / `AutoModelForQuestionAnswering` | `hfl/chinese-macbert-base`（CMRC2018） | **EM / F1**（squad metric） |
| 04a | [`04-sentence_similarity/cross_model.ipynb`](04-sentence_similarity/cross_model.ipynb) | 句相似度（cross-encoder 回歸） | encoder-only / `AutoModelForSequenceClassification(num_labels=1)` | `hfl/chinese-macbert-base` | **MSE / Pearson**（回歸，非分類！） |
| 04b | [`04-sentence_similarity/dual_model.ipynb`](04-sentence_similarity/dual_model.ipynb) | 句相似度（dual-encoder 對比學習） | encoder-only 雙塔 / `CosineEmbeddingLoss` | `hfl/chinese-macbert-base` | Spearman / Pearson |
| 05 | [`05-retrieval_chatbot/retrieval_bot.ipynb`](05-retrieval_chatbot/retrieval_bot.ipynb) | 稠密檢索 + 重排序 FAQ 機器人 | dual-encoder + cross-encoder + FAISS | dual + cross 兩段式 | **Recall@k / MRR / NDCG** |
| 06a | [`06-language_model/causal_lm.ipynb`](06-language_model/causal_lm.ipynb) | 因果語言模型 CLM | decoder-only / `AutoModelForCausalLM` | `bigscience/bloom-389m` | perplexity |
| 06b | [`06-language_model/masked_lm.ipynb`](06-language_model/masked_lm.ipynb) | 遮罩語言模型 MLM | encoder-only / `AutoModelForMaskedLM` | `hfl/chinese-macbert-base` | perplexity / fill-mask 定性 |
| 07a | [`07-text_summarization/summarization.ipynb`](07-text_summarization/summarization.ipynb) | 抽象式摘要（T5） | encoder-decoder / `AutoModelForSeq2SeqLM` | `Langboat/mengzi-t5-base`（NLPCC2017） | **ROUGE-1/2/L** |
| 07b | [`07-text_summarization/summarization_glm.ipynb`](07-text_summarization/summarization_glm.ipynb) | 抽象式摘要（GLM） | prefix-LM / `AutoModelForSeq2SeqLM` | `THUDM/glm-large-chinese` | ROUGE |
| 08 | [`08-generative_chatbot/chatbot.ipynb`](08-generative_chatbot/chatbot.ipynb) | 生成式 chatbot（指令微調入門） | decoder-only SFT / `AutoModelForCausalLM` | `bigscience/bloom-389m`（Alpaca-zh） | 定性 + perplexity |

> 模組 01 的 `01 train opti classification_demo.ipynb` 與 `01-Component` 的分類示範同源，差別在這裡額外教**記憶體優化**（梯度累積、凍結層、混合精度），並自然帶入 `03-PEFT` 的 LoRA 動機。

---

## 4. 評測指標對照：用對指標是任務正確性的一半

| 指標 | 適用任務 | 為什麼用它 | `evaluate` 載入 |
| :--- | :--- | :--- | :--- |
| accuracy + **F1（precision/recall）** | 分類 | 類別不平衡時 accuracy 會騙人，F1 才看得到少數類 | `evaluate.load("f1")` |
| **seqeval** | NER / 序列標註 | 必須以「實體」為單位做精確匹配，不能用 token-level accuracy | `evaluate.load("seqeval")` |
| **EM / F1（squad）** | 抽取式 QA | EM 看完全相符，F1 看 token 重疊；中文要做 offset 對齊 | `evaluate.load("cmrc2018")` 或 squad |
| **MSE / Pearson** | 句相似度（回歸） | 相似度是連續分數，用 threshold 當分類是錯的 | `evaluate.load("mse")` / `pearsonr` |
| **Recall@k / MRR / NDCG** | 檢索 | 檢索看「正確答案有沒有排進前 k 名、排多前面」 | 自訂 / `evaluate` |
| **ROUGE-1/2/L** | 摘要 | 看生成摘要與參考摘要的 n-gram / 最長共同子序列重疊 | `evaluate.load("rouge")` |
| **perplexity** | 語言模型 | 衡量模型對測試文字的「困惑程度」，越低越好 | 由 eval loss 取 `exp()` |
| **WER / CER** | （延伸）語音辨識 | 詞 / 字錯誤率，見 `05-Multimodal` 的 Whisper | `evaluate.load("wer")` |

---

## 5. 最常見的兩個觀念錯誤（重點警示）

### 5.1 回歸 vs. 分類：句相似度不是二分類

`cross_model.ipynb` 訓練的是 `num_labels=1` 的**回歸模型**（輸出一個連續相似度分數），舊寫法卻把輸出 threshold 成 `p > 0.5` 再用 F1 評測——這在理論上是錯的。

```python
# Before（錯誤：把回歸當分類）
preds = (logits > 0.5).astype(int)
f1 = f1_metric.compute(predictions=preds, references=labels)

# After（正確：回歸用 MSE / Pearson 相關係數）
import evaluate
from scipy.stats import pearsonr
mse = evaluate.load("mse").compute(predictions=logits, references=labels)
pearson = pearsonr(logits, labels).statistic   # 相似度任務的標準指標
```

**為什麼重要**：相似度是 `[0, 1]` 或 `[-1, 1]` 的連續量，硬切閾值會丟掉「差一點」與「差很多」的資訊，且 0.5 這個閾值毫無理論依據。回歸任務就該用回歸指標。

### 5.2 NER 的 `-100` 標籤遮罩與 word_ids 對齊

NER 的 label 必須對齊到「子詞 token」，但一個詞被切成多個子詞時，只有第一個子詞算 loss，其餘設 `-100`（cross-entropy 的 `ignore_index`）。

```python
def align_labels(word_ids, labels):
    aligned = []
    prev = None
    for wid in word_ids:                 # tokenizer(...).word_ids()
        if wid is None:                  # [CLS]/[SEP]/[PAD]
            aligned.append(-100)
        elif wid != prev:                # 詞的第一個子詞
            aligned.append(labels[wid])
        else:                            # 同一個詞的後續子詞
            aligned.append(-100)         # 不計 loss
        prev = wid
    return aligned
```

**為什麼設 `-100`**：避免同一個實體被重複計分、避免特殊 token 污染 loss。這個 `-100` 慣例會在 `06-language_model`、`08-generative_chatbot`、`03-PEFT`、`04-kbits-tuning` 反覆出現（指令微調時只在 response token 算 loss），是貫穿後續模組的核心概念。

---

## 6. 檢索系統：dual-encoder + cross-encoder 兩段式

`retrieval_bot.ipynb` 教的是 2026 仍然主流的 FAQ 檢索架構，核心是**速度 vs. 準度的取捨**：

```
使用者 query
      │
      ▼
 ┌──────────────┐   快、可預先建索引
 │ dual-encoder │   query 與所有 FAQ 各自獨立編碼成向量
 └──────┬───────┘
        │ FAISS IndexFlatIP（內積 = cosine，需先 normalize_L2）
        ▼
   取回 top-k 候選（粗篩，召回優先）
        │
        ▼
 ┌──────────────┐   慢、但準
 │ cross-encoder│   query + 候選成對輸入，逐一精算相關分數
 └──────┬───────┘
        ▼
     重排序 → 最終答案
```

**為什麼要兩段？** dual-encoder 可以把上萬條 FAQ **預先**編碼進 FAISS，查詢時只算 query 向量再做近鄰搜尋，毫秒級回應；但雙塔各自獨立編碼，犧牲了 query 與文件的交互資訊。cross-encoder 讓兩者一起進模型做完整 attention，準度高但無法預先索引（每對都要現算）。先用 dual-encoder 粗篩 top-k，再用 cross-encoder 精排——這就是「先廣後精」的好設計。

> `dual_model.py` 是 `04-sentence_similarity/dual_model.ipynb` 雙塔模型的可重用版本，檢索 notebook 直接 import 它，體現「相似度模型即檢索器」的延續性。

---

## 7. 2026 現代化說明：相對舊版改了什麼

本模組所有 notebook 已從 2024 年初的寫法升級。以下是最值得理解的幾項：

### 7.1 純監督任務：移除手刻訓練迴圈，改用 Trainer

```python
# Before（手刻 epoch/batch，散落 .cuda()，用 Adam，無排程、無 amp）
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
for epoch in range(3):
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()

# After（Trainer 一把抓：分散式 / 混合精度 / checkpoint / 排程）
args = TrainingArguments(
    output_dir="out",
    bf16=True,                       # 為何 bf16 > fp16：動態範圍大、不易 overflow
    learning_rate=2e-5,
    warmup_ratio=0.1,                # 暖身避免初期大梯度破壞預訓練權重
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    optim="adamw_torch_fused",       # AdamW（解耦 weight decay）優於 Adam
    eval_strategy="steps",
    load_best_model_at_end=True,
    save_safetensors=True,
    seed=42,
)
trainer = Trainer(model, args, train_dataset=..., eval_dataset=...,
                  compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
trainer.train()
```

**為什麼 AdamW 不是 Adam**：Adam 把 weight decay 混進梯度，AdamW 將其解耦成獨立的權重衰減，這對 Transformer 微調的泛化更穩定，已是業界預設。

**effective batch size** = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`。記憶體不夠時用梯度累積換取等效大 batch。

### 7.2 資料管線：datasets 3.x + 動態 padding

```python
# Before（pandas + 自製 Dataset + 固定 padding 到 128）
# After（load_dataset + batched map + 動態 padding）
ds = load_dataset("csv", data_files="data.csv")
ds = ds.map(tokenize_fn, batched=True, num_proc=4)        # 快 3-5 倍（向量化）
ds = ds["train"].train_test_split(test_size=0.2,
                                  stratify_by_column="label",  # 解類別不平衡
                                  seed=42)
collator = DataCollatorWithPadding(tokenizer)             # 每個 batch padding 到自身最長
```

**為什麼 `batched=True` 快**：tokenizer 的 Rust 後端對一整批文字向量化處理，省去 Python 逐筆迴圈的開銷。**為什麼動態 padding**：固定 padding 到 128 會在短句上浪費大量算力，按 batch 內最長序列 padding 更省。

### 7.3 摘要：用 evaluate 的 rouge 取代第三方 rouge_chinese

```python
# Before
from rouge_chinese import Rouge          # 第三方、需手動 jieba 斷詞、維護不佳
# After
import evaluate
rouge = evaluate.load("rouge")           # 統一生態、與 Trainer compute_metrics 無縫接
```

### 7.4 指令微調（08 / 銜接 03、04）：apply_chat_template + 可選 SFTTrainer

```python
# Before（硬寫模板，訓練/推論容易不一致）
prompt = f"Human: {instruction}\nAssistant: {response}"

# After（模板是跨模型可攜的單一抽象；訓練與推論用同一套）
messages = [{"role": "user", "content": instruction},
            {"role": "assistant", "content": response}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# 推論時：apply_chat_template(messages, add_generation_prompt=True)
```

**為什麼這是關鍵橋樑**：同一套 chat 模板機制在 `05-Multimodal` 會延伸到含 image / audio token 的多模態訊息。學會它，等於提前掌握多模態對話的輸入格式。

---

## 8. 通往下一個模組

完成本模組後，你已經會用 `Trainer` 做監督微調、用 `apply_chat_template` 做指令微調入門、並理解 `-100` 標籤遮罩。接下來：

- **[`../03-PEFT/README.md`](../03-PEFT/README.md)**：當模型變大、全參數微調太貴時，用 LoRA / IA3 只訓練少量參數。本模組 08 的指令微調會直接升級成 `SFTTrainer + peft_config`。
- **[`../04-kbits-tuning/README.md`](../04-kbits-tuning/README.md)**：用 `BitsAndBytesConfig` 做 4/8-bit 量化載入大型 LLM，配合 LoRA 完成 QLoRA。
- **[`../05-Multimodal/README.md`](../05-Multimodal/README.md)**：本模組的稠密檢索（05）會升級為跨模態 RAG；句相似度雙塔（04b）會升級為 CLIP 圖文雙塔；指令微調（08）會升級為 VLM 的視覺問答。

---

## 附錄：本模組目錄結構

```
02-Adv-tasks/
├── 01-finetune_optimize/   微調與記憶體優化（分類 + 凍結/梯度累積）
├── 02-token_classification/ NER（含 ner_data/ 與 seqeval_metric.py）
├── 03-question_answering/   抽取式 QA（含 cmrc_eval.py）
├── 04-sentence_similarity/  句相似度（cross-encoder + dual-encoder）
├── 05-retrieval_chatbot/    稠密檢索 + 重排序（含 dual_model.py）
├── 06-language_model/       CLM 與 MLM
├── 07-text_summarization/   摘要（T5 與 GLM 兩種）
└── 08-generative_chatbot/   生成式 chatbot（指令微調入門）
```
