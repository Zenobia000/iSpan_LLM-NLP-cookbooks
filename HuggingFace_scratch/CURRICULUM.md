# 教材總綱 Curriculum：HuggingFace Transformers 中文實戰 Cookbook（2026 版）

> 本文件是整套教材的**學習地圖**。它告訴你：每個模組教什麼、為什麼要照這個順序學、要先具備什麼、大約要花多少時間，以及從 NLP 基礎一路通往多模態（Multimodal）的完整路徑。
>
> 各模組的細節導覽請見對應的 `README.md`；全 repo 共用的工程慣例與術語請見 `00-Setup-and-Foundations/` 下的參考文件。本文件只負責「把它們串成一條路」。

---

## 1. 這套教材是什麼，寫給誰

這是一套以 **HuggingFace Transformers** 為核心、以**繁體中文 NLP** 為主軸的教學 cookbook。它從最基礎的元件（`pipeline`、`tokenizer`、`model`）出發，逐步走到進階任務（NER、QA、摘要、檢索、生成式 chatbot）、參數高效微調（PEFT / LoRA / IA3）、量化微調（QLoRA），最後延伸到 2026 年最具實戰價值的**多模態**（影像、語音、視覺語言模型 VLM、跨模態 RAG）。

**目標讀者**

- 已具備 Python 3.10+ 基礎，懂基本的 PyTorch 與深度學習概念（張量、反向傳播、loss、optimizer）。
- 想用現代（2026）的 HuggingFace 生態系做事，而不是抄 2024 年初的手刻訓練迴圈。
- 對中文 NLP 任務有實際需求（情感分析、NER、問答、摘要、對話、檢索）。

**這套教材的核心設計理念：消除特殊情況。**

> 「好代碼沒有特殊情況。」

整套教材的每個 notebook 都遵循**同一套**載入、訓練、推論、儲存的慣例。你在 01 模組學到的 `device_map='auto'`、`apply_chat_template()`、`Trainer/SFTTrainer`、`safetensors`，到了 04 量化微調、05 多模態仍然是同一套寫法。**單模態的心智模型可以平滑延伸到多模態**——這不是另起爐灶，而是把 `tokenizer` 升級成 `AutoProcessor`、把 `input_ids` 擴充成 `pixel_values` / audio features、把 `Trainer` 換成支援多模態 collator 的 `SFTTrainer`。

---

## 2. 學習路徑圖（模組依賴關係）

```
00-Setup-and-Foundations  （環境鎖版本 + 共用慣例聖經）
        │
        ▼
01-Component              （pipeline → tokenizer → model → datasets → evaluate → trainer）
        │
        ▼
02-Adv-tasks             （NER / QA / 相似度 / 檢索 / LM / 摘要 / chatbot）
        │
        ▼
03-PEFT                  （LoRA / IA3：參數高效微調）
        │
        ▼
04-kbits-tuning          （16/8/4-bit 量化 + QLoRA）
        │
        ▼
05-Multimodal            （ViT → CLIP → VLM → Whisper → 多模態 RAG → VLM 微調）
```

**為什麼是這個順序？**

- **00 先行**：零文件、零環境鎖定是目前最大的可用性風險。沒有一致的環境與慣例，後面每個 notebook 都會各自為政。
- **01 是範式來源**：所有後續模組的「載入模型 → 處理資料 → 訓練 → 評測 → 儲存」管線，都是 01 模組教過的同一套。
- **02 → 03 → 04 是一條微調深化線**：先學全量微調與各類任務（02），再學「只調一小撮參數」（03 PEFT），最後學「在量化後的模型上只調一小撮參數」（04 QLoRA）。記憶體需求逐步下降，能訓練的模型逐步變大。
- **05 是延伸而非斷裂**：05 模組刻意回收 01–04 的每一個觀念（見 §7 知識回收對照表）。

---

## 3. 模組總覽表（一句話定位）

| 模組 | 一句話定位 | 性質 | 導覽文件 |
| :--- | :--- | :--- | :--- |
| `00-Setup-and-Foundations` | 鎖定 2026 環境、確立全 repo 共用的工程慣例與術語 | 新增（docs + 1 環境 notebook） | [`00-Setup-and-Foundations/00-environment-setup.md`](00-Setup-and-Foundations/00-environment-setup.md) |
| `01-Component` | HF 六大元件：`pipeline` / `tokenizer` / `model` / `datasets` / `evaluate` / `Trainer` | 現代化既有 | [`01-Component/README.md`](01-Component/README.md) |
| `02-Adv-tasks` | 七類進階 NLP 任務的 2026 寫法與架構選型 | 現代化既有 | [`02-Adv-tasks/README.md`](02-Adv-tasks/README.md) |
| `03-PEFT` | 參數高效微調：LoRA 與 IA3 | 現代化既有 | [`03-PEFT/README.md`](03-PEFT/README.md) |
| `04-kbits-tuning` | 量化階梯（16/8/4-bit）與 QLoRA 全流程 | 現代化既有 | [`04-kbits-tuning/README.md`](04-kbits-tuning/README.md) |
| `05-Multimodal` | 多模態：影像分類 / CLIP / VLM / ASR / 多模態 RAG / VLM 微調 | 全新 | [`05-Multimodal/README.md`](05-Multimodal/README.md) |

---

## 4. 環境與鎖定版本（2026）

整套教材在**同一組鎖定版本**上運行。完整安裝步驟與疑難排解見 [`00-Setup-and-Foundations/00-environment-setup.md`](00-Setup-and-Foundations/00-environment-setup.md)。

```text
# core (lock these)
torch>=2.4
transformers>=4.46
datasets>=3.0
trl>=0.12
peft>=0.13
accelerate>=1.0
bitsandbytes>=0.44
evaluate>=0.4
safetensors>=0.4

# multimodal extras (05-Multimodal)
torchvision        # ViT / 影像前處理
faiss-cpu          # CLIP / 多模態 RAG 檢索（有 GPU 可用 faiss-gpu）
pillow             # 影像載入
soundfile          # Whisper 音訊載入
```

**為什麼要鎖版本？** 因為 2024 年初的寫法在 2026 已大量棄用或行為改變：`load_in_4bit=True` 裸參數在 transformers 4.42+ 已棄用、`device=0` 整數裝置語意被 `device_map='auto'` 取代、`pytorch_model.bin`（pickle）被 `.safetensors` 取代。鎖版本是可重現性的前提。

> 4 大共用慣例（**所有 notebook 都遵守**），完整說明見 [`00-Setup-and-Foundations/01-2026-conventions.md`](00-Setup-and-Foundations/01-2026-conventions.md)：
> 1. `device_map='auto'` + `torch_dtype=torch.bfloat16` 統一載入
> 2. `safetensors` 取代 pickle
> 3. `tokenizer.apply_chat_template()` 統一對話格式
> 4. `Trainer` / `trl.SFTTrainer` 取代手刻訓練迴圈

---

## 5. 快速開始

```bash
# 1. 取得程式碼
git clone <repo-url> && cd HuggingFace_scratch

# 2. 建立隔離環境（建議 uv 或 venv）
uv venv && source .venv/bin/activate     # 或 python -m venv .venv

# 3. 安裝鎖定版本
uv pip install -r requirements.txt        # 版本見 §4 與 setup 文件

# 4. 驗證 GPU/CUDA 與 import
python -c "import torch, transformers; print(torch.cuda.is_available(), transformers.__version__)"

# 5. 跑第一個 notebook（從 01 模組的 pipeline 開始）
jupyter lab "01-Component/01pipeline/01.pipeline.ipynb"
```

登入 HuggingFace Hub（下載受限模型、push 訓練產物時需要）：

```bash
huggingface-cli login        # 或設定環境變數 HF_TOKEN
export HF_HOME=/path/to/large/cache   # 統一快取位置，避免塞爆家目錄
```

---

## 6. 各模組詳解：學習目標、前置、時間估計

> 時間估計指「閱讀 + 跑通 + 理解 WHY」的合計，假設讀者具備前置知識、使用單張 16–24GB GPU。

---

### 模組 00 — Setup and Foundations

**定位**：把全 repo 共用的東西寫成單一參考文件，後面所有 notebook 引用它而非各自重複。這是「消除特殊情況」的核心。

**前置**：Python 3.10+、基本 PyTorch 與深度學習概念、命令列操作。

**學習目標**

- 建立鎖定版本的 2026 HF 環境並驗證 GPU/CUDA。
- 理解 4 大共用慣例（`device_map='auto'`+`torch_dtype`、`safetensors`、`apply_chat_template`、`Trainer`/`SFTTrainer`）。
- 會用 `huggingface_hub` 登入、檢視 model card、設定 `HF_HOME` 快取。
- 建立可重現性基線：`set_seed(42)`、移除硬路徑、`pathlib` + 環境變數。

**內含文件**

- [`00-environment-setup.md`](00-Setup-and-Foundations/00-environment-setup.md) — 環境建置與疑難排解
- [`01-2026-conventions.md`](00-Setup-and-Foundations/01-2026-conventions.md) — 2026 慣例聖經（含舊寫法→新寫法對照表）
- [`02-glossary-and-architectures.md`](00-Setup-and-Foundations/02-glossary-and-architectures.md) — 術語與架構速查

**時間估計**：1–1.5 小時。

---

### 模組 01 — Component（HF 六大元件）

**定位**：把 HuggingFace 拆成六個可獨立理解的元件，建立「載入 → 處理 → 訓練 → 評測 → 儲存」的範式。**這是後面所有模組的基礎心智模型。**

**前置**：模組 00。

**學習順序與每個 notebook 教什麼**

| 子目錄 / Notebook | 教什麼（核心觀念） |
| :--- | :--- |
| [`01pipeline/01.pipeline.ipynb`](01-Component/01pipeline/01.pipeline.ipynb) | `pipeline` 高階抽象 vs 底層 `model.forward()`；何時用哪個；`device_map='auto'` 取代 `device=0`；順帶引入 `AutoProcessor` / `AutoImageProcessor`（zero-shot-object-detection 的 OWL-ViT 段落）作為多模態前置 |
| [`02tokenizer/`](01-Component/02tokenizer) | tokenizer 三件套（`input_ids` / `attention_mask` / `token_type_ids`）、特殊 token；processor 是 tokenizer 的多模態超集 |
| [`03Model/03.Model.ipynb`](01-Component/03Model/03.Model.ipynb) | `AutoConfig` / `AutoModel` / `AutoModelForX`；config.json 的意義；encoder-only/enc-dec/decoder-only 區別；`last_hidden_state` vs `pooler_output` |
| [`03Model/03 Model classification_demo.ipynb`](01-Component/03Model/03%20Model%20classification_demo.ipynb) | 用 `Trainer` 取代手刻訓練迴圈做中文情感分類；AdamW vs Adam；為何 bf16 |
| [`04Datasets/04 Datasets.ipynb`](01-Component/04Datasets/04%20Datasets.ipynb) | `load_dataset` + `.map(batched=True, num_proc=N)` + 動態 padding；為何 `batched=True` 快 3–5 倍 |
| [`05evaluate/05 evaluate.ipynb`](01-Component/05evaluate/05%20evaluate.ipynb) | `evaluate.load` + `compute_metrics`；accuracy/precision/recall/F1；混淆矩陣錯誤分析 |
| [`06Trainer/06 Trainer classification_demo.ipynb`](01-Component/06Trainer/06%20Trainer%20classification_demo.ipynb) | 完整 `TrainingArguments`（bf16 / warmup_ratio / cosine / `save_safetensors` / eval_strategy / `load_best_model_at_end`）+ `EarlyStoppingCallback` |

**學習目標**

- 掌握 `pipeline` 抽象與底層 `model.forward()` 的關係，知道何時用哪一個。
- 理解 tokenizer 三件套與特殊 token；認識 processor 是 tokenizer 的多模態超集。
- 用 datasets 3.x 管線取代手刻 `Dataset` / `collate_fn`。
- 用 `Trainer` + 完整 `TrainingArguments` 做監督微調。
- 用 `evaluate.load` + `compute_metrics` 算指標並做混淆矩陣錯誤分析。
- 理解 `device_map` / `torch_dtype` / `safetensors` 載入慣例。

**現代化重點（before/after）**

舊寫法（2024 手刻迴圈）：

```python
model = AutoModelForSequenceClassification.from_pretrained(model_id).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
for epoch in range(3):
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}   # per-batch device move, slow
        loss = model(**batch).loss
        loss.backward(); optimizer.step(); optimizer.zero_grad()
```

新寫法（2026 `Trainer`）：

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, id2label=id2label, label2id=label2id,
    device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True,
)
args = TrainingArguments(
    output_dir="out", bf16=True, learning_rate=2e-5, num_train_epochs=3,
    warmup_ratio=0.1, lr_scheduler_type="cosine", max_grad_norm=1.0,
    eval_strategy="steps", load_best_model_at_end=True,
    save_safetensors=True, optim="adamw_torch_fused", seed=42,
)
trainer = Trainer(model, args, train_dataset=ds["train"], eval_dataset=ds["test"],
                  data_collator=DataCollatorWithPadding(tokenizer),
                  compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
trainer.train()
```

**WHY**：`Trainer` 一行接手裝置/精度/混合精度/梯度累積/checkpoint/early-stopping。手刻迴圈不是「更透明」，而是「更多會出錯的特殊情況」。

**時間估計**：6–8 小時。

---

### 模組 02 — Adv-tasks（進階 NLP 任務）

**定位**：依任務類型組織，學會分辨「哪種架構配哪種任務、哪種指標」。這裡開始接觸生成、檢索、與指令微調的入門。

**前置**：模組 01。

**任務地圖**

| Notebook | 任務 | 架構選型 | 評測指標 | 關鍵觀念 / 修正 |
| :--- | :--- | :--- | :--- | :--- |
| [`02-token_classification/ner.ipynb`](02-Adv-tasks/02-token_classification/ner.ipynb) | 中文 NER | encoder-only + token head | seqeval（precision/recall/F1） | `word_ids()` 子詞對齊、`-100` 忽略 padding、IOB2 標註 |
| [`03-question_answering/mrc_simple_version.ipynb`](02-Adv-tasks/03-question_answering/mrc_simple_version.ipynb) | 抽取式 QA | encoder-only + span head | EM / F1（squad） | `return_offsets_mapping`、`truncation='only_second'`、span 預測 |
| [`04-sentence_similarity/cross_model.ipynb`](02-Adv-tasks/04-sentence_similarity/cross_model.ipynb) | 句相似度（cross-encoder） | encoder-only 句對 | **MSE / Pearson**（回歸） | 修正：回歸任務不可用 threshold 當分類 |
| [`04-sentence_similarity/dual_model.ipynb`](02-Adv-tasks/04-sentence_similarity/dual_model.ipynb) | 句相似度（dual-encoder） | 雙塔 + cosine | Pearson / Spearman | `CosineEmbeddingLoss`、對比學習、為何標籤是 {-1,1} |
| [`05-retrieval_chatbot/retrieval_bot.ipynb`](02-Adv-tasks/05-retrieval_chatbot/retrieval_bot.ipynb) | 檢索式 QA | dual-encoder + cross-encoder rerank | **Recall@k / MRR / NDCG** | FAISS `IndexFlatIP`、dense retrieval + rerank（→ 05 多模態 RAG 的基礎） |
| [`06-language_model/causal_lm.ipynb`](02-Adv-tasks/06-language_model/causal_lm.ipynb) | 因果語言模型 | decoder-only | perplexity | `mlm=False`、`pad_token` 設定、packing |
| [`06-language_model/masked_lm.ipynb`](02-Adv-tasks/06-language_model/masked_lm.ipynb) | 遮罩語言模型 | encoder-only | perplexity | `mlm=True, mlm_probability=0.15`、80/10/10 masking |
| [`07-text_summarization/summarization.ipynb`](02-Adv-tasks/07-text_summarization/summarization.ipynb) | 摘要（T5） | encoder-decoder | **evaluate 的 rouge**（取代 `rouge_chinese`） | task prefix、`-100` label 遮罩、`DataCollatorForSeq2Seq` |
| [`07-text_summarization/summarization_glm.ipynb`](02-Adv-tasks/07-text_summarization/summarization_glm.ipynb) | 摘要（GLM） | prefix-LM | rouge | `apply_chat_template` 取代硬寫 prompt、批次推論 |
| [`08-generative_chatbot/chatbot.ipynb`](02-Adv-tasks/08-generative_chatbot/chatbot.ipynb) | 生成式 chatbot | decoder-only | 定性 / perplexity | `apply_chat_template` + 入門 `SFTTrainer`、`-100` 只在 response 算 loss |

**學習目標**

- 完成上述七類任務的 2026 寫法。
- 區分 encoder-only / encoder-decoder / decoder-only 與對應任務。
- 正確處理回歸 vs 分類（句相似度用 MSE/Pearson 而非 threshold）。
- 建立 dense retrieval + cross-encoder rerank，並以 Recall@k / MRR / NDCG 評測。
- 用 `apply_chat_template` 與（可選）`SFTTrainer` 做指令微調入門。
- 理解 `-100` 標籤遮罩：只在 response token 計算 loss。

**WHY（最關鍵的橋樑）**：`apply_chat_template()` 是**跨模型可攜的單一抽象**。它取代所有硬寫的 `'Human:/Assistant:'`、`f'<s>[INST]...[/INST]'`、ChatGLM 的 `build_chat_input()`。**到了 05 多模態，含 image/audio token 的訊息也是同一套機制**——這是通往多模態的關鍵橋樑。

**時間估計**：10–14 小時。

---

### 模組 03 — PEFT（參數高效微調）

**定位**：當全量微調太貴時，只訓練一小撮新參數。這是邁向「在消費級 GPU 上微調大模型」的第一步。

**前置**：模組 02。

| Notebook | 教什麼 |
| :--- | :--- |
| [`01-LoRA/chatbot_lora.ipynb`](03-PEFT/01-LoRA/chatbot_lora.ipynb) | LoRA 指令微調：明確 `LoraConfig(r, lora_alpha, target_modules, lora_dropout)` + `get_peft_model`；`apply_chat_template` + `SFTTrainer` |
| [`01-LoRA/lora_inference.ipynb`](03-PEFT/01-LoRA/lora_inference.ipynb) | 載入 base + adapter；`merge_and_unload()`；adapter-only vs merged 部署取捨 |
| [`02-IA3/chatbot_ia3.ipynb`](03-PEFT/02-IA3/chatbot_ia3.ipynb) | IA3（element-wise 縮放）vs LoRA 的取捨：參數量、訓練穩定性、推論延遲 |

**學習目標**

- 理解 LoRA 原理與超參（`r` / `lora_alpha` / `target_modules` / `dropout`）的選擇依據。
- 用 `get_peft_model` + 明確 `LoraConfig` 取代手刻凍結；對照 IA3 與 LoRA。
- 用 `apply_chat_template` + `SFTTrainer` 做標準指令微調。
- 掌握 adapter 載入 / 合併（`merge_and_unload`）與 adapter-only 部署的取捨。
- 建立評測協定（perplexity / 定性評估）避免訓練盲飛。

**WHY**：LoRA 把「更新整個權重矩陣」改成「學一個低秩增量 ΔW = BA」。`r` 是容量旋鈕，`lora_alpha/r` 是縮放。凍結原權重 → 記憶體省、可攜（adapter 只有幾 MB）、可熱插拔。

**時間估計**：5–7 小時。

---

### 模組 04 — kbits-tuning（量化 + QLoRA）

**定位**：把模型權重壓到 16/8/4-bit，讓 13B–20B 模型能在單卡微調。QLoRA = 4-bit 載入 + LoRA。

**前置**：模組 03。

| 子目錄 / Notebook | 教什麼 |
| :--- | :--- |
| [`01-llm_download/download.ipynb`](04-kbits-tuning/01-llm_download/download.ipynb) | 從 Hub 安全載入大型 LLM；`AutoModelForCausalLM` + `device_map='auto'` + `torch_dtype`；`trust_remote_code` 安全說明；`apply_chat_template` 互動對話 |
| [`01-llm_download/chatglm2_load.ipynb`](04-kbits-tuning/01-llm_download/chatglm2_load.ipynb) | 修正：`AutoModelForCausalLM`（非 `AutoModel`）、Hub id（非 Windows 硬路徑） |
| [`02-16bits_training/chatglm3_lora_16bit.ipynb`](04-kbits-tuning/02-16bits_training/chatglm3_lora_16bit.ipynb) · [`llama2_lora_16bit.ipynb`](04-kbits-tuning/02-16bits_training/llama2_lora_16bit.ipynb) | 16-bit LoRA：`SFTTrainer` + `apply_chat_template`；何時 16-bit |
| [`03-8bits_training/chatglm3_lora_8bit.ipynb`](04-kbits-tuning/03-8bits_training/chatglm3_lora_8bit.ipynb) · [`llama2_lora_8bit.ipynb`](04-kbits-tuning/03-8bits_training/llama2_lora_8bit.ipynb) | 8-bit：`BitsAndBytesConfig` + `prepare_model_for_kbit_training()` |
| [`04-4bits_training/chatglm3_qlora_4bit.ipynb`](04-kbits-tuning/04-4bits_training/chatglm3_qlora_4bit.ipynb) · [`internlm_qlora_4bit.ipynb`](04-kbits-tuning/04-4bits_training/internlm_qlora_4bit.ipynb) · [`llama2_qlora_4bit.ipynb`](04-kbits-tuning/04-4bits_training/llama2_qlora_4bit.ipynb) | QLoRA 完整流程：nf4 + double_quant + compute_dtype |
| [`04-4bits_training/model_weights_distribution.ipynb`](04-kbits-tuning/04-4bits_training/model_weights_distribution.ipynb) | 量化前後權重分布視覺化、`get_memory_footprint()` |
| [`LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb`](04-kbits-tuning/LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb) | 端到端 QLoRA + `SFTTrainer` 參考實作 |

**學習目標**

- 用 `BitsAndBytesConfig` 設定 16/8/4-bit（nf4、compute_dtype、double_quant）並說明何時用哪一檔。
- 完成 QLoRA 流程：4-bit 載入 + `prepare_model_for_kbit_training` + LoRA + `SFTTrainer`。
- 理解量化的記憶體/速度/品質取捨與權重分布視覺化。
- 正確載入大型 LLM（`device_map='auto'`、`torch_dtype`、`trust_remote_code` 安全說明、Hub id）。
- 以 safetensors 儲存與 `push_to_hub`，建立可重現的微調產物。

**現代化重點（before/after）**

舊寫法（裸 kwargs，4.42+ 已棄用）：

```python
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, torch_dtype=torch.half)
```

新寫法（2026 `BitsAndBytesConfig`）：

```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)   # 正確順序：在套 LoRA 之前
model = get_peft_model(model, lora_config)
```

**WHY**：nf4（4-bit NormalFloat）對常態分布權重比 fp4 更精準；double_quant 連量化常數本身也量化，再省一點記憶體；`compute_dtype=bfloat16` 在反量化後以 bf16 計算保持數值穩定。

**時間估計**：8–12 小時。

---

### 模組 05 — Multimodal（多模態，全新）

**定位**：把前四個模組的單模態心智模型**平滑延伸**到影像、語音、視覺語言。學習弧線從視覺基礎一路到整合性的多模態 RAG 與 VLM 微調。

**前置**：模組 04（量化 / PEFT）、模組 02 的 retrieval（給多模態 RAG）。

**學習弧線與六個 notebook**

| # | Notebook | 能力 | 2026 推薦模型 | 回收的舊知識 |
| :-- | :--- | :--- | :--- | :--- |
| 1 | [`01-image_classification/vit_image_classification.ipynb`](05-Multimodal/01-image_classification/vit_image_classification.ipynb) | 影像分類（ViT） | `google/vit-base-patch16-224`、`facebook/dinov2-base`、`microsoft/swinv2-tiny-patch4-window8-256` | 01 的 `Trainer` 管線：把 tokenizer 換成 `AutoImageProcessor`、`input_ids` 換成 `pixel_values`，其餘不變 |
| 2 | [`02-clip_retrieval/clip_image_text_retrieval.ipynb`](05-Multimodal/02-clip_retrieval/clip_image_text_retrieval.ipynb) | CLIP 圖文檢索 | `google/siglip2-base-patch16-224`、`openai/clip-vit-base-patch32`、`jinaai/jina-clip-v2` | 02 `dual_model` 的雙塔 + cosine + 02 retrieval 的 FAISS |
| 3 | [`03-vlm_vqa_captioning/vlm_vqa_captioning.ipynb`](05-Multimodal/03-vlm_vqa_captioning/vlm_vqa_captioning.ipynb) | VQA & 看圖說話 | `Qwen/Qwen2.5-VL-7B-Instruct`、`llava-hf/llava-onevision-qwen2-7b-ov-hf`、`Salesforce/blip2-opt-2.7b`、`microsoft/Florence-2-large` | 02/04 的 `apply_chat_template`（含 image token）+ 04 的 `BitsAndBytesConfig` 4-bit 載入 |
| 4 | [`04-asr_whisper/whisper_asr.ipynb`](05-Multimodal/04-asr_whisper/whisper_asr.ipynb) | ASR（Whisper） | `openai/whisper-large-v3-turbo`、`openai/whisper-large-v3`、`facebook/wav2vec2-base` | `AutoProcessor` 的音訊分支；轉錄後串接 01 分類 / 02 摘要 |
| 5 | [`05-multimodal_rag/multimodal_embeddings_rag.ipynb`](05-Multimodal/05-multimodal_rag/multimodal_embeddings_rag.ipynb) | 多模態 RAG | `jinaai/jina-clip-v2`、`vidore/colpali-v1.3`、`Qwen/Qwen2.5-VL-7B-Instruct` | 02 文字 RAG + 本模組 CLIP 嵌入 + VLM 生成 |
| 6 | [`06-vlm_finetuning/vlm_lora_finetune.ipynb`](05-Multimodal/06-vlm_finetuning/vlm_lora_finetune.ipynb) | VLM 上的 LoRA/QLoRA | `Qwen/Qwen2.5-VL-7B-Instruct`、`llava-hf/llava-1.5-7b-hf`、`HuggingFaceTB/SmolVLM-Instruct`、`google/paligemma2-3b-pt-224` | 03/04 的 LoRA/QLoRA + `SFTTrainer` 遷移到 VLM |

**學習目標**

- 把單模態心智模型延伸到多模態：tokenizer→`AutoProcessor`、`input_ids`→`pixel_values`/audio features、`Trainer`→多模態 `SFTTrainer`。
- 完成影像分類（ViT）：`image_processor` + `with_transform` + `Trainer`。
- 建立 CLIP/SigLIP 共享嵌入空間並做 zero-shot 分類與圖文雙向檢索（Recall@k / MRR）。
- 用 Qwen2.5-VL / LLaVA / BLIP-2 做 VQA 與中文 captioning，理解 image token 如何進 chat 模板。
- 用 Whisper 做 ASR（chunking / 時間戳 / WER）並串接既有文字模型成跨模態管線。
- 建立跨模態 embedding 與多模態 RAG（含 ColPali 文件視覺檢索）。
- 在 VLM 上做 4-bit QLoRA 微調：多模態 `target_modules`、image token 遮罩、processor collator。

**WHY（單一抽象升級）**：純文字用 `tokenizer`；多模態用 `AutoProcessor = tokenizer + image_processor / feature_extractor`。它同時產生 `input_ids`（文字 token）與 `pixel_values`（影像）或 log-mel（音訊）。VLM 的多模態訊息就是把 image/audio 當成序列中的特殊 token：

```python
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "這張圖是什麼?"},
]}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, ...)
output = model.generate(**inputs)
```

這就是 02 學過的 `apply_chat_template` 的同一套機制——只是訊息裡多了 image。

前置理論（六個 notebook 共用）見 [`05-Multimodal/00-multimodal-foundations.md`](05-Multimodal/00-multimodal-foundations.md)。

**時間估計**：14–20 小時。

---

## 7. 知識回收對照表（單模態 → 多模態）

05 模組刻意不另起爐灶。下表是「你已經會的 → 在多模態怎麼延伸」：

| 你在 01–04 學過 | 在 05 多模態的對應 |
| :--- | :--- |
| `AutoTokenizer` | `AutoProcessor` / `AutoImageProcessor`（tokenizer 的多模態超集） |
| `input_ids` | `pixel_values`（影像）、log-mel features（音訊） |
| `Trainer` 全量分類（01） | ViT 影像分類，同一 `Trainer` + `compute_metrics`（05-01） |
| dual_model 雙塔 cosine（02） | CLIP/SigLIP 影像塔 + 文字塔，同一對比學習（05-02） |
| FAISS dense retrieval（02） | 跨模態 FAISS：同一索引存圖文統一嵌入（05-02、05-05） |
| `apply_chat_template`（02/04） | 含 image/audio token 的多模態 chat 模板（05-03、05-06） |
| 文字 RAG（02） | 多模態 retriever + VLM generator（05-05） |
| `BitsAndBytesConfig` 4-bit（04） | 4-bit 載入 7B VLM（05-03、05-06） |
| LoRA/QLoRA + `SFTTrainer`（03/04） | VLM 上的 QLoRA：`target_modules` 含 LLM + projector，凍結 vision encoder（05-06） |
| `-100` 只在 response 算 loss（02） | image token 與 prompt 設 `-100`，只在 response 算 loss（05-06） |
| evaluate F1 / rouge（01/02） | Recall@k / MRR（檢索）、WER（語音）、CIDEr（captioning）（05 全程） |

---

## 8. 2026 現代化說明（相對舊版改了什麼）

整套教材從 2024 年初寫法升級到 2026。八項橫向現代化主題（完整對照見 [`00-Setup-and-Foundations/01-2026-conventions.md`](00-Setup-and-Foundations/01-2026-conventions.md)）：

| # | 主題 | 舊寫法 → 新寫法 |
| :-- | :--- | :--- |
| 1 | 裝置與精度載入 | `.cuda()` / `device=0` / `torch.half` → `device_map='auto'` + `torch_dtype=bfloat16` + `use_safetensors=True` |
| 2 | 量化設定 | 裸 `load_in_4bit=True` → `quantization_config=BitsAndBytesConfig(...)` + `prepare_model_for_kbit_training()` |
| 3 | 對話格式 | 硬寫 `'Human:/Assistant:'` / `build_chat_input()` → `tokenizer.apply_chat_template()` |
| 4 | 指令微調 | 手刻 `Trainer` + `DataCollatorForSeq2Seq` + 手刻 `-100` → `trl.SFTTrainer` + `SFTConfig` |
| 5 | 監督任務 | 手刻 epoch/batch 迴圈 + `Adam` → `Trainer` + 完整 `TrainingArguments` + `AdamW` |
| 6 | 資料與評測 | pandas + 固定 padding + `rouge_chinese` → `load_dataset` + `map(batched=True)` + 動態 padding + `evaluate.load` |
| 7 | 可重現性與 Hub | Google Drive / `d:/...` 硬路徑 → Hub id / `pathlib` + 環境變數、`set_seed(42)`、`push_to_hub` |
| 8 | processor 抽象 | 把影像當 PIL 物件 → `AutoProcessor` / `AutoImageProcessor`（多模態前置） |

---

## 9. 建議學習節奏

| 學習者類型 | 建議路徑 | 累計時間 |
| :--- | :--- | :--- |
| 完全初學（NLP 入門） | 00 → 01 → 02，先把元件與任務打穩，暫不碰微調 | ~18–24 小時 |
| 想學微調 | 00 → 01（速讀）→ 02（重點）→ 03 → 04 | ~30 小時 |
| 想學多模態 | 完成 02（retrieval）與 04（量化/PEFT）後 → 05 全部 | 前置 + 14–20 小時 |
| 完整通關 | 00 → 01 → 02 → 03 → 04 → 05 | ~50–65 小時 |

> 建議每個模組跑完後，回到對應 `README.md` 的「常見陷阱」段落自我檢核，再進下一模組。

---

## 10. 授權與致謝

- 教學內容以繁體中文撰寫，程式碼註解使用英文。
- 模型與資料集版權歸原作者所有；使用前請檢視各 model card 的 license 與使用條款。
- 本教材以 HuggingFace 生態系（transformers / datasets / evaluate / peft / trl / accelerate / bitsandbytes）為基礎，感謝開源社群。

---

**相關文件快速連結**

- 環境建置：[`00-Setup-and-Foundations/00-environment-setup.md`](00-Setup-and-Foundations/00-environment-setup.md)
- 2026 慣例聖經：[`00-Setup-and-Foundations/01-2026-conventions.md`](00-Setup-and-Foundations/01-2026-conventions.md)
- 術語與架構速查：[`00-Setup-and-Foundations/02-glossary-and-architectures.md`](00-Setup-and-Foundations/02-glossary-and-architectures.md)
- 各模組導覽：[`01-Component/README.md`](01-Component/README.md) · [`02-Adv-tasks/README.md`](02-Adv-tasks/README.md) · [`03-PEFT/README.md`](03-PEFT/README.md) · [`04-kbits-tuning/README.md`](04-kbits-tuning/README.md) · [`05-Multimodal/README.md`](05-Multimodal/README.md)
- 多模態前置理論：[`05-Multimodal/00-multimodal-foundations.md`](05-Multimodal/00-multimodal-foundations.md)
