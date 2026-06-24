# 2024 → 2026 現代化遷移指南（Modernization Guide）

> 本文件是整個 cookbook 的「現代化聖經」。它把散落在 36 個 notebook 中的舊寫法，逐一對映到 2026 年的標準寫法，並解釋**為什麼**要這樣改，而不只是怎麼改。
>
> 配合閱讀：
> - 環境建置：[`00-Setup-and-Foundations/00-environment-setup.md`](00-Setup-and-Foundations/00-environment-setup.md)
> - 共用慣例速查：[`00-Setup-and-Foundations/01-2026-conventions.md`](00-Setup-and-Foundations/01-2026-conventions.md)
> - 術語與架構：[`00-Setup-and-Foundations/02-glossary-and-architectures.md`](00-Setup-and-Foundations/02-glossary-and-architectures.md)
> - 專案總覽：[`README.md`](README.md)

---

## 為什麼要做這次現代化

這套教材原本停在 2024 年初的寫法。當時可以跑，現在多半也還能跑——但它教給學習者的是一堆**已被官方棄用、彼此不一致、且無法延伸到多模態**的習慣。問題不在「跑不動」，而在「學錯」。

舊版的三個系統性病灶：

1. **每個 notebook 長得都不一樣。** 有的 `model.cuda()`、有的 `device=0`、有的 `.to('cuda')`；有的存 `pytorch_model.bin`、有的不存。學習者每換一個檔案就要重新適應一套寫法，認知負擔全花在無意義的差異上。
2. **手刻取代抽象。** 大量手刻訓練迴圈、手刻 `-100` 標籤遮罩、手刻 chat 模板字串。這些不是教學深度，而是 2024 年還沒有好工具時的權宜之計。2026 年 `Trainer` / `SFTTrainer` / `apply_chat_template()` 已經把這些變成單行呼叫。
3. **沒有通往多模態的橋。** 全 repo 把影像當 PIL 物件、把 `tokenizer` 當唯一入口。2026 年的核心抽象是 `AutoProcessor`，它把 text / image / audio 統一起來——舊寫法完全沒鋪這條路。

**現代化的第一原則：消除特殊情況。** 好的教材沒有特殊情況。每個 notebook 都該用同一套載入、同一套訓練、同一套儲存。下面 8 個主題就是把這些特殊情況一個一個消掉。

---

## 2026 鎖定版本

所有 notebook 統一在以下版本基線上運行。請勿使用未鎖版本的 `pip install`。

```txt
transformers>=4.46
datasets>=3.0
trl>=0.12
peft>=0.13
accelerate>=1.0
bitsandbytes>=0.44
evaluate>=0.4
safetensors>=0.4
torch>=2.4
```

詳細安裝步驟與 CUDA / `bitsandbytes` 疑難排解，見 [`00-Setup-and-Foundations/00-environment-setup.md`](00-Setup-and-Foundations/00-environment-setup.md)。

---

## 主題對照總表

| # | 主題 | 舊寫法 | 2026 寫法 | 主要受影響模組 |
|---|------|--------|-----------|----------------|
| 1 | 裝置與精度載入 | `model.cuda()`、`device=0`、`torch.half` | `device_map='auto'` + `torch_dtype=torch.bfloat16` + `use_safetensors=True` | 01, 02, 04 |
| 2 | 量化設定 | `load_in_4bit=True` 裸 kwarg | `quantization_config=BitsAndBytesConfig(...)` | 04 |
| 3 | chat 模板 | 手寫 `Human:/Assistant:`、`build_chat_input()`、`model.chat()` | `tokenizer.apply_chat_template()` | 02, 03, 04 |
| 4 | 指令微調 | 手刻 `Trainer` + 手刻 `-100` 遮罩 | `trl.SFTTrainer` + `SFTConfig` | 02, 03, 04 |
| 5 | 監督訓練 | 手刻 epoch/batch 迴圈 | `Trainer` + 完整 `TrainingArguments` | 01, 02 |
| 6 | 資料與評測 | pandas + 自製 Dataset + `rouge_chinese` | `load_dataset` + `map(batched=True)` + `evaluate.load` | 01, 02 |
| 7 | 環境與可重現性 | Colab mount、硬路徑、無 seed | 鎖版本 + `pathlib` + `set_seed(42)` + Hub | 01, 02, 04 |
| 8 | processor 抽象 | `tokenizer` 為唯一入口、影像當 PIL | `AutoProcessor` / `AutoImageProcessor` | 01（為 05-Multimodal 鋪路） |

---

## 主題 1：統一裝置與精度載入

### 為什麼

舊版用三種互不相容的方式把模型搬上 GPU：`model.cuda()`（寫死 CUDA，沒 GPU 就崩）、`pipeline(device=0)`（用整數，語意不透明——為什麼 0 是 GPU、−1 是 CPU？）、以及零散的 `.to(device)`。同時用 `torch.half`（fp16）卻不解釋為什麼。

2026 的單一慣例：`device_map='auto'` 由 `accelerate` 自動把層分配到 GPU / CPU / disk（記憶體不夠會自動 offload），`torch_dtype=torch.bfloat16` 用數值範圍更穩的 bf16，`use_safetensors=True` 用安全且載入更快的格式。

**為什麼 bf16 優於 fp16/fp32：** fp32 佔記憶體最多但沒必要；fp16 的指數位太少，訓練大模型容易溢位（NaN）；bf16 的指數位與 fp32 相同，數值穩定，是 A100/H100 級硬體的預設選擇。

**為什麼 safetensors 取代 pickle：** `pytorch_model.bin` 是 pickle 格式，載入時會執行任意程式碼（安全風險），且載入慢。`.safetensors` 是純資料格式，零程式碼執行、記憶體映射載入更快。

### Before / After

```python
# Before — 寫死 CUDA、整數 device、fp16 不解釋
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.half)
model = model.cuda()
pipe = pipeline("text-classification", model=model, device=0)
model.save_pretrained("output")  # 存成 pytorch_model.bin
```

```python
# After — 單一慣例，任何硬體都能跑
import torch
from transformers import AutoModelForCausalLM, pipeline

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
pipe = pipeline("text-classification", model=model_id, device_map="auto")
model.save_pretrained("output", safe_serialization=True)  # 存成 .safetensors
```

### 受影響 notebook

- [`01-Component/01pipeline/01.pipeline.ipynb`](01-Component/01pipeline/01.pipeline.ipynb)（`device=0/-1` → `device_map='auto'`）
- [`01-Component/03Model/03.Model.ipynb`](01-Component/03Model/03.Model.ipynb)
- [`01-Component/03Model/03 Model classification_demo.ipynb`](01-Component/03Model/03%20Model%20classification_demo.ipynb)
- [`02-Adv-tasks/02-token_classification/ner.ipynb`](02-Adv-tasks/02-token_classification/ner.ipynb)
- [`02-Adv-tasks/06-language_model/causal_lm.ipynb`](02-Adv-tasks/06-language_model/causal_lm.ipynb)
- [`04-kbits-tuning/01-llm_download/chatglm2_load.ipynb`](04-kbits-tuning/01-llm_download/chatglm2_load.ipynb)（`AutoModel` → `AutoModelForCausalLM`，補 `device_map`/`torch_dtype`）
- [`04-kbits-tuning/02-16bits_training/chatglm3_infer.ipynb`](04-kbits-tuning/02-16bits_training/chatglm3_infer.ipynb)（移除已棄用的 `low_cpu_mem_usage=True`）
- [`04-kbits-tuning/02-16bits_training/llama2_lora_16bit.ipynb`](04-kbits-tuning/02-16bits_training/llama2_lora_16bit.ipynb)

---

## 主題 2：以 BitsAndBytesConfig 取代裸 kwargs 量化

### 為什麼

`from_pretrained(load_in_4bit=True)` 與 `load_in_8bit=True` 這類裸參數在 transformers 4.42+ 已被棄用，未來會移除。它們也無法表達現代量化需要的細節（quant type、compute dtype、double quant）。2026 統一用 `BitsAndBytesConfig` 物件。

這是 04 模組現有最大的教學缺口。請務必把以下取捨講清楚：

- **4-bit vs 8-bit vs 16-bit 何時用：** VRAM < 24GB 用 4-bit（QLoRA）；24–48GB 可用 8-bit；充足時用 16-bit 訓練品質最好。
- **nf4 vs fp4：** `nf4`（NormalFloat4）針對常態分佈的權重設計，幾乎總是優於 `fp4`，是 QLoRA 論文預設。
- **double_quant：** 對量化常數再做一次量化，每參數再省約 0.4 bit，品質損失可忽略，建議開啟。

另外補上正確順序：4-bit 載入後、套 LoRA 前，必須呼叫 `prepare_model_for_kbit_training()`（處理 layer norm 升精度、開啟 gradient checkpointing 相容性、`enable_input_require_grads()`）。

### Before / After

```python
# Before — 裸 kwarg（已棄用），torch.half，順序不對
model = AutoModelForCausalLM.from_pretrained(
    model_path, load_in_4bit=True, torch_dtype=torch.half, device_map="auto"
)
model = get_peft_model(model, lora_config)  # 缺 prepare_model_for_kbit_training
```

```python
# After — BitsAndBytesConfig + 正確順序
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

### 受影響 notebook

- [`04-kbits-tuning/02-16bits_training/chatglm3_lora_16bit.ipynb`](04-kbits-tuning/02-16bits_training/chatglm3_lora_16bit.ipynb)
- [`04-kbits-tuning/03-8bits_training/chatglm3_lora_8bit.ipynb`](04-kbits-tuning/03-8bits_training/chatglm3_lora_8bit.ipynb)
- [`04-kbits-tuning/03-8bits_training/llama2_lora_8bit.ipynb`](04-kbits-tuning/03-8bits_training/llama2_lora_8bit.ipynb)
- [`04-kbits-tuning/04-4bits_training/chatglm3_qlora_4bit.ipynb`](04-kbits-tuning/04-4bits_training/chatglm3_qlora_4bit.ipynb)
- [`04-kbits-tuning/04-4bits_training/internlm_qlora_4bit.ipynb`](04-kbits-tuning/04-4bits_training/internlm_qlora_4bit.ipynb)
- [`04-kbits-tuning/04-4bits_training/llama2_qlora_4bit.ipynb`](04-kbits-tuning/04-4bits_training/llama2_qlora_4bit.ipynb)
- [`04-kbits-tuning/04-4bits_training/model_weights_distribution.ipynb`](04-kbits-tuning/04-4bits_training/model_weights_distribution.ipynb)（量化前後權重分佈視覺化）
- [`04-kbits-tuning/LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb`](04-kbits-tuning/LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb)

---

## 主題 3：chat 一律走 apply_chat_template()

### 為什麼

舊版散落各種脆弱的、模型專屬的對話格式：硬寫 `"Human: ... Assistant:"`、`f"<s>[INST] {prompt} [/INST]"`、ChatGLM 專屬的 `tokenizer.build_chat_input()` 與 `model.chat()`。問題是每個模型的特殊 token 與格式都不同，手寫一定會在換模型時出錯，而且**訓練用的格式跟推論用的格式很容易不一致**（這是微調後模型「變笨」的常見原因）。

`apply_chat_template()` 把這套格式抽象成模型自帶的 Jinja 模板（存在 `tokenizer.chat_template`）。訓練側用它組標籤、推論側用它組 prompt，保證一致。更關鍵的是：**2026 的多模態訊息（image / audio token）走的是同一套機制**——這是通往 05-Multimodal 的橋樑（見 [`05-Multimodal/00-multimodal-foundations.md`](05-Multimodal/00-multimodal-foundations.md)）。

### Before / After

```python
# Before — 硬寫格式（脆弱，換模型即壞），或 ChatGLM 專屬方法
prompt = f"<s>[INST] {instruction} [/INST]"
# 或
inputs = tokenizer.build_chat_input(query, history=[])
response, _ = model.chat(tokenizer, query, history=[])
```

```python
# After — 跨模型可攜的單一抽象
messages = [{"role": "user", "content": instruction}]

# 推論：組 prompt
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512)

# 多模態同一套機制（見 05-Multimodal）
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "這張圖是什麼?"},
]}]
```

### 受影響 notebook

- [`02-Adv-tasks/08-generative_chatbot/chatbot.ipynb`](02-Adv-tasks/08-generative_chatbot/chatbot.ipynb)
- [`03-PEFT/01-LoRA/chatbot_lora.ipynb`](03-PEFT/01-LoRA/chatbot_lora.ipynb)
- [`03-PEFT/01-LoRA/lora_inference.ipynb`](03-PEFT/01-LoRA/lora_inference.ipynb)
- [`03-PEFT/02-IA3/chatbot_ia3.ipynb`](03-PEFT/02-IA3/chatbot_ia3.ipynb)
- [`04-kbits-tuning/01-llm_download/download.ipynb`](04-kbits-tuning/01-llm_download/download.ipynb)（同時修掉 `'assistance'` 拼字 bug → `'assistant'`）
- [`04-kbits-tuning/02-16bits_training/chatglm3_infer.ipynb`](04-kbits-tuning/02-16bits_training/chatglm3_infer.ipynb)
- [`04-kbits-tuning/02-16bits_training/chatglm3_lora_16bit.ipynb`](04-kbits-tuning/02-16bits_training/chatglm3_lora_16bit.ipynb)
- [`04-kbits-tuning/04-4bits_training/internlm_qlora_4bit.ipynb`](04-kbits-tuning/04-4bits_training/internlm_qlora_4bit.ipynb)
- [`04-kbits-tuning/LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb`](04-kbits-tuning/LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb)

---

## 主題 4：指令微調一律用 trl.SFTTrainer

### 為什麼

舊版的 SFT / 指令微調全是手刻：自己用 `process_func` 把 instruction 與 response 拼起來、自己算 `-100` 標籤遮罩、自己配 `DataCollatorForSeq2Seq`。這不只囉嗦，還反覆出現「手刻標籤但從不解釋為什麼」的教學缺口。

`SFTTrainer` 把這些變成設定：用 `formatting_func` 或 chat 欄位自動套模板、自動 packing（把短樣本打包成滿序列，提升 3–8 倍吞吐）、自動 response-only 標籤遮罩。

**保留一個「底層手刻版」作對照教學**，明確解釋 `-100` 遮罩的意義：cross-entropy loss 會忽略標籤為 `-100` 的位置，所以我們把 instruction / prompt 部分設成 `-100`，**只在 response token 上算 loss**——這樣模型學的是「怎麼回答」而不是「怎麼複述問題」。但預設路徑用 `SFTTrainer`。

### Before / After

```python
# Before — 手刻 process_func + 手刻 -100 + 手刻 collator + 通用 Trainer
def process_func(example):
    instruction = tokenizer("Human: " + example["instruction"] + "\n\nAssistant: ")
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

ds = ds.map(process_func, remove_columns=ds.column_names)
trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True))
```

```python
# After — SFTTrainer 自動處理模板 / packing / response-only 遮罩
from trl import SFTTrainer, SFTConfig

def formatting_func(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

sft_config = SFTConfig(
    output_dir="output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,   # effective batch = 2 × 16 = 32
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True,
    packing=True,
    save_safetensors=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    seed=42,
)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds,
    peft_config=lora_config,
    formatting_func=formatting_func,
    processing_class=tokenizer,   # 2026：取代舊的 tokenizer= 參數
)
trainer.train()
```

### 受影響 notebook

- [`02-Adv-tasks/06-language_model/causal_lm.ipynb`](02-Adv-tasks/06-language_model/causal_lm.ipynb)
- [`02-Adv-tasks/08-generative_chatbot/chatbot.ipynb`](02-Adv-tasks/08-generative_chatbot/chatbot.ipynb)
- [`03-PEFT/01-LoRA/chatbot_lora.ipynb`](03-PEFT/01-LoRA/chatbot_lora.ipynb)
- [`03-PEFT/02-IA3/chatbot_ia3.ipynb`](03-PEFT/02-IA3/chatbot_ia3.ipynb)
- [`04-kbits-tuning/02-16bits_training/chatglm3_lora_16bit.ipynb`](04-kbits-tuning/02-16bits_training/chatglm3_lora_16bit.ipynb)
- [`04-kbits-tuning/02-16bits_training/llama2_lora_16bit.ipynb`](04-kbits-tuning/02-16bits_training/llama2_lora_16bit.ipynb)
- [`04-kbits-tuning/03-8bits_training/llama2_lora_8bit.ipynb`](04-kbits-tuning/03-8bits_training/llama2_lora_8bit.ipynb)
- [`04-kbits-tuning/04-4bits_training/llama2_qlora_4bit.ipynb`](04-kbits-tuning/04-4bits_training/llama2_qlora_4bit.ipynb)

---

## 主題 5：純監督任務一律用 Trainer + 完整 TrainingArguments

### 為什麼

01 / 02 模組充斥手刻訓練迴圈：手寫 epoch/batch 迴圈、`torch.optim.Adam`、每個 batch 都 `.cuda()`。這是 2024 之前沒有好工具的遺跡。`Trainer` 一次處理 device 搬移、混合精度、梯度累積、checkpoint、early stopping、logging——五行設定取代五十行樣板。

順手修掉的教學缺口：

- **Adam → AdamW：** transformer 微調的標準是 AdamW，它把 weight decay 與梯度更新解耦，正則化更正確。
- **為何 warmup：** 訓練初期權重隨機，大學習率會破壞預訓練表徵；warmup 讓 lr 從 0 線性爬升，穩定起步（`warmup_ratio=0.1`）。
- **effective batch：** `per_device_batch_size × gradient_accumulation_steps × GPU 數`。小顯卡靠梯度累積撐出大有效 batch。

### Before / After

```python
# Before — 手刻迴圈、Adam、per-batch .cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
model = model.cuda()
for epoch in range(3):
    for batch in train_loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

```python
# After — Trainer + 現代化必填參數
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    bf16=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    optim="adamw_torch_fused",     # AdamW，非 Adam
    eval_strategy="steps",
    eval_steps=100,
    save_safetensors=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    seed=42,
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()
```

### 受影響 notebook

- [`01-Component/03Model/03 Model classification_demo.ipynb`](01-Component/03Model/03%20Model%20classification_demo.ipynb)
- [`01-Component/04Datasets/04 Datasets classification_demo.ipynb`](01-Component/04Datasets/04%20Datasets%20classification_demo.ipynb)
- [`01-Component/05evaluate/05 evaluate classification_demo.ipynb`](01-Component/05evaluate/05%20evaluate%20classification_demo.ipynb)
- [`01-Component/05evaluate/05 evaluate classification_demo_tweets.ipynb`](01-Component/05evaluate/05%20evaluate%20classification_demo_tweets.ipynb)
- [`01-Component/06Trainer/06 Trainer classification_demo.ipynb`](01-Component/06Trainer/06%20Trainer%20classification_demo.ipynb)（已用 Trainer，補齊 `bf16`/`warmup_ratio`/`max_grad_norm`/`save_safetensors`/`seed`，修 `f1_metirc` 拼字）
- [`02-Adv-tasks/01-finetune_optimize/01 train opti classification_demo.ipynb`](02-Adv-tasks/01-finetune_optimize/01%20train%20opti%20classification_demo.ipynb)

---

## 主題 6：資料與評測現代化

### 為什麼

舊版用 pandas 載入 + 自製 `Dataset` / `collate_fn`，並把 padding 寫死成固定 128。評測各自為政，摘要還依賴第三方 `rouge_chinese`。

2026 統一管線：`load_dataset` → `dataset.map(tokenize_fn, batched=True, num_proc=N)` → `DataCollatorWithPadding`（動態 padding，依 batch 內最長序列填補，比固定 128 快且省記憶體）。

教學重點：

- **為何 `batched=True` 快 3–5 倍：** tokenizer 走 Rust 後端做向量化批次處理，避免逐筆 Python 迴圈。
- **為何要 stratify：** 類別不平衡（如 ChnSentiCorp 正負約 2.5:1）時，隨機切分會讓 minority class 在驗證集比例失真。用 `train_test_split(stratify_by_column="label", seed=42)`。
- **回歸 vs 分類的修正：** 句相似度是**回歸**任務（`num_labels=1`），舊版卻用 threshold 把輸出當二分類算 F1，理論上錯誤。改用 MSE / Pearson 相關係數。
- **`rouge_chinese` → `evaluate.load("rouge")`：** 統一進官方 evaluate 生態。

### Before / After

```python
# Before — pandas + 固定 padding + 第三方 rouge + threshold 當分類
df = pd.read_csv("data.csv")
encodings = tokenizer(list(df.text), padding="max_length", max_length=128, truncation=True)
# 句相似度回歸卻用 threshold：
preds = (logits > 0.5).astype(int); f1 = f1_score(labels, preds)
# 摘要：
from rouge_chinese import Rouge
```

```python
# After — datasets 3.x + 動態 padding + evaluate + 正確指標
from datasets import load_dataset
from transformers import DataCollatorWithPadding
import evaluate

ds = load_dataset("csv", data_files="data.csv")["train"]
ds = ds.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True)   # 動態 padding 留給 collator

ds = ds.map(tokenize_fn, batched=True, num_proc=4)
collator = DataCollatorWithPadding(tokenizer)          # 依 batch 最長序列填補

# 分類：accuracy + precision/recall/F1
acc, f1 = evaluate.load("accuracy"), evaluate.load("f1")
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return {**acc.compute(predictions=preds, references=p.label_ids),
            **f1.compute(predictions=preds, references=p.label_ids, average="macro")}

# 回歸（句相似度）：用 MSE / Pearson，不要 threshold
mse, pearson = evaluate.load("mse"), evaluate.load("pearsonr")

# 摘要：官方 rouge
rouge = evaluate.load("rouge")
```

### 受影響 notebook

- [`01-Component/03Model/dataset/ChnSentiCorp_htl_all.ipynb`](01-Component/03Model/dataset/ChnSentiCorp_htl_all.ipynb)（pandas + 手動平衡 → `load_dataset` + stratified split）
- [`01-Component/04Datasets/04 Datasets.ipynb`](01-Component/04Datasets/04%20Datasets.ipynb)
- [`01-Component/05evaluate/05 evaluate.ipynb`](01-Component/05evaluate/05%20evaluate.ipynb)
- [`02-Adv-tasks/04-sentence_similarity/cross_model.ipynb`](02-Adv-tasks/04-sentence_similarity/cross_model.ipynb)（修回歸當分類的錯誤）
- [`02-Adv-tasks/04-sentence_similarity/dual_model.ipynb`](02-Adv-tasks/04-sentence_similarity/dual_model.ipynb)（cosine 相似度改用 Pearson/Spearman 而非硬 threshold 0.7）
- [`02-Adv-tasks/07-text_summarization/summarization.ipynb`](02-Adv-tasks/07-text_summarization/summarization.ipynb)（`rouge_chinese` → `evaluate`）
- [`02-Adv-tasks/07-text_summarization/summarization_glm.ipynb`](02-Adv-tasks/07-text_summarization/summarization_glm.ipynb)

> NER 任務（[`02-Adv-tasks/02-token_classification/ner.ipynb`](02-Adv-tasks/02-token_classification/ner.ipynb)）保留 `evaluate.load("seqeval")`，但 `compute_metrics` 應同時回傳 precision / recall / F1，並以 `word_ids()` 對齊 subword 標籤、padding 設 `-100`。

---

## 主題 7：環境鎖版本、Hub 整合、可重現性

### 為什麼

舊版散落 Google Drive mount、`d:/Pretrained_models/` 等硬路徑，換一台機器就壞；沒有 seed，結果無法重現；還有一堆小 bug。可重現性與可攜性是教材的基本品質，不是加分項。

固定動作：

- **鎖版本**（見本文頂部清單），禁止無版本 `pip install`。
- **移除硬路徑**：改用 HF Hub model id，或 `pathlib.Path` + 環境變數（`HF_HOME` 控制快取位置）。
- **`set_seed(42)`** 放在每個訓練 notebook 開頭。
- **`push_to_hub` + 最小 model card**：持久化 `id2label`/`label2id`、標註 language / license / task tag。
- **修掉散落 bug**：`f1_metirc` → `f1_metric`、`'assistance'` 角色 → `'assistant'`、cell 96/194 未定義類別引用、`MinMaxScaler` 在 split 前造成的資料洩漏。

### Before / After

```python
# Before — Colab mount、硬路徑、無 seed、洩漏
from google.colab import drive; drive.mount("/content/drive")
model_path = "d:/Pretrained_models/chatglm3-6b"
scaler = MinMaxScaler().fit(all_data)   # 在 split 前 fit → 測試集資訊洩漏
```

```python
# After — Hub id / pathlib、固定 seed、無洩漏
import os
from pathlib import Path
from transformers import set_seed

set_seed(42)
model_id = os.environ.get("MODEL_ID", "THUDM/chatglm3-6b")
data_dir = Path(os.environ.get("DATA_DIR", "./data"))

# 先 split 再 fit，避免洩漏
train, test = ds.train_test_split(test_size=0.2, seed=42).values()
scaler = MinMaxScaler().fit(train_features)   # 只 fit 訓練集
```

### 受影響 notebook

- [`01-Component/01pipeline/01.pipeline.ipynb`](01-Component/01pipeline/01.pipeline.ipynb)（修 cell 96/194 未定義類別引用）
- [`01-Component/05evaluate/05 evaluate classification_demo_tweets.ipynb`](01-Component/05evaluate/05%20evaluate%20classification_demo_tweets.ipynb)（修 `MinMaxScaler` 洩漏）
- [`01-Component/06Trainer/06 Trainer classification_demo.ipynb`](01-Component/06Trainer/06%20Trainer%20classification_demo.ipynb)（修 `f1_metirc`）
- [`02-Adv-tasks/01-finetune_optimize/01 train opti classification_demo.ipynb`](02-Adv-tasks/01-finetune_optimize/01%20train%20opti%20classification_demo.ipynb)
- [`04-kbits-tuning/01-llm_download/chatglm2_load.ipynb`](04-kbits-tuning/01-llm_download/chatglm2_load.ipynb)（硬路徑 → Hub id；補 `trust_remote_code` 安全說明）
- [`04-kbits-tuning/01-llm_download/download.ipynb`](04-kbits-tuning/01-llm_download/download.ipynb)（ModelScope → Hub；修 `'assistance'` 拼字）
- [`04-kbits-tuning/LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb`](04-kbits-tuning/LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb)（Google Drive → `push_to_hub`；train/valid 用同一檔的洩漏問題）

> **`trust_remote_code=True` 的安全說明：** ChatGLM / InternLM 等模型需要它來載入自訂建模程式碼。請理解這會執行模型 repo 中的 Python，務必只對信任來源開啟，並在 notebook 中加上明確警示。

---

## 主題 8：從 tokenizer 到 AutoProcessor（多模態前置）

### 為什麼

舊版把 `tokenizer` 當唯一入口，影像（如 OWL-ViT zero-shot 物件偵測段落）只當 PIL 物件處理。但 2026 的核心抽象是 `AutoProcessor`：

> **processor = tokenizer + feature_extractor / image_processor**

文字模型用 tokenizer 把字轉成 `input_ids`；影像模型用 `image_processor` 把圖轉成 `pixel_values`；音訊模型用 `feature_extractor` 把波形轉成 log-mel。多模態模型需要同時處理這些，`AutoProcessor` 就是統一入口。

在 01 模組就引入這個概念（而非另開主題），把既有被當 PIL 處理的影像段落升級成正確的 processor 慣例，為 05-Multimodal 鋪路。

### Before / After

```python
# Before — 影像當 PIL 物件手動塞給模型
from PIL import Image
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")   # 沒解釋 processor 是什麼
```

```python
# After — 明確引入 AutoProcessor / AutoImageProcessor 抽象
from transformers import AutoImageProcessor, AutoProcessor

# 純視覺：image_processor 處理 resize / normalize（須與模型對齊 mean/std）
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

# 多模態：AutoProcessor 同時處理 input_ids 與 pixel_values
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
inputs = processor(text=["一隻貓"], images=image, return_tensors="pt", padding=True)
# inputs 同時含 input_ids 與 pixel_values
```

### 受影響 notebook

- [`01-Component/01pipeline/01.pipeline.ipynb`](01-Component/01pipeline/01.pipeline.ipynb)（zero-shot-object-detection 段落正式引入 `AutoProcessor`）
- [`01-Component/03Model/03.Model.ipynb`](01-Component/03Model/03.Model.ipynb)
- [`01-Component/02tokenizer`](01-Component/02tokenizer)（補 processor 是 tokenizer 多模態超集的說明）

延伸閱讀：[`05-Multimodal/00-multimodal-foundations.md`](05-Multimodal/00-multimodal-foundations.md)、[`05-Multimodal/README.md`](05-Multimodal/README.md)。

---

## 反模式速查表（舊 → 新）

| 舊寫法（反模式） | 2026 寫法 |
|------------------|-----------|
| `model.cuda()` / `.to('cuda')` | `device_map='auto'`（在 `from_pretrained`） |
| `pipeline(..., device=0)` | `pipeline(..., device_map='auto')` |
| `torch_dtype=torch.half` | `torch_dtype=torch.bfloat16` |
| `low_cpu_mem_usage=True` | `device_map='auto'`（已涵蓋） |
| 存檔成 `pytorch_model.bin` | `save_pretrained(safe_serialization=True)` |
| `load_in_4bit=True`（裸 kwarg） | `quantization_config=BitsAndBytesConfig(...)` |
| 套 LoRA 前不呼叫 | `prepare_model_for_kbit_training(model)` |
| `"Human: ... Assistant:"` 硬寫 | `tokenizer.apply_chat_template(...)` |
| `tokenizer.build_chat_input()` / `model.chat()` | `apply_chat_template` + `model.generate()` |
| 手刻 epoch 迴圈 + `torch.optim.Adam` | `Trainer` + `optim='adamw_torch_fused'` |
| 手刻 `-100` 遮罩 + `DataCollatorForSeq2Seq` | `SFTTrainer`（自動 response-only 遮罩） |
| `SFTTrainer(..., tokenizer=...)` | `SFTTrainer(..., processing_class=...)` |
| `padding='max_length', max_length=128` | `DataCollatorWithPadding`（動態 padding） |
| 逐筆 `map(tokenize_fn)` | `map(tokenize_fn, batched=True, num_proc=N)` |
| `pd.read_csv` + 自製 Dataset | `load_dataset("csv", ...)` |
| 隨機 split | `train_test_split(stratify_by_column=..., seed=42)` |
| 回歸用 threshold 算 F1 | `evaluate.load("mse" / "pearsonr")` |
| `from rouge_chinese import Rouge` | `evaluate.load("rouge")` |
| Google Drive mount / 硬路徑 | HF Hub id 或 `pathlib` + 環境變數 |
| 無 seed | `set_seed(42)` |
| 手動存 model 後不發布 | `push_to_hub` + 最小 model card |
| `tokenizer` 當唯一入口 | `AutoProcessor` / `AutoImageProcessor` |

---

## 各模組現代化重點與延伸

| 模組 | 現代化重點 | 導覽文件 |
|------|-----------|----------|
| 01-Component | 主題 1、5、6、8（載入、Trainer、資料評測、processor 抽象） | [`01-Component/README.md`](01-Component/README.md) |
| 02-Adv-tasks | 主題 3、4、6（chat 模板、SFTTrainer、評測指標修正） | [`02-Adv-tasks/README.md`](02-Adv-tasks/README.md) |
| 03-PEFT | 主題 3、4（apply_chat_template + SFTTrainer 標準流程、adapter 合併） | [`03-PEFT/README.md`](03-PEFT/README.md) |
| 04-kbits-tuning | 主題 1、2、3、4（量化階梯、BitsAndBytesConfig、QLoRA 全流程） | [`04-kbits-tuning/README.md`](04-kbits-tuning/README.md) |
| 05-Multimodal（全新） | 主題 8 的延伸：把單模態心智模型平滑延伸到多模態 | [`05-Multimodal/README.md`](05-Multimodal/README.md) |

完成橫向現代化後，請接續 05-Multimodal 模組：它把本文教的所有抽象（`device_map`、`BitsAndBytesConfig`、`apply_chat_template`、`SFTTrainer`、`AutoProcessor`）延伸到影像分類、CLIP 檢索、VLM 視覺問答、Whisper 語音辨識、多模態 RAG 與 VLM 微調。

---

**最後更新：** 2026-06-25　**適用範圍：** 整個 HuggingFace_scratch cookbook
