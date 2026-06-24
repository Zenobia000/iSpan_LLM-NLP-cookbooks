# 03-PEFT：參數高效微調（Parameter-Efficient Fine-Tuning）

> 模組定位：在 [02-Adv-tasks](../02-Adv-tasks/README.md) 你已經會用 `Trainer` 對整個模型做監督微調；本模組教你**只訓練極少數參數**就能把一個通用 LLM 調成你要的樣子。這是進入 [04-kbits-tuning](../04-kbits-tuning/README.md)（量化 + QLoRA）與 [05-Multimodal](../05-Multimodal/README.md)（VLM 微調）的核心前置能力。

- 前置需求：完成 [02-Adv-tasks](../02-Adv-tasks/README.md)，理解 causal LM、`-100` 標籤遮罩、`apply_chat_template`。
- 共用慣例：本模組所有寫法遵循 [00-Setup-and-Foundations/01-2026-conventions.md](../00-Setup-and-Foundations/01-2026-conventions.md)，術語見 [02-glossary-and-architectures.md](../00-Setup-and-Foundations/02-glossary-and-architectures.md)。

---

## 1. 為什麼需要 PEFT（先講 WHY）

全參數微調（full fine-tuning）一個 7B 模型，你要面對三個現實成本：

1. **記憶體爆炸**：7B 參數用 bf16 存權重就要約 14 GB，再加上 AdamW 的兩份 optimizer state（momentum + variance）與梯度，實際訓練峰值約是權重的 **4 倍**（約 56 GB），單張消費級 GPU 直接 OOM。
2. **儲存與部署不可攜**：每微調一個任務就複製一份完整 14 GB 權重。十個任務就是 140 GB，且每次切換任務都要重新載入整個模型。
3. **災難性遺忘風險**：動到全部參數，容易把預訓練學到的通用能力洗掉。

PEFT 的核心洞察：**預訓練模型已經很好，下游適配所需的「改變量」其實是低秩（low-rank）、稀疏的**。所以我們**凍結原始權重**，只在旁邊掛上一小撮可訓練參數。

| 項目 | 全參數微調 | LoRA（PEFT） |
| :--- | :--- | :--- |
| 7B 模型可訓練參數 | ~7,000 M | ~4–20 M（< 0.5%） |
| 訓練峰值記憶體 | ~56 GB | ~16 GB（再配 4-bit 量化可 < 10 GB） |
| 每任務儲存產物 | ~14 GB 完整權重 | ~10–80 MB adapter |
| 任務切換 | 重載整個模型 | 換掛一個 adapter |
| 災難性遺忘 | 高 | 低（原權重不動） |

> 這就是 Linus 式的「消除特殊情況」：與其為每個任務維護一份完整模型（N 個特殊情況），不如維護**一份共用 base + N 個輕量 adapter**，讓部署邏輯永遠長一樣。

---

## 2. LoRA 原理：低秩分解

對某個原始權重矩陣 `W ∈ R^{d×k}`，全參數微調學的是一個更新量 `ΔW`（同樣是 `d×k`）。LoRA 假設這個 `ΔW` 是低秩的，把它分解成兩個小矩陣：

```
ΔW = B · A      其中 A ∈ R^{r×k}, B ∈ R^{d×r},  r << min(d, k)
前向：h = W·x + (lora_alpha / r) · B · A · x
```

- `W` 被**凍結**（不算梯度）。
- 只訓練 `A`（用高斯初始化）與 `B`（初始化為零，所以訓練起點等價於原模型，不破壞既有行為）。
- 參數量從 `d×k` 降到 `r×(d+k)`。以 `d=k=4096, r=8` 為例，從 1,600 萬降到 6.5 萬，約 0.4%。

### 關鍵超參（這幾個就是你會反覆調的旋鈕）

| 超參 | 意義 | 選擇直覺 |
| :--- | :--- | :--- |
| `r`（rank） | 適配的「容量」 | 小資料/簡單任務 `r=8`；困難或資料多 `r=16/32/64`。越大越像全微調，也越耗記憶體。 |
| `lora_alpha` | 縮放因子，實際縮放為 `alpha/r` | 慣例 `alpha = 2 × r`（如 `r=8, alpha=16`）。調 `r` 時通常等比調 `alpha`。 |
| `target_modules` | 把 LoRA 掛在哪些層 | decoder-only 常掛注意力投影 `q_proj/k_proj/v_proj/o_proj`；想更強再加 FFN 的 `gate/up/down_proj`。**注意架構差異**：BLOOM/ChatGLM 是融合的 `query_key_value`，Llama 系是分開的 `q_proj/v_proj`。 |
| `lora_dropout` | adapter 上的 dropout | `0.05` 是安全預設，資料少時可略升以防過擬合。 |
| `bias` | 是否訓練 bias | 通常 `'none'`，最省。 |
| `modules_to_save` | 額外解凍並完整儲存的層 | 換了分類頭或 `lm_head` 時要列入，否則新頭不會被存。 |

`peft` 2026 標準寫法（明確指定所有超參，杜絕「最小 config」的不可重現）：

```python
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,          # 明確宣告任務型別
    r=16,
    lora_alpha=32,                          # alpha = 2 * r
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama 系；BLOOM/ChatGLM 用 ["query_key_value"]
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()          # 印出 trainable% 確認真的只訓練 < 1%
```

> **永遠呼叫 `print_trainable_parameters()`**。如果看到 trainable 百分比異常高，通常是 `target_modules` 寫錯或 base model 沒凍結，這是最常見的「訓練盲飛」起點。

---

## 3. IA3 vs LoRA：另一種更輕的選擇

IA3（Infused Adapters by Inhibiting and Amplifying Inner activations）不加旁路矩陣，而是學三組**逐元素縮放向量**，乘到 key、value 與 FFN 的中間激活上：

```
k' = l_k ⊙ k,   v' = l_v ⊙ v,   ffn' = l_ff ⊙ ffn
```

- 參數量比 LoRA **更少**（只是向量不是矩陣），記憶體更省。
- 通常無需縮放 alpha，訓練超參更少、更穩定。
- 代價：表達能力上限較低，困難任務的天花板通常不如 LoRA。

| 取捨 | LoRA | IA3 |
| :--- | :--- | :--- |
| 可訓練參數 | 少（< 1%） | 極少（比 LoRA 更少） |
| 表達能力 | 較高 | 較低 |
| 超參調校 | `r/alpha/target_modules` | 幾乎免調 |
| 適用情境 | 預設首選、通用 | 極度資源受限、簡單適配 |

```python
from peft import IA3Config, TaskType, get_peft_model

ia3_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],      # 必須是 target_modules 的子集
)
model = get_peft_model(model, ia3_config)
```

> 心智模型：**LoRA 加旁路、IA3 調音量**。先用 LoRA 當預設；只有在 VRAM 真的卡到極限、或任務很簡單時才考慮 IA3。

---

## 4. 2026 標準訓練流程：`apply_chat_template` + `SFTTrainer`

本模組現有 notebook 是 2024 寫法：手刻 `Human:/Assistant:` 模板、用 `Trainer + DataCollatorForSeq2Seq` 並手動拼 `-100` 標籤。2026 的兩個關鍵升級：

### 4.1 chat 一律走 `apply_chat_template()`

硬寫 `"Human: ... Assistant:"` 是 fragile 的：換個模型模板就錯，且訓練/推論很容易用到不同格式。改用 tokenizer 內建模板，**同一套機制在訓練（組標籤）與推論（組 prompt）兩端共用**，保證一致。這也是通往 [05-Multimodal](../05-Multimodal/00-multimodal-foundations.md) 的橋樑——多模態的 image/audio token 是同一套模板機制。

**Before（fragile，03-PEFT/01-LoRA/chatbot_lora.ipynb 現況）**
```python
prompt = f"Human: {example['instruction']}\n\nAssistant: "
text = prompt + example["output"]
# 再手動算哪些 token 設 -100...
```

**After（2026）**
```python
messages = [
    {"role": "user", "content": example["instruction"]},
    {"role": "assistant", "content": example["output"]},
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# 推論端：add_generation_prompt=True 自動補上 assistant 起始
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
)
```

### 4.2 指令微調一律用 `trl.SFTTrainer`

`SFTTrainer` 自動套 chat 模板、自動 packing、自動只在 response token 算 loss（即 response-only 標籤遮罩），把 02/03/04 反覆出現的「手刻 `-100` 但不解釋」一次收掉。

**Before（手刻 Trainer，現況）**
```python
trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer),  # 手動 -100 邏輯散落各處
)
```

**After（2026）**
```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./out",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,          # effective batch = 2 × 8 = 16
    learning_rate=2e-4,                     # LoRA 慣用 lr 比全微調高（1e-4 ~ 3e-4）
    bf16=True,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    save_safetensors=True,
    eval_strategy="steps",
    seed=42,
)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_config,                # 直接傳 LoRA config，省去手動 get_peft_model
    processing_class=tokenizer,             # 2026 取代舊的 tokenizer= 參數
)
trainer.train()
```

> **保留一份「底層手刻版」作對照教學**很有價值：它解釋 `-100` 遮罩為何只算 response token（CrossEntropy 對 `-100` 索引不計 loss，所以模型只學「該怎麼回」而不去背「使用者問了什麼」）。但**預設路徑請用 `SFTTrainer`**。

### `Trainer` vs `SFTTrainer` 何時用

| 用 `Trainer` | 用 `SFTTrainer` |
| :--- | :--- |
| 分類/NER/QA 等監督任務（見 [01-Component](../01-Component/README.md)） | 指令微調 / chat SFT |
| 你要完全掌控 collator 與 loss | 想自動套 chat 模板 + packing + response-only 遮罩 |
| 不涉及 PEFT 也可 | 可直接吃 `peft_config`，一行接上 LoRA/IA3 |

---

## 5. Adapter 的載入、合併與部署取捨

訓練完你得到的是一個輕量 adapter（含 `adapter_config.json` + `adapter_model.safetensors`）。部署有兩條路：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

base = AutoModelForCausalLM.from_pretrained(
    BASE_ID, device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True,
)

# 路線 A：adapter 分離部署（彈性，可熱切換多個任務）
model = PeftModel.from_pretrained(base, "./my-adapter")

# 路線 B：合併進 base（推論最快、無 adapter 額外開銷）
model = PeftModel.from_pretrained(base, "./my-adapter").merge_and_unload()
model.save_pretrained("./merged", safe_serialization=True)
```

| 取捨 | adapter 分離（路線 A） | `merge_and_unload`（路線 B） |
| :--- | :--- | :--- |
| 推論延遲 | 略高（多一次旁路運算） | 最低 |
| 記憶體 | 一份 base 可掛多 adapter | 每個合併模型一份完整權重 |
| 任務切換 | 熱切換、極省 | 需重載整個合併模型 |
| 適用 | 多任務服務、A/B 測試 | 單一任務、追求最低延遲的部署 |

> **鐵律：推論用的 chat 模板必須和訓練時完全一致**。訓練用 `apply_chat_template`，推論就也要用，否則模型看到沒見過的格式，輸出會崩。這是 `03-PEFT/01-LoRA/lora_inference.ipynb` 現況用手寫 `"Human:/Assistant:"` 最該修的點。

---

## 6. 評測協定：別讓訓練盲飛

PEFT 最常見的失敗是「loss 有降但模型沒變好」。最低限度要有：

- **量化指標**：對 causal LM 用 perplexity（`exp(eval_loss)`）追蹤泛化；摘要/翻譯類任務用 `evaluate.load("rouge")`。
- **定性評估**：準備一組固定的 held-out prompt，**訓練前後跑同一批**並肉眼對照輸出（最便宜也最有效）。
- **eval split**：用 `train_test_split(seed=42)` 切出驗證集，在 `SFTConfig` 設 `eval_strategy="steps"`，搭配 `load_best_model_at_end=True` 與 `EarlyStoppingCallback` 避免過擬合。

```python
import math
metrics = trainer.evaluate()
print("perplexity:", math.exp(metrics["eval_loss"]))
```

---

## 7. 本模組 notebook 對照表

| Notebook | 教什麼 | 方法 | base model | 2026 現代化重點 |
| :--- | :--- | :--- | :--- | :--- |
| [01-LoRA/chatbot_lora.ipynb](01-LoRA/chatbot_lora.ipynb) | LoRA 指令微調全流程 | LoRA + SFT | BLOOM-1B | `apply_chat_template` 取代手刻模板；`SFTTrainer` 取代 `Trainer`；明確 `LoraConfig(r/alpha/target_modules)`；`save_safetensors=True` |
| [01-LoRA/lora_inference.ipynb](01-LoRA/lora_inference.ipynb) | 載入 base + adapter 推論、合併 | adapter 載入 / `merge_and_unload` | BLOOM 系 | 以 `device_map='auto'` 取代 `low_cpu_mem_usage`；推論 prompt 改用 `apply_chat_template`；`GenerationConfig` 取代散落 kwargs |
| [02-IA3/chatbot_ia3.ipynb](02-IA3/chatbot_ia3.ipynb) | IA3 微調與 vs LoRA 對照 | IA3 + SFT | BLOOM 系 | 同上現代化；補 LoRA/IA3 的參數量、VRAM、延遲對照 |

---

## 8. 2026 版本與環境須知

本模組假設你已依 [00-Setup-and-Foundations/00-environment-setup.md](../00-Setup-and-Foundations/00-environment-setup.md) 鎖定版本：

```text
transformers>=4.46
trl>=0.12
peft>=0.13
accelerate>=1.0
datasets>=3.0
safetensors>=0.4
torch>=2.4
bitsandbytes>=0.44   # 04 模組 QLoRA 用；本模組做 4-bit 載入時也需要
```

幾個 2026 的 API 變化要注意：

- `SFTTrainer` 的 `tokenizer=` 已改為 `processing_class=`（為了統一文字 tokenizer 與多模態 processor）。
- 儲存一律 `safe_serialization=True`（safetensors），取代舊的 `pytorch_model.bin`（pickle）。safetensors 載入更快、不執行任意程式碼，更安全。
- 模型載入一律 `from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, use_safetensors=True)`，不要再用 `model.cuda()` 或整數 `device=0`。

---

## 9. 學習路徑與銜接

```
02-Adv-tasks（全參數微調、-100 遮罩、chat 模板）
        │
        ▼
03-PEFT（本模組）── LoRA / IA3：凍結 base、只訓練 adapter
        │
        ▼
04-kbits-tuning ── 量化 + QLoRA：4-bit 載入後再掛 LoRA，把 7B 塞進單卡
        │
        ▼
05-Multimodal ── 把 LoRA/QLoRA 遷移到 VLM（多模態 target_modules、image token 遮罩）
```

下一步請接 [04-kbits-tuning/README.md](../04-kbits-tuning/README.md)，學如何在 LoRA 之前先用 `BitsAndBytesConfig` 把模型量化到 4-bit（QLoRA），把本模組的記憶體成本再壓一個量級。

---

## 學習目標檢核

讀完並跑完本模組，你應該能：

- [ ] 說清楚 PEFT 為何重要：記憶體、可攜性、抗遺忘三個面向。
- [ ] 理解 LoRA 低秩分解原理，並能依任務選 `r / lora_alpha / target_modules`。
- [ ] 用 `get_peft_model` + 明確 `LoraConfig` 取代手刻凍結，並用 `print_trainable_parameters()` 驗證。
- [ ] 對照 IA3 與 LoRA 的取捨，知道各自適用情境。
- [ ] 用 `apply_chat_template` + `SFTTrainer` 跑標準指令微調，保證訓練/推論模板一致。
- [ ] 掌握 adapter 載入 / `merge_and_unload` 與部署取捨。
- [ ] 建立 perplexity + 定性評估協定，避免訓練盲飛。
