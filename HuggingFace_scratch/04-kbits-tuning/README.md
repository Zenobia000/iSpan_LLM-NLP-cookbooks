# 04 - 量化與 k-bit 微調（Quantization & k-bit Training）

> 模組定位：用量化（quantization）把大型語言模型塞進你手上有限的顯示卡，並在量化後的權重上掛 LoRA 做參數高效微調（QLoRA）。這是整套 cookbook 從「能載入模型」走到「能在單張消費級 GPU 上微調 7B / 13B / 20B 模型」的關鍵一章。
>
> 前置：[03-PEFT](../03-PEFT/README.md)（LoRA / IA3 原理與 `SFTTrainer` 流程）。
> 後續：[05-Multimodal](../05-Multimodal/README.md)（把這裡學到的 4-bit QLoRA 遷移到 VLM）。

---

## 目錄

1. [這個模組要解決的真問題](#這個模組要解決的真問題)
2. [核心觀念：為什麼要量化](#核心觀念為什麼要量化)
3. [記憶體數學：先算清楚你需要多少 VRAM](#記憶體數學先算清楚你需要多少-vram)
4. [量化階梯：16 / 8 / 4-bit 何時用哪一檔](#量化階梯16--8--4-bit-何時用哪一檔)
5. [`BitsAndBytesConfig` 完整參數](#bitsandbytesconfig-完整參數)
6. [QLoRA 完整流程](#qlora-完整流程)
7. [本模組 notebook 對照表](#本模組-notebook-對照表)
8. [大型 LLM 安全載入](#大型-llm-安全載入)
9. [2026 版本與環境設定](#2026-版本與環境設定)
10. [儲存與發布](#儲存與發布)
11. [常見陷阱與反模式對照](#常見陷阱與反模式對照)
12. [延伸閱讀與交叉連結](#延伸閱讀與交叉連結)

---

## 這個模組要解決的真問題

一個 7B 參數的模型，若用 FP32（每個參數 4 bytes）載入，光是權重就要 `7e9 × 4 = 28 GB`。再加上微調時的梯度、optimizer 狀態、activation，全參數微調一個 7B 模型輕鬆吃掉 100 GB 以上的 VRAM——這不是消費級顯卡（RTX 4090 = 24 GB、RTX 3060 = 12 GB）能負擔的。

量化（quantization）與 QLoRA 就是回答這個問題的工程答案：

- **量化**：把權重從 16/32-bit 浮點壓到 8-bit 或 4-bit 整數，記憶體直接砍半甚至砍到 1/4。
- **QLoRA**：把整個 base model 凍結成 4-bit（不訓練它），只在旁邊掛上極小的 LoRA adapter（可訓練參數通常 < 1%），梯度與 optimizer 狀態只作用在這一小撮參數上。

兩者結合，讓「在單張 24 GB 卡上微調 13B 模型」「在 48 GB 上微調 20B 模型」變成可行。本模組的 notebook 正是這條路徑的逐步示範：從載入（`01`）→ 16-bit LoRA（`02`）→ 8-bit LoRA（`03`）→ 4-bit QLoRA（`04`）。

---

## 核心觀念：為什麼要量化

### 量化的本質：用更少的 bit 表示同一個數值範圍

浮點數（FP32 / FP16 / BF16）用指數與尾數記錄一個連續範圍的數值。量化做的事情是：把這個連續範圍切成有限個離散桶（int8 = 256 個桶、int4 = 16 個桶），每個權重對應到最近的桶，再記錄一個 scale 因子把整數還原回近似的浮點值。

```
原始權重 (FP16):  0.0123, -0.0456, 0.0789, ...
量化 (int4):      把 [-max, +max] 切成 16 格 → 每個權重存成 0~15 的整數 + 一個 scale
反量化:           整數 × scale ≈ 原始浮點值（有誤差，但通常 < 2% 任務品質損失）
```

**關鍵 WHY**：量化犧牲一點精度換取記憶體與頻寬。對推論與 LoRA 微調而言，這個誤差通常在可接受範圍內，因為：

1. LLM 權重對精度本來就有冗餘（這也是為何 distillation / pruning 可行）。
2. QLoRA 的可訓練 adapter 是用較高精度（BF16）計算的，會「補償」base model 量化帶來的誤差。

### compute dtype vs storage dtype

這是初學者最容易混淆的點：4-bit 量化指的是**權重的儲存格式**，不代表運算也用 4-bit。實際前向計算時，bitsandbytes 會把需要用到的權重區塊**即時反量化（dequantize）回 `bnb_4bit_compute_dtype`（通常 BF16）**再做矩陣乘法。

- `load_in_4bit` → storage：4-bit（省 VRAM）
- `bnb_4bit_compute_dtype=torch.bfloat16` → compute：BF16（保品質）

---

## 記憶體數學：先算清楚你需要多少 VRAM

教學上，永遠先讓學生會「估算」再動手，避免盲目 OOM。

### 1. 權重佔用 = 參數量 × 每參數 bytes

| 精度 | 每參數 bytes | 7B 權重 | 13B 權重 | 20B 權重 |
| :--- | :--- | :--- | :--- | :--- |
| FP32 | 4 | 28 GB | 52 GB | 80 GB |
| FP16 / BF16 | 2 | 14 GB | 26 GB | 40 GB |
| INT8 | 1 | 7 GB | 13 GB | 20 GB |
| INT4 (nf4) | 0.5 | ~3.5 GB | ~6.5 GB | ~10 GB |

### 2. 微調時的額外開銷

全參數微調用 AdamW，每個可訓練參數還要存：梯度（1×）+ 一階動量 m（1×）+ 二階動量 v（1×）。以 FP32 optimizer 狀態算，光 optimizer 就是權重的數倍。

**QLoRA 的省法**：base model 4-bit 凍結（無梯度、無 optimizer 狀態），只有 LoRA adapter（< 1% 參數）需要梯度與 optimizer。這就是為什麼下表中 QLoRA 的微調額外開銷幾乎可以忽略，主要瓶頸回到 activation。

### 3. 實務 VRAM 估算總表（含微調，batch 小、gradient checkpointing 開啟）

| 模型 | 全參數微調 (BF16) | LoRA (16-bit) | QLoRA (4-bit) | 可跑的卡 |
| :--- | :--- | :--- | :--- | :--- |
| 7B | ~80 GB+ | ~18 GB | ~7-10 GB | QLoRA: RTX 3060 12G / 4090 |
| 13B | ~150 GB+ | ~30 GB | ~12-16 GB | QLoRA: RTX 4090 24G |
| 20B | 多卡 | ~45 GB | ~20-24 GB | QLoRA: 單張 24-48G |

> 數字會隨 `max_length`、batch size、gradient checkpointing 與是否 `double_quant` 浮動。教學重點是讓學生建立「先算再跑」的習慣，而非記死數字。可用 `model.get_memory_footprint()` 驗證載入後的實際佔用。

---

## 量化階梯：16 / 8 / 4-bit 何時用哪一檔

本模組刻意用「逐級降低精度」的目錄結構（`02` → `03` → `04`）來教這個取捨：

| 檔位 | 對應目錄 | 記憶體 | 品質 | 速度 | 何時用 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **16-bit (BF16/FP16)** | [`02-16bits_training`](./02-16bits_training/) | 基準 | 最佳 | 最快 | VRAM 充足、追求最佳品質、模型 ≤ 7B 且有 24G+ 卡 |
| **8-bit (int8)** | [`03-8bits_training`](./03-8bits_training/) | ~½ | 幾乎無損 | 略慢 | VRAM 中等、想保品質又要省一半 |
| **4-bit (nf4, QLoRA)** | [`04-4bits_training`](./04-4bits_training/) | ~¼ | <2% 損失 | 反量化有 overhead | VRAM 吃緊、要在消費級卡跑 13B/20B |

**決策心法**：

- 先問「全 BF16 跑得動嗎？」跑得動就別量化，品質與速度都最好。
- 跑不動 → 先試 8-bit（品質損失最小）。
- 還是不夠 → 上 4-bit QLoRA（這是 2026 在單卡微調大模型的事實標準）。

### nf4 vs fp4

`bnb_4bit_quant_type` 有兩個選擇：

- **`nf4`（NormalFloat4）**：QLoRA 論文提出，假設權重近似常態分布，把 16 個量化桶按常態分位數切分。對「常態分布的神經網路權重」是資訊理論上更優的配置。**預設且推薦用 nf4。**
- **`fp4`（Float4）**：均勻切分的 4-bit 浮點。一般略遜於 nf4。

### double quantization 的成本效益

`bnb_4bit_use_double_quant=True` 會**對 quantization 的 scale 因子本身再做一次量化**。每個權重區塊都有一個 scale，這些 scale 加起來也佔記憶體。double quant 把它們再壓一層，每個參數約再省 0.4 bit（7B 約省 ~0.4 GB）。代價極小、幾乎無品質影響，**建議開啟**。

---

## `BitsAndBytesConfig` 完整參數

> 鐵律：2026 一律用 `BitsAndBytesConfig` 物件設定量化。`from_pretrained(..., load_in_4bit=True)` 與 `load_in_8bit=True` 的裸 kwargs **在 transformers 4.42+ 已棄用**，本模組所有 notebook 都要改成下列寫法。

### Before（本模組 notebook 的舊寫法，已棄用）

```python
# 04-4bits_training/llama2_qlora_4bit.ipynb 等的舊寫法
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,                  # deprecated bare kwarg
    bnb_4bit_compute_dtype=torch.half,  # half is fragile; prefer bfloat16
    bnb_4bit_quant_type="nf4",
    device_map="auto",
)
```

### After（2026 標準）

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",              # NormalFloat4: best for weight distributions
    bnb_4bit_compute_dtype=torch.bfloat16,  # dequantize to BF16 for matmul stability
    bnb_4bit_use_double_quant=True,         # quantize the scales too, ~0.4 bit/param saved
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,                  # prefer HF Hub id over local Windows path
    quantization_config=bnb_config,
    device_map="auto",         # auto place across GPU/CPU/disk
    use_safetensors=True,
)
```

8-bit 版本同理：

```python
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # int8 不需要 compute_dtype / quant_type；bitsandbytes 自動以 LLM.int8() 處理 outlier
)
```

| 參數 | 作用 | 推薦值 |
| :--- | :--- | :--- |
| `load_in_4bit` / `load_in_8bit` | 啟用哪一檔量化 | 二擇一 |
| `bnb_4bit_quant_type` | 量化型別 | `"nf4"` |
| `bnb_4bit_compute_dtype` | 反量化後的運算精度 | `torch.bfloat16`（A100/4090 等 Ampere+），舊卡退 `torch.float16` |
| `bnb_4bit_use_double_quant` | 二次量化 scale | `True` |

---

## QLoRA 完整流程

QLoRA = 4-bit 量化載入 + `prepare_model_for_kbit_training()` + LoRA adapter + `SFTTrainer`。順序錯了會悄悄失敗（梯度不流動、loss 不降），這是本模組最大的教學重點。

### 完整骨架（取代手刻 Trainer + 手刻 -100 標籤）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. 4-bit 量化載入
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2. 關鍵步驟：在掛 LoRA 之前先 prepare（順序不可顛倒）
model = prepare_model_for_kbit_training(model)  # enables grad on inputs, casts layernorm, etc.

# 3. LoRA 設定（target_modules 依架構而定，見下）
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Llama 系；ChatGLM 用 ["query_key_value"]
)

# 4. 用 SFTTrainer，模板/標籤遮罩交給它，不要手刻
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,   # effective batch = 2 × 16 = 32
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",          # paged optimizer 避免量化訓練 OOM 尖峰
        save_safetensors=True,
        seed=42,
    ),
    peft_config=lora_config,
    processing_class=tokenizer,
)
trainer.train()
```

### 為什麼 `prepare_model_for_kbit_training()` 必須在 LoRA 之前

量化後的 base layer 是凍結且不可微的。這個函式做三件事，確保梯度能正確流到 LoRA：

1. 把 layernorm 等不穩定層 cast 回 FP32（量化下數值穩定性需要）。
2. 啟用 input embedding 的 `requires_grad`（讓梯度能反傳通過凍結層到達 adapter）。
3. 設定 gradient checkpointing 相容性。

若先 `get_peft_model` 再 prepare，prepare 的設定會作用在錯的層上，導致訓練「看起來在跑但 loss 不動」。

### 為什麼用 `SFTTrainer` 取代手刻 `Trainer` + 手刻 -100

本模組現有 notebook（如 `llama2_qlora_4bit.ipynb`、`chatglm3_lora_8bit.ipynb`）都手刻 instruction/response 拼接與 `-100` 標籤遮罩。問題：

- 手刻容易出錯（漏 EOS、padding side 設錯、遮罩邊界算錯）。
- 各模型的 chat 格式不同，硬寫 `Human:/Assistant:` 不可攜。

`SFTTrainer` 配合 `tokenizer.apply_chat_template()` 自動套模板、自動只在 response token 算 loss（`-100` 遮罩 prompt 部分）。教學上保留一個「底層手刻版」對照，解釋 `-100` 為何只算 response token（CrossEntropyLoss 的 `ignore_index=-100`），但**預設路徑用 `SFTTrainer`**。詳見 [03-PEFT 的 SFTTrainer 流程](../03-PEFT/README.md)。

### target_modules 速查

| 模型家族 | 對應 notebook | 建議 `target_modules` |
| :--- | :--- | :--- |
| Llama-2 / Taiwan-LLM | `llama2_*` | `["q_proj", "k_proj", "v_proj", "o_proj"]` |
| ChatGLM3 | `chatglm3_*` | `["query_key_value"]` |
| InternLM | `internlm_qlora_4bit` | `["q_proj","k_proj","v_proj","o_proj"]`（依該版架構確認） |

---

## 本模組 notebook 對照表

| Notebook | 主題 | 教什麼 | 2026 重點修正 |
| :--- | :--- | :--- | :--- |
| [`01-llm_download/download.ipynb`](./01-llm_download/download.ipynb) | 從 Hub 載入 LLM 並對話 | `AutoModelForCausalLM` + `pipeline` + `apply_chat_template` + 多輪對話 | 移除 ModelScope 與硬路徑改用 Hub id；修 `'assistance'` → `'assistant'` 拼字 bug；`device_map='auto'` + `torch_dtype=bfloat16` |
| [`01-llm_download/chatglm2_load.ipynb`](./01-llm_download/chatglm2_load.ipynb) | 載入 ChatGLM2-6B | `trust_remote_code` 載入大模型 | `AutoModel` → `AutoModelForCausalLM`；補 `device_map`/`torch_dtype`（否則 FP32 OOM）；補 `trust_remote_code` 安全說明 |
| [`02-16bits_training/chatglm3_infer.ipynb`](./02-16bits_training/chatglm3_infer.ipynb) | ChatGLM3 16-bit 推論 | fp16 推論、chat 模板探索 | `model.chat()`/`build_chat_input()` → `apply_chat_template()` + `generate()`；解釋 `torch.half` 取捨 |
| [`02-16bits_training/chatglm3_lora_16bit.ipynb`](./02-16bits_training/chatglm3_lora_16bit.ipynb) | ChatGLM3 16-bit LoRA | LoRA + 梯度累積 | 補 `eval_dataset`/評測；`build_chat_input()` → `apply_chat_template()`；`SFTTrainer`；safetensors |
| [`02-16bits_training/llama2_lora_16bit.ipynb`](./02-16bits_training/llama2_lora_16bit.ipynb) | Taiwan-LLM 16-bit LoRA | 全序列 SFT、`-100` 遮罩 | 顯式 `LoraConfig`（r/alpha/target_modules）；`SFTTrainer`；解釋 `padding_side='right'` 對 decoder-only 的必要性 |
| [`03-8bits_training/chatglm3_lora_8bit.ipynb`](./03-8bits_training/chatglm3_lora_8bit.ipynb) | ChatGLM3 8-bit LoRA | int8 量化 + LoRA | 裸 `load_in_8bit` → `BitsAndBytesConfig`；補 `gradient_checkpointing`；`SFTTrainer` |
| [`03-8bits_training/llama2_lora_8bit.ipynb`](./03-8bits_training/llama2_lora_8bit.ipynb) | Llama-2-7B 8-bit LoRA | int8 + `merge_and_unload` | 補 `prepare_model_for_kbit_training()`；`torch.half` → `bfloat16`；`max_grad_norm`/`warmup_ratio`/`cosine` |
| [`04-4bits_training/chatglm3_qlora_4bit.ipynb`](./04-4bits_training/chatglm3_qlora_4bit.ipynb) | ChatGLM3 QLoRA 4-bit | 完整 QLoRA | `BitsAndBytesConfig` + `nf4` + double quant；`apply_chat_template`；`SFTTrainer` |
| [`04-4bits_training/internlm_qlora_4bit.ipynb`](./04-4bits_training/internlm_qlora_4bit.ipynb) | InternLM-20B QLoRA | 20B 在單卡微調 | `BitsAndBytesConfig`；`use_cache=False`（與 gradient checkpointing 相容）；`apply_chat_template` |
| [`04-4bits_training/llama2_qlora_4bit.ipynb`](./04-4bits_training/llama2_qlora_4bit.ipynb) | Llama-2-13B QLoRA | 13B QLoRA + adapter 合併 | `BitsAndBytesConfig`；`pad_token_id` 用屬性而非硬寫 2；`merge_and_unload` + safetensors |
| [`04-4bits_training/model_weights_distribution.ipynb`](./04-4bits_training/model_weights_distribution.ipynb) | 權重分布視覺化 | 量化前後權重直方圖 | 用 `named_parameters()` 做逐層直方圖；連結分布形狀與 int4 量化桶（nf4 為何適合常態分布） |
| [`LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb`](./LLaMA2-prompt-tuning/Fine_Tunning_Llama_2.0.ipynb) | QLoRA + SFTTrainer 範例 | 端到端 QLoRA | `f"<s>[INST]..."` → `apply_chat_template`；`packing=True` + `max_seq_length`；修正 train/valid 用同檔的資料洩漏；`save_safetensors` |

---

## 大型 LLM 安全載入

### `trust_remote_code` 的安全意涵

ChatGLM、InternLM 等模型在 Hub repo 內附帶**自訂的 modeling Python 程式碼**，`from_pretrained(..., trust_remote_code=True)` 會直接執行這些程式碼。這等於在你的機器上跑來路的 code。

**教學原則**：

- 只對你信任的官方 repo（如 `THUDM/chatglm3-6b`、`internlm/internlm-20b`）開啟。
- 開啟前可先到 Hub 查看 `modeling_*.py` 內容。
- 釘住 `revision`（commit hash）避免日後 repo 更新引入非預期程式碼。

### 用 Hub id，不要用本機硬路徑

本模組多個 notebook 殘留 `d:/Pretrained_models/...` 等 Windows 硬路徑，在 Linux/Mac/Colab 全部失效。一律改用 Hub model id（透過 `HF_HOME` 控制快取位置）：

```python
# Before（不可攜）
model_path = "d:/Pretrained_models/chatglm3-6b"

# After（可攜、可重現）
model_id = "THUDM/chatglm3-6b"   # 由 HF_HOME 決定快取在哪
```

---

## 2026 版本與環境設定

完整環境建置見 [00-Setup-and-Foundations/00-environment-setup.md](../00-Setup-and-Foundations/00-environment-setup.md)；全 repo 共用慣例見 [01-2026-conventions.md](../00-Setup-and-Foundations/01-2026-conventions.md)。本模組的版本鎖定：

```text
torch>=2.4
transformers>=4.46
datasets>=3.0
trl>=0.12
peft>=0.13
accelerate>=1.0
bitsandbytes>=0.44
safetensors>=0.4
evaluate>=0.4
```

> `bitsandbytes` 對 CUDA 版本敏感，安裝後務必驗證：
>
> ```python
> import torch, bitsandbytes
> print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
> ```
>
> 若 import 報 CUDA 版本不符，多半是 `bitsandbytes` wheel 與本機 CUDA toolkit 不匹配，排查見 [環境設定文件](../00-Setup-and-Foundations/00-environment-setup.md) 的「bitsandbytes/CUDA 版本」一節。

---

## 儲存與發布

### 一律存成 safetensors

```python
# adapter-only（小、可攜，部署時再疊回 base model）
trainer.model.save_pretrained("./my-lora-adapter", safe_serialization=True)

# 或合併後存（部署簡單，但檔案大、失去 adapter 靈活性）
merged = trainer.model.merge_and_unload()  # 注意：不可逆，且量化模型合併有額外注意事項
merged.save_pretrained("./my-merged-model", safe_serialization=True)
```

**WHY safetensors**：取代 pickle（`pytorch_model.bin`）。pickle 載入時會執行任意程式碼（安全風險），safetensors 純資料、載入更快、zero-copy mmap。

### adapter-only vs merged：部署取捨

| | adapter-only | merged |
| :--- | :--- | :--- |
| 檔案大小 | 小（幾十 MB） | 與 base 同等（GB 級） |
| 部署 | 需 base + adapter | 單一模型 |
| 靈活性 | 可換多個 adapter | 固定 |
| 推論延遲 | 略高（多一層） | 略低 |

### 推論模板必須與訓練一致

訓練用 `apply_chat_template` 組標籤，推論就必須用**同一個模板**組 prompt，否則模型看到的格式與訓練時不同，輸出會崩。這是 QLoRA 微調後最常見的「訓練 loss 很低但推論很爛」的根因。

### push_to_hub 與最小 model card

```python
trainer.push_to_hub()  # 自動產生 model card 骨架
```

model card 至少標注：base model、量化方式（nf4 4-bit）、訓練資料、語言（zh-TW）、授權、任務 tag。

---

## 常見陷阱與反模式對照

| 舊寫法（反模式） | 2026 正確寫法 | 為什麼 |
| :--- | :--- | :--- |
| `load_in_4bit=True`（裸 kwarg） | `quantization_config=BitsAndBytesConfig(...)` | 裸 kwarg 在 4.42+ 棄用 |
| `bnb_4bit_compute_dtype=torch.half` | `torch.bfloat16`（Ampere+） | bf16 動態範圍大、訓練更穩 |
| 先 `get_peft_model` 再 prepare | 先 `prepare_model_for_kbit_training()` 再掛 LoRA | 順序錯則梯度不流動 |
| 手刻 `Human:/Assistant:` 拼接 | `tokenizer.apply_chat_template()` | 可攜、訓練/推論一致 |
| 手刻 Trainer + 手刻 `-100` | `SFTTrainer` + `SFTConfig` | 自動模板、自動 response-only 遮罩 |
| `AutoModel`（載 LLM） | `AutoModelForCausalLM` | 語意正確、拿到 LM head |
| `model.cuda()` / FP32 預設 | `device_map='auto'` + `torch_dtype` | 否則 6B 模型 FP32 直接 OOM |
| `d:/Pretrained_models/...` | Hub model id + `HF_HOME` | 可攜、可重現 |
| `pytorch_model.bin`（pickle） | `safe_serialization=True` | 安全、載入快 |
| train 與 valid 用同一檔 | 真正切分 train/valid（`seed=42`） | 避免資料洩漏，能偵測 overfit |
| `model.chat()` / `build_chat_input()` | `apply_chat_template()` + `generate()` | 跨模型可攜，且銜接多模態 |

---

## 延伸閱讀與交叉連結

- 上一站：[03-PEFT](../03-PEFT/README.md) — LoRA / IA3 原理、`SFTTrainer` 標準流程、adapter 合併部署。
- 下一站：[05-Multimodal](../05-Multimodal/README.md) — 把本模組的 4-bit QLoRA 遷移到 VLM（[`06-vlm_finetuning/vlm_lora_finetune.ipynb`](../05-Multimodal/06-vlm_finetuning/vlm_lora_finetune.ipynb)）：target_modules 要含 LLM 與 projector、用 `processor` 而非 `tokenizer`、image token 的標籤遮罩。
- 共用慣例：[00-Setup-and-Foundations/01-2026-conventions.md](../00-Setup-and-Foundations/01-2026-conventions.md) — `BitsAndBytesConfig`、`apply_chat_template`、safetensors 的統一用法。
- 術語速查：[00-Setup-and-Foundations/02-glossary-and-architectures.md](../00-Setup-and-Foundations/02-glossary-and-architectures.md) — fp32/bf16/fp16/int8/nf4、LoRA r/alpha/target_modules。
- repo 總覽：[../README.md](../README.md)。

> 小結：量化是「用可接受的精度損失換 VRAM」，QLoRA 是「在 4-bit 凍結 base 上掛 BF16 LoRA」。記住順序（量化載入 → `prepare_model_for_kbit_training` → LoRA → `SFTTrainer`）與一致性（訓練/推論同模板），你就能在一張消費級卡上微調過去需要叢集才跑得動的模型。
