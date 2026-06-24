# HuggingFace Transformers 中文教學 Cookbook（2026 版）

> 一套以 **HuggingFace Transformers** 為核心、文字導向（多為中文 NLP）並延伸至多模態的實作教學集。
> 從最基礎的 `pipeline` 元件，一路帶到 LoRA / QLoRA 量化微調，最後跨入影像、語音與視覺語言模型（VLM）。

---

## 專案簡介與目標讀者

這個 repo 把「會用 HuggingFace」這件事拆成一條可以照著走的學習弧線：每個模組只引入少量新概念，並且重複使用同一套心智模型。讀完之後，你應該能：

- 看懂任一 HuggingFace 任務 notebook，並判斷它用的是哪一層抽象（`pipeline` / `AutoModel` / `Trainer` / `SFTTrainer`）。
- 用 2026 的標準寫法（而非 2024 的手刻寫法）載入模型、處理資料、訓練、評測、發布。
- 把單模態（純文字）的能力平滑延伸到多模態（影像 / 語音 / 圖文）。

**目標讀者**：已具備 Python 3.10+ 基礎、理解基本 PyTorch 與深度學習概念、能操作命令列的學習者。不需要事先懂 Transformer 內部數學，理論會在 [`00-Setup-and-Foundations/`](./00-Setup-and-Foundations/) 的速查文件中補齊。

> 為什麼要有這份 README？因為「沒有導覽」本身就是最大的可用性問題。在動手跑任何 notebook 之前，先在這裡建立全域地圖，能避免你在 36 個 notebook 之間迷路。

---

## 學習路徑圖（模組依賴關係）

模組是**有方向的**：後面的模組假設你已經內化前面的概念。請依序學習，不要跳過 `00`。

```
00-Setup-and-Foundations   （環境 + 全 repo 共用慣例 + 術語速查）
          │
          ▼
01-Component               （pipeline / tokenizer / model / datasets / evaluate / Trainer）
          │
          ▼
02-Adv-tasks               （NER / QA / 相似度 / 檢索 / LM / 摘要 / chatbot）
          │
          ▼
03-PEFT                    （LoRA / IA3 參數高效微調）
          │
          ▼
04-kbits-tuning            （16 / 8 / 4-bit 量化 + QLoRA）
          │
          ▼
05-Multimodal              （ViT / CLIP / VLM / Whisper / 多模態 RAG / VLM 微調）
```

核心觀念：**多模態不是另起爐灶，而是單模態的延伸**。

| 單模態（01–04 學到的） | 多模態（05 延伸成的） |
| :--- | :--- |
| `tokenizer` | `AutoProcessor`（tokenizer + image/feature processor） |
| `input_ids` | `pixel_values` / audio features + `input_ids` |
| `Trainer` | 多模態 `SFTTrainer` + 多模態 collator |
| 文字 RAG（dense retrieval + rerank） | 跨模態 RAG（CLIP/SigLIP 嵌入 + VLM 生成） |
| `apply_chat_template`（純文字 messages） | 同一套模板，含 image/audio token 的 messages |

---

## 模組總覽表

| 模組 | 一句話定位 | 導覽文件 |
| :--- | :--- | :--- |
| `00-Setup-and-Foundations` | 鎖版本環境 + 全 repo 共用的 4 大慣例 + 術語/架構速查 | [README 區段](#00-setup-and-foundations新增) |
| `01-Component` | HuggingFace 六大元件：`pipeline → tokenizer → model → datasets → evaluate → Trainer` | [`01-Component/README.md`](./01-Component/README.md) |
| `02-Adv-tasks` | 依任務組織的進階 NLP：NER、QA、相似度、檢索、語言模型、摘要、chatbot | [`02-Adv-tasks/README.md`](./02-Adv-tasks/README.md) |
| `03-PEFT` | 參數高效微調：LoRA、IA3，以及 `SFTTrainer` 標準指令微調流程 | [`03-PEFT/README.md`](./03-PEFT/README.md) |
| `04-kbits-tuning` | 量化階梯（16/8/4-bit）與 QLoRA 全流程，含 VRAM 估算與大模型載入 | [`04-kbits-tuning/README.md`](./04-kbits-tuning/README.md) |
| `05-Multimodal` | 視覺、語音、圖文：ViT、CLIP/SigLIP、Qwen2.5-VL、Whisper、多模態 RAG、VLM 微調 | [`05-Multimodal/README.md`](./05-Multimodal/README.md) |

> 註：`00` 與 `05` 為 2026 版新增模組。`01–04` 為既有教材的現代化版本（見下方「2026 現代化說明」）。

---

## 環境需求與鎖定版本

### 硬體

| 用途 | 最低需求 | 建議 |
| :--- | :--- | :--- |
| `01`–`02` 監督微調（BERT 級） | 8 GB VRAM 或 CPU | 12 GB+ VRAM |
| `03`–`04` LoRA / QLoRA（1B–13B） | 4-bit 下 8–16 GB VRAM | 24 GB VRAM（A100 / 3090 / 4090） |
| `05` VLM 推論 / 微調（7B 級） | 4-bit 下 12–16 GB VRAM | 24 GB+ VRAM |

bf16 需要 GPU compute capability ≥ 8（Ampere 以後，如 A100 / 30 系 / 40 系）。較舊 GPU 請改用 fp16。

### 鎖定版本（2026 基線）

全 repo 共用同一組版本，避免 API 漂移。為什麼鎖版本？因為這些套件每隔幾個月就改 API（`load_in_4bit` 裸參數、`evaluation_strategy` 改名等），鎖死版本是「可重現」的前提。

```txt
# requirements.txt（節錄；完整清單見 00-Setup-and-Foundations/00-environment-setup.md）
torch>=2.4
transformers>=4.46
datasets>=3.0
trl>=0.12
peft>=0.13
accelerate>=1.0
bitsandbytes>=0.44
evaluate>=0.4
safetensors>=0.4
```

各版本門檻對應的關鍵能力：

| 套件 | 版本 | 為什麼是這個門檻 |
| :--- | :--- | :--- |
| `transformers` | `>=4.46` | `BitsAndBytesConfig` 取代裸 `load_in_4bit`、`apply_chat_template` 成熟、`eval_strategy` 新名、`attn_implementation='sdpa'` 預設 |
| `datasets` | `>=3.0` | 預設 `safetensors`、streaming 強化、`Image`/`Audio` feature 與 `cast_column` |
| `trl` | `>=0.12` | `SFTTrainer` + `SFTConfig` 穩定、`processing_class` 取代 `tokenizer` 參數、多模態 collator 支援 |
| `peft` | `>=0.13` | LoRA 對 VLM projector 的支援、`prepare_model_for_kbit_training` 行為一致 |
| `bitsandbytes` | `>=0.44` | nf4 / double quant / `paged_adamw` 穩定 |

---

## 快速開始

```bash
# 1) 取得程式碼
git clone <this-repo-url>
cd HuggingFace_scratch

# 2) 建立環境（擇一）

# 方式 A：uv（推薦，速度快、鎖檔可重現）
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

# 方式 B：venv + pip
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) 驗證安裝
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"

# 4)（可選）登入 HuggingFace Hub 以下載受授權模型 / 發布成果
huggingface-cli login   # 或設定環境變數 HF_TOKEN
```

跑你的第一個 notebook：

```bash
jupyter lab "01-Component/01pipeline/01.pipeline.ipynb"
```

完整環境設定、`HF_HOME` 快取位置、`bitsandbytes`/CUDA 版本疑難排解，見 [`00-Setup-and-Foundations/00-environment-setup.md`](./00-Setup-and-Foundations/00-environment-setup.md)。

---

## 2026 現代化說明（相對舊版改了什麼）

舊版 notebook 停留在 2024 年初的寫法。2026 版把**反覆出現的 8 個慣例**抽成單一參考文件 [`00-Setup-and-Foundations/01-2026-conventions.md`](./00-Setup-and-Foundations/01-2026-conventions.md)，每個 notebook 引用它而非各自重複。核心精神：**消除特殊情況，讓每個 notebook 都長得一樣**。

### 反模式對照表（舊寫法 → 新寫法）

| 主題 | 舊寫法（2024） | 新寫法（2026） | 為什麼 |
| :--- | :--- | :--- | :--- |
| 裝置與精度 | `model.cuda()`、`device=0`、`torch.half` | `from_pretrained(model_id, device_map='auto', torch_dtype=torch.bfloat16)` | 自動 CPU/disk offload；bf16 數值更穩、範圍更大 |
| 序列化 | `pytorch_model.bin`（pickle） | `save_pretrained(safe_serialization=True)`（safetensors） | 載入更快、無任意程式碼執行風險 |
| 量化 | `load_in_4bit=True`（裸參數，4.42+ 已棄用） | `quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)` | 棄用參數會消失；config 物件可重現 |
| 對話格式 | 手寫 `'Human:/Assistant:'`、`f'<s>[INST]...'`、`model.chat()` | `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` | 跨模型可攜；訓練/推論模板一致；也是多模態的橋樑 |
| 指令微調 | 手刻 `Trainer` + `DataCollatorForSeq2Seq` + 手刻 `-100` | `SFTTrainer(model, args=SFTConfig(...), peft_config=lora_config, processing_class=tokenizer)` | 自動套模板、packing、response-only 遮罩 |
| 監督任務訓練 | 手刻 epoch/batch 迴圈、`torch.optim.Adam` | `Trainer` + 完整 `TrainingArguments`（`bf16`、`warmup_ratio`、cosine、`save_safetensors`） | 自動處理裝置/混精/checkpoint；AdamW 是 transformer 標準 |
| 資料管線 | `pandas` + 自製 `Dataset`/`collate_fn`、固定 padding | `load_dataset` + `map(fn, batched=True)` + `DataCollatorWithPadding`（動態 padding） | `batched=True` 快 3–5 倍；動態 padding 省記憶體 |
| 評測 | 手刻 accuracy、第三方 `rouge_chinese` | `evaluate.load()` + `compute_metrics`（含 P/R/F1、混淆矩陣、ROUGE、WER） | 一致 API；補齊回歸（MSE/Pearson）與檢索（Recall@k/MRR）指標 |

### 還修掉了什麼

- 移除所有 Google Drive mount 與 `d:/Pretrained_models` 等硬路徑，改用 HF Hub model id 或 `pathlib` + 環境變數。
- 補上 `set_seed(42)` 可重現性基線。
- 修掉散落的 bug：`f1_metirc` 拼字、`'assistance'` 角色拼字、`MinMaxScaler` 在 split 前造成資料洩漏、句相似度把回歸當分類 threshold 的錯誤。

---

## 各模組詳解與 notebook 對照

### 00-Setup-and-Foundations（新增）

純文件模組（外加一個環境驗證 notebook），是所有後續內容的共同基礎。

| 文件 | 教什麼 |
| :--- | :--- |
| [`00-environment-setup.md`](./00-Setup-and-Foundations/00-environment-setup.md) | 硬體/CUDA 需求、`uv`/`venv` 建環境、鎖版本、HF 登入與 `HF_HOME`、安裝排錯 |
| [`01-2026-conventions.md`](./00-Setup-and-Foundations/01-2026-conventions.md) | 全 repo 共用的 4 大慣例（device_map+dtype、safetensors、`apply_chat_template`、`Trainer`/`SFTTrainer`）與反模式對照表 |
| [`02-glossary-and-architectures.md`](./00-Setup-and-Foundations/02-glossary-and-architectures.md) | Transformer 三型架構、tokenizer 三件套、attention mask、量化術語、PEFT 術語、processor 抽象 |

**學習目標**：建立鎖版本環境並驗證 GPU；理解 4 大共用慣例；會用 `huggingface_hub` 登入、檢視 model card、設定快取。

### 01-Component（現代化既有）

依抽象由高到低學習六大元件。詳見 [`01-Component/README.md`](./01-Component/README.md)。

| Notebook | 教什麼 | 核心 API |
| :--- | :--- | :--- |
| [`01pipeline/01.pipeline.ipynb`](./01-Component/01pipeline/01.pipeline.ipynb) | `pipeline` 抽象、任務選擇、裝置管理；初探 `AutoProcessor` | `pipeline`、`SUPPORTED_TASKS` |
| `02tokenizer/` | tokenizer 三件套、特殊 token；processor 是其多模態超集 | `AutoTokenizer` |
| [`03Model/03.Model.ipynb`](./01-Component/03Model/03.Model.ipynb) | `AutoModel`/`AutoConfig` 載入、輸出語意（`last_hidden_state` vs `pooler_output`） | `AutoModel`、`AutoConfig` |
| [`03Model/03 Model classification_demo.ipynb`](./01-Component/03Model/03%20Model%20classification_demo.ipynb) | 序列分類微調（由手刻迴圈改為 `Trainer`） | `AutoModelForSequenceClassification` |
| [`04Datasets/04 Datasets.ipynb`](./01-Component/04Datasets/04%20Datasets.ipynb) | `datasets` 3.x：載入、`map(batched=True)`、動態 padding | `load_dataset`、`DataCollatorWithPadding` |
| [`05evaluate/05 evaluate.ipynb`](./01-Component/05evaluate/05%20evaluate.ipynb) | `evaluate` 指標載入、合併、`compute_metrics` | `evaluate.load`、`evaluate.combine` |
| [`06Trainer/06 Trainer classification_demo.ipynb`](./01-Component/06Trainer/06%20Trainer%20classification_demo.ipynb) | 完整 `TrainingArguments`（bf16/warmup/cosine/safetensors/early stopping） | `Trainer`、`TrainingArguments` |

### 02-Adv-tasks（現代化既有）

依任務類型組織，重點在「架構選型」與「正確的評測指標」。詳見 [`02-Adv-tasks/README.md`](./02-Adv-tasks/README.md)。

| Notebook | 任務 | 架構 / 指標 |
| :--- | :--- | :--- |
| [`02-token_classification/ner.ipynb`](./02-Adv-tasks/02-token_classification/ner.ipynb) | 中文 NER | encoder-only + `seqeval`、`word_ids()` 對齊、`-100` 遮罩 |
| [`03-question_answering/mrc_simple_version.ipynb`](./02-Adv-tasks/03-question_answering/mrc_simple_version.ipynb) | 抽取式 QA | offset mapping + span 預測、SQuAD EM/F1 |
| [`04-sentence_similarity/dual_model.ipynb`](./02-Adv-tasks/04-sentence_similarity/dual_model.ipynb) | 句相似度（雙塔） | `CosineEmbeddingLoss`（接 05 的 CLIP） |
| [`04-sentence_similarity/cross_model.ipynb`](./02-Adv-tasks/04-sentence_similarity/cross_model.ipynb) | 句相似度（回歸） | 修正為 MSE/Pearson，非分類 threshold |
| [`05-retrieval_chatbot/retrieval_bot.ipynb`](./02-Adv-tasks/05-retrieval_chatbot/retrieval_bot.ipynb) | 檢索式 QA | dense + cross-encoder rerank、FAISS、Recall@k/MRR/NDCG |
| [`06-language_model/causal_lm.ipynb`](./02-Adv-tasks/06-language_model/causal_lm.ipynb) | CLM 微調 | decoder-only、`mlm=False` |
| [`06-language_model/masked_lm.ipynb`](./02-Adv-tasks/06-language_model/masked_lm.ipynb) | MLM 預訓練 | encoder-only、`mlm_probability=0.15` |
| [`07-text_summarization/summarization.ipynb`](./02-Adv-tasks/07-text_summarization/summarization.ipynb) | 摘要 | enc-dec（T5）、`evaluate` 的 ROUGE 取代 `rouge_chinese` |
| [`08-generative_chatbot/chatbot.ipynb`](./02-Adv-tasks/08-generative_chatbot/chatbot.ipynb) | 生成式 chatbot | `apply_chat_template` + （可選）`SFTTrainer` |

### 03-PEFT（現代化既有）

LoRA / IA3 原理與超參選擇。詳見 [`03-PEFT/README.md`](./03-PEFT/README.md)。

| Notebook | 教什麼 |
| :--- | :--- |
| [`01-LoRA/chatbot_lora.ipynb`](./03-PEFT/01-LoRA/chatbot_lora.ipynb) | 明確 `LoraConfig`（`r`/`lora_alpha`/`target_modules`）+ `SFTTrainer` |
| [`01-LoRA/lora_inference.ipynb`](./03-PEFT/01-LoRA/lora_inference.ipynb) | adapter 載入、`merge_and_unload()`、合併 vs adapter-only 取捨 |
| [`02-IA3/chatbot_ia3.ipynb`](./03-PEFT/02-IA3/chatbot_ia3.ipynb) | IA3 與 LoRA 的取捨 |

### 04-kbits-tuning（現代化既有）

量化階梯與 QLoRA 全流程。詳見 [`04-kbits-tuning/README.md`](./04-kbits-tuning/README.md)。

| Notebook | 教什麼 |
| :--- | :--- |
| [`01-llm_download/download.ipynb`](./04-kbits-tuning/01-llm_download/download.ipynb) | 大模型載入、`apply_chat_template`、互動式對話 |
| [`02-16bits_training/llama2_lora_16bit.ipynb`](./04-kbits-tuning/02-16bits_training/llama2_lora_16bit.ipynb) | 16-bit LoRA SFT |
| [`03-8bits_training/llama2_lora_8bit.ipynb`](./04-kbits-tuning/03-8bits_training/llama2_lora_8bit.ipynb) | 8-bit + `prepare_model_for_kbit_training` |
| [`04-4bits_training/llama2_qlora_4bit.ipynb`](./04-kbits-tuning/04-4bits_training/llama2_qlora_4bit.ipynb) | QLoRA：4-bit + LoRA + `SFTTrainer` |
| [`04-4bits_training/model_weights_distribution.ipynb`](./04-kbits-tuning/04-4bits_training/model_weights_distribution.ipynb) | 量化前後權重分布視覺化 |

### 05-Multimodal（全新）

把單模態心智模型延伸到多模態。詳見 [`05-Multimodal/README.md`](./05-Multimodal/README.md) 與前置理論 [`05-Multimodal/00-multimodal-foundations.md`](./05-Multimodal/00-multimodal-foundations.md)。

| Notebook | 能力 | 2026 推薦模型 |
| :--- | :--- | :--- |
| [`01-image_classification/vit_image_classification.ipynb`](./05-Multimodal/01-image_classification/vit_image_classification.ipynb) | 影像分類（純視覺基礎） | `google/vit-base-patch16-224`、`facebook/dinov2-base` |
| [`02-clip_retrieval/clip_image_text_retrieval.ipynb`](./05-Multimodal/02-clip_retrieval/clip_image_text_retrieval.ipynb) | 圖文檢索 / zero-shot 分類 | `google/siglip2-base-patch16-224`、`jinaai/jina-clip-v2` |
| [`03-vlm_vqa_captioning/vlm_vqa_captioning.ipynb`](./05-Multimodal/03-vlm_vqa_captioning/vlm_vqa_captioning.ipynb) | VQA / 看圖說話 | `Qwen/Qwen2.5-VL-7B-Instruct`、`Salesforce/blip2-opt-2.7b` |
| [`04-asr_whisper/whisper_asr.ipynb`](./05-Multimodal/04-asr_whisper/whisper_asr.ipynb) | 語音辨識（ASR） | `openai/whisper-large-v3-turbo` |
| [`05-multimodal_rag/multimodal_embeddings_rag.ipynb`](./05-Multimodal/05-multimodal_rag/multimodal_embeddings_rag.ipynb) | 跨模態 RAG | `jinaai/jina-clip-v2`、`vidore/colpali-v1.3` + Qwen2.5-VL |
| [`06-vlm_finetuning/vlm_lora_finetune.ipynb`](./05-Multimodal/06-vlm_finetuning/vlm_lora_finetune.ipynb) | VLM 上的 QLoRA 微調 | `Qwen/Qwen2.5-VL-7B-Instruct`、`HuggingFaceTB/SmolVLM-Instruct` |

---

## 如何使用本 Cookbook

1. **先讀 `00`**：不要跳過。環境鎖版本與 4 大慣例是後面一切的前提。
2. **照模組順序**：`01 → 02 → 03 → 04 → 05`。每個模組的 `README.md` 會說明前置與學習目標。
3. **讀 prose、跑 code**：散文（zh-TW）解釋「為什麼」；程式碼註解（English）解釋「怎麼做」。
4. **遇到不懂的術語**：查 [`02-glossary-and-architectures.md`](./00-Setup-and-Foundations/02-glossary-and-architectures.md)，不在每個 notebook 重述。
5. **看到舊寫法**：對照 [`01-2026-conventions.md`](./00-Setup-and-Foundations/01-2026-conventions.md) 的反模式表理解差異。

一段「新慣例」的最小範例（所有 notebook 共用的載入起手式）：

```python
# Before (2024)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.cuda()                      # hardcoded device, fp32, .bin format

# After (2026)
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",            # auto CPU/disk offload
    torch_dtype=torch.bfloat16,   # wider range, more stable than fp16
    use_safetensors=True,         # fast, safe loading
)
```

---

## 授權與致謝

- **教材內容**：供教學與學習用途。各 notebook 使用的預訓練模型與資料集，請遵循其各自在 HuggingFace Hub 上的授權條款（特別是 Llama、ChatGLM、Qwen 等有額外條款的模型）。
- **致謝**：本教材建立於 HuggingFace（`transformers`、`datasets`、`evaluate`、`accelerate`、`peft`、`trl`）與 `bitsandbytes`、`faiss` 等開源生態之上。
- 相關背景資料：根目錄 `Transformers_hugging_face.pdf`。

---

> 有問題或發現錯誤？歡迎開 issue。本 cookbook 以 zh-TW 撰寫、程式碼註解使用英文，並鎖定 2026 版本以確保可重現性。
