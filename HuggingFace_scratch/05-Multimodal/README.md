# 05-Multimodal｜多模態擴充模組

> 本模組是整套 cookbook 的縱向擴充：把你在 `01`～`04` 練熟的「單模態心智模型」平滑延伸到影像、音訊與圖文混合的世界。
>
> 核心主張只有一句話：**多模態不是另起爐灶，而是把 `tokenizer` 換成 `AutoProcessor`、把 `input_ids` 換成 `pixel_values` / audio features，其餘 `Trainer`、`apply_chat_template`、`BitsAndBytesConfig`、LoRA 的流程幾乎原封不動。**

---

## 1. 模組定位：多模態是單模態的延伸，不是平行宇宙

學多模態最大的認知陷阱，是以為它是一套全新的 API、全新的訓練流程、全新的概念。實際上 2026 的 HuggingFace 生態刻意把多模態設計成單模態的「超集」：

| 你在 01-04 學到的 | 在 05 對應到的 | 本質差異 |
| :--- | :--- | :--- |
| `AutoTokenizer` | `AutoProcessor` / `AutoImageProcessor` | processor = tokenizer + image/feature 前處理，是同一個抽象的擴展 |
| `input_ids` / `attention_mask` | `pixel_values`、`input_features`（log-mel）、外加 `input_ids` | 多了一條視覺/音訊通道，但仍然餵進同一個 `forward()` |
| `AutoModelForSequenceClassification` | `AutoModelForImageClassification` | 換 head，不換訓練迴圈 |
| `apply_chat_template(messages)` | 同一個 `apply_chat_template`，content 內含 `{"type": "image"}` | **同一套 chat 模板機制**，image/audio 只是序列裡的特殊 token |
| `Trainer` + `TrainingArguments` | 同一個 `Trainer` / `SFTTrainer`，collator 換多模態版 | 訓練流程不變 |
| 文字 dense retrieval（[`retrieval_bot`](../02-Adv-tasks/05-retrieval_chatbot/retrieval_bot.ipynb)） | CLIP/SigLIP 跨模態檢索 + 同一個 FAISS | 嵌入即介面，索引邏輯一致 |
| LoRA / QLoRA（[`03-PEFT`](../03-PEFT/README.md)、[`04-kbits-tuning`](../04-kbits-tuning/README.md)） | VLM 上的 LoRA / QLoRA | `target_modules` 多含 projector，其餘相同 |

**為什麼這個設計重要（WHY）**：如果你把每個模態當成獨立技能去背 API，學習成本是線性疊加；但只要抓住「processor 抽象 + chat 模板 + 共享嵌入空間」三根支柱，多模態就只是你既有知識的「再應用」。本模組所有 notebook 都會反覆提醒你：這一步在純文字版本裡你已經做過了。

理論細節（processor 抽象、共享嵌入空間、image/audio token 在序列中的角色、多模態評測指標）統一寫在 [`00-multimodal-foundations.md`](./00-multimodal-foundations.md)，6 個 notebook 共用，避免在每個檔案重述。

---

## 2. 學習弧線與 notebook 依賴圖

本模組依「視覺 → 嵌入 → 生成 → 音訊 → 整合 → 微調」的順序鋪陳，刻意讓每一節都回收前一節的成果：

```
01 image_classification (ViT)          視覺的最小起點：image_processor 取代 tokenizer
        │
        ▼
02 clip_retrieval (CLIP / SigLIP)      共享嵌入空間 + 跨模態檢索（回收 dual_model 的對比學習）
        │
        ├──────────────┐
        ▼              ▼
03 vlm_vqa_captioning  04 asr_whisper   生成端（image token 進 chat 模板） / 音訊端（log-mel）
        │              │
        └──────┬───────┘
               ▼
05 multimodal_rag                       整合：02 的嵌入檢索 + 03 的 VLM 生成（+ 04 音訊可選）
               │
               ▼
06 vlm_finetuning (LoRA / QLoRA)        閉合學習弧：把 03/04 的 LoRA/QLoRA 遷移到 VLM
```

**建議學習順序**：嚴格照 `01 → 02 → 03 → 04 → 05 → 06` 走。`05` 依賴 `02`（檢索）與 `03`（生成），`06` 依賴 `03`（VLM 載入）與你在 [`04-kbits-tuning`](../04-kbits-tuning/README.md) 學到的量化知識。若時間有限，`04 ASR` 與 `05/06` 弱耦合，可在 `03` 之後跳過先看 `06`，但不建議。

### Notebook 一覽表

| 順序 | Notebook | 教什麼（核心能力） | 回收哪個既有模組 |
| :--- | :--- | :--- | :--- |
| 01 | [`01-image_classification/vit_image_classification.ipynb`](./01-image_classification/vit_image_classification.ipynb) | 影像分類：`AutoImageProcessor` + `with_transform` + `Trainer` | [`01-Component`](../01-Component/README.md) 的 Trainer/評測管線 |
| 02 | [`02-clip_retrieval/clip_image_text_retrieval.ipynb`](./02-clip_retrieval/clip_image_text_retrieval.ipynb) | CLIP/SigLIP 共享嵌入、zero-shot 分類、圖文雙向檢索（Recall@k / MRR） | [`02-Adv-tasks/04-sentence_similarity/dual_model.ipynb`](../02-Adv-tasks/04-sentence_similarity/dual_model.ipynb) 的對比學習 + [`retrieval_bot`](../02-Adv-tasks/05-retrieval_chatbot/retrieval_bot.ipynb) 的 FAISS |
| 03 | [`03-vlm_vqa_captioning/vlm_vqa_captioning.ipynb`](./03-vlm_vqa_captioning/vlm_vqa_captioning.ipynb) | VLM 視覺問答與中文 captioning，image token 進 chat 模板 | [`02-Adv-tasks/08-generative_chatbot/chatbot.ipynb`](../02-Adv-tasks/08-generative_chatbot/chatbot.ipynb) 的 `apply_chat_template` |
| 04 | [`04-asr_whisper/whisper_asr.ipynb`](./04-asr_whisper/whisper_asr.ipynb) | Whisper ASR（chunking / 時間戳 / WER），串接既有文字模型 | [`01-Component`](../01-Component/README.md) 分類、[`02-Adv-tasks/07-text_summarization`](../02-Adv-tasks/07-text_summarization/summarization.ipynb) 摘要 |
| 05 | [`05-multimodal_rag/multimodal_embeddings_rag.ipynb`](./05-multimodal_rag/multimodal_embeddings_rag.ipynb) | 跨模態 embedding + 多模態 RAG（含 ColPali 文件視覺檢索） | [`retrieval_bot`](../02-Adv-tasks/05-retrieval_chatbot/retrieval_bot.ipynb) 的 dense retrieval + rerank |
| 06 | [`06-vlm_finetuning/vlm_lora_finetune.ipynb`](./06-vlm_finetuning/vlm_lora_finetune.ipynb) | VLM 上的 4-bit QLoRA 微調：多模態 `target_modules`、image token 遮罩 | [`03-PEFT/01-LoRA`](../03-PEFT/01-LoRA/chatbot_lora.ipynb) + [`04-kbits-tuning/04-4bits_training`](../04-kbits-tuning/04-4bits_training/llama2_qlora_4bit.ipynb) |

---

## 3. 核心抽象：`AutoProcessor` 統一 text / image / audio

純文字時代，你用 `AutoTokenizer` 把字串轉成 `input_ids`。多模態時代，每個模態都有自己的「前處理器」，而 `AutoProcessor` 是把它們組合在一起的傘狀類別：

- **文字** → `tokenizer` → `input_ids`、`attention_mask`
- **影像** → `image_processor`（resize / rescale / normalize，使用模型訓練時的 image mean/std）→ `pixel_values`
- **音訊** → `feature_extractor`（重採樣到 16kHz、轉 log-mel spectrogram）→ `input_features`

```python
# Before（純文字，01-04 的寫法）
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("一段文字", return_tensors="pt")
# inputs => {input_ids, attention_mask}

# After（多模態，05 的寫法）
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(model_id)
inputs = processor(images=image, text="這張圖是什麼?", return_tensors="pt")
# inputs => {pixel_values, input_ids, attention_mask}
```

**WHY 用 processor 而非各自呼叫**：VLM / CLIP 要求影像與文字的前處理「對齊」——影像被切成幾個 patch、產生幾個 image token、這些 token 插在序列哪個位置，全部由 processor 一致決定。手動拼接 `pixel_values` 與 `input_ids` 幾乎必然踩到 token 數對不上的坑。這正是本 repo 過去把影像當成裸 `PIL.Image` 處理（見 [`01-Component`](../01-Component/01pipeline/01.pipeline.ipynb) 的 zero-shot-object-detection 段落）的最大缺口。

---

## 4. 2026 推薦模型與 VRAM 需求總表

本模組所有模型都鎖定 2026 可用、且優先選中文友善者。表中 VRAM 為「4-bit 量化推論」的概略下限，實際依序列長度與 batch 而定。

| 能力 | 2026 首選 | 經典/輕量對照 | 概略 VRAM（4-bit 推論） |
| :--- | :--- | :--- | :--- |
| 影像分類 | `google/vit-base-patch16-224` | `facebook/dinov2-base`、`microsoft/swinv2-tiny-patch4-window8-256`、`timm/convnext_tiny.fb_in22k` | < 2 GB |
| 圖文檢索 | `google/siglip2-base-patch16-224` | `openai/clip-vit-base-patch32`、`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`、`jinaai/jina-clip-v2`（多語/中文） | 2–6 GB |
| VLM 問答/看圖 | `Qwen/Qwen2.5-VL-7B-Instruct`（中文強） | `llava-hf/llava-onevision-qwen2-7b-ov-hf`、`Salesforce/blip2-opt-2.7b`、`microsoft/Florence-2-large`（grounding/OCR） | 7B 約 6–8 GB |
| ASR | `openai/whisper-large-v3-turbo`（快） | `openai/whisper-large-v3`、`Systran/faster-whisper-large-v3`、`facebook/wav2vec2-base`（CTC 對照） | 2–6 GB |
| 多模態 embedding / RAG | `jinaai/jina-clip-v2`（圖文統一/多語） | `google/siglip2-base-patch16-224`、`vidore/colpali-v1.3`（PDF/截圖）、生成端 `Qwen/Qwen2.5-VL-7B-Instruct` | 檢索 2–4 GB + 生成 6–8 GB |
| VLM 微調 | `Qwen/Qwen2.5-VL-7B-Instruct`（4-bit QLoRA） | `llava-hf/llava-1.5-7b-hf`、`HuggingFaceTB/SmolVLM-Instruct`（低 VRAM）、`google/paligemma2-3b-pt-224` | 7B QLoRA 約 10–16 GB（含訓練狀態） |

> 若你的 GPU < 12 GB，優先用 `SmolVLM-Instruct`（VLM）與 `whisper-large-v3-turbo`（ASR），並全程套 4-bit。VRAM 估算與量化階梯的原理見 [`04-kbits-tuning/README.md`](../04-kbits-tuning/README.md)。

---

## 5. 環境與版本鎖定（2026）

本模組沿用全 repo 的鎖定版本，**不另開一套**。請先完成 [`00-Setup-and-Foundations/00-environment-setup.md`](../00-Setup-and-Foundations/00-environment-setup.md)：

```text
transformers>=4.46
datasets>=3.0
trl>=0.12
peft>=0.13
accelerate>=1.0
bitsandbytes>=0.44
evaluate>=0.4
safetensors>=0.4
torch>=2.4
# 多模態額外需要：
torchvision        # 影像 transform
faiss-cpu          # 檢索索引（有 GPU 可換 faiss-gpu）
librosa soundfile  # 音訊 I/O 與重採樣
pillow             # 影像 I/O
```

載入慣例與 01-04 完全一致（細節見 [`00-Setup-and-Foundations/01-2026-conventions.md`](../00-Setup-and-Foundations/01-2026-conventions.md)）：

```python
# 全模組統一：device_map + bfloat16 + safetensors
model = AutoModelForImageClassification.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
```

---

## 6. 各能力的應用場景與設計重點

### 6.1 影像分類（ViT）— 視覺的最小起點

**WHY 從這裡開始**：它和你在 [`01-Component`](../01-Component/README.md) 做的中文情感分類在訓練程式碼上幾乎一字不差，認知落差最小。唯一的差別是前處理：

```python
# Before（文字分類）
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True)
dataset = dataset.map(tokenize_fn, batched=True)

# After（影像分類）
image_processor = AutoImageProcessor.from_pretrained(model_id)
def transform(batch):
    batch["pixel_values"] = image_processor(batch["image"], return_tensors="pt")["pixel_values"]
    return batch
dataset = dataset.with_transform(transform)   # lazy，不預先展開整個影像張量
```

`with_transform` 取代 `map` 的原因：影像張量遠大於 token，預先 `map` 會吃爆記憶體與磁碟；`with_transform` 在每次取 batch 時才即時前處理。`Trainer`、`compute_metrics`、混淆矩陣錯誤分析全部沿用 01。

### 6.2 CLIP / SigLIP 跨模態檢索 — 嵌入即介面

**WHY 接在分類之後**：它直接延伸 [`dual_model.ipynb`](../02-Adv-tasks/04-sentence_similarity/dual_model.ipynb) 的雙塔對比學習（`CosineEmbeddingLoss`）——把「文字塔 + 文字塔」換成「影像塔 + 文字塔」，loss、normalize、FAISS 檢索的心智模型完全相同。zero-shot 分類用 prompt 模板 `一張{label}的照片` 算 `logits_per_image`，這是「不訓練就能分類」的關鍵直覺。SigLIP 與 CLIP 的差別（sigmoid loss vs softmax）是本節的理論亮點。檢索評測補上 Recall@k / MRR——這正是 [`retrieval_bot`](../02-Adv-tasks/05-retrieval_chatbot/retrieval_bot.ipynb) 缺的指標。

### 6.3 VLM 視覺問答與看圖說話 — chat 模板跨模態運作

**WHY 是本模組的核心**：2026 多模態的主力能力。重點在於讓你看到「同一套 `apply_chat_template`」如何承載影像：

```python
# Before（純文字 chatbot，見 02-Adv-tasks/08）
messages = [{"role": "user", "content": "這句話什麼意思?"}]

# After（VLM，content 變成 list，含 image）
messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "這張圖是什麼?"},
    ],
}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
)
```

模型架構心智模型：`vision encoder → projector → LLM`，影像被編碼後經 projector 投影成「LLM 看得懂的 token」，插進序列。7B VLM 用 `BitsAndBytesConfig` 4-bit 載入（呼應 [`04-kbits-tuning`](../04-kbits-tuning/README.md)）。本節同時涵蓋多輪含影像對話的 history 管理與多模態幻覺討論。

### 6.4 Whisper ASR — 音訊也是 processor 抽象

**WHY 需要它**：補齊音訊模態，並示範跨模態管線。`WhisperProcessor` = `feature_extractor`（log-mel）+ `tokenizer`，與文字/影像同源。重點技術：16kHz 取樣、`chunk_length_s=30` 長音訊切塊、`return_timestamps=True`、`language`/`task`（transcribe vs translate）。最有價值的設計是「Whisper 轉錄 → 餵進 01 的分類或 02 的摘要」，把音訊重新接回你已經會的文字能力。評測補 WER/CER。

### 6.5 多模態 embedding 與跨模態 RAG — 本模組能力的整合

**WHY 放在後段**：它把 [`retrieval_bot`](../02-Adv-tasks/05-retrieval_chatbot/retrieval_bot.ipynb) 的純文字 RAG 升級成跨模態：用 `jina-clip-v2` / SigLIP 把圖與文編進「同一個嵌入空間」存進 FAISS，query 取回後組進 VLM 的多模態 messages 生成有依據的中文回答。進階引入 `vidore/colpali-v1.3` 做 PDF/截圖的 late-interaction 視覺檢索（免 OCR 的 DocQA）。一句話總結：**RAG = 檢索（嵌入） + 生成（VLM）**，整合了 02 的視覺、03 的生成、可選 04 的音訊。

### 6.6 VLM 上的 LoRA / QLoRA — 閉合學習弧線

**WHY 是終點**：把 [`03-PEFT`](../03-PEFT/README.md) 與 [`04-kbits-tuning`](../04-kbits-tuning/README.md) 的知識遷移到 VLM，完成「單模態微調 → 多模態微調」的弧線。三個關鍵差異：

1. **`target_modules` 要含 LLM 與 projector**（LLM 的 `q/k/v/o` + 視覺 projector），通常凍結 vision encoder。
2. **標籤遮罩擴展**：image token 與 prompt 設 `-100`，只在 response 算 loss（延伸 04 的 `-100` 概念到影像）。
3. **用 `processor` 而非 `tokenizer`**，collator 要同時處理 `pixel_values + input_ids + labels`。

```python
# Before（純文字 QLoRA target_modules，見 04-kbits-tuning/04）
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# After（VLM，加上 projector，凍結 vision tower）
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "multi_modal_projector"]
# vision_tower 不放進 target_modules，且設為 requires_grad=False
```

訓練走 `SFTTrainer(peft_config=...)` + `SFTConfig(bf16=True, gradient_checkpointing=True, save_safetensors=True)`，與 04 同形。

---

## 7. 與 01-04 模組的知識回收對照

這張表是本模組的「省力證明」——多模態幾乎沒有新概念，只有舊概念的再應用：

| 舊概念（學自） | 在 05 的再應用 |
| :--- | :--- |
| `Trainer` + `TrainingArguments`（[`01-Component`](../01-Component/06Trainer/06%20Trainer%20classification_demo.ipynb)） | ViT 影像分類訓練（01-image_classification） |
| `compute_metrics` + 混淆矩陣（[`01-Component`](../01-Component/05evaluate/05%20evaluate.ipynb)） | 影像分類評測、檢索 Recall@k、ASR 的 WER |
| 雙塔 `CosineEmbeddingLoss`（[`dual_model.ipynb`](../02-Adv-tasks/04-sentence_similarity/dual_model.ipynb)） | CLIP/SigLIP 影像塔 + 文字塔對比學習 |
| FAISS dense retrieval（[`retrieval_bot`](../02-Adv-tasks/05-retrieval_chatbot/retrieval_bot.ipynb)） | 跨模態檢索索引、多模態 RAG |
| `apply_chat_template`（[`chatbot.ipynb`](../02-Adv-tasks/08-generative_chatbot/chatbot.ipynb)） | VLM 多模態 messages（image token） |
| LoRA `get_peft_model` + `LoraConfig`（[`03-PEFT`](../03-PEFT/README.md)） | VLM 的 `target_modules`（含 projector） |
| `BitsAndBytesConfig` 4-bit + `prepare_model_for_kbit_training`（[`04-kbits-tuning`](../04-kbits-tuning/README.md)） | 7B VLM 載入與 QLoRA 微調 |
| `-100` 標籤遮罩（[`04-kbits-tuning`](../04-kbits-tuning/04-4bits_training/llama2_qlora_4bit.ipynb)） | VLM 微調對 image token + prompt 遮罩 |

---

## 8. 延伸閱讀

- 多模態理論前置：[`00-multimodal-foundations.md`](./00-multimodal-foundations.md)
- 全 repo 2026 慣例聖經：[`../00-Setup-and-Foundations/01-2026-conventions.md`](../00-Setup-and-Foundations/01-2026-conventions.md)
- 術語與架構速查：[`../00-Setup-and-Foundations/02-glossary-and-architectures.md`](../00-Setup-and-Foundations/02-glossary-and-architectures.md)
- 量化與 QLoRA：[`../04-kbits-tuning/README.md`](../04-kbits-tuning/README.md)
- repo 總入口：[`../README.md`](../README.md)

---

> 本模組為 2026 版重構新增內容，所有 notebook 維持 zh-TW 教學、英文程式碼註解、版本鎖定。若你是第一次接觸多模態，請務必先讀完本頁第 1 與第 3 節再開始 `01-image_classification`。
