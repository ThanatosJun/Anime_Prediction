# component_image_text_description

動畫圖片 **文字描述推論**：輸入圖片 → ToriiGate-0.5 → 輸出文字描述。

基於 **Qwen3.5-4B**，專為動畫/數位藝術設計，支援 10 種輸出格式與角色辨識。

---

## 檔案一覽

| 檔案 | 用途 |
|------|------|
| `describer.py` | **主入口**。`ToriiGateDescriber` class + CLI |
| `prompts.py` | 10 種 caption prompt 模板 + `make_user_query()` |
| `config.py` | 讀取設定檔 |
| `torii_config.yaml` | 設定（模型路徑、caption 模式） |
| `run_poc_describe.py` | POC 腳本：對 N 部動畫 cover 批次生成描述 → `poc_descriptions.csv` |

## 模型權重（兩種後端，二擇一）

| 後端 | 目錄 | 內容 | 觸發條件 |
|------|------|------|---------|
| **transformers**（POC 用）| `model-torii-hf/` | HF safetensors（`model.safetensors` ~10GB + config/tokenizer）| 目錄內**無** `.gguf` |
| GGUF（llama-cpp）| `model-torii/` | `ToriiGate-0.5_Q8_0.gguf`（5.1GB Q8 量化）| 目錄內**有** `.gguf` |

> `describer._detect_backend()`：路徑為 `.gguf` 檔或資料夾內含 `.gguf` → GGUF 後端；否則 → transformers。
> ⚠️ 因此 HF 權重須放 `model-torii-hf/`（不可與 gguf 同目錄，否則會誤選 GGUF 後端）。

---

## 安裝依賴

```bash
# transformers 後端（HF safetensors，POC 用）
pip install transformers accelerate    # accelerate 為 device_map 載入所需

# GGUF 後端（可選，需 CUDA build）
pip install llama-cpp-python
```

> v0.5 **不需要** `qwen-vl-utils`（v0.4 才需要）

---

## 快速使用

### Python 呼叫

```python
from describer import ToriiGateDescriber

# transformers 後端（HF safetensors）；GGUF 後端改指向 model-torii/
describer = ToriiGateDescriber('src_2/component_image_text_description/model-torii-hf')

# 基本描述（long_thoughts_v2，最詳細）
text = describer.describe('image.jpg')

# 簡短 caption
text = describer.describe('image.jpg', mode='short')

# 加入角色資訊提升準確度
text = describer.describe(
    'image.jpg',
    mode='long_thoughts_v2',
    characters=['hatsune_miku', 'megurine_luka'],
    tags=['2girls', 'blue_hair', 'pink_hair', 'holding_hands'],
)
```

### CLI

```bash
python describer.py path/to/image.jpg           # 預設 long_thoughts_v2
python describer.py path/to/image.jpg short
python describer.py path/to/image.jpg json
```

### POC 批次描述（驗證 VLM 描述能否補強 text 分支）

```bash
# 對 train split 隨機 50 部 cover 生成描述 → poc_descriptions.csv
python src_2/component_image_text_description/run_poc_describe.py --n 50 --mode short
```

目的：將 VLM 描述當作 text 分支的**替代/並接輸入來源**（非補 Swin 視覺），
驗證是否比官方 description 帶更多預測訊號（text Captum 貢獻僅 0.027，rag_text 0.004）。
輸出 `poc_descriptions.csv`（id, title, has_cover, description）供人工檢視語意品質。

---

## Caption 模式

| mode | 說明 |
|------|------|
| `long_thoughts_v2` | **預設**。6 段結構化：角色分析 / 關鍵細節 / 詳細描述 / 各角色描述 |
| `long_thoughts` | 6 段：含一般描述、各部件清單、背景特效 |
| `long` | 2~5 段自然文字，詳細生動 |
| `short` | 簡短 caption，適合 diffusion model |
| `json` | JSON 格式（character / background / atmosphere） |
| `min_structured_json` | 簡化 JSON，短描述優先 |
| `min_structured_md` | 3 段 Markdown（含推理段） |
| `chroma-style` | 4 段：Regular / Individual Parts / Midjourney / DeviantArt |
| `md_comic` | Markdown 漫畫格式，逐格描述 |
| `json_comic` | JSON 漫畫格式，逐格描述 |

---

## Grounding 參數（可選，提升角色辨識準確度）

```python
describer.describe(
    'image.jpg',
    mode='long_thoughts_v2',
    use_names=True,                        # 啟用角色名稱辨識（預設 True）
    tags=['2girls', 'blue_hair'],          # booru tags
    characters=['hatsune_miku'],           # 圖中角色名稱
    character_tags={                       # 各角色的代表 tags
        'chars': {'hatsune_miku': ['twintails', 'blue_hair', 'headphones']},
        'skins': {}
    },
)
```

---

## v0.4 → v0.5 差異

| | v0.4 | v0.5 |
|--|--|--|
| 基底模型 | Qwen2-VL-7B | Qwen3.5-4B |
| 模型 class | `Qwen2VLForConditionalGeneration` | `Qwen3_5ForConditionalGeneration` |
| Processor | `Qwen2VLProcessor` | `AutoProcessor` |
| 依賴 | `qwen-vl-utils` | 不需要 |
| pixel 單位 | `× 28×28` | `× 32×32` |
| 預設模式 | `long` | `long_thoughts_v2` |
| Caption 模式數 | 5 | 10 |
