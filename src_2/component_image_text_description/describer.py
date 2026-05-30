"""
ToriiGate-0.5 推論模組（GGUF / safetensors 自動判斷）

輸入：圖片路徑 / PIL.Image
輸出：圖片文字描述

使用方式：
    from describer import ToriiGateDescriber
    d = ToriiGateDescriber()
    print(d.describe('image.jpg'))
    print(d.describe('image.jpg', mode='short'))
"""
import base64
import io
import sys
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image

from config import load_config
from prompts import make_user_query, system_prompt, prompts_b

VALID_MODES = list(prompts_b.keys())


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


class ToriiGateDescriber:
    """
    Args:
        model_path : 模型路徑。
                     - 資料夾（含 config.json + *.safetensors）→ transformers 載入
                     - *.gguf 檔案                            → llama-cpp-python 載入
                     未傳入時從 torii_config.yaml 讀取
        config     : dict（可選），未傳入時從 torii_config.yaml 讀取
    """

    def __init__(self, model_path: str = None, config: dict = None):
        if config is None:
            config = load_config()

        cfg  = config['model']
        path = model_path or cfg['model_path']

        self.max_new_tokens = cfg.get('max_new_tokens', 1000)
        self._backend       = self._detect_backend(path)

        if self._backend == 'gguf':
            self._load_gguf(path, cfg)
        else:
            self._load_transformers(path, cfg)

    # ── 後端偵測 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_backend(path: str) -> str:
        p = Path(path)
        if p.suffix == '.gguf' or (p.is_dir() and list(p.glob('*.gguf'))):
            return 'gguf'
        return 'transformers'

    # ── 載入 GGUF（llama-cpp-python）─────────────────────────────────────────

    def _load_gguf(self, path: str, cfg: dict):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("需要安裝 llama-cpp-python：pip install llama-cpp-python")

        # 若傳入資料夾，自動找第一個 .gguf 檔
        p = Path(path)
        if p.is_dir():
            gguf_files = list(p.glob('*.gguf'))
            if not gguf_files:
                raise FileNotFoundError(f"資料夾 {path} 內找不到 .gguf 檔案")
            path = str(gguf_files[0])

        n_gpu = -1 if cfg.get('device', 'cuda') == 'cuda' else 0

        self._llm = Llama(
            model_path=path,
            n_gpu_layers=n_gpu,
            n_ctx=cfg.get('n_ctx', 4096),
            verbose=False,
        )

    # ── 載入 safetensors（transformers）──────────────────────────────────────

    def _load_transformers(self, path: str, cfg: dict):
        try:
            from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError("需要安裝 transformers：pip install transformers")

        device = cfg.get('device', 'cuda')
        dtype  = getattr(torch, cfg.get('dtype', 'bfloat16'))
        min_px = cfg.get('min_pixels', 256) * 32 * 32
        max_px = cfg.get('max_pixels', 512) * 32 * 32

        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map=device,
        )
        self._processor = AutoProcessor.from_pretrained(
            path,
            min_pixels=min_px,
            max_pixels=max_px,
            padding_side='right',
        )
        self._device = device
        self._model.eval()

    # ── 核心推論 ──────────────────────────────────────────────────────────────

    def describe(
        self,
        image: Union[str, Path, Image.Image],
        mode: str = 'long_thoughts_v2',
        use_names: bool = True,
        tags: Optional[list] = None,
        characters: Optional[list] = None,
        character_tags: Optional[dict] = None,
        character_descriptions: Optional[dict] = None,
    ) -> str:
        """
        Args:
            image                  : 圖片路徑或 PIL.Image
            mode                   : caption 模式（見 VALID_MODES）
            use_names              : True → 嘗試辨識角色名稱
            tags                   : booru tags list（可選）
            characters             : 角色名稱 list（可選）
            character_tags         : {'chars': {name: [tags]}, 'skins': {...}}（可選）
            character_descriptions : {'chars': {name: descr},  'skins': {...}}（可選）
        """
        if mode not in VALID_MODES:
            raise ValueError(f"mode 必須是 {VALID_MODES} 其中之一，收到：{mode!r}")

        item = {
            'tags':       tags or [],
            'characters': characters or [],
            'char_p_tags': character_tags or {'chars': {}, 'skins': {}},
            'char_descr':  character_descriptions or {'chars': {}, 'skins': {}},
        }
        prompt = make_user_query(
            item,
            c_type=mode,
            use_names=use_names,
            add_tags=bool(tags),
            add_characters=bool(characters),
            add_char_tags=bool(character_tags),
            add_description=bool(character_descriptions),
        )

        img = self._load_image(image)

        if self._backend == 'gguf':
            return self._generate_gguf(img, prompt)
        else:
            return self._generate_transformers(img, prompt)

    def describe_batch(self, images: list, mode: str = 'long_thoughts_v2', **kwargs) -> list:
        return [self.describe(img, mode=mode, **kwargs) for img in images]

    # ── 內部方法 ──────────────────────────────────────────────────────────────

    def _load_image(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert('RGB')
        return Image.open(str(image)).convert('RGB')

    def _generate_gguf(self, img: Image.Image, prompt: str) -> str:
        b64 = _image_to_base64(img)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
        )
        return response['choices'][0]['message']['content']

    def _generate_transformers(self, img: Image.Image, prompt: str) -> str:
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        texts  = [self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)]
        inputs = self._processor(text=texts, images=[img], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        result = self._processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return result[0]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """
    用法：
        python describer.py <image_path> [mode]

    範例：
        python describer.py image.jpg
        python describer.py image.jpg short
        python describer.py image.jpg json
    """
    if len(sys.argv) < 2:
        print("Usage: python describer.py <image_path> [mode]")
        print(f"  mode: {VALID_MODES}  (default: long_thoughts_v2)")
        sys.exit(1)

    image_path = sys.argv[1]
    config     = load_config()
    mode       = sys.argv[2] if len(sys.argv) > 2 else config.get('caption', {}).get('mode', 'long_thoughts_v2')

    describer = ToriiGateDescriber(config=config)
    print(describer.describe(image_path, mode=mode))


if __name__ == '__main__':
    main()
