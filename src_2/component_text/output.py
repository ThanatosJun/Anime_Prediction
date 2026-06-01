"""
Text embedding 推論主入口

輸入：動畫 description 文字（或 CSV）
輸出：e5-base-v2 embedding（768-dim）→ parquet

使用方式：
    from output import TextEmbedder
    embedder = TextEmbedder()
    emb = embedder.embed('A story about a boy who...')  # (768,)

    # CLI
    python output.py                            # 依 text_encoder_config.yaml 批次處理
    python output.py "A story about a boy..."   # 單句直接輸出
"""

import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from config import load_config
from text_preprocessor import TextPreprocessor
from embedding_generator import EmbeddingGenerator


class TextEmbedder:
    """
    Args:
        model_name : HuggingFace model ID 或本地路徑。未傳入時從 config 讀取。
        config     : dict（可選），未傳入時從 text_encoder_config.yaml 讀取。
    """

    def __init__(self, model_name: str = None, config: dict = None):
        if config is None:
            config = load_config()

        model_cfg  = config.get('model', {})
        preproc_cfg = config.get('preprocessing', {})

        self._preprocessor = TextPreprocessor(
            lowercase             = preproc_cfg.get('lowercase', True),
            remove_urls           = preproc_cfg.get('remove_urls', True),
            remove_marketing      = preproc_cfg.get('remove_marketing', False),
            remove_extra_whitespace = preproc_cfg.get('remove_extra_whitespace', True),
            min_length            = int(preproc_cfg.get('min_length', 10)),
            max_length            = int(preproc_cfg.get('max_length', 512)),
        )

        self._generator = EmbeddingGenerator(
            model_name = model_name or model_cfg.get('model_name', 'intfloat/e5-base-v2'),
            device     = model_cfg.get('device', 'auto'),
            batch_size = int(model_cfg.get('batch_size', 16)),
        )

    @property
    def embedding_dim(self) -> int:
        return self._generator.embedding_dim

    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        單句文字 → embedding。
        Returns None if text is empty after preprocessing.
        """
        cleaned = self._preprocessor.clean(text)
        if cleaned is None:
            return None
        return self._generator.encode([cleaned], show_progress_bar=False)[0]

    def embed_batch(self, texts: list) -> np.ndarray:
        """
        多句文字 → (N, 768) array。
        空/清洗後為 None 的句子以零向量填補。
        """
        cleaned = [self._preprocessor.clean(t) or '' for t in texts]
        return self._generator.encode(cleaned, show_progress_bar=True)

    def embed_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = 'description',
        id_col: str = 'id',
    ) -> dict:
        """
        對整個 DataFrame 批次 embed。
        Returns: {anime_id: np.ndarray(768,), ...}
        """
        results = {}
        valid_mask = df[text_col].notna()
        sub = df.loc[valid_mask].copy()

        cleaned = sub[text_col].apply(self._preprocessor.clean)
        valid2 = cleaned.notna()
        sub = sub.loc[valid2]
        cleaned = cleaned.loc[valid2]

        if sub.empty:
            return results

        embs = self._generator.encode(cleaned.tolist(), show_progress_bar=True)
        for idx, anime_id in enumerate(sub[id_col]):
            results[anime_id] = embs[idx]

        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    """
    用法：
        python output.py                      # 批次處理（依 config）
        python output.py "some text here"     # 單句 embed，印出前 8 維
    """
    config = load_config()

    # 單句模式
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        embedder = TextEmbedder(config=config)
        emb = embedder.embed(text)
        if emb is None:
            print("Text too short or empty after preprocessing.")
        else:
            print(f"Embedding dim: {emb.shape[0]}")
            print(f"First 8 values: {emb[:8]}")
        return

    # 批次模式
    inf_cfg   = config.get('inference', {})
    input_csv = inf_cfg.get('input_csv')
    out_path  = inf_cfg.get('output_path', 'src_2/data/text_embeddings/text_embeddings.parquet')
    text_col  = inf_cfg.get('text_column', 'description')
    id_col    = inf_cfg.get('id_column', 'id')

    if not input_csv:
        print("inference.input_csv not set in text_encoder_config.yaml")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    embedder = TextEmbedder(config=config)
    id_to_emb = embedder.embed_dataframe(df, text_col=text_col, id_col=id_col)

    # 組成 parquet 格式：每維一欄 emb_000 … emb_767
    emb_dim = embedder.embedding_dim
    emb_cols = [f'emb_{i:03d}' for i in range(emb_dim)]

    rows = []
    for anime_id, emb in id_to_emb.items():
        row = {id_col: anime_id}
        row.update(dict(zip(emb_cols, emb.tolist())))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} embeddings → {out_path}")


if __name__ == '__main__':
    main()
