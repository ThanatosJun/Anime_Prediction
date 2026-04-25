"""
Embedding Generation Module

Generates sentence embeddings using Sentence-Transformers.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Generates embeddings from text using Sentence-Transformers.
    
    Features:
    - Batch processing for efficiency
    - Device management (GPU/CPU)
    - Caching support
    - Reproducible random seed
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 16,
        random_seed: int = 42,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence-Transformers model identifier
            device: Device to use ("auto", "cuda", or "cpu")
            batch_size: Batch size for processing
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(random_seed)
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded on {self.device}. Embedding dimension: {self.embedding_dim}")

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve runtime device in a CPU-safe way for non-CUDA environments."""
        requested_device = (requested_device or "auto").lower()

        if requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if requested_device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"

        if requested_device not in {"cuda", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, cpu")

        return requested_device
    
    def encode(self, texts: List[str], show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode a list of texts to embeddings.
        
        Args:
            texts: List of text strings
            show_progress_bar: Show progress bar during encoding
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )
        return embeddings
    
    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "description",
        output_column: str = "text_embedding",
        skip_null: bool = True,
    ) -> pd.DataFrame:
        """
        Encode text column in dataframe and add embeddings.
        
        Args:
            df: Input dataframe
            text_column: Column containing text to encode
            output_column: Column name for embeddings (stored as list)
            skip_null: Skip rows with null text
            
        Returns:
            Dataframe with embedding column
        """
        df = df.copy()
        
        # Identify rows to encode
        if skip_null:
            mask = df[text_column].notna()
            texts_to_encode = df.loc[mask, text_column].tolist()
            indices_to_encode = df.loc[mask].index
        else:
            texts_to_encode = df[text_column].fillna("").tolist()
            indices_to_encode = df.index
        
        print(f"Encoding {len(texts_to_encode)} texts...")
        embeddings = self.encode(texts_to_encode, show_progress_bar=True)
        
        # Initialize embedding column with None
        df[output_column] = None
        
        # Assign embeddings
        for idx, embedding in zip(indices_to_encode, embeddings):
            df.loc[idx, output_column] = embedding
        
        return df
    
    def get_model_info(self) -> dict:
        """
        Get model information for reproducibility tracking.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed,
        }
