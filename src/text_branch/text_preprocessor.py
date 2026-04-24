"""
Text Preprocessing Module

Handles cleaning and normalization of anime synopsis text.
"""

import re
from typing import Optional
import pandas as pd


class TextPreprocessor:
    """
    Preprocesses anime synopsis text for embedding generation.
    
    Features:
    - URL removal
    - Whitespace normalization
    - Length constraints
    - Null/empty handling
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_extra_whitespace: bool = True,
        min_length: int = 10,
        max_length: int = 512,
    ):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_extra_whitespace: Normalize whitespace
            min_length: Minimum text length to retain
            max_length: Maximum text length (for truncation)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        self.max_length = max_length
    
    def clean(self, text: Optional[str]) -> Optional[str]:
        """
        Clean a single text string.
        
        Args:
            text: Input text or None
            
        Returns:
            Cleaned text or None if empty
        """
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return None
        
        if isinstance(text, str):
            text = str(text).strip()
        else:
            return None
        
        # Check empty after strip
        if not text:
            return None
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)
        
        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Check minimum length
        if len(text) < self.min_length:
            return None
        
        # Truncate to max length
        if len(text) > self.max_length:
            text = text[:self.max_length]
        
        return text
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "description",
        output_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process text column in a dataframe.
        
        Args:
            df: Input dataframe
            text_column: Column name containing text
            output_column: Output column name (default: text_column)
            
        Returns:
            Dataframe with cleaned text column
        """
        output_column = output_column or text_column
        df = df.copy()
        df[output_column] = df[text_column].apply(self.clean)
        return df
    
    def get_clean_stats(self, df: pd.DataFrame, text_column: str = "description") -> dict:
        """
        Get preprocessing statistics.
        
        Args:
            df: Input dataframe
            text_column: Column name containing text
            
        Returns:
            Dictionary with cleaning statistics
        """
        total = len(df)
        null_before = df[text_column].isna().sum()
        
        # Apply cleaning
        cleaned = df[text_column].apply(self.clean)
        null_after = cleaned.isna().sum()
        
        kept = total - null_after
        
        return {
            "total_rows": total,
            "null_before": null_before,
            "null_after": null_after,
            "rows_kept": kept,
            "rows_dropped": null_after,
            "retention_rate": kept / total if total > 0 else 0,
        }
