"""
Anime synopsis text preprocessing（推論用）

來源：src/text_branch/text_preprocessor.py（Text branch）
"""

import re
from typing import Optional

import pandas as pd


class TextPreprocessor:
    # (Source: AniList), (Written by MAL Rewrite), etc.
    _SOURCE_TAG_RE = re.compile(
        r'\(\s*(?:source|written by|edit(?:ed)?(?: by)?|adapted from)\s*[^)]*\)',
        re.IGNORECASE,
    )
    # Streaming platform credit lines.
    _PLATFORM_RE = re.compile(
        r'\b(?:streaming|available|watch|simulcast|licensed)[\w\s,]*'
        r'(?:crunchyroll|funimation|netflix|hidive|amazon prime|disney\+|hulu|bilibili|aniplus|wakanim)[^\n.!?]*[.!?]?',
        re.IGNORECASE,
    )
    # "Based on the manga/light novel/…" phrases.
    _BASED_ON_RE = re.compile(
        r'based on (?:the )?(?:manga|light novel|visual novel|game|novel|web novel|webtoon)\b[^.!?]*[.!?]?',
        re.IGNORECASE,
    )
    # Disc/home-video release notes.
    _DISC_RE = re.compile(
        r'\b(?:blu[- ]?ray|dvd|home video)\b[^.!?]*(?:includes?|comes? with|features?)[^.!?]*[.!?]?',
        re.IGNORECASE,
    )
    # HTML tags.
    _HTML_TAG_RE = re.compile(r'<[^>]+>')

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_marketing: bool = False,
        remove_extra_whitespace: bool = True,
        min_length: int = 10,
        max_length: int = 512,
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_marketing = remove_marketing
        self.remove_extra_whitespace = remove_extra_whitespace
        self.min_length = min_length
        self.max_length = max_length

    def clean(self, text: Optional[str]) -> Optional[str]:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return None
        if not isinstance(text, str):
            return None

        text = text.strip()
        if not text:
            return None

        text = self._HTML_TAG_RE.sub(' ', text)

        if self.remove_marketing:
            text = self._SOURCE_TAG_RE.sub(' ', text)
            text = self._PLATFORM_RE.sub(' ', text)
            text = self._BASED_ON_RE.sub(' ', text)
            text = self._DISC_RE.sub(' ', text)

        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)

        if self.lowercase:
            text = text.lower()

        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < self.min_length:
            return None
        if len(text) > self.max_length:
            text = text[:self.max_length]

        return text

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'description',
        output_column: Optional[str] = None,
    ) -> pd.DataFrame:
        output_column = output_column or text_column
        df = df.copy()
        df[output_column] = df[text_column].apply(self.clean)
        return df
