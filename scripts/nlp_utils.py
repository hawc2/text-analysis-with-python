"""Small NLP utilities for tokenization and corpus preparation.

This module adapts helpers from the top-level `nlp/` folder for reuse
inside `Hawc2/text-analysis-with-python` scripts.
"""
from typing import List
import csv
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

_tokenizer = RegexpTokenizer(r"\w+")
_stopwords = set(stopwords.words('english'))


def tokenize(text: str) -> List[str]:
    """Tokenize a piece of text into word tokens (alpha/numeric).

    Lowercases and returns a list of tokens.
    """
    if not isinstance(text, str):
        return []
    tokens = _tokenizer.tokenize(text)
    return [t.lower() for t in tokens]


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove English stopwords from a token list."""
    return [t for t in tokens if t not in _stopwords]


def normalize_document(text: str) -> List[str]:
    """Full normalize pipeline: tokenize + remove stopwords."""
    return remove_stopwords(tokenize(text))


def corpus_from_csv(path: str, text_col: str = 'review') -> List[List[str]]:
    """Read a CSV and return a list of token lists from `text_col`.

    Uses pandas for convenience; falls back to csv reader if pandas fails.
    """
    try:
        df = pd.read_csv(path, encoding='utf-8')
        texts = df[text_col].astype(str).tolist()
    except Exception:
        texts = []
        with open(path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                texts.append(r.get(text_col, ''))
    return [normalize_document(t) for t in texts]
