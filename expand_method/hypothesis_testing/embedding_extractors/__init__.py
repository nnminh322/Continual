"""
Embedding extractors for SRT routing.
"""
from .clip_extractor import CLIPVisionExtractor
from .sentence_bert import SentenceBERTExtractor

__all__ = ["CLIPVisionExtractor", "SentenceBERTExtractor"]
