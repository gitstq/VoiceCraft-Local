"""Tests for text processor module."""

import pytest
from src.core.text_processor import TextProcessor


class TestTextProcessor:
    """Test cases for TextProcessor."""
    
    def test_initialization(self):
        """Test text processor initialization."""
        processor = TextProcessor(language="zh")
        assert processor.language == "zh"
        assert processor.vocab_size > 0
    
    def test_normalize_text(self):
        """Test text normalization."""
        processor = TextProcessor()
        
        # Test whitespace normalization
        text = "Hello   world"
        normalized = processor._normalize_text(text)
        assert normalized == "Hello world"
        
        # Test special character removal
        text = "Hello @#$ world"
        normalized = processor._normalize_text(text)
        assert "@" not in normalized
        assert "#" not in normalized
    
    def test_text_to_phonemes_chinese(self):
        """Test Chinese text to phonemes conversion."""
        processor = TextProcessor(language="zh")
        
        text = "你好"
        phonemes = processor.text_to_phonemes(text, language="zh")
        
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0
    
    def test_text_to_phonemes_english(self):
        """Test English text to phonemes conversion."""
        processor = TextProcessor(language="en")
        
        text = "hello"
        phonemes = processor.text_to_phonemes(text, language="en")
        
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0
    
    def test_phonemes_to_ids(self):
        """Test phoneme to ID conversion."""
        processor = TextProcessor()
        
        phonemes = ["AA", "B", "K"]
        ids = processor.phonemes_to_ids(phonemes)
        
        assert isinstance(ids, list)
        assert len(ids) == len(phonemes) + 2  # +2 for SOS and EOS
        assert ids[0] == processor.PHONEME_VOCAB["<SOS>"]
        assert ids[-1] == processor.PHONEME_VOCAB["<EOS>"]
    
    def test_ids_to_phonemes(self):
        """Test ID to phoneme conversion."""
        processor = TextProcessor()
        
        ids = [40, 0, 15, 23, 41]  # SOS, AA, B, K, EOS
        phonemes = processor.ids_to_phonemes(ids)
        
        assert isinstance(phonemes, list)
        assert len(phonemes) == len(ids)
    
    def test_roundtrip_conversion(self):
        """Test roundtrip conversion of phonemes to IDs and back."""
        processor = TextProcessor()
        
        original_phonemes = ["AA", "B", "K"]
        ids = processor.phonemes_to_ids(original_phonemes)
        recovered_phonemes = processor.ids_to_phonemes(ids[1:-1])  # Remove SOS/EOS
        
        assert recovered_phonemes == original_phonemes
