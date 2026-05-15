"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path

from src.core.config import TTSConfig


class TestTTSConfig:
    """Test cases for TTSConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TTSConfig()
        
        assert config.model_name == "voicecraft-base"
        assert config.sample_rate == 22050
        assert config.speed == 1.0
        assert config.pitch == 1.0
        assert config.language == "zh"
        assert config.output_format == "wav"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TTSConfig(
            model_name="voicecraft-fast",
            sample_rate=16000,
            speed=1.5,
            pitch=0.9,
            language="en"
        )
        
        assert config.model_name == "voicecraft-fast"
        assert config.sample_rate == 16000
        assert config.speed == 1.5
        assert config.pitch == 0.9
        assert config.language == "en"
    
    def test_invalid_speed(self):
        """Test validation of invalid speed values."""
        with pytest.raises(ValueError, match="Speed must be between"):
            TTSConfig(speed=3.0)
        
        with pytest.raises(ValueError, match="Speed must be between"):
            TTSConfig(speed=0.3)
    
    def test_invalid_pitch(self):
        """Test validation of invalid pitch values."""
        with pytest.raises(ValueError, match="Pitch must be between"):
            TTSConfig(pitch=3.0)
        
        with pytest.raises(ValueError, match="Pitch must be between"):
            TTSConfig(pitch=0.3)
    
    def test_yaml_roundtrip(self):
        """Test YAML save and load."""
        config = TTSConfig(
            model_name="test-model",
            speed=1.2,
            language="ja"
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save to YAML
            config.to_yaml(temp_path)
            
            # Load from YAML
            loaded_config = TTSConfig.from_yaml(temp_path)
            
            assert loaded_config.model_name == config.model_name
            assert loaded_config.speed == config.speed
            assert loaded_config.language == config.language
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TTSConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "model_name" in config_dict
        assert "sample_rate" in config_dict
        assert "speed" in config_dict
