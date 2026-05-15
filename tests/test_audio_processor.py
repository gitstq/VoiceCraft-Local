"""Tests for audio processor module."""

import pytest
import numpy as np
from src.core.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test cases for AudioProcessor."""
    
    def test_initialization(self):
        """Test audio processor initialization."""
        processor = AudioProcessor(
            sample_rate=22050,
            hop_length=256,
            n_mels=80
        )
        
        assert processor.sample_rate == 22050
        assert processor.hop_length == 256
        assert processor.n_mels == 80
    
    def test_change_speed(self):
        """Test speed change functionality."""
        processor = AudioProcessor(sample_rate=22050)
        
        # Create test waveform (1 second)
        duration = 1.0
        t = np.linspace(0, duration, int(processor.sample_rate * duration))
        waveform = np.sin(2 * np.pi * 440 * t)
        
        # Test speed up
        fast = processor.change_speed(waveform, 2.0)
        assert len(fast) < len(waveform)
        
        # Test slow down
        slow = processor.change_speed(waveform, 0.5)
        assert len(slow) > len(waveform)
        
        # Test no change
        same = processor.change_speed(waveform, 1.0)
        assert len(same) == len(waveform)
    
    def test_change_pitch(self):
        """Test pitch change functionality."""
        processor = AudioProcessor(sample_rate=22050)
        
        # Create test waveform
        duration = 1.0
        t = np.linspace(0, duration, int(processor.sample_rate * duration))
        waveform = np.sin(2 * np.pi * 440 * t)
        
        # Test pitch change (should not change length)
        higher = processor.change_pitch(waveform, 1.5)
        assert len(higher) == len(waveform)
        
        lower = processor.change_pitch(waveform, 0.7)
        assert len(lower) == len(waveform)
        
        # Test no change
        same = processor.change_pitch(waveform, 1.0)
        assert len(same) == len(waveform)
    
    def test_normalize_volume(self):
        """Test volume normalization."""
        processor = AudioProcessor()
        
        # Create quiet waveform
        waveform = np.random.randn(10000) * 0.1
        
        # Normalize
        normalized = processor.normalize_volume(waveform, target_db=-20)
        
        # Check that waveform is not clipped
        assert np.max(np.abs(normalized)) <= 1.0
        
        # Check that volume increased
        original_rms = np.sqrt(np.mean(waveform ** 2))
        normalized_rms = np.sqrt(np.mean(normalized ** 2))
        assert normalized_rms > original_rms
    
    def test_fade_in_out(self):
        """Test fade in/out functionality."""
        processor = AudioProcessor(sample_rate=22050)
        
        # Create test waveform
        waveform = np.ones(10000)
        
        # Apply fade
        faded = processor.fade_in_out(waveform, fade_duration=0.1)
        
        # Check that start and end are faded
        fade_samples = int(0.1 * 22050)
        assert faded[0] < 1.0
        assert faded[-1] < 1.0
        assert np.all(faded[fade_samples:-fade_samples] == 1.0)
    
    def test_get_duration(self):
        """Test duration calculation."""
        processor = AudioProcessor(sample_rate=22050)
        
        # Create 2 second waveform
        waveform = np.zeros(44100)
        
        duration = processor.get_duration(waveform)
        assert duration == 2.0
    
    def test_add_pause(self):
        """Test adding pause to waveform."""
        processor = AudioProcessor(sample_rate=22050)
        
        waveform = np.ones(1000)
        pause_duration = 0.5
        
        with_pause = processor.add_pause(waveform, pause_duration)
        
        expected_length = len(waveform) + int(pause_duration * 22050)
        assert len(with_pause) == expected_length
        
        # Check that pause is silence
        assert np.all(with_pause[len(waveform):] == 0)
    
    def test_apply_effects(self):
        """Test applying multiple effects."""
        processor = AudioProcessor()
        
        waveform = np.random.randn(10000)
        
        # Apply multiple effects
        processed = processor.apply_effects(
            waveform,
            speed=1.2,
            pitch=1.1,
            volume=0.8,
            trim=False,
            fade=True
        )
        
        # Check that output is valid
        assert len(processed) > 0
        assert np.max(np.abs(processed)) <= 1.0
