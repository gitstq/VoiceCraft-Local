"""Audio playback functionality."""

import numpy as np
from typing import Optional


class AudioPlayer:
    """Simple audio player using sounddevice."""
    
    def __init__(self, sample_rate: int = 22050):
        """Initialize audio player.
        
        Args:
            sample_rate: Audio sample rate.
        """
        self.sample_rate = sample_rate
        self._stream = None
        
        try:
            import sounddevice as sd
            self.sd = sd
            self.available = True
        except ImportError:
            self.available = False
    
    def play(self, waveform: np.ndarray, blocking: bool = True) -> None:
        """Play audio waveform.
        
        Args:
            waveform: Audio waveform to play.
            blocking: Whether to block until playback completes.
            
        Raises:
            RuntimeError: If sounddevice is not available.
        """
        if not self.available:
            raise RuntimeError(
                "Audio playback not available. "
                "Install sounddevice: pip install sounddevice"
            )
        
        self.sd.play(waveform, self.sample_rate)
        
        if blocking:
            self.sd.wait()
    
    def stop(self) -> None:
        """Stop current playback."""
        if self.available:
            self.sd.stop()
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        if not self.available:
            return False
        return self.sd.get_status() is not None
