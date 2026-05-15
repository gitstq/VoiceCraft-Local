"""Audio recording functionality for voice cloning."""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import soundfile as sf


class AudioRecorder:
    """Record audio from microphone."""
    
    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        """Initialize audio recorder.
        
        Args:
            sample_rate: Recording sample rate.
            channels: Number of audio channels (1=mono, 2=stereo).
        """
        self.sample_rate = sample_rate
        self.channels = channels
        
        try:
            import sounddevice as sd
            self.sd = sd
            self.available = True
        except ImportError:
            self.available = False
    
    def record(
        self,
        duration: float,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Record audio for specified duration.
        
        Args:
            duration: Recording duration in seconds.
            output_path: Optional path to save recording.
            
        Returns:
            Recorded audio waveform.
            
        Raises:
            RuntimeError: If sounddevice is not available.
        """
        if not self.available:
            raise RuntimeError(
                "Audio recording not available. "
                "Install sounddevice: pip install sounddevice"
            )
        
        print(f"Recording for {duration} seconds...")
        
        num_samples = int(duration * self.sample_rate)
        recording = self.sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        )
        
        self.sd.wait()
        
        # Flatten if mono
        if self.channels == 1:
            recording = recording.squeeze()
        
        print("Recording complete!")
        
        # Save if path provided
        if output_path:
            sf.write(output_path, recording, self.sample_rate)
            print(f"Saved to: {output_path}")
        
        return recording
    
    def record_with_vad(
        self,
        max_duration: float = 30.0,
        silence_threshold: float = 0.01,
        min_silence_duration: float = 2.0,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """Record audio with voice activity detection.
        
        Records until silence is detected for specified duration.
        
        Args:
            max_duration: Maximum recording duration.
            silence_threshold: Energy threshold for silence detection.
            min_silence_duration: Minimum silence duration to stop recording.
            output_path: Optional path to save recording.
            
        Returns:
            Recorded audio waveform.
        """
        if not self.available:
            raise RuntimeError("Audio recording not available.")
        
        print("Recording... (speak now, stop to end)")
        
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(chunk_duration * self.sample_rate)
        
        recording = []
        silence_samples = 0
        min_silence_samples = int(min_silence_duration * self.sample_rate)
        max_samples = int(max_duration * self.sample_rate)
        
        with self.sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32
        ) as stream:
            total_samples = 0
            
            while total_samples < max_samples:
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk.squeeze()
                recording.append(chunk)
                total_samples += len(chunk)
                
                # Check energy level
                energy = np.sqrt(np.mean(chunk ** 2))
                
                if energy < silence_threshold:
                    silence_samples += len(chunk)
                    if silence_samples >= min_silence_samples:
                        print("Silence detected, stopping...")
                        break
                else:
                    silence_samples = 0
        
        # Concatenate chunks
        waveform = np.concatenate(recording)
        
        print(f"Recorded {len(waveform) / self.sample_rate:.2f} seconds")
        
        # Save if path provided
        if output_path:
            sf.write(output_path, waveform, self.sample_rate)
            print(f"Saved to: {output_path}")
        
        return waveform
    
    def list_devices(self) -> None:
        """List available audio input devices."""
        if not self.available:
            print("Audio recording not available.")
            return
        
        devices = self.sd.query_devices()
        print("\nAudio Input Devices:")
        print("-" * 50)
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}")
                print(f"   Channels: {device['max_input_channels']}")
                print(f"   Sample Rate: {device['default_samplerate']}")
                print()
