"""Audio processing utilities for TTS."""

import numpy as np
import librosa
from typing import Tuple, Optional


class AudioProcessor:
    """Process audio waveforms for TTS."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        n_mels: int = 80
    ):
        """Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate.
            hop_length: Hop length for STFT.
            win_length: Window length for STFT.
            n_fft: FFT size.
            n_mels: Number of mel frequency bins.
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.n_mels = n_mels
    
    def wav_to_mel(
        self,
        waveform: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """Convert waveform to mel spectrogram.
        
        Args:
            waveform: Audio waveform.
            normalize: Whether to normalize the output.
            
        Returns:
            Mel spectrogram.
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        if normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec
    
    def mel_to_wav(
        self,
        mel_spec: np.ndarray,
        n_iter: int = 32
    ) -> np.ndarray:
        """Convert mel spectrogram to waveform using Griffin-Lim.
        
        Args:
            mel_spec: Mel spectrogram.
            n_iter: Number of Griffin-Lim iterations.
            
        Returns:
            Reconstructed waveform.
        """
        # Convert from log scale
        mel_spec = librosa.db_to_power(mel_spec)
        
        # Convert mel to linear spectrogram
        linear_spec = librosa.feature.inverse.mel_to_stft(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft
        )
        
        # Griffin-Lim reconstruction
        waveform = librosa.griffinlim(
            linear_spec,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        return waveform
    
    def change_speed(
        self,
        waveform: np.ndarray,
        speed_factor: float
    ) -> np.ndarray:
        """Change audio speed without changing pitch.
        
        Args:
            waveform: Input audio waveform.
            speed_factor: Speed multiplier (>1 = faster, <1 = slower).
            
        Returns:
            Speed-adjusted waveform.
        """
        if speed_factor == 1.0:
            return waveform
        
        # Use librosa's time stretch
        return librosa.effects.time_stretch(
            waveform,
            rate=speed_factor
        )
    
    def change_pitch(
        self,
        waveform: np.ndarray,
        pitch_factor: float
    ) -> np.ndarray:
        """Change audio pitch without changing speed.
        
        Args:
            waveform: Input audio waveform.
            pitch_factor: Pitch multiplier (>1 = higher, <1 = lower).
            
        Returns:
            Pitch-adjusted waveform.
        """
        if pitch_factor == 1.0:
            return waveform
        
        # Convert factor to semitones
        n_steps = 12 * np.log2(pitch_factor)
        
        return librosa.effects.pitch_shift(
            waveform,
            sr=self.sample_rate,
            n_steps=n_steps
        )
    
    def normalize_volume(
        self,
        waveform: np.ndarray,
        target_db: float = -20.0
    ) -> np.ndarray:
        """Normalize audio volume to target dB.
        
        Args:
            waveform: Input audio waveform.
            target_db: Target volume in dB.
            
        Returns:
            Normalized waveform.
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(waveform ** 2))
        
        if rms < 1e-10:
            return waveform
        
        # Calculate gain needed
        current_db = 20 * np.log10(rms)
        gain_db = target_db - current_db
        gain = 10 ** (gain_db / 20)
        
        return waveform * gain
    
    def trim_silence(
        self,
        waveform: np.ndarray,
        threshold_db: float = -40.0
    ) -> np.ndarray:
        """Trim silence from beginning and end.
        
        Args:
            waveform: Input audio waveform.
            threshold_db: Silence threshold in dB.
            
        Returns:
            Trimmed waveform.
        """
        # Find non-silent intervals
        intervals = librosa.effects.split(
            waveform,
            top_db=abs(threshold_db)
        )
        
        if len(intervals) == 0:
            return waveform
        
        # Concatenate all non-silent parts
        start = intervals[0][0]
        end = intervals[-1][1]
        
        return waveform[start:end]
    
    def add_pause(
        self,
        waveform: np.ndarray,
        pause_duration: float
    ) -> np.ndarray:
        """Add silence pause to end of audio.
        
        Args:
            waveform: Input audio waveform.
            pause_duration: Pause duration in seconds.
            
        Returns:
            Waveform with added pause.
        """
        pause_samples = int(pause_duration * self.sample_rate)
        pause = np.zeros(pause_samples)
        
        return np.concatenate([waveform, pause])
    
    def fade_in_out(
        self,
        waveform: np.ndarray,
        fade_duration: float = 0.05
    ) -> np.ndarray:
        """Apply fade in and fade out.
        
        Args:
            waveform: Input audio waveform.
            fade_duration: Fade duration in seconds.
            
        Returns:
            Faded waveform.
        """
        fade_samples = int(fade_duration * self.sample_rate)
        
        if len(waveform) <= 2 * fade_samples:
            return waveform
        
        # Create fade curves
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        # Apply fades
        waveform = waveform.copy()
        waveform[:fade_samples] *= fade_in
        waveform[-fade_samples:] *= fade_out
        
        return waveform
    
    def resample(
        self,
        waveform: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Resample audio to different sample rate.
        
        Args:
            waveform: Input audio waveform.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.
            
        Returns:
            Resampled waveform.
        """
        if orig_sr == target_sr:
            return waveform
        
        return librosa.resample(
            waveform,
            orig_sr=orig_sr,
            target_sr=target_sr
        )
    
    def get_duration(self, waveform: np.ndarray) -> float:
        """Get audio duration in seconds.
        
        Args:
            waveform: Audio waveform.
            
        Returns:
            Duration in seconds.
        """
        return len(waveform) / self.sample_rate
    
    def apply_effects(
        self,
        waveform: np.ndarray,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        trim: bool = False,
        fade: bool = True
    ) -> np.ndarray:
        """Apply multiple audio effects.
        
        Args:
            waveform: Input audio waveform.
            speed: Speed factor.
            pitch: Pitch factor.
            volume: Volume multiplier.
            trim: Whether to trim silence.
            fade: Whether to apply fade in/out.
            
        Returns:
            Processed waveform.
        """
        if trim:
            waveform = self.trim_silence(waveform)
        
        if speed != 1.0:
            waveform = self.change_speed(waveform, speed)
        
        if pitch != 1.0:
            waveform = self.change_pitch(waveform, pitch)
        
        if volume != 1.0:
            waveform = waveform * volume
            waveform = np.clip(waveform, -1.0, 1.0)
        
        if fade:
            waveform = self.fade_in_out(waveform)
        
        return waveform
