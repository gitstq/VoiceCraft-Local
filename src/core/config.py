"""Configuration management for VoiceCraft-Local."""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class TTSConfig:
    """TTS Engine configuration."""
    
    # Model settings
    model_name: str = "voicecraft-base"
    model_path: Optional[str] = None
    device: str = "cpu"  # cpu, cuda
    
    # Audio settings
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    
    # Synthesis settings
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    
    # Language settings
    language: str = "zh"  # zh, en, ja, ko
    
    # Output settings
    output_format: str = "wav"  # wav, mp3, ogg
    output_dir: str = "./output"
    
    # Advanced settings
    use_gpu: bool = False
    num_threads: int = 4
    max_text_length: int = 5000
    
    # Voice cloning settings
    clone_voice: bool = False
    reference_audio: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        if self.model_path is None:
            self.model_path = self._get_default_model_path()
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate settings
        if self.speed < 0.5 or self.speed > 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")
        
        if self.pitch < 0.5 or self.pitch > 2.0:
            raise ValueError("Pitch must be between 0.5 and 2.0")
    
    def _get_default_model_path(self) -> str:
        """Get default model path based on model name."""
        base_dir = Path(__file__).parent.parent.parent
        return str(base_dir / "models" / f"{self.model_name}.onnx")
    
    @classmethod
    def from_yaml(cls, path: str) -> "TTSConfig":
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        data = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "speed": self.speed,
            "pitch": self.pitch,
            "volume": self.volume,
            "language": self.language,
            "output_format": self.output_format,
            "output_dir": self.output_dir,
            "use_gpu": self.use_gpu,
            "num_threads": self.num_threads,
            "max_text_length": self.max_text_length,
            "clone_voice": self.clone_voice,
            "reference_audio": self.reference_audio,
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_mels": self.n_mels,
            "speed": self.speed,
            "pitch": self.pitch,
            "volume": self.volume,
            "language": self.language,
            "output_format": self.output_format,
            "output_dir": self.output_dir,
            "use_gpu": self.use_gpu,
            "num_threads": self.num_threads,
            "max_text_length": self.max_text_length,
            "clone_voice": self.clone_voice,
            "reference_audio": self.reference_audio,
        }
