"""
VoiceCraft-Local: Lightning-Fast, On-Device, Multilingual TTS
A local text-to-speech engine powered by ONNX Runtime
"""

__version__ = "1.0.0"
__author__ = "VoiceCraft Team"
__license__ = "MIT"

from .core.engine import TTSEngine
from .core.config import TTSConfig

__all__ = ["TTSEngine", "TTSConfig"]
