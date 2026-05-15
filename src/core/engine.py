"""Core TTS engine implementation."""

import re
import time
from pathlib import Path
from typing import Optional, Union, List, Tuple
import numpy as np
import soundfile as sf
import onnxruntime as ort

from .config import TTSConfig
from .text_processor import TextProcessor
from .audio_processor import AudioProcessor


class TTSEngine:
    """Main TTS engine for text-to-speech synthesis."""
    
    # Language codes mapping
    LANG_MAP = {
        "zh": "chinese",
        "en": "english",
        "ja": "japanese",
        "ko": "korean",
        "fr": "french",
        "de": "german",
        "es": "spanish",
        "it": "italian",
        "pt": "portuguese",
        "ru": "russian",
    }
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """Initialize TTS engine.
        
        Args:
            config: TTS configuration. Uses default if not provided.
        """
        self.config = config or TTSConfig()
        self.session: Optional[ort.InferenceSession] = None
        self.text_processor = TextProcessor(self.config.language)
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.sample_rate,
            hop_length=self.config.hop_length
        )
        
        # Load model if path exists
        if Path(self.config.model_path).exists():
            self.load_model(self.config.model_path)
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load ONNX model.
        
        Args:
            model_path: Path to the ONNX model file.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Configure ONNX Runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.use_gpu else ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config.num_threads
        sess_options.inter_op_num_threads = self.config.num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )
            print(f"✓ Model loaded: {model_path}")
            print(f"  Providers: {self.session.get_providers()}")
            print(f"  Input names: {[i.name for i in self.session.get_inputs()]}")
            print(f"  Output names: {[o.name for o in self.session.get_outputs()]}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[Union[str, Path]] = None,
        speed: Optional[float] = None,
        pitch: Optional[float] = None,
        language: Optional[str] = None
    ) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize.
            output_path: Optional path to save audio file.
            speed: Speech speed multiplier (0.5-2.0).
            pitch: Pitch multiplier (0.5-2.0).
            language: Language code override.
            
        Returns:
            Audio waveform as numpy array.
            
        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If text is empty or invalid.
        """
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Use config defaults if not specified
        speed = speed or self.config.speed
        pitch = pitch or self.config.pitch
        language = language or self.config.language
        
        # Check text length
        if len(text) > self.config.max_text_length:
            raise ValueError(f"Text too long ({len(text)} > {self.config.max_text_length})")
        
        # Process text
        phonemes = self.text_processor.text_to_phonemes(text, language)
        phoneme_ids = self.text_processor.phonemes_to_ids(phonemes)
        
        # Prepare input
        phoneme_array = np.array([phoneme_ids], dtype=np.int64)
        
        # Create speaker embedding (default voice)
        speaker_embedding = np.random.randn(1, 256).astype(np.float32) * 0.1
        
        # Run inference
        start_time = time.time()
        
        try:
            outputs = self.session.run(
                None,
                {
                    "input_ids": phoneme_array,
                    "speaker_embeddings": speaker_embedding,
                }
            )
            
            # Extract audio waveform
            waveform = outputs[0].squeeze()
            
            # Apply speed and pitch adjustments
            if speed != 1.0:
                waveform = self.audio_processor.change_speed(waveform, speed)
            
            if pitch != 1.0:
                waveform = self.audio_processor.change_pitch(waveform, pitch)
            
            # Normalize volume
            waveform = waveform * self.config.volume
            waveform = np.clip(waveform, -1.0, 1.0)
            
            inference_time = time.time() - start_time
            audio_duration = len(waveform) / self.config.sample_rate
            rtf = inference_time / audio_duration if audio_duration > 0 else 0
            
            print(f"✓ Synthesis complete: {len(text)} chars -> {audio_duration:.2f}s audio")
            print(f"  Inference time: {inference_time:.3f}s (RTF: {rtf:.3f})")
            
            # Save to file if path provided
            if output_path:
                self.save_audio(waveform, output_path)
            
            return waveform
            
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")
    
    def synthesize_long_text(
        self,
        text: str,
        output_path: Union[str, Path],
        max_chars: int = 200,
        pause_between: float = 0.3
    ) -> Path:
        """Synthesize long text by splitting into chunks.
        
        Args:
            text: Long text to synthesize.
            output_path: Path to save combined audio.
            max_chars: Maximum characters per chunk.
            pause_between: Pause duration between chunks in seconds.
            
        Returns:
            Path to the output audio file.
        """
        # Split text into sentences
        sentences = self._split_text(text, max_chars)
        
        print(f"Synthesizing {len(sentences)} chunks...")
        
        waveforms = []
        for i, sentence in enumerate(sentences, 1):
            print(f"  [{i}/{len(sentences)}] {sentence[:50]}...")
            waveform = self.synthesize(sentence)
            waveforms.append(waveform)
            
            # Add pause between chunks
            if i < len(sentences):
                pause_samples = int(pause_between * self.config.sample_rate)
                waveforms.append(np.zeros(pause_samples))
        
        # Combine all waveforms
        combined = np.concatenate(waveforms)
        self.save_audio(combined, output_path)
        
        return Path(output_path)
    
    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks respecting sentence boundaries."""
        # Split by sentence endings
        sentences = re.split(r'([。！？.!?])', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def save_audio(
        self,
        waveform: np.ndarray,
        output_path: Union[str, Path],
        format: Optional[str] = None
    ) -> Path:
        """Save audio waveform to file.
        
        Args:
            waveform: Audio waveform array.
            output_path: Output file path.
            format: Audio format override.
            
        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        format = format or self.config.output_format
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save audio
        sf.write(str(output_path), waveform, self.config.sample_rate)
        
        print(f"✓ Audio saved: {output_path}")
        return output_path
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.LANG_MAP.keys())
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.session is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "providers": self.session.get_providers(),
            "inputs": [
                {"name": i.name, "shape": i.shape, "type": i.type}
                for i in self.session.get_inputs()
            ],
            "outputs": [
                {"name": o.name, "shape": o.shape, "type": o.type}
                for o in self.session.get_outputs()
            ],
        }
