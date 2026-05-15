"""Model management for VoiceCraft-Local."""

import os
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm
import platformdirs


class ModelManager:
    """Manages TTS model downloads and caching."""
    
    # Model registry with download URLs and checksums
    MODEL_REGISTRY = {
        "voicecraft-base": {
            "url": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/model.onnx",
            "checksum": "placeholder",
            "size": "120MB",
            "languages": ["zh", "en", "ja", "ko"],
            "description": "Base multilingual TTS model",
        },
        "voicecraft-fast": {
            "url": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/model_fast.onnx",
            "checksum": "placeholder",
            "size": "80MB",
            "languages": ["zh", "en"],
            "description": "Fast inference model for Chinese and English",
        },
        "voicecraft-clone": {
            "url": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/model_clone.onnx",
            "checksum": "placeholder",
            "size": "150MB",
            "languages": ["zh", "en", "ja", "ko"],
            "description": "Voice cloning capable model",
        },
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize model manager.
        
        Args:
            cache_dir: Directory to cache downloaded models.
                      Defaults to platform-specific cache directory.
        """
        if cache_dir is None:
            cache_dir = platformdirs.user_cache_dir("voicecraft-local", "VoiceCraft")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_models(self) -> List[Dict]:
        """List all available models with their info."""
        models = []
        for name, info in self.MODEL_REGISTRY.items():
            model_info = {
                "name": name,
                "size": info["size"],
                "languages": info["languages"],
                "description": info["description"],
                "downloaded": self.is_model_downloaded(name),
            }
            models.append(model_info)
        return models
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        model_path = self.cache_dir / f"{model_name}.onnx"
        return model_path.exists()
    
    def get_model_path(self, model_name: str) -> Path:
        """Get the local path for a model."""
        return self.cache_dir / f"{model_name}.onnx"
    
    def download_model(
        self,
        model_name: str,
        force: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """Download a model from the registry.
        
        Args:
            model_name: Name of the model to download.
            force: Whether to re-download if already exists.
            progress_callback: Optional callback for progress updates.
            
        Returns:
            Path to the downloaded model file.
            
        Raises:
            ValueError: If model_name is not in registry.
            RuntimeError: If download fails.
        """
        if model_name not in self.MODEL_REGISTRY:
            available = ", ".join(self.MODEL_REGISTRY.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        
        model_path = self.get_model_path(model_name)
        
        # Check if already downloaded
        if model_path.exists() and not force:
            print(f"Model '{model_name}' already exists at {model_path}")
            return model_path
        
        # Download model
        model_info = self.MODEL_REGISTRY[model_name]
        url = model_info["url"]
        
        print(f"Downloading model '{model_name}'...")
        print(f"URL: {url}")
        print(f"Destination: {model_path}")
        
        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {model_name}"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                                if progress_callback:
                                    progress_callback(len(chunk))
                else:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            print(f"✓ Model '{model_name}' downloaded successfully!")
            return model_path
            
        except requests.RequestException as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(f"Failed to download model: {e}")
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model.
        
        Args:
            model_name: Name of the model to delete.
            
        Returns:
            True if model was deleted, False if it didn't exist.
        """
        model_path = self.get_model_path(model_name)
        if model_path.exists():
            model_path.unlink()
            print(f"✓ Model '{model_name}' deleted.")
            return True
        return False
    
    def get_cache_size(self) -> int:
        """Get total size of cached models in bytes."""
        total_size = 0
        for model_file in self.cache_dir.glob("*.onnx"):
            total_size += model_file.stat().st_size
        return total_size
    
    def clear_cache(self) -> int:
        """Clear all cached models.
        
        Returns:
            Number of models deleted.
        """
        count = 0
        for model_file in self.cache_dir.glob("*.onnx"):
            model_file.unlink()
            count += 1
        print(f"✓ Cleared {count} models from cache.")
        return count
    
    def verify_checksum(self, model_name: str) -> bool:
        """Verify model file checksum.
        
        Args:
            model_name: Name of the model to verify.
            
        Returns:
            True if checksum matches, False otherwise.
        """
        if model_name not in self.MODEL_REGISTRY:
            return False
        
        expected_checksum = self.MODEL_REGISTRY[model_name]["checksum"]
        if expected_checksum == "placeholder":
            # Skip verification for placeholder checksums
            return True
        
        model_path = self.get_model_path(model_name)
        if not model_path.exists():
            return False
        
        # Calculate SHA256 checksum
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest() == expected_checksum
