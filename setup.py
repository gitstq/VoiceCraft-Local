"""Setup script for VoiceCraft-Local."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="voicecraft-local",
    version="1.0.0",
    author="VoiceCraft Team",
    author_email="contact@voicecraft.dev",
    description="Lightning-Fast, On-Device, Multilingual TTS powered by ONNX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/voicecraft-local",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gui": [
            "PyQt6>=6.5.0",
        ],
        "audio": [
            "sounddevice>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "voicecraft=cli.main:cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
