"""Main CLI entry point for VoiceCraft-Local."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.engine import TTSEngine
from core.config import TTSConfig
from core.model_manager import ModelManager


console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="voicecraft")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """🎙️ VoiceCraft-Local: Lightning-Fast, On-Device, Multilingual TTS
    
    A local text-to-speech engine powered by ONNX Runtime.
    No cloud required - your voice, your data, your device.
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('text')
@click.option('--output', '-o', type=click.Path(), help='Output audio file path')
@click.option('--model', '-m', default='voicecraft-base', help='Model name to use')
@click.option('--language', '-l', default='zh', help='Language code (zh/en/ja/ko)')
@click.option('--speed', '-s', default=1.0, type=float, help='Speech speed (0.5-2.0)')
@click.option('--pitch', '-p', default=1.0, type=float, help='Pitch adjustment (0.5-2.0)')
@click.option('--play', is_flag=True, help='Play audio after synthesis')
@click.pass_context
def speak(ctx, text, output, model, language, speed, pitch, play):
    """Synthesize speech from text."""
    verbose = ctx.obj.get('verbose', False)
    
    # Print banner
    console.print(Panel.fit(
        "[bold cyan]🎙️ VoiceCraft-Local TTS Engine[/bold cyan]\n"
        "[dim]Lightning-Fast, On-Device, Multilingual Text-to-Speech[/dim]",
        border_style="cyan"
    ))
    
    # Validate inputs
    if speed < 0.5 or speed > 2.0:
        console.print("[red]Error: Speed must be between 0.5 and 2.0[/red]")
        sys.exit(1)
    
    if pitch < 0.5 or pitch > 2.0:
        console.print("[red]Error: Pitch must be between 0.5 and 2.0[/red]")
        sys.exit(1)
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Check/download model
    if not model_manager.is_model_downloaded(model):
        console.print(f"[yellow]Model '{model}' not found. Downloading...[/yellow]")
        try:
            model_manager.download_model(model)
        except Exception as e:
            console.print(f"[red]Failed to download model: {e}[/red]")
            sys.exit(1)
    
    # Set up configuration
    config = TTSConfig(
        model_name=model,
        model_path=str(model_manager.get_model_path(model)),
        language=language,
        speed=speed,
        pitch=pitch,
    )
    
    # Set default output path
    if output is None:
        output = f"output_{click.utils.generate_filename()}.wav"
    
    # Initialize engine
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading TTS engine...", total=None)
        
        try:
            engine = TTSEngine(config)
            progress.update(task, description="Engine ready!")
        except Exception as e:
            console.print(f"[red]Failed to initialize engine: {e}[/red]")
            sys.exit(1)
    
    # Synthesize
    console.print(f"\n[bold]Input:[/bold] {text}")
    console.print(f"[bold]Language:[/bold] {language}")
    console.print(f"[bold]Speed:[/bold] {speed}x")
    console.print(f"[bold]Pitch:[/bold] {pitch}x")
    console.print()
    
    try:
        waveform = engine.synthesize(text, output_path=output)
        console.print(f"\n[green]✓ Audio saved to:[/green] {output}")
        
        if play:
            console.print("[yellow]Playing audio...[/yellow]")
            try:
                import sounddevice as sd
                sd.play(waveform, config.sample_rate)
                sd.wait()
            except ImportError:
                console.print("[dim]Install sounddevice to enable playback: pip install sounddevice[/dim]")
                
    except Exception as e:
        console.print(f"[red]Synthesis failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output audio file path')
@click.option('--model', '-m', default='voicecraft-base', help='Model name to use')
@click.option('--language', '-l', help='Language code (auto-detect if not specified)')
@click.option('--speed', '-s', default=1.0, type=float, help='Speech speed (0.5-2.0)')
@click.pass_context
def file(ctx, input_file, output, model, language, speed):
    """Synthesize speech from a text file."""
    verbose = ctx.obj.get('verbose', False)
    
    # Read input file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        console.print(f"[red]Failed to read file: {e}[/red]")
        sys.exit(1)
    
    # Set default output
    if output is None:
        input_path = Path(input_file)
        output = str(input_path.with_suffix('.wav'))
    
    # Call speak command
    ctx.invoke(
        speak,
        text=text,
        output=output,
        model=model,
        language=language or 'zh',
        speed=speed,
        pitch=1.0,
        play=False
    )


@cli.command()
def models():
    """List available TTS models."""
    model_manager = ModelManager()
    available_models = model_manager.list_available_models()
    
    table = Table(title="Available TTS Models")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Size", style="green")
    table.add_column("Languages", style="yellow")
    table.add_column("Description", style="white")
    table.add_column("Status", style="magenta")
    
    for model in available_models:
        status = "✓ Downloaded" if model['downloaded'] else "Not downloaded"
        status_style = "green" if model['downloaded'] else "dim"
        
        table.add_row(
            model['name'],
            model['size'],
            ", ".join(model['languages']),
            model['description'],
            f"[{status_style}]{status}[/{status_style}]"
        )
    
    console.print(table)
    
    # Show cache info
    cache_size = model_manager.get_cache_size()
    console.print(f"\n[dim]Cache size: {cache_size / 1024 / 1024:.1f} MB[/dim]")


@cli.command()
@click.argument('model_name')
def download(model_name):
    """Download a TTS model."""
    model_manager = ModelManager()
    
    if model_name not in model_manager.MODEL_REGISTRY:
        available = ", ".join(model_manager.MODEL_REGISTRY.keys())
        console.print(f"[red]Unknown model: {model_name}[/red]")
        console.print(f"[dim]Available models: {available}[/dim]")
        sys.exit(1)
    
    if model_manager.is_model_downloaded(model_name):
        console.print(f"[yellow]Model '{model_name}' is already downloaded.[/yellow]")
        if not click.confirm("Re-download?"):
            return
    
    try:
        model_manager.download_model(model_name, force=True)
        console.print(f"[green]✓ Model '{model_name}' downloaded successfully![/green]")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear the model cache?')
def clean():
    """Clear downloaded model cache."""
    model_manager = ModelManager()
    count = model_manager.clear_cache()
    console.print(f"[green]✓ Cleared {count} models from cache.[/green]")


@cli.command()
def config():
    """Show current configuration."""
    cfg = TTSConfig()
    
    table = Table(title="Default Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    
    for key, value in cfg.to_dict().items():
        table.add_row(key, str(value))
    
    console.print(table)


@cli.command()
def languages():
    """List supported languages."""
    engine = TTSEngine()
    languages = engine.get_supported_languages()
    
    table = Table(title="Supported Languages")
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="white")
    
    lang_names = {
        'zh': 'Chinese (中文)',
        'en': 'English',
        'ja': 'Japanese (日本語)',
        'ko': 'Korean (한국어)',
        'fr': 'French (Français)',
        'de': 'German (Deutsch)',
        'es': 'Spanish (Español)',
        'it': 'Italian (Italiano)',
        'pt': 'Portuguese (Português)',
        'ru': 'Russian (Русский)',
    }
    
    for code in languages:
        table.add_row(code, lang_names.get(code, code))
    
    console.print(table)


if __name__ == '__main__':
    cli()
