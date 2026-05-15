#!/usr/bin/env python3
"""Build script for VoiceCraft-Local.

This script handles:
- Cross-platform executable building
- Dependency installation
- Package creation
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path


# Build configuration
APP_NAME = "VoiceCraft-Local"
APP_VERSION = "1.0.0"
ENTRY_POINT = "src/cli/main.py"
ICON_PATH = "assets/icon.ico"  # Windows icon


def run_command(cmd, cwd=None):
    """Run a shell command."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    
    dirs_to_remove = ['build', 'dist', '__pycache__', '.pytest_cache']
    for dir_name in dirs_to_remove:
        path = Path(dir_name)
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed {dir_name}")
    
    # Remove .pyc files
    for pyc_file in Path('.').rglob('*.pyc'):
        pyc_file.unlink()
    
    # Remove .spec files
    for spec_file in Path('.').glob('*.spec'):
        spec_file.unlink()
    
    print("✓ Clean complete")


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    if not run_command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt']):
        print("✗ Failed to install dependencies")
        return False
    
    print("✓ Dependencies installed")
    return True


def build_windows():
    """Build Windows executable."""
    print("Building Windows executable...")
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', f'{APP_NAME}-{APP_VERSION}-Windows',
        '--onefile',
        '--console',
        '--clean',
    ]
    
    if Path(ICON_PATH).exists():
        cmd.extend(['--icon', ICON_PATH])
    
    cmd.append(ENTRY_POINT)
    
    if not run_command(cmd):
        print("✗ Windows build failed")
        return False
    
    print("✓ Windows build complete")
    return True


def build_linux():
    """Build Linux executable."""
    print("Building Linux executable...")
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', f'{APP_NAME}-{APP_VERSION}-Linux',
        '--onefile',
        '--console',
        '--clean',
        ENTRY_POINT
    ]
    
    if not run_command(cmd):
        print("✗ Linux build failed")
        return False
    
    print("✓ Linux build complete")
    return True


def build_macos():
    """Build macOS executable."""
    print("Building macOS executable...")
    
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', f'{APP_NAME}-{APP_VERSION}-macOS',
        '--onefile',
        '--console',
        '--clean',
        '--target-architecture', 'universal2',
        ENTRY_POINT
    ]
    
    if not run_command(cmd):
        print("✗ macOS build failed")
        return False
    
    print("✓ macOS build complete")
    return True


def build_all():
    """Build for all platforms (from current platform)."""
    current_platform = platform.system()
    
    print(f"Building on {current_platform}...")
    
    if current_platform == 'Windows':
        return build_windows()
    elif current_platform == 'Linux':
        return build_linux()
    elif current_platform == 'Darwin':
        return build_macos()
    else:
        print(f"Unsupported platform: {current_platform}")
        return False


def run_tests():
    """Run test suite."""
    print("Running tests...")
    
    if not run_command([sys.executable, '-m', 'pytest', 'tests/', '-v']):
        print("✗ Tests failed")
        return False
    
    print("✓ All tests passed")
    return True


def create_package():
    """Create distribution package."""
    print("Creating distribution package...")
    
    # Create package directory
    pkg_dir = Path(f'dist/{APP_NAME}-{APP_VERSION}')
    pkg_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    files_to_copy = [
        'README.md',
        'LICENSE',
        'requirements.txt',
    ]
    
    for file in files_to_copy:
        src = Path(file)
        if src.exists():
            shutil.copy(src, pkg_dir)
    
    # Copy source
    if Path('src').exists():
        shutil.copytree('src', pkg_dir / 'src', dirs_exist_ok=True)
    
    # Create archive
    archive_name = f'{APP_NAME}-{APP_VERSION}-{platform.system()}'
    shutil.make_archive(f'dist/{archive_name}', 'zip', pkg_dir)
    
    print(f"✓ Package created: dist/{archive_name}.zip")
    return True


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build VoiceCraft-Local')
    parser.add_argument('--clean', action='store_true', help='Clean build artifacts')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    parser.add_argument('--test', action='store_true', help='Run tests')
    parser.add_argument('--build', action='store_true', help='Build executable')
    parser.add_argument('--package', action='store_true', help='Create package')
    parser.add_argument('--all', action='store_true', help='Run full build pipeline')
    
    args = parser.parse_args()
    
    # Default to --all if no arguments
    if not any([args.clean, args.install, args.test, args.build, args.package]):
        args.all = True
    
    if args.all:
        args.clean = True
        args.install = True
        args.test = True
        args.build = True
        args.package = True
    
    # Execute requested actions
    if args.clean:
        clean_build()
    
    if args.install:
        if not install_dependencies():
            sys.exit(1)
    
    if args.test:
        if not run_tests():
            sys.exit(1)
    
    if args.build:
        if not build_all():
            sys.exit(1)
    
    if args.package:
        if not create_package():
            sys.exit(1)
    
    print("\n✓ Build process complete!")


if __name__ == '__main__':
    main()
