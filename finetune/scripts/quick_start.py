#!/usr/bin/env python3
"""
Kronos Finetune Quick Start

The easiest way to get started with Kronos finetuning.
Automatically detects your environment and runs the most appropriate configuration.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_gpu_availability():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def get_recommended_config():
    """Get recommended configuration based on system."""
    has_gpu = check_gpu_availability()
    
    if has_gpu:
        return "base"  # GPU available, use base config
    else:
        return "dev"   # CPU only, use fast dev config

def main():
    """Main quick start function."""
    print("🚀 Kronos Finetune Quick Start")
    print("="*50)
    
    # Check environment
    has_gpu = check_gpu_availability()
    recommended_config = get_recommended_config()
    
    print(f"GPU Available: {'✅ Yes' if has_gpu else '❌ No'}")
    print(f"Recommended Config: {recommended_config}")
    print()
    
    # Ask user for confirmation
    response = input(f"Run complete finetune with '{recommended_config}' config? [Y/n]: ").strip().lower()
    
    if response in ['', 'y', 'yes']:
        print(f"Starting complete finetune pipeline with '{recommended_config}' config...")
        print()
        
        # Run the complete pipeline
        cmd = [
            sys.executable, 
            "run_complete_finetune.py", 
            "--config", recommended_config,
            "--verbose"
        ]
        
        try:
            result = subprocess.run(cmd, check=True)
            print("\n🎉 Finetune completed successfully!")
            print("Your models are ready to use!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Finetune failed with exit code {e.returncode}")
            print("Check the logs for details.")
        except KeyboardInterrupt:
            print("\n⏹️  Finetune interrupted by user")
    else:
        print("For manual control, use:")
        print(f"  python run_complete_finetune.py --config {recommended_config}")
        print("Or see docs/FINETUNE_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()