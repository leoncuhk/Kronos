#!/usr/bin/env python3
"""
Clean Predictor Training Script

This script provides a simple, unified interface for training the Kronos predictor.
It replaces the multiple scattered predictor training implementations.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from finetune.config import BaseConfig, ProductionConfig, DevelopmentConfig
from finetune.training import UnifiedTrainer
from finetune.data import DataManager

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Kronos Predictor")
    
    parser.add_argument(
        '--config', 
        choices=['dev', 'base', 'production'], 
        default='base',
        help='Configuration to use (default: base)'
    )
    
    parser.add_argument(
        '--device',
        default='auto',
        help='Device to use (auto, cpu, cuda, cuda:0, etc.)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    
    parser.add_argument(
        '--tokenizer-path',
        help='Path to finetuned tokenizer (if not using default)'
    )
    
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Prepare training data before training'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving checkpoints (for testing)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_config(config_type: str):
    """Load configuration based on type."""
    if config_type == 'dev':
        return DevelopmentConfig()
    elif config_type == 'production':
        return ProductionConfig()
    else:
        return BaseConfig()


def check_tokenizer_availability(config, args):
    """Check if finetuned tokenizer is available."""
    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path)
        if tokenizer_path.exists():
            config.pretrained_tokenizer_path = str(tokenizer_path)
            logger.info(f"Using custom tokenizer: {tokenizer_path}")
            return True
        else:
            logger.error(f"Custom tokenizer path not found: {tokenizer_path}")
            return False
    
    # Check for finetuned tokenizer in default location
    finetuned_path = Path(config.finetuned_tokenizer_path)
    if finetuned_path.exists():
        config.pretrained_tokenizer_path = str(finetuned_path)
        logger.info(f"Using finetuned tokenizer: {finetuned_path}")
        return True
    
    # Fall back to pretrained tokenizer
    logger.info(f"Using pretrained tokenizer: {config.pretrained_tokenizer_path}")
    logger.warning("Consider training the tokenizer first for better performance")
    return True


def prepare_data_if_needed(config, args):
    """Prepare training data if requested or if data doesn't exist."""
    data_manager = DataManager(config)
    
    if args.prepare_data:
        logger.info("Preparing training data...")
        success = data_manager.create_and_save_mock_data()
        if not success:
            logger.error("Failed to prepare data")
            return False
    else:
        # Check if data exists
        health = data_manager.check_data_health()
        if not all(health['datasets_exist'].values()):
            logger.info("Training data not found. Preparing data...")
            success = data_manager.create_and_save_mock_data()
            if not success:
                logger.error("Failed to prepare data")
                return False
        else:
            logger.info("Using existing training data")
    
    return True


def main():
    """Main training function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("🚀 Starting Kronos Predictor Training")
    logger.info(f"Configuration: {args.config}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.epochs:
            config.epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.device != 'auto':
            os.environ['CUDA_VISIBLE_DEVICES'] = args.device.replace('cuda:', '')
        
        logger.info(f"Training for {config.epochs} epochs with batch size {config.batch_size}")
        
        # Check tokenizer availability
        if not check_tokenizer_availability(config, args):
            return 1
        
        # Prepare data
        if not prepare_data_if_needed(config, args):
            return 1
        
        # Initialize trainer
        trainer = UnifiedTrainer(config, model_type="predictor")
        
        # Disable saving if requested
        if args.no_save:
            trainer.save_checkpoint = lambda *args, **kwargs: logger.info("Checkpoint saving disabled")
        
        # Run training
        results = trainer.train()
        
        if results['success']:
            logger.info("✅ Predictor training completed successfully!")
            logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
            logger.info(f"Training epochs: {results['final_epoch']}")
            
            # Print training summary
            print("\n📊 Training Summary:")
            print("-" * 50)
            print(f"Configuration: {args.config}")
            print(f"Epochs: {results['final_epoch']}")
            print(f"Best validation loss: {results['best_val_loss']:.4f}")
            print(f"Tokenizer: {config.pretrained_tokenizer_path}")
            print(f"Model saved to: {config.save_path}/{config.predictor_save_folder_name}")
            
            return 0
        else:
            logger.error(f"❌ Training failed: {results.get('error', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)