#!/usr/bin/env python3
"""
Complete Finetune Pipeline Script

This script runs the complete end-to-end finetune pipeline:
1. Data preparation and validation
2. Tokenizer training
3. Predictor training  
4. Model validation and performance testing
"""

import argparse
import sys
import time
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
    parser = argparse.ArgumentParser(description="Run Complete Kronos Finetune Pipeline")
    
    parser.add_argument(
        '--config',
        choices=['dev', 'base', 'production'],
        default='base',
        help='Configuration to use (default: base)'
    )
    
    parser.add_argument(
        '--skip-data-prep',
        action='store_true',
        help='Skip data preparation (use existing data)'
    )
    
    parser.add_argument(
        '--skip-tokenizer',
        action='store_true',
        help='Skip tokenizer training (use existing or pretrained)'
    )
    
    parser.add_argument(
        '--skip-predictor',
        action='store_true',
        help='Skip predictor training'
    )
    
    parser.add_argument(
        '--device',
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
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


def prepare_data(config, args):
    """Prepare training data."""
    logger.info("=" * 60)
    logger.info("🔄 STEP 1: DATA PREPARATION")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("DRY RUN: Would prepare training data")
        return True
    
    data_manager = DataManager(config)
    
    if args.skip_data_prep:
        # Check if data exists
        health = data_manager.check_data_health()
        if all(health['datasets_exist'].values()):
            logger.info("✅ Using existing training data")
            return True
        else:
            logger.info("⚠️  Existing data incomplete, preparing new data...")
    
    # Prepare data
    start_time = time.time()
    success = data_manager.create_and_save_mock_data()
    prep_time = time.time() - start_time
    
    if success:
        logger.info(f"✅ Data preparation completed in {prep_time:.2f}s")
        
        # Show data health
        health = data_manager.check_data_health()
        logger.info(f"   📊 Dataset health:")
        for dataset, exists in health['datasets_exist'].items():
            logger.info(f"      {dataset}: {'✅' if exists else '❌'}")
        
        if 'data_quality' in health and health['data_quality']:
            quality = health['data_quality']
            logger.info(f"   📈 Data quality:")
            logger.info(f"      Symbols: {quality.get('total_symbols', 'N/A')}")
            logger.info(f"      Missing rate: {quality.get('missing_rate', 0):.4%}")
            logger.info(f"      Date range: {quality.get('date_range', {}).get('start', 'N/A')} to {quality.get('date_range', {}).get('end', 'N/A')}")
        
        return True
    else:
        logger.error("❌ Data preparation failed")
        return False


def train_tokenizer(config, args):
    """Train tokenizer model."""
    logger.info("=" * 60)
    logger.info("🤖 STEP 2: TOKENIZER TRAINING")
    logger.info("=" * 60)
    
    if args.skip_tokenizer:
        # Check if finetuned tokenizer exists
        finetuned_path = Path(config.finetuned_tokenizer_path)
        if finetuned_path.exists():
            logger.info("✅ Using existing finetuned tokenizer")
            return True
        else:
            logger.info("⚠️  No existing finetuned tokenizer, will use pretrained")
            return True
    
    if args.dry_run:
        logger.info("DRY RUN: Would train tokenizer")
        return True
    
    # Train tokenizer
    start_time = time.time()
    trainer = UnifiedTrainer(config, model_type="tokenizer")
    results = trainer.train()
    train_time = time.time() - start_time
    
    if results['success']:
        logger.info(f"✅ Tokenizer training completed in {train_time:.2f}s")
        logger.info(f"   📊 Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"   📂 Model saved to: {config.save_path}/{config.tokenizer_save_folder_name}")
        return True
    else:
        logger.error(f"❌ Tokenizer training failed: {results.get('error', 'Unknown error')}")
        return False


def train_predictor(config, args):
    """Train predictor model."""
    logger.info("=" * 60)
    logger.info("🎯 STEP 3: PREDICTOR TRAINING")
    logger.info("=" * 60)
    
    if args.skip_predictor:
        logger.info("⏭️  Skipping predictor training")
        return True
    
    if args.dry_run:
        logger.info("DRY RUN: Would train predictor")
        return True
    
    # Check if finetuned tokenizer exists and update config
    finetuned_tokenizer = Path(config.finetuned_tokenizer_path)
    if finetuned_tokenizer.exists():
        config.pretrained_tokenizer_path = str(finetuned_tokenizer)
        logger.info(f"🔗 Using finetuned tokenizer: {finetuned_tokenizer}")
    else:
        logger.info(f"🔗 Using pretrained tokenizer: {config.pretrained_tokenizer_path}")
    
    # Train predictor
    start_time = time.time()
    trainer = UnifiedTrainer(config, model_type="predictor")
    results = trainer.train()
    train_time = time.time() - start_time
    
    if results['success']:
        logger.info(f"✅ Predictor training completed in {train_time:.2f}s")
        logger.info(f"   📊 Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"   📂 Model saved to: {config.save_path}/{config.predictor_save_folder_name}")
        return True
    else:
        logger.error(f"❌ Predictor training failed: {results.get('error', 'Unknown error')}")
        return False


def validate_pipeline(config, args):
    """Validate the complete pipeline."""
    logger.info("=" * 60)
    logger.info("✅ STEP 4: PIPELINE VALIDATION")
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("DRY RUN: Would validate pipeline")
        return True
    
    validation_results = {
        'data_ready': False,
        'tokenizer_ready': False,
        'predictor_ready': False
    }
    
    # Check data
    data_manager = DataManager(config)
    health = data_manager.check_data_health()
    validation_results['data_ready'] = all(health['datasets_exist'].values())
    
    # Check tokenizer
    tokenizer_path = Path(config.finetuned_tokenizer_path)
    validation_results['tokenizer_ready'] = tokenizer_path.exists()
    
    # Check predictor
    predictor_path = Path(config.finetuned_predictor_path)
    validation_results['predictor_ready'] = predictor_path.exists()
    
    # Report results
    logger.info("📋 Pipeline Validation Results:")
    for component, ready in validation_results.items():
        status = "✅ READY" if ready else "❌ NOT READY"
        logger.info(f"   {component.replace('_', ' ').title()}: {status}")
    
    all_ready = all(validation_results.values())
    if all_ready:
        logger.info("🎉 Complete pipeline validation PASSED!")
    else:
        logger.warning("⚠️  Pipeline validation found missing components")
    
    return all_ready


def print_final_summary(config, total_time, success, args):
    """Print final pipeline summary."""
    logger.info("=" * 60)
    logger.info("📊 PIPELINE SUMMARY")
    logger.info("=" * 60)
    
    status = "✅ SUCCESS" if success else "❌ FAILED"
    logger.info(f"Status: {status}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    if success:
        logger.info(f"\n📂 Output Locations:")
        logger.info(f"   Data: {config.dataset_path}")
        logger.info(f"   Models: {config.save_path}")
        logger.info(f"   Logs: {Path(config.outputs_dir) / 'logs'}")
        
        logger.info(f"\n🚀 Next Steps:")
        logger.info(f"   1. Test your finetuned models with prediction examples")
        logger.info(f"   2. Run backtesting to evaluate performance")
        logger.info(f"   3. Consider longer training for production use")
    else:
        logger.info(f"\n🔧 Troubleshooting:")
        logger.info(f"   1. Check logs for detailed error messages")
        logger.info(f"   2. Verify GPU availability and memory")
        logger.info(f"   3. Try with smaller batch size or dev config")


def main():
    """Main pipeline function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    start_time = time.time()
    
    logger.info("🚀 KRONOS COMPLETE FINETUNE PIPELINE")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {args.device}")
    
    if args.dry_run:
        logger.info("🧪 DRY RUN MODE - No actual training will occur")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        success = True
        
        # Step 1: Data preparation
        if not prepare_data(config, args):
            success = False
        
        # Step 2: Tokenizer training
        if success and not train_tokenizer(config, args):
            success = False
        
        # Step 3: Predictor training
        if success and not train_predictor(config, args):
            success = False
        
        # Step 4: Validation
        if success:
            validate_pipeline(config, args)
        
        # Final summary
        total_time = time.time() - start_time
        print_final_summary(config, total_time, success, args)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)