#!/usr/bin/env python3
"""
极简微调脚本 - 专为"微调完有效就应用"场景设计

使用方法:
  python simple_train.py                    # 标准微调，约30分钟
  python simple_train.py --quick            # 快速测试，约3分钟  
  python simple_train.py --intensive        # 强化训练，约2小时
  python simple_train.py --epochs 8         # 自定义轮数
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from finetune.config import SimpleConfig, get_quick_config, get_standard_config, get_intensive_config
from finetune.training import UnifiedTrainer
from finetune.data import DataManager

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Kronos极简微调")
    
    # 预设模式
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick', action='store_true', 
                           help='快速模式: 3分钟测试代码')
    mode_group.add_argument('--intensive', action='store_true',
                           help='强化模式: 2小时追求最佳效果')
    
    # 自定义参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--skip-tokenizer', action='store_true',
                       help='跳过tokenizer训练')
    parser.add_argument('--skip-predictor', action='store_true', 
                       help='跳过predictor训练')
    parser.add_argument('--prepare-data', action='store_true',
                       help='重新准备训练数据')
    
    return parser.parse_args()


def get_config(args):
    """根据参数获取配置"""
    kwargs = {}
    if args.epochs:
        kwargs['epochs'] = args.epochs
    if args.batch_size:
        kwargs['batch_size'] = args.batch_size
    
    if args.quick:
        config = get_quick_config(**kwargs)
        logger.info("🚀 使用快速模式 - 适合代码测试")
    elif args.intensive:
        config = get_intensive_config(**kwargs)
        logger.info("💪 使用强化模式 - 追求最佳效果")
    else:
        config = get_standard_config(**kwargs)
        logger.info("📊 使用标准模式 - 平衡效率与效果")
    
    return config


def prepare_data_if_needed(config, args):
    """准备训练数据"""
    data_manager = DataManager(config)
    
    if args.prepare_data:
        logger.info("准备新的训练数据...")
        return data_manager.create_and_save_mock_data()
    
    # 检查数据是否存在
    health = data_manager.check_data_health()
    if all(health['datasets_exist'].values()):
        logger.info("✅ 发现已有训练数据")
        return True
    else:
        logger.info("📊 准备训练数据...")
        return data_manager.create_and_save_mock_data()


def train_model(config, model_type, args):
    """训练指定类型的模型"""
    model_name = "Tokenizer" if model_type == "tokenizer" else "Predictor"
    
    if getattr(args, f'skip_{model_type}', False):
        logger.info(f"⏭️  跳过{model_name}训练")
        return True
    
    logger.info(f"🎯 开始训练{model_name}...")
    
    # 检查是否使用微调后的tokenizer
    if model_type == "predictor":
        tokenizer_path = Path(config.finetuned_tokenizer_path)
        if tokenizer_path.exists():
            config.pretrained_tokenizer_path = str(tokenizer_path)
            logger.info(f"🔗 使用微调后的Tokenizer: {tokenizer_path}")
    
    # 训练
    trainer = UnifiedTrainer(config, model_type)
    results = trainer.train()
    
    if results['success']:
        logger.info(f"✅ {model_name}训练完成!")
        logger.info(f"   最佳验证损失: {results['best_val_loss']:.4f}")
        return True
    else:
        logger.error(f"❌ {model_name}训练失败: {results.get('error', '未知错误')}")
        return False


def main():
    args = parse_args()
    
    try:
        print("🚀 Kronos极简微调")
        print("=" * 50)
        
        # 1. 获取配置
        config = get_config(args)
        config.summary()
        print()
        
        # 2. 准备数据
        logger.info("📊 第一步: 数据准备")
        if not prepare_data_if_needed(config, args):
            logger.error("数据准备失败")
            return 1
        print()
        
        # 3. 训练Tokenizer
        logger.info("🤖 第二步: Tokenizer训练")
        if not train_model(config, "tokenizer", args):
            return 1
        print()
        
        # 4. 训练Predictor  
        logger.info("🎯 第三步: Predictor训练")
        if not train_model(config, "predictor", args):
            return 1
        print()
        
        # 5. 完成总结
        print("🎉 微调完成!")
        print("=" * 50)
        print("📂 模型保存位置:")
        print(f"   Tokenizer: {config.finetuned_tokenizer_path}")
        print(f"   Predictor: {config.finetuned_predictor_path}")
        print()
        print("🚀 现在可以使用你的微调模型了!")
        print("   参考 examples/ 目录中的预测示例代码")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⏹️  用户中断训练")
        return 1
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)