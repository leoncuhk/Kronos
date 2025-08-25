#!/usr/bin/env python3
"""
快速训练测试脚本 - 验证Kronos微调流程
基于原项目train_tokenizer.py和train_predictor.py简化而来
"""

import sys
sys.path.append("..")

import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from config import Config
from dataset import QlibDataset
from model import Kronos, KronosTokenizer

def quick_tokenizer_test():
    """快速测试tokenizer训练"""
    print("🔧 快速测试Tokenizer训练...")
    
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 1. 加载预训练tokenizer
        print("📥 加载预训练tokenizer...")
        tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
        tokenizer = tokenizer.to(device)
        
        # 2. 加载数据集
        print("📊 加载训练数据...")
        train_dataset = QlibDataset('train')
        
        if len(train_dataset) == 0:
            print("❌ 训练数据集为空，请先运行数据预处理")
            return False
        
        # 3. 测试前向传播
        print("🧪 测试前向传播...")
        sample_batch = train_dataset[0]  # 获取一个样本
        if isinstance(sample_batch, tuple):
            batch_x, _ = sample_batch
        else:
            batch_x = sample_batch
        
        # 处理不同的数据类型
        if isinstance(batch_x, np.ndarray):
            batch_x = torch.from_numpy(batch_x).float().unsqueeze(0).to(device)
        elif isinstance(batch_x, torch.Tensor):
            batch_x = batch_x.float().unsqueeze(0).to(device)
        else:
            raise TypeError(f"Unsupported batch_x type: {type(batch_x)}")
        
        with torch.no_grad():
            zs, bsq_loss, _, _ = tokenizer(batch_x)
            print(f"✅ 前向传播成功! BSQ Loss: {bsq_loss.item():.4f}")
        
        print("🎉 Tokenizer测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_predictor_test():
    """快速测试predictor训练"""
    print("🔧 快速测试Predictor训练...")
    
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 1. 加载预训练模型
        print("📥 加载预训练模型...")
        tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
        model = Kronos.from_pretrained(config.pretrained_predictor_path)
        
        tokenizer = tokenizer.to(device)
        model = model.to(device)
        
        # 2. 测试数据
        print("📊 加载测试数据...")
        train_dataset = QlibDataset('train')
        
        if len(train_dataset) == 0:
            print("❌ 训练数据集为空")
            return False
        
        # 3. 测试完整流程
        print("🧪 测试预测流程...")
        sample_batch = train_dataset[0]
        if isinstance(sample_batch, tuple):
            batch_x, batch_x_stamp = sample_batch
        else:
            batch_x = sample_batch
            batch_x_stamp = None
        
        # 处理不同的数据类型
        if isinstance(batch_x, np.ndarray):
            batch_x = torch.from_numpy(batch_x).float().unsqueeze(0).to(device)
        elif isinstance(batch_x, torch.Tensor):
            batch_x = batch_x.float().unsqueeze(0).to(device)
        else:
            raise TypeError(f"Unsupported batch_x type: {type(batch_x)}")
            
        if batch_x_stamp is not None:
            if isinstance(batch_x_stamp, np.ndarray):
                batch_x_stamp = torch.from_numpy(batch_x_stamp).float().unsqueeze(0).to(device)
            elif isinstance(batch_x_stamp, torch.Tensor):
                batch_x_stamp = batch_x_stamp.float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Tokenize
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            # Predict
            if batch_x_stamp is not None:
                logits = model(token_seq_0[:, :-1], token_seq_1[:, :-1], batch_x_stamp[:, :-1, :])
            else:
                logits = model(token_seq_0[:, :-1], token_seq_1[:, :-1])
            
            print(f"✅ 预测成功! 输出形状: {[l.shape for l in logits]}")
        
        print("🎉 Predictor测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ Predictor测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 Kronos微调快速测试")
    print("=" * 50)
    
    # 检查数据
    config = Config()
    data_dir = Path(config.dataset_path)
    if not data_dir.exists():
        print("❌ 数据目录不存在，请先运行 mock_data_preprocess.py")
        return False
    
    # 测试Tokenizer
    tokenizer_ok = quick_tokenizer_test()
    if not tokenizer_ok:
        print("❌ Tokenizer测试失败")
        return False
    
    print()  # 空行分隔
    
    # 测试Predictor
    predictor_ok = quick_predictor_test()
    if not predictor_ok:
        print("❌ Predictor测试失败")
        return False
    
    print("\n🎉 所有测试通过! Kronos微调环境配置成功!")
    print("💡 接下来可以:")
    print("   1. 运行完整的tokenizer训练: python train_tokenizer.py")
    print("   2. 运行完整的predictor训练: python train_predictor.py")
    print("   3. 运行回测验证: python qlib_test.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
