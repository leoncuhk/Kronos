#!/usr/bin/env python3
"""
简化版Tokenizer训练脚本 - 去除复杂依赖，专注核心训练功能
基于train_tokenizer.py简化而来，用于验证微调流程
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# 添加项目路径
sys.path.append("../")
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer

def simple_train_tokenizer(device="cuda", epochs=2, save_models=True):
    """简化的tokenizer训练函数"""
    print("🚀 开始简化版Tokenizer训练")
    print("="*60)
    
    # 1. 配置和初始化
    config = Config()
    print(f"使用设备: {device}")
    print(f"训练轮数: {epochs}")
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # 2. 创建数据加载器
    print("📊 加载训练数据...")
    train_dataset = QlibDataset('train')
    val_dataset = QlibDataset('val')
    
    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   验证集大小: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 3. 加载预训练模型
    print("🧠 加载预训练Tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    tokenizer = tokenizer.to(device)
    
    print(f"   模型参数量: {sum(p.numel() for p in tokenizer.parameters()):,}")
    print(f"   可训练参数: {sum(p.numel() for p in tokenizer.parameters() if p.requires_grad):,}")
    
    # 4. 优化器和调度器
    optimizer = torch.optim.AdamW(
        tokenizer.parameters(),
        lr=config.tokenizer_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    
    # 5. 训练循环
    print("🎓 开始训练...")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\\n📖 Epoch {epoch+1}/{epochs}")
        
        # 训练阶段
        tokenizer.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if isinstance(batch_data, (tuple, list)):
                batch_x = batch_data[0]
                if len(batch_data) > 1:
                    _ = batch_data[1]  # timestamps if available
            else:
                batch_x = batch_data
            
            # 确保是tensor
            if isinstance(batch_x, list):
                batch_x = torch.stack(batch_x)
            elif not isinstance(batch_x, torch.Tensor):
                batch_x = torch.tensor(batch_x)
                
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            zs, bsq_loss, _, _ = tokenizer(batch_x)
            z_pre, z = zs
            
            # 计算重构损失
            recon_loss_pre = F.mse_loss(z_pre, batch_x)
            recon_loss_all = F.mse_loss(z, batch_x)
            
            # 总损失
            total_loss = (recon_loss_pre + recon_loss_all + bsq_loss) / 2
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 20 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"   Batch {batch_idx:4d}/{len(train_loader):4d} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"BSQ: {bsq_loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        tokenizer.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (tuple, list)):
                    batch_x = batch_data[0]
                    if len(batch_data) > 1:
                        _ = batch_data[1]  # timestamps if available
                else:
                    batch_x = batch_data
                
                # 确保是tensor
                if isinstance(batch_x, list):
                    batch_x = torch.stack(batch_x)
                elif not isinstance(batch_x, torch.Tensor):
                    batch_x = torch.tensor(batch_x)
                    
                batch_x = batch_x.to(device)
                
                zs, bsq_loss, _, _ = tokenizer(batch_x)
                z_pre, z = zs
                
                recon_loss_pre = F.mse_loss(z_pre, batch_x)
                recon_loss_all = F.mse_loss(z, batch_x)
                total_loss = (recon_loss_pre + recon_loss_all + bsq_loss) / 2
                
                val_loss += total_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"   训练损失: {avg_train_loss:.4f}")
        print(f"   验证损失: {avg_val_loss:.4f}")
        print(f"   耗时: {epoch_time:.2f}秒")
        
        # 保存最佳模型
        if save_models and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = Path(config.save_path) / config.tokenizer_save_folder_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = save_dir / f"best_tokenizer_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': tokenizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            
            # 同时保存为HuggingFace格式
            hf_save_dir = save_dir / "best_model"
            hf_save_dir.mkdir(exist_ok=True)
            tokenizer.save_pretrained(hf_save_dir)
            
            print(f"   ✅ 保存最佳模型: {checkpoint_path}")
    
    print("\\n🎉 Tokenizer训练完成!")
    print(f"   最佳验证损失: {best_val_loss:.4f}")
    
    if save_models:
        print(f"   模型保存位置: {save_dir}")
        print(f"   HuggingFace格式: {hf_save_dir}")
    
    return {
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化版Tokenizer训练")
    parser.add_argument("--device", type=str, default="cuda", 
                      help="训练设备 (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=2,
                      help="训练轮数")
    parser.add_argument("--no-save", action="store_true",
                      help="不保存模型")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    try:
        results = simple_train_tokenizer(
            device=device,
            epochs=args.epochs,
            save_models=not args.no_save
        )
        
        print("\\n📊 训练结果:")
        print(f"   最终训练损失: {results['final_train_loss']:.4f}")
        print(f"   最终验证损失: {results['final_val_loss']:.4f}")
        print(f"   最佳验证损失: {results['best_val_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    else:
        print("\\n✅ 简化版Tokenizer训练成功完成! 🎉")
