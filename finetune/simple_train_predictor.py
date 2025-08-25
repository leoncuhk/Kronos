#!/usr/bin/env python3
"""
简化版Predictor训练脚本 - 基于微调后的Tokenizer训练Predictor
使用真实格式数据进行训练
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.append("../")
from config import Config
from dataset import QlibDataset
from model.kronos import Kronos, KronosTokenizer

def create_real_data_dataset(data_type='train'):
    """创建使用真实格式数据的数据集"""
    class RealDataDataset(QlibDataset):
        def __init__(self, data_type):
            # 直接调用父类初始化，但修改数据文件名
            self.data_type = data_type
            self.config = Config()
            self.feature_list = self.config.feature_list
            self.time_feature_list = self.config.time_feature_list
            self.window = self.config.lookback_window
            self.data = {}
            self.indices = []
            self.py_rng = np.random.RandomState(self.config.seed)
            
            # 使用真实格式数据
            self.load_real_data()
            self._precompute_sample_indices()

        def load_real_data(self):
            """加载真实格式数据"""
            import pickle
            
            data_file = Path(self.config.dataset_path) / f"{self.data_type}_data_real.pkl"
            if not data_file.exists():
                raise FileNotFoundError(f"真实格式数据文件不存在: {data_file}")
            
            print(f"[{self.data_type.upper()}] 加载真实格式数据: {data_file}")
            with open(data_file, 'rb') as f:
                raw_data = pickle.load(f)
            
            # 处理数据格式
            for symbol, df in raw_data.items():
                # 确保有所需的列
                if not all(col in df.columns for col in self.feature_list):
                    print(f"   警告: {symbol} 缺少某些特征列，使用可用列")
                
                # 添加时间特征
                df = df.copy()
                df.reset_index(inplace=True)
                
                if 'datetime' not in df.columns:
                    df['datetime'] = df.index
                
                df['minute'] = df['datetime'].dt.minute
                df['hour'] = df['datetime'].dt.hour 
                df['weekday'] = df['datetime'].dt.weekday
                df['day'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                
                # 只保留需要的列
                available_features = [col for col in self.feature_list if col in df.columns]
                all_columns = available_features + self.time_feature_list
                
                self.data[symbol] = df[all_columns]
                print(f"   ✅ {symbol}: {len(df)}条记录")
        
        def _precompute_sample_indices(self):
            """预计算样本索引"""
            print(f"[{self.data_type.upper()}] Pre-computing sample indices...")
            self.indices = []
            
            for symbol in self.data.keys():
                df = self.data[symbol]
                num_samples = len(df) - self.window + 1
                if num_samples > 0:
                    # Add all valid starting indices for this symbol to the global list.
                    for i in range(num_samples):
                        self.indices.append((symbol, i))
            
            # Limit the number of samples per epoch if specified.
            total_samples = len(self.indices)
            if self.data_type == 'train':
                epoch_samples = min(self.config.n_train_iter // self.config.batch_size * self.config.batch_size, total_samples)
            elif self.data_type == 'val':
                epoch_samples = min(self.config.n_val_iter // self.config.batch_size * self.config.batch_size, total_samples)
            else:
                epoch_samples = total_samples
            
            self.epoch_length = epoch_samples
            self.n_samples = epoch_samples  # 添加n_samples属性
            print(f"[{self.data_type.upper()}] Found {total_samples} possible samples. Using {epoch_samples} per epoch.")
        
        def __len__(self):
            """返回数据集大小"""
            return self.n_samples
        
        def __getitem__(self, index):
            """获取数据项"""
            if index >= self.n_samples:
                raise IndexError("Index out of range")
            
            # 使用随机采样（类似原数据集的行为）
            random_idx = self.py_rng.randint(0, len(self.indices) - 1)
            symbol, start_idx = self.indices[random_idx]
            
            df = self.data[symbol]
            end_idx = start_idx + self.window
            
            # 获取特征数据
            feature_data = df.iloc[start_idx:end_idx][self.feature_list].values
            time_data = df.iloc[start_idx:end_idx][self.time_feature_list].values
            
            return torch.tensor(feature_data, dtype=torch.float32), torch.tensor(time_data, dtype=torch.float32)
    
    return RealDataDataset(data_type)

def simple_train_predictor(device="cuda", epochs=2, save_models=True):
    """简化的Predictor训练函数"""
    print("🚀 开始简化版Predictor训练")
    print("="*60)
    
    # 1. 配置和初始化
    config = Config()
    print(f"使用设备: {device}")
    print(f"训练轮数: {epochs}")
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # 2. 创建数据加载器 - 使用真实格式数据
    print("📊 加载真实格式训练数据...")
    try:
        train_dataset = create_real_data_dataset('train')
        val_dataset = create_real_data_dataset('val')
    except FileNotFoundError as e:
        print(f"❌ 真实格式数据不存在: {e}")
        print("💡 请先运行 python get_real_data.py 生成真实格式数据")
        return None
    
    print(f"   训练集大小: {len(train_dataset)}")
    print(f"   验证集大小: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 使用单进程避免pickle问题
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # 使用单进程避免pickle问题
        pin_memory=True
    )
    
    # 3. 加载模型
    print("🧠 加载Tokenizer和Predictor...")
    
    # 检查是否有微调后的tokenizer
    finetuned_tokenizer_path = Path(config.save_path) / config.tokenizer_save_folder_name / "best_model"
    if finetuned_tokenizer_path.exists():
        print(f"   使用微调后的Tokenizer: {finetuned_tokenizer_path}")
        tokenizer = KronosTokenizer.from_pretrained(str(finetuned_tokenizer_path))
    else:
        print(f"   使用预训练Tokenizer: {config.pretrained_tokenizer_path}")
        tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    
    # 加载预训练Predictor
    predictor = Kronos.from_pretrained(config.pretrained_predictor_path)
    
    tokenizer = tokenizer.to(device)
    predictor = predictor.to(device)
    
    # 冻结tokenizer参数
    for param in tokenizer.parameters():
        param.requires_grad = False
    tokenizer.eval()
    
    print(f"   Predictor参数量: {sum(p.numel() for p in predictor.parameters()):,}")
    print(f"   可训练参数: {sum(p.numel() for p in predictor.parameters() if p.requires_grad):,}")
    
    # 4. 优化器和调度器
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )
    
    # 5. 训练循环
    print("🎓 开始Predictor训练...")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\\n📖 Epoch {epoch+1}/{epochs}")
        
        # 训练阶段
        predictor.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 处理批次数据
            if isinstance(batch_data, (tuple, list)):
                batch_x = batch_data[0]
                if len(batch_data) > 1:
                    batch_x_stamp = batch_data[1]
                else:
                    batch_x_stamp = None
            else:
                batch_x = batch_data
                batch_x_stamp = None
            
            # 确保是tensor
            if isinstance(batch_x, list):
                batch_x = torch.stack(batch_x)
            elif not isinstance(batch_x, torch.Tensor):
                batch_x = torch.tensor(batch_x)
            
            batch_x = batch_x.to(device)
            
            if batch_x_stamp is not None:
                if isinstance(batch_x_stamp, list):
                    batch_x_stamp = torch.stack(batch_x_stamp)
                elif not isinstance(batch_x_stamp, torch.Tensor):
                    batch_x_stamp = torch.tensor(batch_x_stamp)
                batch_x_stamp = batch_x_stamp.to(device)
            
            optimizer.zero_grad()
            
            # 使用tokenizer编码（冻结）
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            # 准备输入和目标序列
            input_s1 = token_seq_0[:, :-1]
            input_s2 = token_seq_1[:, :-1]
            target_s1 = token_seq_0[:, 1:]
            target_s2 = token_seq_1[:, 1:]
            
            # Predictor前向传播
            if batch_x_stamp is not None:
                s1_logits, s2_logits = predictor(input_s1, input_s2, batch_x_stamp[:, :-1, :])
            else:
                s1_logits, s2_logits = predictor(input_s1, input_s2)
            
            # 计算损失
            s1_loss = F.cross_entropy(s1_logits.reshape(-1, s1_logits.size(-1)), target_s1.reshape(-1))
            s2_loss = F.cross_entropy(s2_logits.reshape(-1, s2_logits.size(-1)), target_s2.reshape(-1))
            total_loss = (s1_loss + s2_loss) / 2
            
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            num_batches += 1
            
            # 打印进度
            if batch_idx % 20 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"   Batch {batch_idx:4d}/{len(train_loader):4d} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"S1: {s1_loss.item():.4f} | "
                      f"S2: {s2_loss.item():.4f} | "
                      f"LR: {current_lr:.2e}")
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        predictor.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # 处理数据（与训练阶段相同）
                if isinstance(batch_data, (tuple, list)):
                    batch_x = batch_data[0]
                    batch_x_stamp = batch_data[1] if len(batch_data) > 1 else None
                else:
                    batch_x = batch_data
                    batch_x_stamp = None
                
                if isinstance(batch_x, list):
                    batch_x = torch.stack(batch_x)
                elif not isinstance(batch_x, torch.Tensor):
                    batch_x = torch.tensor(batch_x)
                
                batch_x = batch_x.to(device)
                
                if batch_x_stamp is not None:
                    if isinstance(batch_x_stamp, list):
                        batch_x_stamp = torch.stack(batch_x_stamp)
                    elif not isinstance(batch_x_stamp, torch.Tensor):
                        batch_x_stamp = torch.tensor(batch_x_stamp)
                    batch_x_stamp = batch_x_stamp.to(device)
                
                # 编码和预测
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                
                input_s1 = token_seq_0[:, :-1]
                input_s2 = token_seq_1[:, :-1]
                target_s1 = token_seq_0[:, 1:]
                target_s2 = token_seq_1[:, 1:]
                
                if batch_x_stamp is not None:
                    s1_logits, s2_logits = predictor(input_s1, input_s2, batch_x_stamp[:, :-1, :])
                else:
                    s1_logits, s2_logits = predictor(input_s1, input_s2)
                
                s1_loss = F.cross_entropy(s1_logits.reshape(-1, s1_logits.size(-1)), target_s1.reshape(-1))
                s2_loss = F.cross_entropy(s2_logits.reshape(-1, s2_logits.size(-1)), target_s2.reshape(-1))
                total_loss = (s1_loss + s2_loss) / 2
                
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
            save_dir = Path(config.save_path) / config.predictor_save_folder_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = save_dir / f"best_predictor_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            
            # 同时保存为HuggingFace格式
            hf_save_dir = save_dir / "best_model"
            hf_save_dir.mkdir(exist_ok=True)
            predictor.save_pretrained(hf_save_dir)
            
            print(f"   ✅ 保存最佳Predictor模型: {checkpoint_path}")
    
    print("\\n🎉 Predictor训练完成!")
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
    parser = argparse.ArgumentParser(description="简化版Predictor训练")
    parser.add_argument("--device", type=str, default="cuda", 
                      help="训练设备 (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=2,
                      help="训练轮数")
    parser.add_argument("--no-save", action="store_true",
                      help="不保存模型")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    try:
        results = simple_train_predictor(
            device=device,
            epochs=args.epochs,
            save_models=not args.no_save
        )
        
        if results is None:
            return False
            
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
        print("\\n✅ 简化版Predictor训练成功完成! 🎉")
