#!/usr/bin/env python3
"""
模拟数据预处理器 - 用于测试Kronos微调流程
基于qlib_data_preprocess.py的简化版本
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from config import Config

class MockDataPreprocessor:
    """模拟数据预处理器"""
    
    def __init__(self):
        self.config = Config()
        self.data_fields = ['open', 'close', 'high', 'low', 'vol', 'vwap']
        self.data = {}
        
    def create_synthetic_data(self):
        """创建合成数据用于测试"""
        print("🎭 创建合成测试数据...")
        
        # 生成时间序列
        start_date = pd.Timestamp(self.config.dataset_begin_time)
        end_date = pd.Timestamp(self.config.dataset_end_time)
        
        # 创建业务日历（简化版）
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        business_days = dates[dates.dayofweek < 5][:1000]  # 限制数据量用于测试
        
        # 生成10只模拟股票的数据
        symbols = [f"SH{600000+i:06d}" for i in range(10)]
        
        for symbol in symbols:
            print(f"   生成 {symbol} 的数据...")
            
            n_days = len(business_days)
            np.random.seed(hash(symbol) % 1000)
            
            # 基础价格和波动率
            base_price = 10 + np.random.random() * 20
            returns = np.random.normal(0.0005, 0.02, n_days)
            
            # 生成价格序列
            log_prices = np.cumsum(returns) + np.log(base_price)
            close_prices = np.exp(log_prices)
            
            # OHLC数据
            open_prices = close_prices * (1 + np.random.normal(0, 0.003, n_days))
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
            
            # 交易量和VWAP
            volumes = (1000000 + np.random.randint(0, 2000000, n_days)).astype(float)
            vwap = (high_prices + low_prices + close_prices) / 3
            
            # 存储数据
            self.data[symbol] = pd.DataFrame({
                'datetime': business_days,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'vol': volumes,  # 修正列名为 'vol'
                'vwap': vwap
            }).set_index('datetime')
        
        print(f"✅ 已生成 {len(symbols)} 个股票的合成数据")
        return True
    
    def prepare_features(self):
        """准备特征数据"""
        print("🔧 准备特征数据...")
        
        processed_data = {}
        for symbol, df in self.data.items():
            # 基础特征
            features = df[self.data_fields].copy()
            
            # 添加金额特征（如果需要）  
            if 'amt' not in features.columns:
                features['amt'] = features['close'] * features['vol']
            
            # 数据标准化（可选）
            # features = (features - features.mean()) / features.std()
            
            processed_data[symbol] = features
        
        self.data = processed_data
        print("✅ 特征准备完成")
        return True
    
    def split_and_save_datasets(self):
        """分割并保存数据集"""
        print("💾 分割并保存数据集...")
        
        # 确保输出目录存在
        os.makedirs(self.config.dataset_path, exist_ok=True)
        
        # 时间分割
        train_symbols = {}
        val_symbols = {}
        test_symbols = {}
        
        for symbol, df in self.data.items():
            total_len = len(df)
            train_end = int(total_len * 0.7)
            val_end = int(total_len * 0.85)
            
            train_symbols[symbol] = df.iloc[:train_end]
            val_symbols[symbol] = df.iloc[train_end:val_end]
            test_symbols[symbol] = df.iloc[val_end:]
        
        # 保存数据集
        datasets = {
            'train': train_symbols,
            'val': val_symbols,
            'test': test_symbols
        }
        
        for split_name, data in datasets.items():
            file_path = Path(self.config.dataset_path) / f"{split_name}_data.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"   保存 {split_name} 数据集: {file_path}")
        
        print("✅ 数据集分割和保存完成")
        return True
    
    def run_full_pipeline(self):
        """运行完整的数据预处理管道"""
        print("🚀 开始模拟数据预处理...")
        
        try:
            # 1. 创建合成数据
            self.create_synthetic_data()
            
            # 2. 准备特征
            self.prepare_features()
            
            # 3. 分割和保存数据集
            self.split_and_save_datasets()
            
            print("🎉 模拟数据预处理完成!")
            return True
            
        except Exception as e:
            print(f"❌ 预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    preprocessor = MockDataPreprocessor()
    success = preprocessor.run_full_pipeline()
    if success:
        print("\n✅ 可以继续进行模型微调训练!")
    else:
        print("\n❌ 预处理失败，请检查错误信息")
