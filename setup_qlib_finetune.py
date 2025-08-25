#!/usr/bin/env python3
"""
🏗️ Qlib + Kronos微调环境配置脚本
一键完成qlib数据准备、配置修改、模型微调的完整流程

基于项目原有架构，遵循finetune目录的设计思路
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import qlib
from qlib.config import REG_CN
from qlib.data import D

class QlibKronosSetup:
    """Qlib + Kronos微调环境配置器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.finetune_dir = self.project_root / "finetune"
        self.outputs_dir = self.project_root / "outputs"
        self.data_dir = self.outputs_dir / "qlib_data"
        
        # 创建必要目录
        self.outputs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        print("🚀 Qlib + Kronos微调环境配置器")
        print("="*60)
    
    def step_1_download_qlib_data(self):
        """步骤1: 下载和配置qlib数据"""
        print("\n📊 步骤1: 配置qlib数据")
        print("-" * 40)
        
        try:
            # 方案A: 使用内置数据下载器（推荐小规模测试）
            print("🔍 检查qlib数据可用性...")
            
            # 初始化qlib（使用临时目录）
            temp_data_dir = self.data_dir / "qlib_cn_data"
            temp_data_dir.mkdir(exist_ok=True)
            
            # 尝试初始化qlib
            try:
                qlib.init(provider_uri=str(temp_data_dir), region=REG_CN)
                print("✅ Qlib初始化成功")
            except Exception as e:
                print(f"⚠️  Qlib初始化失败: {e}")
                print("📥 尝试下载示例数据...")
                
                # 方案B: 生成模拟数据用于测试
                self._create_mock_data()
                return True
            
            # 尝试访问日历数据
            try:
                cal = D.calendar()
                if len(cal) > 0:
                    print(f"✅ 找到qlib日历数据: {len(cal)}条记录")
                    print(f"   时间范围: {cal[0]} 到 {cal[-1]}")
                    return True
                else:
                    print("⚠️  日历数据为空，生成模拟数据")
                    self._create_mock_data()
                    return True
            except Exception as e:
                print(f"⚠️  无法访问qlib数据: {e}")
                print("📥 生成模拟数据用于测试...")
                self._create_mock_data()
                return True
                
        except Exception as e:
            print(f"❌ 数据配置失败: {e}")
            return False
    
    def _create_mock_data(self):
        """创建模拟数据用于测试"""
        print("🎭 创建模拟CSI300数据...")
        
        # 生成时间序列
        start_date = pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp('2024-12-31')
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 过滤工作日
        business_days = dates[dates.dayofweek < 5]  # 周一到周五
        
        # 模拟股票代码
        symbols = [f"{i:06d}.SH" for i in range(600000, 600010)]  # 10只股票用于测试
        
        mock_data_dir = self.data_dir / "mock_csi300_data"
        mock_data_dir.mkdir(exist_ok=True)
        
        for symbol in symbols:
            # 为每只股票生成价格数据
            np.random.seed(hash(symbol) % 1000)  # 固定种子确保可重现
            
            n_days = len(business_days)
            base_price = 10 + np.random.random() * 20  # 基础价格10-30元
            
            # 生成价格序列（使用随机游走）
            returns = np.random.normal(0.0002, 0.02, n_days)  # 年化收益2.5%，波动率31%
            log_prices = np.cumsum(returns) + np.log(base_price)
            prices = np.exp(log_prices)
            
            # 生成OHLC数据
            close_prices = prices
            open_prices = close_prices * (1 + np.random.normal(0, 0.005, n_days))
            high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
            low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
            
            # 生成交易量
            base_volume = 1000000 + np.random.randint(0, 5000000)
            volumes = base_volume * (1 + np.random.normal(0, 0.5, n_days))
            volumes = np.abs(volumes).astype(int)
            
            # 计算vwap
            vwap = (high_prices + low_prices + close_prices) / 3
            
            # 创建DataFrame
            stock_data = pd.DataFrame({
                'datetime': business_days,
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices,
                'volume': volumes,
                'vwap': vwap
            })
            
            # 保存数据
            symbol_file = mock_data_dir / f"{symbol.replace('.', '_')}.csv"
            stock_data.to_csv(symbol_file, index=False)
        
        print(f"✅ 已生成{len(symbols)}只股票的模拟数据")
        print(f"   数据保存位置: {mock_data_dir}")
        print(f"   时间范围: {business_days[0]} 到 {business_days[-1]}")
        
        return mock_data_dir
    
    def step_2_update_config(self):
        """步骤2: 更新配置文件"""
        print("\n⚙️  步骤2: 更新配置文件")
        print("-" * 40)
        
        config_file = self.finetune_dir / "config.py"
        if not config_file.exists():
            print(f"❌ 配置文件不存在: {config_file}")
            return False
        
        # 读取原配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        
        # 创建备份
        backup_file = config_file.with_suffix('.py.backup')
        shutil.copy2(config_file, backup_file)
        print(f"📋 已备份原配置: {backup_file}")
        
        # 修改配置
        updated_config = config_content.replace(
            'self.qlib_data_path = "~/.qlib/qlib_data/cn_data"',
            f'self.qlib_data_path = r"{self.data_dir}"'
        ).replace(
            'self.dataset_path = "./data/processed_datasets"',
            f'self.dataset_path = r"{self.outputs_dir / "processed_datasets"}"'
        ).replace(
            'self.save_path = "./outputs/models"',
            f'self.save_path = r"{self.outputs_dir / "models"}"'
        ).replace(
            'self.backtest_result_path = "./outputs/backtest_results"',
            f'self.backtest_result_path = r"{self.outputs_dir / "backtest_results"}"'
        ).replace(
            'self.pretrained_tokenizer_path = "path/to/your/Kronos-Tokenizer-base"',
            'self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"'
        ).replace(
            'self.pretrained_predictor_path = "path/to/your/Kronos-small"',
            'self.pretrained_predictor_path = "NeoQuasar/Kronos-small"'
        ).replace(
            'self.epochs = 30',
            'self.epochs = 2  # 减少训练轮数用于测试'
        ).replace(
            'self.n_train_iter = 2000 * self.batch_size',
            'self.n_train_iter = 100 * self.batch_size  # 减少训练步数用于测试'
        ).replace(
            'self.n_val_iter = 400 * self.batch_size',
            'self.n_val_iter = 20 * self.batch_size  # 减少验证步数用于测试'
        )
        
        # 写入更新后的配置
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(updated_config)
        
        print("✅ 配置文件已更新")
        print("   主要修改:")
        print(f"   - qlib数据路径: {self.data_dir}")
        print(f"   - 输出路径: {self.outputs_dir}")
        print("   - 预训练模型: 使用HuggingFace Hub")
        print("   - 训练参数: 优化用于快速测试")
        
        return True
    
    def step_3_create_mock_data_preprocessor(self):
        """步骤3: 创建模拟数据预处理器"""
        print("\n🔄 步骤3: 创建数据预处理脚本")
        print("-" * 40)
        
        mock_preprocess_script = self.finetune_dir / "mock_data_preprocess.py"
        
        script_content = '''#!/usr/bin/env python3
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
        self.data_fields = ['open', 'close', 'high', 'low', 'volume', 'vwap']
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
                'volume': volumes,
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
                features['amt'] = features['close'] * features['volume']
            
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
        print("\\n✅ 可以继续进行模型微调训练!")
    else:
        print("\\n❌ 预处理失败，请检查错误信息")
'''
        
        with open(mock_preprocess_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ 已创建模拟数据预处理脚本: {mock_preprocess_script}")
        return True
    
    def step_4_test_preprocessing(self):
        """步骤4: 测试数据预处理"""
        print("\n🧪 步骤4: 测试数据预处理")
        print("-" * 40)
        
        try:
            os.chdir(self.finetune_dir)
            result = subprocess.run(
                [sys.executable, "mock_data_preprocess.py"], 
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                print("✅ 数据预处理测试成功!")
                print("标准输出:", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                return True
            else:
                print("❌ 数据预处理失败")
                print("错误输出:", result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print("⏱️  预处理超时，可能需要更多时间")
            return False
        except Exception as e:
            print(f"❌ 预处理测试异常: {e}")
            return False
        finally:
            os.chdir(self.project_root)
    
    def step_5_prepare_training_script(self):
        """步骤5: 准备训练脚本"""
        print("\n🎓 步骤5: 准备快速训练脚本")
        print("-" * 40)
        
        quick_train_script = self.finetune_dir / "quick_train_test.py"
        
        script_content = '''#!/usr/bin/env python3
"""
快速训练测试脚本 - 验证Kronos微调流程
基于原项目train_tokenizer.py和train_predictor.py简化而来
"""

import sys
sys.path.append("..")

import os
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
        
        batch_x = torch.from_numpy(batch_x).float().unsqueeze(0).to(device)
        
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
        
        batch_x = torch.from_numpy(batch_x).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Tokenize
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            # Predict
            if batch_x_stamp is not None:
                batch_x_stamp = torch.from_numpy(batch_x_stamp).float().unsqueeze(0).to(device)
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
    
    print("\\n🎉 所有测试通过! Kronos微调环境配置成功!")
    print("💡 接下来可以:")
    print("   1. 运行完整的tokenizer训练: python train_tokenizer.py")
    print("   2. 运行完整的predictor训练: python train_predictor.py")
    print("   3. 运行回测验证: python qlib_test.py")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
'''
        
        with open(quick_train_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"✅ 已创建快速训练测试脚本: {quick_train_script}")
        return True
    
    def run_complete_setup(self):
        """运行完整的配置流程"""
        print("🔧 开始完整的Qlib + Kronos微调环境配置...")
        
        steps = [
            ("📊 数据配置", self.step_1_download_qlib_data),
            ("⚙️  配置更新", self.step_2_update_config),
            ("🔄 预处理脚本", self.step_3_create_mock_data_preprocessor),
            ("🧪 预处理测试", self.step_4_test_preprocessing),
            ("🎓 训练脚本", self.step_5_prepare_training_script),
        ]
        
        for step_name, step_func in steps:
            print(f"\\n{step_name}")
            if not step_func():
                print(f"❌ {step_name} 失败!")
                return False
        
        print("\\n🎉 Qlib + Kronos微调环境配置完成!")
        print("\\n📋 接下来可以执行:")
        print(f"   cd {self.finetune_dir}")
        print("   python mock_data_preprocess.py    # 数据预处理")
        print("   python quick_train_test.py        # 快速训练测试")
        print("   python train_tokenizer.py         # 完整tokenizer训练")
        print("   python train_predictor.py         # 完整predictor训练")
        
        return True

if __name__ == "__main__":
    setup = QlibKronosSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\\n✅ 配置成功！现在可以开始微调实验了! 🎉")
    else:
        print("\\n❌ 配置过程中遇到问题，请检查上述错误信息。")
