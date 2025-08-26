#!/usr/bin/env python3
"""
🎯 Kronos完整微调演示脚本
展示从数据准备到模型训练到预测验证的完整流程

功能：
✅ 真实格式数据生成和验证
✅ Tokenizer微调训练
✅ Predictor微调训练
✅ 微调后模型预测演示
✅ 性能对比和结果分析
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append("../")
from config import Config
from model import Kronos, KronosTokenizer, KronosPredictor

class KronosFineTuneDemo:
    """Kronos微调完整演示"""
    
    def __init__(self):
        self.config = Config()
        self.results = {}
        print("🎯 Kronos完整微调演示")
        print("="*80)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print()
    
    def step_1_prepare_data(self):
        """步骤1：准备真实格式数据"""
        print("📊 步骤1：准备真实格式数据")
        print("-" * 50)
        
        try:
            # 检查是否已有真实格式数据
            data_info_file = Path(self.config.dataset_path) / "data_info_real.json"
            if data_info_file.exists():
                import json
                with open(data_info_file, 'r', encoding='utf-8') as f:
                    data_info = json.load(f)
                
                print("✅ 发现已有真实格式数据:")
                print(f"   股票数量: {data_info['total_symbols']}")
                print(f"   时间范围: {data_info['date_range']['start']} 到 {data_info['date_range']['end']}")
                print(f"   交易日数: {data_info['total_trading_days']}")
                
                self.results['data_ready'] = True
                return True
            else:
                print("⚠️  未发现真实格式数据，请运行:")
                print("   python get_real_data.py")
                self.results['data_ready'] = False
                return False
                
        except Exception as e:
            print(f"❌ 数据检查失败: {e}")
            self.results['data_ready'] = False
            return False
    
    def step_2_train_tokenizer(self):
        """步骤2：Tokenizer微调训练"""
        print("\\n🔧 步骤2：Tokenizer微调训练")
        print("-" * 50)
        
        # 检查是否已有训练好的tokenizer
        tokenizer_path = Path(self.config.save_path) / self.config.tokenizer_save_folder_name / "best_model"
        
        if tokenizer_path.exists():
            print("✅ 发现已训练的Tokenizer:")
            print(f"   路径: {tokenizer_path}")
            
            # 验证模型可以加载
            try:
                tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
                print(f"   参数量: {sum(p.numel() for p in tokenizer.parameters()):,}")
                self.results['tokenizer_trained'] = True
                return True
            except Exception as e:
                print(f"⚠️  Tokenizer加载失败: {e}")
                print("建议重新训练: python simple_train_tokenizer.py")
                self.results['tokenizer_trained'] = False
                return False
        else:
            print("⚠️  未发现训练好的Tokenizer，建议运行:")
            print("   python simple_train_tokenizer.py --device cuda --epochs 2")
            self.results['tokenizer_trained'] = False
            return False
    
    def step_3_train_predictor(self):
        """步骤3：Predictor微调训练"""
        print("\\n🧠 步骤3：Predictor微调训练")
        print("-" * 50)
        
        # 检查是否已有训练好的predictor
        predictor_path = Path(self.config.save_path) / self.config.predictor_save_folder_name / "best_model"
        
        if predictor_path.exists():
            print("✅ 发现已训练的Predictor:")
            print(f"   路径: {predictor_path}")
            
            # 验证模型可以加载
            try:
                predictor = Kronos.from_pretrained(str(predictor_path))
                print(f"   参数量: {sum(p.numel() for p in predictor.parameters()):,}")
                self.results['predictor_trained'] = True
                return True
            except Exception as e:
                print(f"⚠️  Predictor加载失败: {e}")
                print("建议重新训练: python simple_train_predictor.py")
                self.results['predictor_trained'] = False
                return False
        else:
            print("⚠️  未发现训练好的Predictor，建议运行:")
            print("   python simple_train_predictor.py --device cuda --epochs 2")
            self.results['predictor_trained'] = False
            return False
    
    def step_4_prediction_demo(self):
        """步骤4：预测演示对比"""
        print("\\n🔮 步骤4：预测演示对比")
        print("-" * 50)
        
        if not (self.results.get('tokenizer_trained') and self.results.get('predictor_trained')):
            print("❌ 微调模型不可用，跳过预测演示")
            return False
        
        try:
            # 加载微调后的模型
            tokenizer_path = Path(self.config.save_path) / self.config.tokenizer_save_folder_name / "best_model"
            predictor_path = Path(self.config.save_path) / self.config.predictor_save_folder_name / "best_model"
            
            finetuned_tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
            finetuned_predictor = Kronos.from_pretrained(str(predictor_path))
            
            # 加载预训练模型对比
            pretrained_tokenizer = KronosTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
            pretrained_predictor = Kronos.from_pretrained(self.config.pretrained_predictor_path)
            
            # 创建预测器
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            finetuned_predictor_obj = KronosPredictor(
                finetuned_predictor, finetuned_tokenizer, device=device, max_context=512
            )
            
            pretrained_predictor_obj = KronosPredictor(
                pretrained_predictor, pretrained_tokenizer, device=device, max_context=512
            )
            
            # 准备测试数据
            test_data = self._prepare_test_data()
            if test_data is None:
                return False
            
            # 进行预测对比
            results = self._run_prediction_comparison(
                test_data, finetuned_predictor_obj, pretrained_predictor_obj
            )
            
            # 可视化结果
            self._visualize_comparison(test_data, results)
            
            self.results['prediction_demo'] = True
            return True
            
        except Exception as e:
            print(f"❌ 预测演示失败: {e}")
            import traceback
            traceback.print_exc()
            self.results['prediction_demo'] = False
            return False
    
    def _prepare_test_data(self):
        """准备测试数据"""
        try:
            # 加载真实格式测试数据
            import pickle
            test_file = Path(self.config.dataset_path) / "test_data_real.pkl"
            
            if not test_file.exists():
                print("❌ 测试数据不存在")
                return None
            
            with open(test_file, 'rb') as f:
                test_data_dict = pickle.load(f)
            
            # 选择一只股票进行演示（选择贵州茅台）
            symbol = "600519.SH"  # 贵州茅台
            if symbol not in test_data_dict:
                symbol = list(test_data_dict.keys())[0]  # 如果没有茅台，选择第一只
            
            df = test_data_dict[symbol].copy()
            df.reset_index(inplace=True)
            
            # 确保有datetime列
            if 'datetime' not in df.columns:
                df['datetime'] = df.index
            
            print(f"✅ 准备测试数据: {symbol}")
            print(f"   数据长度: {len(df)}")
            print(f"   时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
            
            return {'symbol': symbol, 'data': df}
            
        except Exception as e:
            print(f"❌ 测试数据准备失败: {e}")
            return None
    
    def _run_prediction_comparison(self, test_data, finetuned_predictor, pretrained_predictor):
        """运行预测对比"""
        symbol = test_data['symbol']
        df = test_data['data']
        
        # 设置预测参数
        lookback = 100
        pred_len = 50
        
        if len(df) < lookback + pred_len:
            print(f"⚠️  数据不够，调整参数")
            lookback = len(df) // 2
            pred_len = len(df) - lookback
        
        # 准备输入数据
        x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'vol', 'amt']]
        x_timestamp = pd.to_datetime(df.iloc[:lookback]['datetime'])
        y_timestamp = pd.to_datetime(df.iloc[lookback:lookback+pred_len]['datetime'])
        
        # 真实数据（用于对比）
        y_true = df.iloc[lookback:lookback+pred_len][['open', 'high', 'low', 'close', 'vol', 'amt']]
        
        print(f"预测配置: 历史{lookback}天 → 预测{pred_len}天")
        
        # 微调模型预测
        print("   运行微调模型预测...")
        start_time = time.time()
        
        finetuned_pred = finetuned_predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.8,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )
        
        finetuned_time = time.time() - start_time
        
        # 预训练模型预测
        print("   运行预训练模型预测...")
        start_time = time.time()
        
        pretrained_pred = pretrained_predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.8,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )
        
        pretrained_time = time.time() - start_time
        
        # 计算预测误差
        def calc_mape(y_true, y_pred):
            """计算MAPE"""
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        finetuned_mape = calc_mape(y_true['close'].values, finetuned_pred['close'].values)
        pretrained_mape = calc_mape(y_true['close'].values, pretrained_pred['close'].values)
        
        print(f"\\n📊 预测结果对比:")
        print(f"   微调模型 MAPE: {finetuned_mape:.2f}% (耗时: {finetuned_time:.2f}s)")
        print(f"   预训练模型 MAPE: {pretrained_mape:.2f}% (耗时: {pretrained_time:.2f}s)")
        
        if finetuned_mape < pretrained_mape:
            improvement = ((pretrained_mape - finetuned_mape) / pretrained_mape) * 100
            print(f"   🎉 微调模型表现更好，改进 {improvement:.1f}%!")
        else:
            print(f"   📝 预训练模型表现更好")
        
        return {
            'finetuned_pred': finetuned_pred,
            'pretrained_pred': pretrained_pred,
            'y_true': y_true,
            'finetuned_mape': finetuned_mape,
            'pretrained_mape': pretrained_mape,
            'finetuned_time': finetuned_time,
            'pretrained_time': pretrained_time
        }
    
    def _visualize_comparison(self, test_data, results):
        """可视化对比结果"""
        print("\\n📈 生成对比图表...")
        
        symbol = test_data['symbol']
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 上图：价格对比
        time_index = results['y_true'].index
        
        ax1.plot(time_index, results['y_true']['close'], 
                label='真实价格', color='blue', linewidth=2)
        ax1.plot(time_index, results['finetuned_pred']['close'], 
                label=f'微调模型 (MAPE: {results["finetuned_mape"]:.2f}%)', 
                color='red', linestyle='--', linewidth=2)
        ax1.plot(time_index, results['pretrained_pred']['close'], 
                label=f'预训练模型 (MAPE: {results["pretrained_mape"]:.2f}%)', 
                color='green', linestyle=':', linewidth=2)
        
        ax1.set_title(f'{symbol} 股价预测对比 - 微调 vs 预训练模型', fontsize=14, weight='bold')
        ax1.set_ylabel('价格 (元)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：预测误差
        finetuned_error = np.abs(results['y_true']['close'] - results['finetuned_pred']['close'])
        pretrained_error = np.abs(results['y_true']['close'] - results['pretrained_pred']['close'])
        
        ax2.plot(time_index, finetuned_error, 
                label='微调模型误差', color='red', linewidth=2)
        ax2.plot(time_index, pretrained_error, 
                label='预训练模型误差', color='green', linewidth=2)
        
        ax2.set_title('预测误差对比', fontsize=12)
        ax2.set_ylabel('绝对误差 (元)')
        ax2.set_xlabel('时间')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = Path("../outputs") / "finetune_comparison.png"
        chart_path.parent.mkdir(exist_ok=True)
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 图表已保存: {chart_path}")
    
    def step_5_performance_summary(self):
        """步骤5：性能总结"""
        print("\\n📋 步骤5：性能总结")
        print("-" * 50)
        
        print("🎯 Kronos微调完成情况:")
        print(f"   ✅ 数据准备: {'完成' if self.results.get('data_ready') else '未完成'}")
        print(f"   ✅ Tokenizer训练: {'完成' if self.results.get('tokenizer_trained') else '未完成'}")
        print(f"   ✅ Predictor训练: {'完成' if self.results.get('predictor_trained') else '未完成'}")
        print(f"   ✅ 预测演示: {'完成' if self.results.get('prediction_demo') else '未完成'}")
        
        # 统计微调模型文件
        model_files = []
        outputs_dir = Path(self.config.save_path)
        
        if outputs_dir.exists():
            for file in outputs_dir.rglob("*.pt"):
                model_files.append(str(file.relative_to(outputs_dir)))
            
            for file in outputs_dir.rglob("pytorch_model.bin"):
                model_files.append(str(file.relative_to(outputs_dir)))
        
        print(f"\\n💾 已生成模型文件 ({len(model_files)}个):")
        for file in model_files[:10]:  # 只显示前10个
            print(f"   {file}")
        if len(model_files) > 10:
            print(f"   ... 还有{len(model_files)-10}个文件")
        
        # 性能建议
        print("\\n💡 性能优化建议:")
        if not self.results.get('data_ready'):
            print("   🔹 运行 python get_real_data.py 获取真实格式数据")
        if not self.results.get('tokenizer_trained'):
            print("   🔹 运行 python simple_train_tokenizer.py 训练Tokenizer")
        if not self.results.get('predictor_trained'):
            print("   🔹 运行 python simple_train_predictor.py 训练Predictor")
        if all(self.results.get(k, False) for k in ['data_ready', 'tokenizer_trained', 'predictor_trained']):
            print("   🎉 所有组件都已就绪！可以用于生产预测")
            print("   🔹 可以在更大数据集上进行更长时间的训练")
            print("   🔹 可以尝试不同的超参数配置")
            print("   🔹 可以集成到实际的交易系统中")
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("🚀 开始Kronos完整微调演示...")
        start_time = time.time()
        
        # 执行所有步骤
        steps = [
            ("数据准备", self.step_1_prepare_data),
            ("Tokenizer训练", self.step_2_train_tokenizer),
            ("Predictor训练", self.step_3_train_predictor),
            ("预测演示", self.step_4_prediction_demo),
            ("性能总结", self.step_5_performance_summary),
        ]
        
        for step_name, step_func in steps:
            try:
                step_func()
            except Exception as e:
                print(f"❌ {step_name}执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        total_time = time.time() - start_time
        
        print("\\n" + "="*80)
        print("🎉 Kronos微调演示完成!")
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        print(f"📅 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

if __name__ == "__main__":
    demo = KronosFineTuneDemo()
    demo.run_complete_demo()
