#!/usr/bin/env python3
"""
🚀 Kronos快速入门实验 - 方案A实现
基于项目原有代码架构，无需Qlib依赖的完整预测实验

特性:
✅ 多数据源支持 (YFinance, Binance, 本地CSV)
✅ 概率预测与不确定性量化  
✅ GPU加速支持
✅ 专业级可视化
✅ 遵循项目原有设计思路

作者: Based on Kronos project architecture
"""

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from tqdm import trange

# 导入项目模型
from model import Kronos, KronosTokenizer, KronosPredictor

# 忽略警告
warnings.filterwarnings('ignore')

class KronosQuickExperiment:
    """Kronos快速实验类 - 遵循项目架构设计"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.predictor = None
        self.setup_environment()
    
    def get_default_config(self):
        """获取默认配置 - 基于项目examples和报告优化"""
        return {
            # === 模型配置 ===
            "model_cache_dir": "./Kronos_model",
            "tokenizer_model": "NeoQuasar/Kronos-Tokenizer-base",
            "predictor_model": "NeoQuasar/Kronos-small",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_context": 512,
            
            # === 数据配置 ===
            "data_source": "yfinance",  # yfinance, binance, local_csv
            "symbol": "BTC-USD",
            "interval": "1h",
            "hist_points": 400,  # 与examples一致
            "pred_horizon": 120,  # 与examples一致
            
            # === 预测配置 ===
            "n_predictions": 30,  # 蒙特卡洛采样数，基于报告demo配置
            "temperature": 1.0,
            "top_p": 0.9,
            "sample_count": 1,
            
            # === 评估配置 ===
            "vol_window": 30,  # 波动率计算窗口
            "confidence_levels": [0.68, 0.95],  # 1σ和2σ置信区间
            
            # === 输出配置 ===
            "save_plots": True,
            "plot_dir": "./outputs",
            "verbose": True
        }
    
    def setup_environment(self):
        """环境设置"""
        if self.config["verbose"]:
            print("🔧 Environment Setup:")
            print(f"   Device: {self.config['device']}")
            print(f"   GPU Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print()
        
        # 创建输出目录
        Path(self.config["plot_dir"]).mkdir(exist_ok=True)
    
    def load_model(self):
        """加载Kronos模型 - 遵循项目加载方式"""
        if self.config["verbose"]:
            print("🧠 Loading Kronos model...")
        
        start_time = time.time()
        
        # 加载分词器和模型
        tokenizer = KronosTokenizer.from_pretrained(
            self.config["tokenizer_model"], 
            cache_dir=self.config["model_cache_dir"]
        )
        model = Kronos.from_pretrained(
            self.config["predictor_model"], 
            cache_dir=self.config["model_cache_dir"]
        )
        
        # 设置评估模式
        tokenizer.eval()
        model.eval()
        
        # 创建预测器
        self.predictor = KronosPredictor(
            model, tokenizer, 
            device=self.config["device"],
            max_context=self.config["max_context"]
        )
        
        load_time = time.time() - start_time
        if self.config["verbose"]:
            print(f"✅ Model loaded successfully in {load_time:.2f}s")
            print()
        
        return self.predictor
    
    def fetch_data_yfinance(self):
        """从YFinance获取数据 - 基于项目yfinance_daily.py架构"""
        if self.config["verbose"]:
            print(f"📊 Fetching {self.config['symbol']} data from YFinance...")
        
        try:
            ticker = yf.Ticker(self.config["symbol"])
            # 根据interval调整获取时间范围
            period_map = {
                "1h": "90d",
                "1d": "2y",
                "5m": "60d"
            }
            period = period_map.get(self.config["interval"], "2y")
            
            data = ticker.history(
                period=period, 
                interval=self.config["interval"], 
                auto_adjust=True
            )
            
            if data.empty:
                raise ValueError("No data retrieved from YFinance")
            
            # 数据格式转换 - 遵循项目标准格式
            data = data.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            data['amount'] = data['close'] * data['volume']
            data.reset_index(inplace=True)
            
            # 处理时间列
            time_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
            data.rename(columns={time_col: 'timestamps'}, inplace=True)
            data['timestamps'] = pd.to_datetime(data['timestamps'])
            data = data.dropna()
            
            if self.config["verbose"]:
                print(f"✅ Data fetched: {len(data)} records")
                print(f"   Time range: {data['timestamps'].min()} to {data['timestamps'].max()}")
                print()
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching YFinance data: {e}")
            raise
    
    def fetch_data_local_csv(self):
        """从本地CSV获取数据 - 基于项目examples数据格式"""
        if self.config["verbose"]:
            print("📁 Loading data from local CSV...")
        
        try:
            # 使用项目示例数据
            csv_path = Path("examples/data/XSHG_5min_600977.csv")
            if not csv_path.exists():
                raise FileNotFoundError(f"Local CSV file not found: {csv_path}")
            
            data = pd.read_csv(csv_path)
            data['timestamps'] = pd.to_datetime(data['timestamps'])
            
            # 确保包含必需的列
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            if self.config["verbose"]:
                print(f"✅ Local data loaded: {len(data)} records")
                print()
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading local CSV: {e}")
            raise
    
    def fetch_data(self):
        """数据获取统一接口"""
        data_source = self.config["data_source"]
        
        if data_source == "yfinance":
            return self.fetch_data_yfinance()
        elif data_source == "local_csv":
            return self.fetch_data_local_csv()
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
    
    def make_probabilistic_prediction(self, data):
        """概率预测 - 基于项目demo的蒙特卡洛方法"""
        # 准备数据 - 遵循项目数据处理方式
        hist_data = data.tail(self.config["hist_points"])
        
        # 生成未来时间戳
        last_timestamp = hist_data['timestamps'].max()
        freq_delta = hist_data['timestamps'].iloc[-1] - hist_data['timestamps'].iloc[-2]
        future_timestamps = [
            last_timestamp + (i+1) * freq_delta 
            for i in range(self.config["pred_horizon"])
        ]
        y_timestamp = pd.Series(future_timestamps)
        
        x_timestamp = hist_data['timestamps']
        x_df = hist_data[['open', 'high', 'low', 'close', 'volume', 'amount']]
        
        if self.config["verbose"]:
            print(f"🔮 Making probabilistic prediction...")
            print(f"   Historical points: {len(hist_data)}")
            print(f"   Prediction horizon: {self.config['pred_horizon']}")
            print(f"   Monte Carlo samples: {self.config['n_predictions']}")
        
        # 蒙特卡洛采样预测 - 遵循项目demo架构
        predictions = {}
        cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        for col in cols:
            predictions[col] = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for i in trange(self.config["n_predictions"], desc="Monte Carlo Sampling", disable=not self.config["verbose"]):
                pred_df = self.predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=self.config["pred_horizon"],
                    T=self.config["temperature"],
                    top_p=self.config["top_p"],
                    sample_count=self.config["sample_count"],
                    verbose=False
                )
                
                for col in cols:
                    predictions[col].append(pred_df[col].values)
        
        pred_time = time.time() - start_time
        
        if self.config["verbose"]:
            print(f"✅ Prediction completed in {pred_time:.2f}s")
            print(f"   Average time per sample: {pred_time/self.config['n_predictions']:.3f}s")
            print()
        
        # 转换为DataFrame格式
        pred_results = {}
        for col in cols:
            pred_results[col] = pd.DataFrame(predictions[col]).T
            pred_results[col].index = y_timestamp
        
        return pred_results, hist_data
    
    def calculate_metrics(self, predictions, hist_data):
        """计算预测指标 - 基于项目报告的指标体系"""
        close_preds = predictions['close']
        current_price = hist_data['close'].iloc[-1]
        
        # 1. 上涨概率
        final_prices = close_preds.iloc[-1]
        upside_prob = (final_prices > current_price).mean()
        
        # 2. 波动率放大概率
        hist_returns = np.log(hist_data['close'] / hist_data['close'].shift(1))
        hist_vol = hist_returns.iloc[-self.config["vol_window"]:].std()
        
        vol_amplifications = []
        for col in close_preds.columns:
            pred_prices = close_preds[col].values
            full_prices = np.concatenate([[current_price], pred_prices])
            pred_returns = np.log(full_prices[1:] / full_prices[:-1])
            pred_vol = np.std(pred_returns)
            vol_amplifications.append(pred_vol > hist_vol)
        
        vol_amp_prob = np.mean(vol_amplifications)
        
        # 3. 预测区间
        close_mean = close_preds.mean(axis=1)
        close_std = close_preds.std(axis=1)
        close_min = close_preds.min(axis=1)
        close_max = close_preds.max(axis=1)
        
        # 4. 置信区间
        confidence_intervals = {}
        for conf_level in self.config["confidence_levels"]:
            alpha = 1 - conf_level
            lower = close_preds.quantile(alpha/2, axis=1)
            upper = close_preds.quantile(1-alpha/2, axis=1)
            confidence_intervals[f"{conf_level:.0%}"] = (lower, upper)
        
        metrics = {
            'upside_probability': upside_prob,
            'volatility_amplification_prob': vol_amp_prob,
            'current_price': current_price,
            'mean_forecast': close_mean.iloc[-1],
            'forecast_range': (close_min.iloc[-1], close_max.iloc[-1]),
            'confidence_intervals': confidence_intervals,
            'prediction_stats': {
                'mean': close_mean,
                'std': close_std,
                'min': close_min,
                'max': close_max
            }
        }
        
        if self.config["verbose"]:
            print("📊 Prediction Metrics:")
            print(f"   Upside Probability: {upside_prob:.1%}")
            print(f"   Volatility Amplification: {vol_amp_prob:.1%}")
            print(f"   Current Price: ${current_price:.2f}")
            print(f"   Mean Forecast: ${close_mean.iloc[-1]:.2f}")
            print(f"   Forecast Range: ${close_min.iloc[-1]:.2f} - ${close_max.iloc[-1]:.2f}")
            print()
        
        return metrics
    
    def create_visualization(self, predictions, hist_data, metrics):
        """创建专业可视化 - 基于项目demo的图表风格"""
        if self.config["verbose"]:
            print("🎨 Creating visualization...")
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # 准备数据
        hist_time = hist_data['timestamps']
        pred_time = predictions['close'].index
        close_preds = predictions['close']
        volume_preds = predictions['volume']
        
        # === 价格预测图 ===
        ax1 = axes[0]
        
        # 历史价格
        ax1.plot(hist_time, hist_data['close'], 
                color='royalblue', label='Historical Price', linewidth=1.5)
        
        # 预测均值
        pred_mean = close_preds.mean(axis=1)
        ax1.plot(pred_time, pred_mean, 
                color='darkorange', label='Mean Forecast', linewidth=2)
        
        # 预测区间
        pred_min = close_preds.min(axis=1)
        pred_max = close_preds.max(axis=1)
        ax1.fill_between(pred_time, pred_min, pred_max, 
                        color='darkorange', alpha=0.2, label='Forecast Range')
        
        # 置信区间
        pred_std = close_preds.std(axis=1)
        ax1.fill_between(pred_time, pred_mean - pred_std, pred_mean + pred_std,
                        color='darkorange', alpha=0.3, label='±1σ')
        
        # 分隔线
        separator_time = hist_time.iloc[-1]
        ax1.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5)
        
        ax1.set_title(f'{self.config["symbol"]} Probabilistic Forecast - Kronos Model', 
                     fontsize=16, weight='bold')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === 交易量图 ===
        ax2 = axes[1]
        
        # 历史交易量
        ax2.bar(hist_time, hist_data['volume'], 
               color='skyblue', label='Historical Volume', alpha=0.7, width=0.8)
        
        # 预测交易量
        volume_mean = volume_preds.mean(axis=1)
        ax2.bar(pred_time, volume_mean, 
               color='sandybrown', label='Forecasted Volume', alpha=0.7, width=0.8)
        
        ax2.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加预测指标文本
        metrics_text = f"""Prediction Metrics:
• Upside Probability: {metrics['upside_probability']:.1%}
• Volatility Amplification: {metrics['volatility_amplification_prob']:.1%}
• Current: ${metrics['current_price']:.2f}
• Forecast: ${metrics['mean_forecast']:.2f}"""
        
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图表
        if self.config["save_plots"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kronos_prediction_{self.config['symbol']}_{timestamp}.png"
            filepath = Path(self.config["plot_dir"]) / filename
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            if self.config["verbose"]:
                print(f"✅ Chart saved: {filepath}")
        
        plt.show()
        
        # 清理内存
        del close_preds, volume_preds
        gc.collect()
    
    def run_experiment(self):
        """运行完整实验流程"""
        print("🚀 Kronos Quick Start Experiment")
        print("="*60)
        print(f"📝 Configuration:")
        for key, value in self.config.items():
            if isinstance(value, dict):
                continue
            print(f"   {key}: {value}")
        print("="*60)
        
        try:
            # 1. 加载模型
            self.load_model()
            
            # 2. 获取数据
            data = self.fetch_data()
            
            # 3. 概率预测
            predictions, hist_data = self.make_probabilistic_prediction(data)
            
            # 4. 计算指标
            metrics = self.calculate_metrics(predictions, hist_data)
            
            # 5. 可视化
            self.create_visualization(predictions, hist_data, metrics)
            
            print("🎉 Experiment completed successfully!")
            return predictions, metrics
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            raise

# ============================================================================
# 🎯 快速启动函数
# ============================================================================

def quick_start_btc_hourly():
    """快速启动：BTC小时级预测"""
    config = {
        "data_source": "yfinance",
        "symbol": "BTC-USD",
        "interval": "1h",
        "hist_points": 400,
        "pred_horizon": 24,  # 24小时预测
        "n_predictions": 10,  # 快速测试用较少采样
        "verbose": True
    }
    
    experiment = KronosQuickExperiment(config)
    return experiment.run_experiment()

def quick_start_local_data():
    """快速启动：本地示例数据"""
    config = {
        "data_source": "local_csv",
        "hist_points": 300,
        "pred_horizon": 100,
        "n_predictions": 15,
        "verbose": True
    }
    
    experiment = KronosQuickExperiment(config)
    return experiment.run_experiment()

if __name__ == "__main__":
    print("选择实验模式:")
    print("1. BTC小时级预测 (YFinance)")
    print("2. 本地示例数据预测")
    print("3. 自定义配置")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        quick_start_btc_hourly()
    elif choice == "2":
        quick_start_local_data()
    elif choice == "3":
        experiment = KronosQuickExperiment()
        experiment.run_experiment()
    else:
        print("使用默认配置...")
        quick_start_btc_hourly()
