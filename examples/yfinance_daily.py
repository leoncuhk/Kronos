#!/usr/bin/env python3
"""
YFinance BTC日线预测 - 基于Demo框架
针对日线数据优化的参数配置和可视化风格
"""

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from tqdm import trange

# 使用项目本地的model实现
sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor

# 忽略警告
warnings.filterwarnings('ignore')

# === 快速配置：修改这里切换不同标的 ===
TARGET_SYMBOL = 'SPY'      # 支持: 'BTC-USD', 'GC=F', 'SPY', 'AAPL', 'TSLA' 等

# --- 配置：针对日线数据优化 ---
Config = {
    "REPO_PATH": Path(__file__).parent.resolve(),
    "MODEL_PATH": "../Kronos_model",
    "SYMBOL": TARGET_SYMBOL,       # 使用上面配置的标的
    "INTERVAL": '1d',              # 日线数据
    "HIST_POINTS": 400,            # 400个交易日历史数据(约1.5年)
    "PRED_HORIZON": 30,            # 预测30天
    "N_PREDICTIONS": 20,           # 20次采样(日线数据波动较小，适当减少)
    "VOL_WINDOW": 30,              # 30天波动率窗口
}

def load_model():
    """加载Kronos模型和分词器"""
    print("Loading Kronos model...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", cache_dir=Config["MODEL_PATH"])
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", cache_dir=Config["MODEL_PATH"])
    tokenizer.eval()
    model.eval()
    predictor = KronosPredictor(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu", max_context=512)
    print("Model loaded successfully.")
    return predictor

def fetch_yfinance_data():
    """从YFinance获取BTC-USD日线数据"""
    print(f"Fetching {Config['SYMBOL']} {Config['INTERVAL']} data from YFinance...")
    
    try:
        ticker = yf.Ticker(Config["SYMBOL"])
        # 获取2年的日线数据
        data = ticker.history(period="2y", interval=Config["INTERVAL"], auto_adjust=True)
        
        if data.empty:
            raise ValueError("No data retrieved from YFinance")
        
        # 数据格式转换
        data = data.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        data['amount'] = data['close'] * data['volume']
        data.reset_index(inplace=True)
        data.rename(columns={'Date': 'timestamps'}, inplace=True)
        data['timestamps'] = pd.to_datetime(data['timestamps'])
        data = data.dropna()
        
        print("Data fetched successfully.")
        return data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise

def make_prediction(df, predictor):
    """生成日线概率预测"""
    # 只使用最后HIST_POINTS个数据点
    df_recent = df.iloc[-Config["HIST_POINTS"]:]
    
    last_timestamp = df_recent['timestamps'].max()
    start_new_range = last_timestamp + pd.Timedelta(days=1)
    new_timestamps_index = pd.date_range(
        start=start_new_range,
        periods=Config["PRED_HORIZON"],
        freq='D'  # 日线频率
    )
    y_timestamp = pd.Series(new_timestamps_index, name='y_timestamp')
    x_timestamp = df_recent['timestamps']
    x_df = df_recent[['open', 'high', 'low', 'close', 'volume', 'amount']]

    with torch.no_grad():
        print("Making daily prediction (T=1.0)...")
        begin_time = time.time()
        
        close_preds_list = []
        volume_preds_list = []
        
        for i in trange(Config["N_PREDICTIONS"], desc="Monte Carlo Sampling"):
            pred_df = predictor.predict(
                df=x_df, 
                x_timestamp=x_timestamp, 
                y_timestamp=y_timestamp,
                pred_len=Config["PRED_HORIZON"], 
                T=1.0, 
                top_p=0.9,  # 日线数据稍微降低随机性
                sample_count=1, 
                verbose=False
            )
            close_preds_list.append(pred_df['close'])
            volume_preds_list.append(pred_df['volume'])
        
        print(f"Daily prediction completed in {time.time() - begin_time:.2f} seconds.")
        
        # 构造预测数据
        close_preds_df = pd.DataFrame(close_preds_list).T
        volume_preds_df = pd.DataFrame(volume_preds_list).T
        
        return close_preds_df, volume_preds_df

def calculate_metrics(hist_df, close_preds_df, v_close_preds_df):
    """计算日线上涨和波动率放大概率"""
    last_close = hist_df['close'].iloc[-1]

    # 1. 上涨概率 (30天视角)
    final_day_preds = close_preds_df.iloc[-1]
    upside_prob = (final_day_preds > last_close).mean()

    # 2. 波动率放大概率 (30天视角)
    hist_log_returns = np.log(hist_df['close'] / hist_df['close'].shift(1))
    historical_vol = hist_log_returns.iloc[-Config["VOL_WINDOW"]:].std()

    amplification_count = 0
    for col in v_close_preds_df.columns:
        pred_series = v_close_preds_df[col].values.flatten()
        full_sequence = np.concatenate([[last_close], pred_series])
        pred_log_returns = np.log(full_sequence[1:] / full_sequence[:-1])
        predicted_vol = np.std(pred_log_returns)
        if predicted_vol > historical_vol:
            amplification_count += 1

    vol_amp_prob = amplification_count / len(v_close_preds_df.columns)

    print(f"Upside Probability (30d): {upside_prob:.2%}, Volatility Amplification Probability: {vol_amp_prob:.2%}")
    return upside_prob, vol_amp_prob

def create_plot(hist_df, close_preds_df, volume_preds_df):
    """生成日线预测图表"""
    print("Generating daily forecast chart...")
    
    # 日线图表设置
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    hist_time = hist_df['timestamps']
    last_hist_time = hist_time.iloc[-1]
    pred_time = pd.to_datetime([last_hist_time + timedelta(days=i + 1) for i in range(len(close_preds_df))])

    # === 价格预测图 ===
    ax1.plot(hist_time, hist_df['close'], color='royalblue', label='Historical Price', linewidth=1.5)
    mean_preds = close_preds_df.mean(axis=1)
    ax1.plot(pred_time, mean_preds, color='darkorange', linestyle='-', label='Mean Forecast')
    ax1.fill_between(pred_time, close_preds_df.min(axis=1), close_preds_df.max(axis=1), 
                    color='darkorange', alpha=0.2, label='Forecast Range (Min-Max)')
    ax1.set_title(f'{Config["SYMBOL"]} Probabilistic Price & Volume Forecast (Next {Config["PRED_HORIZON"]} Days)', 
                 fontsize=16, weight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # === 交易量图 ===
    # 日线数据使用更宽的柱状图
    bar_width = 0.8  # 日线用更宽的柱子
    ax2.bar(hist_time, hist_df['volume'], color='skyblue', label='Historical Volume', width=bar_width)
    ax2.bar(pred_time, volume_preds_df.mean(axis=1), color='sandybrown', label='Mean Forecasted Volume', width=bar_width)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 分隔线
    separator_time = hist_time.iloc[-1] + timedelta(hours=12)  # 日线数据用12小时偏移
    for ax in [ax1, ax2]:
        ax.axvline(x=separator_time, color='red', linestyle='--', linewidth=1.5, label='_nolegend_')
        ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    chart_filename = f'yfinance_daily_forecast_{TARGET_SYMBOL.replace("=", "_").replace("-", "_")}.png'
    chart_path = Config["REPO_PATH"] / chart_filename
    fig.savefig(chart_path, dpi=120)
    plt.show()
    print(f"Chart saved to: {chart_path}")

def main_task():
    """执行完整的日线预测任务"""
    print("\n" + "=" * 60 + f"\nStarting YFinance daily prediction at {datetime.now()}\n" + "=" * 60)
    
    # 1. 加载模型
    predictor = load_model()
    
    # 2. 获取数据
    df_full = fetch_yfinance_data()
    # 直接使用完整数据，make_prediction函数内部会选择合适的数据范围

    # 3. 生成预测
    close_preds, volume_preds = make_prediction(df_full, predictor)

    # 4. 准备绘图数据 - 使用与预测相同的历史数据范围
    hist_df_for_plot = df_full.tail(Config["HIST_POINTS"])
    hist_df_for_metrics = df_full.tail(Config["VOL_WINDOW"])

    # 5. 计算指标
    upside_prob, vol_amp_prob = calculate_metrics(hist_df_for_metrics, close_preds, close_preds)
    
    # 6. 生成图表
    create_plot(hist_df_for_plot, close_preds, volume_preds)

    # 7. 结果总结
    print("\n" + "=" * 60)
    print("🎯 YFinance Daily Prediction Results:")
    print(f"📊 Symbol: {Config['SYMBOL']} (YFinance)")
    print(f"⏱️  Data: {Config['INTERVAL']} interval, {Config['HIST_POINTS']} historical points")
    print(f"🔮 Forecast: {Config['PRED_HORIZON']} days, {Config['N_PREDICTIONS']} Monte Carlo samples")
    print(f"🚀 Upside Probability (30d): {upside_prob:.1%}")
    print(f"📈 Volatility Amplification: {vol_amp_prob:.1%}")
    print(f"💰 Current Price: ${hist_df_for_plot['close'].iloc[-1]:.2f}")
    print(f"🎯 Mean Forecast: ${close_preds.mean().mean():.2f}")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    try:
        main_task()
        print("✅ YFinance daily prediction completed successfully! 🎉")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
