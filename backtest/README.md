# Kronos 量化回测框架

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/framework-Qlib%20Compatible-orange.svg)](https://github.com/microsoft/qlib)

基于四步量化交易法的专业回测框架，将Kronos定位为"概率优势排序引擎"而非单点预测器。

## 🎯 核心理念

### 四步量化交易法
```
1. Universe Selection  → 选择广泛、流动性好的股票池
2. Alpha Generation    → 使用Kronos生成概率化Alpha信号  
3. Portfolio Construction → 构建风险调整后的投资组合
4. Risk Management     → 实时风险监控和交易执行
```

### Kronos作为概率排序引擎
- **多路径采样**：Monte Carlo方法获得预测分布
- **不确定性量化**：提供置信区间而非单点预测
- **风险调整信号**：预期收益率 / 预期波动率
- **系统性排序**：专注相对排序而非绝对预测

## 🏗️ 系统架构

```
backtest/
├── engine.py              # 核心回测引擎
├── example.py              # 使用示例
├── data/                   # 数据接口层
│   ├── qlib_interface.py   # Qlib数据接口
│   └── __init__.py
├── universe/               # 交易域选择
│   ├── universe_selector.py # 股票池选择器
│   ├── filters.py          # 过滤器集合
│   └── __init__.py
├── signals/                # Alpha信号生成
│   ├── kronos_alpha.py     # Kronos信号生成器
│   ├── signal_processor.py # 信号处理器
│   └── __init__.py
├── portfolio/              # 组合构建
│   ├── portfolio_constructor.py # 组合构建器
│   ├── rebalancer.py       # 再平衡器
│   ├── risk_budgeter.py    # 风险预算器
│   └── __init__.py
├── risk_management/        # 风险管理
│   ├── risk_manager.py     # 风险管理器
│   ├── position_sizer.py   # 头寸规模器
│   ├── execution_manager.py # 执行管理器
│   └── __init__.py
└── README.md              # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```python
# 确保已安装项目依赖
pip install pandas numpy scipy scikit-learn torch qlib
```

### 2. 基础回测示例

```python
from backtest import BacktestEngine, BacktestConfig

# 创建回测配置
config = BacktestConfig(
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=1_000_000,
    
    # Kronos Alpha配置
    signal_config={
        'prediction_horizon': 10,    # 预测10天收益
        'sample_count': 20,          # 20次Monte Carlo采样
        'risk_adjustment': True      # 启用风险调整
    },
    
    # 组合配置
    portfolio_config={
        'max_positions': 50,
        'construction_method': 'signal_weighted'
    }
)

# 运行回测
engine = BacktestEngine(config=config)
result = engine.run_backtest()

# 显示结果
engine.print_summary(result)
```

### 3. 运行完整示例

```bash
cd backtest/
python example.py
```

## 📊 核心模块详解

### 1. 数据接口 (Data Interface)
- **QlibDataInterface**: 与主项目qlib数据基础设施集成
- 统一数据访问接口，支持多市场
- 自动数据预处理和清洗
- 与finetune模块共享数据管道

### 2. 交易域选择 (Universe Selection)
- **UniverseSelector**: 专业股票池选择器
- **LiquidityFilter**: 流动性过滤
- **FundamentalsFilter**: 基本面过滤
- **TechnicalFilter**: 技术面过滤

### 3. Alpha信号生成 (Signal Generation)
- **KronosAlphaGenerator**: 核心Kronos信号引擎
- Monte Carlo预测采样获得分布
- 风险调整评分：预期收益/预期波动率
- **SignalProcessor**: 信号质量控制和处理

### 4. 组合构建 (Portfolio Construction)
- **PortfolioConstructor**: 专业组合构建器
- **Rebalancer**: 智能再平衡管理
- **RiskBudgeter**: 风险预算分配
- 支持多种构建方法

### 5. 风险管理 (Risk Management)
- **RiskManager**: 实时风险监控
- **PositionSizer**: 基于风险的头寸规模
- **ExecutionManager**: 智能交易执行
- 多层风险控制体系

## 📈 回测结果分析

### 核心绩效指标
```python
result.total_return          # 总收益率
result.annualized_return     # 年化收益率
result.sharpe_ratio         # 夏普比率
result.max_drawdown         # 最大回撤
result.alpha               # 超额收益
result.information_ratio   # 信息比率
```

## ⚙️ 高级配置

### 策略参数调优
```python
# Kronos Alpha参数
signal_config = {
    'lookback_window': 90,        # 历史数据回望期
    'prediction_horizon': 10,     # 预测时间范围
    'sample_count': 20,           # Monte Carlo采样数
    'temperature': 0.8,           # 采样温度参数
    'risk_adjustment': True       # 风险调整开关
}

# 组合构建参数  
portfolio_config = {
    'max_positions': 50,              # 最大持仓数
    'max_weight_per_stock': 0.05,    # 单股最大权重
    'construction_method': 'signal_weighted'
}

# 风险管理参数
risk_config = {
    'max_portfolio_volatility': 0.20,    # 组合波动率上限
    'max_drawdown_limit': 0.08,          # 最大回撤限制
    'stop_loss_threshold': -0.05         # 止损阈值
}
```

## 📚 最佳实践

### 1. 参数优化建议
- **信号参数**: prediction_horizon=5-20天, sample_count=15-30次
- **组合参数**: max_positions=30-100只, max_weight=3%-10%
- **风险参数**: max_volatility=12%-25%, stop_loss=-3%到-8%

### 2. 回测陷阱避免
- 避免前瞻偏差和数据泄露
- 充分考虑交易成本
- 严格时间序列分割
- 样本外验证防止过拟合

## 🔧 扩展开发

支持自定义信号生成器、组合构建方法和风险指标。框架采用模块化设计，便于扩展和定制。

## 📄 许可证

MIT License

---

**Kronos Team** 🚀 *让量化投资更智能*