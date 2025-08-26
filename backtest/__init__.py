#!/usr/bin/env python3
"""
Kronos量化交易回测框架

基于四步量化交易法的专业回测系统：
1. Universe Selection - 交易域选择
2. Alpha Signal Generation - Alpha信号生成  
3. Portfolio Construction - 投资组合构建
4. Risk Management & Execution - 风险管理与执行

核心模块：
- data: 数据接口和管理
- universe: 股票池选择和过滤
- signals: Alpha信号生成和处理
- portfolio: 组合构建和再平衡
- risk_management: 风险管理和执行
- engine: 回测引擎

作者: Kronos Team
日期: 2025-01-26
"""

from .engine import BacktestEngine, BacktestConfig
from .data import QlibDataInterface
from .universe import UniverseSelector, LiquidityFilter, FundamentalsFilter
from .signals import KronosAlphaGenerator, SignalProcessor
from .portfolio import PortfolioConstructor, Rebalancer, RiskBudgeter
from .risk_management import RiskManager, PositionSizer, ExecutionManager

__version__ = "1.0.0"
__author__ = "Kronos Team"

__all__ = [
    # 核心引擎
    'BacktestEngine',
    'BacktestConfig',
    
    # 数据模块
    'QlibDataInterface',
    
    # 交易域选择
    'UniverseSelector', 
    'LiquidityFilter', 
    'FundamentalsFilter',
    
    # 信号生成
    'KronosAlphaGenerator', 
    'SignalProcessor',
    
    # 组合构建
    'PortfolioConstructor', 
    'Rebalancer', 
    'RiskBudgeter',
    
    # 风险管理
    'RiskManager', 
    'PositionSizer', 
    'ExecutionManager'
]