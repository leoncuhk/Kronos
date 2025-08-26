#!/usr/bin/env python3
"""
股票过滤器模块
为Universe Selection提供专业的过滤功能

包含过滤器：
1. LiquidityFilter - 流动性过滤器
2. FundamentalsFilter - 基本面过滤器
3. TechnicalFilter - 技术面过滤器

作者: Kronos Team  
日期: 2025-01-26
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BaseFilter:
    """过滤器基类"""
    
    def __init__(self, name: str):
        self.name = name
        
    def filter(self, 
               data: Dict[str, pd.DataFrame], 
               symbols: List[str],
               as_of_date: str) -> List[str]:
        """
        过滤股票列表
        
        Args:
            data: 股票历史数据
            symbols: 候选股票列表
            as_of_date: 评估日期
            
        Returns:
            过滤后的股票列表
        """
        raise NotImplementedError("子类必须实现filter方法")

class LiquidityFilter(BaseFilter):
    """
    流动性过滤器
    
    过滤标准：
    1. 最小平均成交量
    2. 价格范围限制  
    3. 成交金额要求
    4. 零成交天数限制
    """
    
    def __init__(self,
                 min_avg_volume: float = 1e6,      # 最小日均成交量
                 min_price: float = 2.0,           # 最小股价
                 max_price: float = 1000.0,        # 最大股价
                 min_avg_amount: float = 1e7,      # 最小日均成交金额
                 max_zero_volume_ratio: float = 0.05,  # 最大零成交天数比例
                 lookback_days: int = 60           # 回望天数
                 ):
        super().__init__("LiquidityFilter")
        
        self.min_avg_volume = min_avg_volume
        self.min_price = min_price
        self.max_price = max_price
        self.min_avg_amount = min_avg_amount
        self.max_zero_volume_ratio = max_zero_volume_ratio
        self.lookback_days = lookback_days
        
        logger.info(f"LiquidityFilter initialized: "
                   f"min_volume={min_avg_volume:.0e}, "
                   f"price_range=[{min_price}, {max_price}], "
                   f"min_amount={min_avg_amount:.0e}")
    
    def filter(self, 
               data: Dict[str, pd.DataFrame], 
               symbols: List[str],
               as_of_date: str) -> List[str]:
        """应用流动性过滤器"""
        
        filtered_symbols = []
        eval_date = pd.to_datetime(as_of_date)
        
        for symbol in symbols:
            if symbol not in data:
                continue
                
            try:
                df = data[symbol]
                
                # 获取评估日期之前的数据
                df_before = df[df.index <= eval_date].tail(self.lookback_days)
                
                if len(df_before) < 20:  # 数据不足
                    continue
                
                # 检查流动性指标
                if self._check_liquidity(df_before, symbol):
                    filtered_symbols.append(symbol)
                    
            except Exception as e:
                logger.debug(f"流动性检查失败 {symbol}: {e}")
                continue
        
        logger.info(f"流动性过滤: {len(filtered_symbols)}/{len(symbols)} 股票通过")
        return filtered_symbols
    
    def _check_liquidity(self, df: pd.DataFrame, symbol: str) -> bool:
        """检查单只股票的流动性"""
        
        # 检查必需的列
        required_cols = ['close', 'vol']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # 1. 价格范围检查
        recent_prices = df['close'].tail(10)
        avg_price = recent_prices.mean()
        
        if avg_price < self.min_price or avg_price > self.max_price:
            return False
        
        # 2. 平均成交量检查
        avg_volume = df['vol'].mean()
        if avg_volume < self.min_avg_volume:
            return False
        
        # 3. 平均成交金额检查（如果有成交金额列）
        if 'amt' in df.columns:
            avg_amount = df['amt'].mean()
            if avg_amount < self.min_avg_amount:
                return False
        else:
            # 用价格*成交量估算
            estimated_amount = (df['close'] * df['vol']).mean()
            if estimated_amount < self.min_avg_amount:
                return False
        
        # 4. 零成交天数比例检查
        zero_volume_days = (df['vol'] == 0).sum()
        zero_volume_ratio = zero_volume_days / len(df)
        
        if zero_volume_ratio > self.max_zero_volume_ratio:
            return False
        
        # 5. 价格连续性检查（避免停牌股）
        price_changes = df['close'].pct_change().dropna()
        zero_change_ratio = (price_changes == 0).sum() / len(price_changes)
        
        if zero_change_ratio > 0.3:  # 超过30%的时间价格不变
            return False
        
        return True

class FundamentalsFilter(BaseFilter):
    """
    基本面过滤器
    
    过滤标准：
    1. 最小市值要求
    2. 财务健康度
    3. 上市时间要求
    """
    
    def __init__(self,
                 min_market_cap: float = 1e9,      # 最小市值
                 min_listing_days: int = 252,      # 最小上市天数（约1年）
                 exclude_st_stocks: bool = True    # 排除ST股票
                 ):
        super().__init__("FundamentalsFilter")
        
        self.min_market_cap = min_market_cap
        self.min_listing_days = min_listing_days
        self.exclude_st_stocks = exclude_st_stocks
        
        logger.info(f"FundamentalsFilter initialized: "
                   f"min_market_cap={min_market_cap:.0e}, "
                   f"min_listing_days={min_listing_days}")
    
    def filter(self, 
               data: Dict[str, pd.DataFrame], 
               symbols: List[str],
               as_of_date: str) -> List[str]:
        """应用基本面过滤器"""
        
        filtered_symbols = []
        eval_date = pd.to_datetime(as_of_date)
        
        for symbol in symbols:
            if symbol not in data:
                continue
                
            try:
                df = data[symbol]
                
                # 检查基本面指标
                if self._check_fundamentals(df, symbol, eval_date):
                    filtered_symbols.append(symbol)
                    
            except Exception as e:
                logger.debug(f"基本面检查失败 {symbol}: {e}")
                continue
        
        logger.info(f"基本面过滤: {len(filtered_symbols)}/{len(symbols)} 股票通过")
        return filtered_symbols
    
    def _check_fundamentals(self, df: pd.DataFrame, symbol: str, eval_date: pd.Timestamp) -> bool:
        """检查单只股票的基本面"""
        
        # 1. 上市时间检查
        if len(df) < self.min_listing_days:
            return False
        
        # 2. ST股票过滤（基于股票代码简单判断）
        if self.exclude_st_stocks and 'ST' in symbol.upper():
            return False
        
        # 3. 市值估算检查
        # 注意：这里是简化的市值估算，实际应该用真实的股本数据
        if 'close' in df.columns and 'vol' in df.columns:
            df_recent = df[df.index <= eval_date].tail(20)
            
            if len(df_recent) > 0:
                avg_price = df_recent['close'].mean()
                avg_volume = df_recent['vol'].mean()
                
                # 简化的市值估算：假设总股本 ≈ 平均成交量 * 100
                # 这是一个粗略的估算，实际应该使用真实股本数据
                estimated_shares = avg_volume * 100
                estimated_market_cap = avg_price * estimated_shares
                
                if estimated_market_cap < self.min_market_cap:
                    return False
        
        # 4. 价格异常检查
        if 'close' in df.columns:
            recent_prices = df[df.index <= eval_date]['close'].tail(60)
            
            if len(recent_prices) > 0:
                # 检查是否有异常的价格波动（可能的数据错误）
                price_std = recent_prices.std()
                price_mean = recent_prices.mean()
                
                # 如果波动率异常高，可能是数据问题
                if price_std / price_mean > 2.0:  # 变异系数 > 200%
                    return False
                
                # 检查是否有连续涨跌停（A股特有）
                if symbol.endswith('.SZ') or symbol.endswith('.SH'):
                    daily_returns = recent_prices.pct_change().dropna()
                    extreme_moves = (daily_returns.abs() > 0.095).sum()  # 接近10%涨跌停
                    
                    if extreme_moves > len(daily_returns) * 0.3:  # 超过30%的天数涨跌停
                        return False
        
        return True

class TechnicalFilter(BaseFilter):
    """
    技术面过滤器
    
    过滤标准：
    1. 技术指标异常值
    2. 价格趋势过滤
    3. 动量指标过滤
    """
    
    def __init__(self,
                 min_rsi: float = 20,          # 最小RSI值
                 max_rsi: float = 80,          # 最大RSI值
                 lookback_days: int = 60       # 技术指标计算周期
                 ):
        super().__init__("TechnicalFilter")
        
        self.min_rsi = min_rsi
        self.max_rsi = max_rsi
        self.lookback_days = lookback_days
        
        logger.info(f"TechnicalFilter initialized: "
                   f"RSI_range=[{min_rsi}, {max_rsi}]")
    
    def filter(self, 
               data: Dict[str, pd.DataFrame], 
               symbols: List[str],
               as_of_date: str) -> List[str]:
        """应用技术面过滤器"""
        
        filtered_symbols = []
        eval_date = pd.to_datetime(as_of_date)
        
        for symbol in symbols:
            if symbol not in data:
                continue
                
            try:
                df = data[symbol]
                df_before = df[df.index <= eval_date].tail(self.lookback_days)
                
                # 检查技术指标
                if self._check_technicals(df_before, symbol):
                    filtered_symbols.append(symbol)
                    
            except Exception as e:
                logger.debug(f"技术面检查失败 {symbol}: {e}")
                continue
        
        logger.info(f"技术面过滤: {len(filtered_symbols)}/{len(symbols)} 股票通过")
        return filtered_symbols
    
    def _check_technicals(self, df: pd.DataFrame, symbol: str) -> bool:
        """检查单只股票的技术指标"""
        
        if 'close' not in df.columns or len(df) < 20:
            return False
        
        try:
            # 计算RSI指标
            rsi = self._calculate_rsi(df['close'], period=14)
            
            if len(rsi) > 0:
                latest_rsi = rsi.iloc[-1]
                
                # RSI过滤：避免极端超买超卖
                if latest_rsi < self.min_rsi or latest_rsi > self.max_rsi:
                    return False
            
            # 检查价格趋势（简单移动平均）
            if len(df) >= 20:
                ma_short = df['close'].rolling(5).mean()
                ma_long = df['close'].rolling(20).mean()
                
                if len(ma_short) > 0 and len(ma_long) > 0:
                    # 避免明显的下降趋势
                    if ma_short.iloc[-1] < ma_long.iloc[-1] * 0.95:  # 短期均线比长期均线低5%以上
                        pass  # 可以根据需要调整策略
            
            return True
            
        except Exception as e:
            logger.debug(f"技术指标计算失败 {symbol}: {e}")
            return False
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi