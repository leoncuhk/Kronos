#!/usr/bin/env python3
"""
交易域选择器 (Universe Selector)
量化交易四步法的第一步：定义交易域

核心功能：
1. 选择广泛、流动性好的股票池
2. 应用流动性和基本面过滤器
3. 确保策略容量和统计意义
4. 支持多市场股票池管理

原则：
- 必须在广泛的股票池中操作，以发挥排序的统计意义
- 设定流动性过滤规则，确保可交易性
- 动态调整股票池，剔除不符合条件的股票

作者: Kronos Team
日期: 2025-01-26
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ..data import QlibDataInterface
from .filters import LiquidityFilter, FundamentalsFilter

logger = logging.getLogger(__name__)

class UniverseSelector:
    """
    交易域选择器
    
    核心职责：
    1. 管理多市场股票池
    2. 应用动态过滤器
    3. 确保策略容量和统计意义
    4. 提供时间序列的股票池变化
    """
    
    def __init__(self, 
                 data_interface: QlibDataInterface,
                 min_market_cap: float = 1e9,      # 最小市值（元或美元）
                 min_avg_volume: float = 1e6,      # 最小日均成交量
                 min_price: float = 2.0,           # 最小股价
                 max_price: float = 1000.0,        # 最大股价
                 liquidity_lookback: int = 60,     # 流动性计算回望期
                 rebalance_freq: str = 'monthly'   # 股票池调整频率
                 ):
        """
        初始化交易域选择器
        
        Args:
            data_interface: 数据接口
            min_market_cap: 最小市值过滤阈值
            min_avg_volume: 最小平均成交量
            min_price: 最小价格
            max_price: 最大价格  
            liquidity_lookback: 流动性计算的回望天数
            rebalance_freq: 股票池重新平衡频率 ('daily', 'weekly', 'monthly', 'quarterly')
        """
        self.data_interface = data_interface
        
        # 过滤器参数
        self.min_market_cap = min_market_cap
        self.min_avg_volume = min_avg_volume
        self.min_price = min_price
        self.max_price = max_price
        self.liquidity_lookback = liquidity_lookback
        self.rebalance_freq = rebalance_freq
        
        # 初始化过滤器
        self.liquidity_filter = LiquidityFilter(
            min_avg_volume=min_avg_volume,
            min_price=min_price,
            max_price=max_price,
            lookback_days=liquidity_lookback
        )
        
        self.fundamentals_filter = FundamentalsFilter(
            min_market_cap=min_market_cap
        )
        
        # 缓存已选择的股票池
        self._universe_cache = {}
        
        logger.info(f"UniverseSelector initialized with filters: "
                   f"min_market_cap={min_market_cap:.0e}, "
                   f"min_avg_volume={min_avg_volume:.0e}, "
                   f"price_range=[{min_price}, {max_price}]")
    
    def select_universe(self, 
                       market: str,
                       date: str,
                       base_universe: Optional[List[str]] = None) -> List[str]:
        """
        选择指定日期的交易域
        
        Args:
            market: 市场代码 ('csi300', 'csi500', 'nasdaq100', 'sp500')
            date: 选择日期 (YYYY-MM-DD)
            base_universe: 基础股票池，如果为None则使用市场默认池
            
        Returns:
            符合条件的股票代码列表
        """
        cache_key = f"{market}_{date}_{self.rebalance_freq}"
        
        # 检查缓存
        if cache_key in self._universe_cache:
            return self._universe_cache[cache_key]
        
        # 获取基础股票池
        if base_universe is None:
            base_universe = self.data_interface.get_universe(market)
        
        logger.info(f"为 {date} 选择 {market} 交易域，基础池大小: {len(base_universe)}")
        
        # 计算数据加载的时间范围（需要历史数据来计算过滤指标）
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=self.liquidity_lookback + 30)  # 额外缓冲
        
        # 加载历史数据
        try:
            historical_data = self.data_interface.load_market_data(
                market=market,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            return base_universe  # 如果数据加载失败，返回原始股票池
        
        # 只保留基础股票池中有数据的股票
        available_symbols = [s for s in base_universe if s in historical_data]
        
        # 应用流动性过滤器
        liquid_symbols = self.liquidity_filter.filter(
            data=historical_data,
            symbols=available_symbols,
            as_of_date=date
        )
        
        logger.info(f"流动性过滤后: {len(liquid_symbols)}/{len(available_symbols)} 股票")
        
        # 应用基本面过滤器（如果有基本面数据）
        try:
            final_symbols = self.fundamentals_filter.filter(
                data=historical_data,
                symbols=liquid_symbols,
                as_of_date=date
            )
            logger.info(f"基本面过滤后: {len(final_symbols)}/{len(liquid_symbols)} 股票")
        except Exception as e:
            logger.warning(f"基本面过滤失败，跳过: {e}")
            final_symbols = liquid_symbols
        
        # 按市值或流动性排序，选择Top N
        final_symbols = self._rank_and_select(
            symbols=final_symbols,
            data=historical_data,
            date=date,
            max_symbols=self._get_max_symbols_for_market(market)
        )
        
        # 缓存结果
        self._universe_cache[cache_key] = final_symbols
        
        logger.info(f"最终选择 {len(final_symbols)} 只股票用于 {market} {date}")
        return final_symbols
    
    def _rank_and_select(self,
                        symbols: List[str],
                        data: Dict[str, pd.DataFrame],
                        date: str,
                        max_symbols: int) -> List[str]:
        """
        根据流动性和市值指标排序并选择Top N股票
        
        Args:
            symbols: 候选股票列表
            data: 历史数据
            date: 评估日期
            max_symbols: 最大选择数量
            
        Returns:
            排序后的Top N股票列表
        """
        if len(symbols) <= max_symbols:
            return symbols
        
        # 计算每只股票的评分指标
        stock_scores = []
        eval_date = pd.to_datetime(date)
        
        for symbol in symbols:
            try:
                df = data[symbol]
                
                # 只取评估日期之前的数据
                df_before = df[df.index <= eval_date].tail(self.liquidity_lookback)
                
                if len(df_before) < 10:  # 数据不足
                    continue
                
                # 计算评分指标
                avg_volume = df_before['vol'].mean() if 'vol' in df_before.columns else 0
                avg_amount = df_before['amt'].mean() if 'amt' in df_before.columns else 0
                price_stability = 1.0 / (df_before['close'].std() / df_before['close'].mean() + 1e-6) if 'close' in df_before.columns else 0
                
                # 综合评分（流动性权重更高）
                score = (
                    0.5 * avg_volume +      # 成交量
                    0.3 * avg_amount +      # 成交额  
                    0.2 * price_stability   # 价格稳定性
                )
                
                stock_scores.append((symbol, score))
                
            except Exception as e:
                logger.debug(f"计算 {symbol} 评分失败: {e}")
                continue
        
        # 按评分排序，选择Top N
        stock_scores.sort(key=lambda x: x[1], reverse=True)
        selected_symbols = [symbol for symbol, score in stock_scores[:max_symbols]]
        
        logger.info(f"从 {len(symbols)} 只股票中选择评分最高的 {len(selected_symbols)} 只")
        return selected_symbols
    
    def _get_max_symbols_for_market(self, market: str) -> int:
        """获取不同市场的最大股票数量限制"""
        market_limits = {
            'csi300': 100,    # CSI300 选择100只
            'csi500': 150,    # CSI500 选择150只  
            'csi800': 200,    # CSI800 选择200只
            'nasdaq100': 80,  # NASDAQ100 选择80只
            'sp500': 150,     # S&P500 选择150只
            'hsi': 30         # 恒生指数选择30只
        }
        return market_limits.get(market, 50)  # 默认50只
    
    def get_universe_changes(self,
                           market: str, 
                           start_date: str,
                           end_date: str) -> Dict[str, Dict[str, List[str]]]:
        """
        获取指定时间段内股票池的变化情况
        
        Args:
            market: 市场代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            {date: {'added': [...], 'removed': [...]}} 格式的变化记录
        """
        rebalance_dates = self._get_rebalance_dates(start_date, end_date)
        
        universe_changes = {}
        previous_universe = set()
        
        for i, date in enumerate(rebalance_dates):
            date_str = date.strftime('%Y-%m-%d')
            current_universe = set(self.select_universe(market, date_str))
            
            if i == 0:
                # 第一次，所有股票都是新增的
                universe_changes[date_str] = {
                    'added': list(current_universe),
                    'removed': []
                }
            else:
                # 计算增加和移除的股票
                added = list(current_universe - previous_universe)
                removed = list(previous_universe - current_universe)
                
                universe_changes[date_str] = {
                    'added': added,
                    'removed': removed
                }
                
                if added or removed:
                    logger.info(f"{date_str}: 新增 {len(added)} 只，移除 {len(removed)} 只股票")
            
            previous_universe = current_universe
        
        return universe_changes
    
    def _get_rebalance_dates(self, start_date: str, end_date: str) -> List[pd.Timestamp]:
        """
        获取股票池重新平衡日期
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            重新平衡日期列表
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if self.rebalance_freq == 'daily':
            freq = 'D'
        elif self.rebalance_freq == 'weekly':
            freq = 'W-MON'  # 每周一
        elif self.rebalance_freq == 'monthly':
            freq = 'MS'     # 每月第一天
        elif self.rebalance_freq == 'quarterly':
            freq = 'QS'     # 每季度第一天
        else:
            freq = 'MS'     # 默认月度
        
        dates = pd.date_range(start=start, end=end, freq=freq)
        return dates.tolist()
    
    def get_universe_statistics(self, market: str, date: str) -> Dict[str, any]:
        """
        获取股票池统计信息
        
        Args:
            market: 市场代码
            date: 统计日期
            
        Returns:
            股票池统计信息字典
        """
        universe = self.select_universe(market, date)
        
        # 加载数据计算统计信息
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=30)
        
        try:
            data = self.data_interface.load_market_data(
                market=market,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # 计算统计指标
            total_market_value = 0
            total_avg_volume = 0
            price_ranges = []
            
            for symbol in universe:
                if symbol in data:
                    df = data[symbol]
                    recent_data = df[df.index <= end_date].tail(20)  # 最近20天
                    
                    if len(recent_data) > 0 and 'close' in recent_data.columns:
                        # 估算市值（简化计算）
                        avg_price = recent_data['close'].mean()
                        avg_volume = recent_data['vol'].mean() if 'vol' in recent_data.columns else 0
                        estimated_market_value = avg_price * avg_volume * 10  # 简化估算
                        
                        total_market_value += estimated_market_value
                        total_avg_volume += avg_volume
                        price_ranges.append(avg_price)
            
            statistics = {
                'universe_size': len(universe),
                'market': market,
                'date': date,
                'estimated_total_market_value': total_market_value,
                'average_daily_volume': total_avg_volume / max(len(universe), 1),
                'price_statistics': {
                    'min': min(price_ranges) if price_ranges else 0,
                    'max': max(price_ranges) if price_ranges else 0,
                    'median': np.median(price_ranges) if price_ranges else 0
                }
            }
            
        except Exception as e:
            logger.warning(f"无法计算统计信息: {e}")
            statistics = {
                'universe_size': len(universe),
                'market': market,
                'date': date,
                'error': str(e)
            }
        
        return statistics