#!/usr/bin/env python3
"""
Qlib数据接口层
与主项目的finetune数据管理系统集成，提供回测专用的数据访问接口

核心功能：
1. 复用finetune的DataManager和配置系统
2. 提供回测专用的数据预处理
3. 支持A股、美股、港股多市场数据
4. 高效的时间序列数据访问

作者: Kronos Team
日期: 2025-01-26
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加主项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from finetune.config import BaseConfig
    from finetune.data import DataManager
except ImportError as e:
    raise ImportError(f"无法导入主项目模块: {e}. 请确保在Kronos项目根目录运行")

logger = logging.getLogger(__name__)

class QlibDataInterface:
    """
    Qlib数据接口类
    复用主项目的数据管理能力，为回测系统提供统一的数据访问接口
    """
    
    def __init__(self, config=None):
        """初始化数据接口"""
        self.config = config if config is not None else BaseConfig()
        self.data_manager = DataManager(self.config)
        
        # 缓存已加载的数据
        self._data_cache = {}
        self._universe_cache = {}
        
        # 支持的市场和对应的股票池
        self.supported_markets = {
            'csi300': self._get_csi300_symbols(),
            'csi500': self._get_csi500_symbols(), 
            'csi800': self._get_csi800_symbols(),
            'nasdaq100': self._get_nasdaq100_symbols(),
            'sp500': self._get_sp500_symbols(),
            'hsi': self._get_hsi_symbols()
        }
        
        logger.info(f"QlibDataInterface initialized with {len(self.supported_markets)} markets")
    
    def _get_csi300_symbols(self) -> List[str]:
        """获取沪深300成分股代码"""
        # 这里返回一个代表性的CSI300股票列表
        # 在实际应用中，应该从qlib或其他数据源动态获取
        return [
            "000001.SZ", "000002.SZ", "000858.SZ", "000568.SZ", "000725.SZ",
            "002415.SZ", "002594.SZ", "300059.SZ", "300750.SZ", 
            "600000.SH", "600036.SH", "600519.SH", "600887.SH", "601166.SH",
            "601318.SH", "601398.SH", "601857.SH", "603259.SH", "688981.SH"
        ]
    
    def _get_csi500_symbols(self) -> List[str]:
        """获取中证500成分股代码"""
        # 简化版本，实际应该从数据源获取
        return [
            "000063.SZ", "000100.SZ", "000338.SZ", "000425.SZ", "000876.SZ",
            "002024.SZ", "002129.SZ", "002271.SZ", "002352.SZ", "002460.SZ"
        ]
    
    def _get_csi800_symbols(self) -> List[str]:
        """获取中证800成分股代码（CSI300 + CSI500）"""
        return self._get_csi300_symbols() + self._get_csi500_symbols()
    
    def _get_nasdaq100_symbols(self) -> List[str]:
        """获取纳斯达克100成分股代码"""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "ADBE", "PYPL", "INTC", "CMCSA", "PEP", "CSCO", "AVGO", "TXN"
        ]
    
    def _get_sp500_symbols(self) -> List[str]:
        """获取标普500代表性股票代码"""
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "BRK.B", "TSLA", "META", "UNH",
            "JNJ", "XOM", "JPM", "PG", "HD", "CVX", "PFE", "BAC", "ABBV", "KO"
        ]
    
    def _get_hsi_symbols(self) -> List[str]:
        """获取恒生指数成分股代码"""
        return [
            "0700.HK", "9988.HK", "0005.HK", "1299.HK", "2318.HK", "1398.HK",
            "0939.HK", "0388.HK", "3988.HK", "0016.HK", "0175.HK", "0883.HK"
        ]
    
    def get_universe(self, market: str) -> List[str]:
        """
        获取指定市场的股票池
        
        Args:
            market: 市场代码 ('csi300', 'csi500', 'nasdaq100', 'sp500', 'hsi')
            
        Returns:
            股票代码列表
        """
        if market not in self.supported_markets:
            raise ValueError(f"不支持的市场: {market}. 支持的市场: {list(self.supported_markets.keys())}")
        
        return self.supported_markets[market].copy()
    
    def load_market_data(self, 
                        market: str,
                        start_date: str,
                        end_date: str,
                        features: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载指定市场的历史数据
        
        Args:
            market: 市场代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            features: 需要的特征列表，默认使用配置中的feature_list
            
        Returns:
            {symbol: DataFrame} 格式的数据字典
        """
        cache_key = f"{market}_{start_date}_{end_date}"
        
        # 检查缓存
        if cache_key in self._data_cache:
            logger.info(f"从缓存加载 {market} 数据")
            return self._data_cache[cache_key]
        
        # 获取股票池
        symbols = self.get_universe(market)
        
        # 使用特征列表
        if features is None:
            features = self.config.feature_list
        
        logger.info(f"加载 {market} 市场数据: {len(symbols)} 只股票, {start_date} 至 {end_date}")
        
        try:
            # 尝试从实际的qlib数据源加载
            market_data = self._load_real_data(symbols, start_date, end_date, features)
        except Exception as e:
            logger.warning(f"无法加载真实数据 ({e})，使用模拟数据")
            # 回退到模拟数据
            market_data = self._load_mock_data(symbols, start_date, end_date, features)
        
        # 缓存数据
        self._data_cache[cache_key] = market_data
        
        logger.info(f"成功加载 {len(market_data)} 只股票的数据")
        return market_data
    
    def _load_real_data(self, 
                       symbols: List[str], 
                       start_date: str, 
                       end_date: str,
                       features: List[str]) -> Dict[str, pd.DataFrame]:
        """尝试加载真实的qlib数据"""
        try:
            import qlib
            from qlib.data import D
            
            # 检查qlib是否初始化
            if not hasattr(qlib, '_default_config') or qlib._default_config is None:
                logger.info("初始化qlib...")
                qlib.init(provider_uri=self.config.qlib_data_path, region='cn')
            
            # 构建qlib字段映射
            field_mapping = {
                'open': '$open',
                'high': '$high', 
                'low': '$low',
                'close': '$close',
                'vol': '$volume',
                'amt': '$amount'
            }
            
            qlib_fields = [field_mapping.get(f, f) for f in features]
            
            # 从qlib加载数据
            data = D.features(
                instruments=symbols,
                fields=qlib_fields,
                start_time=start_date,
                end_time=end_date,
                freq='day'
            )
            
            if data is None or data.empty:
                raise ValueError("Qlib返回空数据")
            
            # 转换为我们需要的格式
            market_data = {}
            for symbol in symbols:
                try:
                    symbol_data = data.xs(symbol, level='instrument')
                    symbol_data.columns = features  # 重命名列
                    
                    # 基本数据清理
                    symbol_data = symbol_data.fillna(method='ffill').fillna(method='bfill')
                    
                    if not symbol_data.empty:
                        market_data[symbol] = symbol_data
                except (KeyError, Exception) as e:
                    logger.warning(f"跳过股票 {symbol}: {e}")
                    continue
            
            return market_data
            
        except Exception as e:
            raise Exception(f"Qlib数据加载失败: {e}")
    
    def _load_mock_data(self,
                       symbols: List[str],
                       start_date: str,
                       end_date: str, 
                       features: List[str]) -> Dict[str, pd.DataFrame]:
        """生成模拟数据作为后备"""
        logger.info("生成模拟数据...")
        
        # 生成交易日历
        trading_days = self.data_manager.get_trading_calendar(start_date, end_date)
        
        mock_data = {}
        
        for symbol in symbols:
            try:
                # 生成该股票的模拟数据
                stock_data = self.data_manager.generate_realistic_stock_data(
                    symbol=symbol,
                    trading_days=trading_days,
                    base_price_range=self._get_price_range_for_symbol(symbol)
                )
                
                # 确保有所需的特征
                available_features = [f for f in features if f in stock_data.columns]
                if available_features:
                    mock_data[symbol] = stock_data[available_features]
                
            except Exception as e:
                logger.warning(f"跳过模拟数据生成 {symbol}: {e}")
                continue
        
        return mock_data
    
    def _get_price_range_for_symbol(self, symbol: str) -> Tuple[float, float]:
        """根据股票代码返回合理的价格范围"""
        if symbol.endswith('.SZ') or symbol.endswith('.SH'):
            # A股价格范围
            if '60052' in symbol:  # 茅台类高价股
                return (1200, 2500)
            elif '00000' in symbol or '60000' in symbol:  # 银行股
                return (8, 20)
            elif '30' in symbol:  # 创业板
                return (15, 80)
            else:
                return (10, 50)
        elif symbol.endswith('.HK'):
            # 港股价格范围
            return (50, 500)
        else:
            # 美股价格范围
            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                return (100, 300)
            elif symbol in ['TSLA', 'NVDA']:
                return (150, 400)
            else:
                return (50, 200)
    
    def get_returns(self, 
                   data: Dict[str, pd.DataFrame], 
                   periods: int = 1) -> Dict[str, pd.Series]:
        """
        计算股票收益率
        
        Args:
            data: 股票价格数据
            periods: 收益率计算周期，默认1天
            
        Returns:
            {symbol: returns_series} 格式的收益率数据
        """
        returns = {}
        
        for symbol, df in data.items():
            if 'close' in df.columns and len(df) > periods:
                symbol_returns = df['close'].pct_change(periods=periods).dropna()
                returns[symbol] = symbol_returns
        
        return returns
    
    def get_price_matrix(self, 
                        data: Dict[str, pd.DataFrame],
                        price_type: str = 'close') -> pd.DataFrame:
        """
        将股票数据转换为价格矩阵 (时间 x 股票)
        
        Args:
            data: 股票数据字典
            price_type: 价格类型 ('open', 'high', 'low', 'close')
            
        Returns:
            价格矩阵 DataFrame
        """
        if not data:
            return pd.DataFrame()
        
        price_series = {}
        
        for symbol, df in data.items():
            if price_type in df.columns:
                price_series[symbol] = df[price_type]
        
        if not price_series:
            return pd.DataFrame()
        
        # 合并所有价格序列，使用外连接
        price_matrix = pd.DataFrame(price_series)
        
        # 前向填充缺失值
        price_matrix = price_matrix.fillna(method='ffill')
        
        return price_matrix
    
    def validate_data_quality(self, data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """
        验证数据质量
        
        Args:
            data: 股票数据字典
            
        Returns:
            数据质量报告
        """
        return self.data_manager.validate_data_quality(data)
    
    def get_market_info(self) -> Dict[str, Dict]:
        """获取所有支持市场的信息"""
        market_info = {}
        
        for market, symbols in self.supported_markets.items():
            market_info[market] = {
                'name': market.upper(),
                'symbol_count': len(symbols),
                'region': self._get_market_region(market),
                'currency': self._get_market_currency(market),
                'timezone': self._get_market_timezone(market)
            }
        
        return market_info
    
    def _get_market_region(self, market: str) -> str:
        """获取市场地区"""
        if market.startswith('csi'):
            return 'China'
        elif market in ['nasdaq100', 'sp500']:
            return 'US'
        elif market == 'hsi':
            return 'HongKong'
        else:
            return 'Unknown'
    
    def _get_market_currency(self, market: str) -> str:
        """获取市场货币"""
        if market.startswith('csi'):
            return 'CNY'
        elif market in ['nasdaq100', 'sp500']:
            return 'USD'
        elif market == 'hsi':
            return 'HKD'
        else:
            return 'Unknown'
    
    def _get_market_timezone(self, market: str) -> str:
        """获取市场时区"""
        if market.startswith('csi'):
            return 'Asia/Shanghai'
        elif market in ['nasdaq100', 'sp500']:
            return 'America/New_York'
        elif market == 'hsi':
            return 'Asia/Hong_Kong'
        else:
            return 'UTC'