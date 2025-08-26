#!/usr/bin/env python3
"""
组合构建器 (Portfolio Constructor)
量化交易四步法的第三步：将Alpha信号转换为实际投资组合

核心功能：
1. 信号到权重的映射策略
2. 风险预算和约束处理
3. 多种组合构建方法（等权重、信号加权、风险平价等）
4. 交易成本和冲击成本考虑

设计原则：
- 风险可控：确保组合风险在预期范围内
- 成本有效：最小化不必要的交易成本
- 稳健性：对信号噪音和市场变化有较强适应性

作者: Kronos Team
日期: 2025-01-26
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class PortfolioConstructor:
    """
    投资组合构建器
    
    核心职责：
    1. 将Alpha信号转换为投资组合权重
    2. 应用风险约束和预算限制
    3. 优化组合构建以降低交易成本
    4. 支持多种组合构建策略
    """
    
    def __init__(self,
                 # 基础参数
                 portfolio_value: float = 1e6,          # 组合总价值
                 max_positions: int = 50,               # 最大持仓数量
                 
                 # 权重约束
                 max_weight_per_stock: float = 0.05,    # 单只股票最大权重
                 min_weight_threshold: float = 0.01,    # 最小权重阈值
                 sector_max_weight: float = 0.3,        # 单个行业最大权重
                 
                 # 构建方法
                 construction_method: str = "signal_weighted",  # 构建方法
                 long_only: bool = True,                # 是否仅做多
                 
                 # 风险控制
                 target_volatility: Optional[float] = 0.15,     # 目标波动率
                 max_tracking_error: Optional[float] = 0.08,    # 最大跟踪误差
                 
                 # 交易成本
                 transaction_cost_rate: float = 0.002,  # 交易成本费率
                 market_impact_coeff: float = 0.001,    # 市场冲击系数
                 
                 # 优化参数
                 risk_aversion: float = 2.0,            # 风险厌恶系数
                 turnover_penalty: float = 0.01         # 换手率惩罚系数
                 ):
        """
        初始化组合构建器
        
        Args:
            portfolio_value: 组合总价值
            max_positions: 最大持仓股票数
            max_weight_per_stock: 单只股票最大权重
            min_weight_threshold: 权重过滤阈值
            sector_max_weight: 行业集中度限制
            construction_method: 构建方法 ('equal_weight', 'signal_weighted', 'risk_parity', 'mean_variance')
            long_only: 是否只做多头
            target_volatility: 目标组合波动率
            max_tracking_error: 最大跟踪误差
            transaction_cost_rate: 交易费用比率
            market_impact_coeff: 市场冲击成本系数
            risk_aversion: 风险厌恶参数
            turnover_penalty: 换手率惩罚系数
        """
        self.portfolio_value = portfolio_value
        self.max_positions = max_positions
        
        # 权重约束
        self.max_weight_per_stock = max_weight_per_stock
        self.min_weight_threshold = min_weight_threshold
        self.sector_max_weight = sector_max_weight
        
        # 构建方法
        self.construction_method = construction_method
        self.long_only = long_only
        
        # 风险控制
        self.target_volatility = target_volatility
        self.max_tracking_error = max_tracking_error
        
        # 成本模型
        self.transaction_cost_rate = transaction_cost_rate
        self.market_impact_coeff = market_impact_coeff
        
        # 优化参数
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        
        # 内部状态
        self._current_positions = {}
        self._covariance_matrix = None
        self._expected_returns = None
        
        logger.info(f"PortfolioConstructor initialized: "
                   f"method={construction_method}, "
                   f"max_positions={max_positions}, "
                   f"portfolio_value={portfolio_value:.0e}")
    
    def construct_portfolio(self,
                          signals_df: pd.DataFrame,
                          market_data: Dict[str, pd.DataFrame],
                          current_positions: Optional[Dict[str, float]] = None,
                          benchmark_weights: Optional[Dict[str, float]] = None) -> Dict[str, any]:
        """
        构建投资组合
        
        Args:
            signals_df: 处理后的Alpha信号数据
            market_data: 市场历史数据（用于风险估计）
            current_positions: 当前持仓 {symbol: weight}
            benchmark_weights: 基准权重（用于跟踪误差控制）
            
        Returns:
            组合信息字典：
            - target_weights: 目标权重
            - trade_instructions: 交易指令
            - portfolio_metrics: 组合指标
            - construction_info: 构建信息
        """
        if signals_df.empty:
            logger.warning("信号数据为空，无法构建组合")
            return self._empty_portfolio()
        
        logger.info(f"构建投资组合，信号数量: {len(signals_df)}")
        
        # 1. 更新当前持仓
        if current_positions:
            self._current_positions = current_positions
        
        # 2. 预处理信号和数据
        processed_signals = self._preprocess_signals(signals_df)
        
        # 3. 估计风险和收益参数
        self._estimate_risk_return_params(processed_signals, market_data)
        
        # 4. 根据构建方法生成权重
        target_weights = self._generate_portfolio_weights(
            processed_signals, benchmark_weights
        )
        
        # 5. 应用约束和优化
        optimized_weights = self._apply_constraints_and_optimize(
            target_weights, processed_signals, benchmark_weights
        )
        
        # 6. 生成交易指令
        trade_instructions = self._generate_trade_instructions(optimized_weights)
        
        # 7. 计算组合指标
        portfolio_metrics = self._calculate_portfolio_metrics(
            optimized_weights, processed_signals
        )
        
        # 8. 构建结果
        result = {
            'target_weights': optimized_weights,
            'trade_instructions': trade_instructions,
            'portfolio_metrics': portfolio_metrics,
            'construction_info': {
                'method': self.construction_method,
                'signal_count': len(processed_signals),
                'position_count': len([w for w in optimized_weights.values() if abs(w) > self.min_weight_threshold]),
                'gross_exposure': sum(abs(w) for w in optimized_weights.values()),
                'net_exposure': sum(optimized_weights.values())
            }
        }
        
        logger.info(f"组合构建完成: {len(optimized_weights)} 持仓, "
                   f"总曝险: {result['construction_info']['gross_exposure']:.2%}")
        
        return result
    
    def _empty_portfolio(self) -> Dict[str, any]:
        """返回空组合"""
        return {
            'target_weights': {},
            'trade_instructions': [],
            'portfolio_metrics': {},
            'construction_info': {'method': self.construction_method, 'signal_count': 0}
        }
    
    def _preprocess_signals(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """预处理Alpha信号"""
        
        processed_df = signals_df.copy()
        
        # 1. 过滤信号质量低的股票
        if 'signal_quality' in processed_df.columns:
            processed_df = processed_df[processed_df['signal_quality'] > 0.3]
        
        # 2. 选择Top N信号（基于最终权重）
        if len(processed_df) > self.max_positions:
            weight_col = 'final_weight_normalized' if 'final_weight_normalized' in processed_df.columns else 'risk_adjusted_score'
            processed_df = processed_df.nlargest(self.max_positions, weight_col)
        
        # 3. 标准化信号强度
        if 'final_weight_normalized' in processed_df.columns:
            signal_col = 'final_weight_normalized'
        else:
            signal_col = 'risk_adjusted_score'
            
        if signal_col in processed_df.columns:
            # 标准化到[-1, 1]范围
            max_abs_signal = processed_df[signal_col].abs().max()
            if max_abs_signal > 0:
                processed_df['normalized_signal'] = processed_df[signal_col] / max_abs_signal
            else:
                processed_df['normalized_signal'] = 0.0
        
        logger.info(f"信号预处理完成: {len(processed_df)} 个有效信号")
        return processed_df
    
    def _estimate_risk_return_params(self, 
                                   signals_df: pd.DataFrame,
                                   market_data: Dict[str, pd.DataFrame]):
        """估计风险收益参数"""
        
        symbols = signals_df['symbol'].tolist()
        
        try:
            # 计算预期收益（来自Alpha信号）
            self._expected_returns = {}
            for idx, row in signals_df.iterrows():
                symbol = row['symbol']
                expected_return = row.get('expected_return', 0.0)
                self._expected_returns[symbol] = expected_return
            
            # 估计协方差矩阵
            self._covariance_matrix = self._estimate_covariance_matrix(symbols, market_data)
            
            logger.info("风险收益参数估计完成")
            
        except Exception as e:
            logger.warning(f"风险收益参数估计失败: {e}")
            # 使用默认参数
            self._expected_returns = {s: 0.0 for s in symbols}
            n = len(symbols)
            self._covariance_matrix = pd.DataFrame(
                np.eye(n) * 0.04,  # 假设20%的年化波动率
                index=symbols,
                columns=symbols
            )
    
    def _estimate_covariance_matrix(self, 
                                  symbols: List[str],
                                  market_data: Dict[str, pd.DataFrame],
                                  lookback_days: int = 252) -> pd.DataFrame:
        """估计协方差矩阵"""
        
        returns_data = {}
        
        # 计算各股票的收益率
        for symbol in symbols:
            if symbol in market_data:
                df = market_data[symbol]
                if 'close' in df.columns and len(df) > 1:
                    returns = df['close'].pct_change().dropna()
                    if len(returns) >= 20:  # 至少需要20个数据点
                        returns_data[symbol] = returns.tail(lookback_days)
        
        if len(returns_data) < 2:
            # 如果数据不足，使用单位矩阵
            n = len(symbols)
            return pd.DataFrame(
                np.eye(n) * 0.04,  # 假设日波动率2%
                index=symbols,
                columns=symbols
            )
        
        # 对齐时间序列
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            # 备用方案
            n = len(symbols)
            return pd.DataFrame(
                np.eye(n) * 0.04,
                index=symbols,
                columns=symbols
            )
        
        # 计算协方差矩阵（年化）
        cov_matrix = returns_df.cov() * 252  # 假设252个交易日
        
        # 确保包含所有symbols
        missing_symbols = set(symbols) - set(cov_matrix.index)
        for symbol in missing_symbols:
            # 对缺失股票，使用平均方差
            avg_var = cov_matrix.values.diagonal().mean() if not cov_matrix.empty else 0.04
            cov_matrix.loc[symbol, :] = 0.0
            cov_matrix.loc[:, symbol] = 0.0
            cov_matrix.loc[symbol, symbol] = avg_var
        
        # 重新排序以匹配symbols顺序
        cov_matrix = cov_matrix.loc[symbols, symbols]
        
        return cov_matrix
    
    def _generate_portfolio_weights(self,
                                  signals_df: pd.DataFrame,
                                  benchmark_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """根据构建方法生成组合权重"""
        
        if self.construction_method == "equal_weight":
            return self._equal_weight_portfolio(signals_df)
        elif self.construction_method == "signal_weighted":
            return self._signal_weighted_portfolio(signals_df)
        elif self.construction_method == "risk_parity":
            return self._risk_parity_portfolio(signals_df)
        elif self.construction_method == "mean_variance":
            return self._mean_variance_portfolio(signals_df, benchmark_weights)
        else:
            logger.warning(f"未知构建方法: {self.construction_method}，使用信号加权")
            return self._signal_weighted_portfolio(signals_df)
    
    def _equal_weight_portfolio(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """等权重组合"""
        symbols = signals_df['symbol'].tolist()
        equal_weight = 1.0 / len(symbols)
        return {symbol: equal_weight for symbol in symbols}
    
    def _signal_weighted_portfolio(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """信号加权组合"""
        weights = {}
        
        # 使用归一化信号作为权重
        if 'normalized_signal' in signals_df.columns:
            signal_col = 'normalized_signal'
        else:
            signal_col = 'risk_adjusted_score'
        
        total_abs_signal = signals_df[signal_col].abs().sum()
        
        if total_abs_signal > 0:
            for idx, row in signals_df.iterrows():
                symbol = row['symbol']
                signal = row[signal_col]
                
                if self.long_only:
                    # 仅做多：负信号设为0
                    weight = max(0, signal) / total_abs_signal
                else:
                    # 多空：保持信号方向
                    weight = signal / total_abs_signal
                
                weights[symbol] = weight
        else:
            # 如果所有信号都为0，使用等权重
            return self._equal_weight_portfolio(signals_df)
        
        # 标准化权重
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _risk_parity_portfolio(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """风险平价组合"""
        symbols = signals_df['symbol'].tolist()
        
        if self._covariance_matrix is None or len(symbols) < 2:
            return self._equal_weight_portfolio(signals_df)
        
        try:
            # 风险平价：每个股票的风险贡献相等
            cov_matrix = self._covariance_matrix.loc[symbols, symbols]
            
            # 计算风险预算权重（逆波动率加权）
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol_weights = 1.0 / volatilities
            inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
            
            weights = dict(zip(symbols, inv_vol_weights))
            
            # 结合信号调整（信号作为倾斜因子）
            if 'normalized_signal' in signals_df.columns:
                signal_dict = dict(zip(signals_df['symbol'], signals_df['normalized_signal']))
                
                for symbol in weights:
                    signal = signal_dict.get(symbol, 0)
                    if self.long_only and signal < 0:
                        weights[symbol] = 0
                    else:
                        # 使用信号作为倾斜因子
                        tilt_factor = 1 + 0.5 * signal  # 调整强度
                        weights[symbol] *= tilt_factor
            
            # 重新标准化
            total_weight = sum(abs(w) for w in weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
            
            return weights
            
        except Exception as e:
            logger.warning(f"风险平价计算失败: {e}")
            return self._signal_weighted_portfolio(signals_df)
    
    def _mean_variance_portfolio(self, 
                               signals_df: pd.DataFrame,
                               benchmark_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """均值方差优化组合"""
        symbols = signals_df['symbol'].tolist()
        
        if self._expected_returns is None or self._covariance_matrix is None:
            return self._signal_weighted_portfolio(signals_df)
        
        try:
            # 设置优化问题
            n = len(symbols)
            expected_returns = np.array([self._expected_returns.get(s, 0) for s in symbols])
            cov_matrix = self._covariance_matrix.loc[symbols, symbols].values
            
            # 目标函数：最大化 expected_return - 0.5 * risk_aversion * variance
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance)
            
            # 约束条件
            constraints = []
            
            # 权重和约束
            if self.long_only:
                constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            else:
                constraints.append({'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1.0})
            
            # 边界约束
            if self.long_only:
                bounds = [(0, self.max_weight_per_stock) for _ in range(n)]
            else:
                bounds = [(-self.max_weight_per_stock, self.max_weight_per_stock) for _ in range(n)]
            
            # 初始权重
            x0 = np.ones(n) / n
            
            # 优化求解
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = dict(zip(symbols, result.x))
                return optimal_weights
            else:
                logger.warning("均值方差优化未收敛，使用信号加权")
                return self._signal_weighted_portfolio(signals_df)
                
        except Exception as e:
            logger.warning(f"均值方差优化失败: {e}")
            return self._signal_weighted_portfolio(signals_df)
    
    def _apply_constraints_and_optimize(self,
                                      raw_weights: Dict[str, float],
                                      signals_df: pd.DataFrame,
                                      benchmark_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """应用约束条件和二次优化"""
        
        optimized_weights = raw_weights.copy()
        
        # 1. 应用最小权重过滤
        optimized_weights = {
            k: v for k, v in optimized_weights.items() 
            if abs(v) >= self.min_weight_threshold
        }
        
        # 2. 应用单只股票权重限制
        for symbol in optimized_weights:
            weight = optimized_weights[symbol]
            if abs(weight) > self.max_weight_per_stock:
                optimized_weights[symbol] = np.sign(weight) * self.max_weight_per_stock
        
        # 3. 重新标准化权重
        if self.long_only:
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}
        else:
            total_abs_weight = sum(abs(v) for v in optimized_weights.values())
            if total_abs_weight > 0:
                optimized_weights = {k: v / total_abs_weight for k, v in optimized_weights.items()}
        
        # 4. 应用组合层面约束
        optimized_weights = self._apply_portfolio_constraints(optimized_weights)
        
        return optimized_weights
    
    def _apply_portfolio_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用组合层面约束"""
        
        # TODO: 可以添加行业集中度约束、跟踪误差约束等
        # 这里先返回基础的权重约束结果
        return weights
    
    def _generate_trade_instructions(self, target_weights: Dict[str, float]) -> List[Dict]:
        """生成交易指令"""
        
        trade_instructions = []
        
        for symbol, target_weight in target_weights.items():
            current_weight = self._current_positions.get(symbol, 0.0)
            
            # 计算权重变化
            weight_change = target_weight - current_weight
            
            if abs(weight_change) > self.min_weight_threshold:
                # 计算交易金额
                trade_value = weight_change * self.portfolio_value
                
                instruction = {
                    'symbol': symbol,
                    'action': 'BUY' if weight_change > 0 else 'SELL',
                    'target_weight': target_weight,
                    'current_weight': current_weight,
                    'weight_change': weight_change,
                    'trade_value': trade_value,
                    'priority': abs(weight_change)  # 权重变化越大，优先级越高
                }
                
                trade_instructions.append(instruction)
        
        # 处理需要清仓的持仓
        for symbol, current_weight in self._current_positions.items():
            if symbol not in target_weights and abs(current_weight) > self.min_weight_threshold:
                instruction = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'target_weight': 0.0,
                    'current_weight': current_weight,
                    'weight_change': -current_weight,
                    'trade_value': -current_weight * self.portfolio_value,
                    'priority': abs(current_weight)
                }
                trade_instructions.append(instruction)
        
        # 按优先级排序
        trade_instructions.sort(key=lambda x: x['priority'], reverse=True)
        
        return trade_instructions
    
    def _calculate_portfolio_metrics(self, 
                                   weights: Dict[str, float],
                                   signals_df: pd.DataFrame) -> Dict[str, any]:
        """计算组合指标"""
        
        metrics = {}
        
        try:
            if not weights:
                return metrics
            
            # 基础指标
            metrics['position_count'] = len(weights)
            metrics['gross_exposure'] = sum(abs(w) for w in weights.values())
            metrics['net_exposure'] = sum(weights.values())
            metrics['leverage'] = metrics['gross_exposure']
            
            # 权重分布统计
            weight_values = list(weights.values())
            metrics['max_weight'] = max(weight_values) if weight_values else 0
            metrics['min_weight'] = min(weight_values) if weight_values else 0
            metrics['weight_std'] = np.std(weight_values) if len(weight_values) > 1 else 0
            
            # 信号覆盖度
            signal_symbols = set(signals_df['symbol'])
            portfolio_symbols = set(weights.keys())
            metrics['signal_coverage'] = len(portfolio_symbols & signal_symbols) / len(signal_symbols) if signal_symbols else 0
            
            # 预期收益和风险指标
            if self._expected_returns and self._covariance_matrix is not None:
                symbols = list(weights.keys())
                weight_array = np.array([weights[s] for s in symbols])
                returns_array = np.array([self._expected_returns.get(s, 0) for s in symbols])
                
                # 预期组合收益
                metrics['expected_return'] = np.dot(weight_array, returns_array)
                
                # 预期组合风险
                try:
                    cov_sub = self._covariance_matrix.loc[symbols, symbols].values
                    metrics['expected_volatility'] = np.sqrt(np.dot(weight_array, np.dot(cov_sub, weight_array)))
                    
                    # 夏普比率预期（假设无风险利率为3%）
                    if metrics['expected_volatility'] > 0:
                        metrics['expected_sharpe'] = (metrics['expected_return'] - 0.03) / metrics['expected_volatility']
                except:
                    pass
            
            # 换手率计算
            current_symbols = set(self._current_positions.keys())
            target_symbols = set(weights.keys())
            
            # 计算换手率
            turnover = 0.0
            all_symbols = current_symbols | target_symbols
            
            for symbol in all_symbols:
                current_w = self._current_positions.get(symbol, 0.0)
                target_w = weights.get(symbol, 0.0)
                turnover += abs(target_w - current_w)
            
            metrics['turnover'] = turnover / 2.0  # 单边换手率
            
            # 预估交易成本
            metrics['estimated_transaction_cost'] = metrics['turnover'] * self.transaction_cost_rate
            
        except Exception as e:
            logger.warning(f"组合指标计算失败: {e}")
        
        return metrics