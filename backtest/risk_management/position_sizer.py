#!/usr/bin/env python3
"""
头寸规模管理器 (Position Sizer)
基于风险控制的头寸规模确定

核心功能：
1. 基于风险的头寸规模计算
2. Kelly公式和相关变种的应用
3. 固定风险百分比法
4. 动态头寸调整
5. 考虑交易成本的头寸优化

设计原则：
- 风险第一：头寸大小基于可承受风险确定
- 期望最大化：在风险约束下最大化期望收益
- 成本敏感：考虑交易成本对头寸大小的影响

作者: Kronos Team
日期: 2025-01-26
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SizingMethod(Enum):
    """头寸规模方法枚举"""
    FIXED_FRACTIONAL = "fixed_fractional"        # 固定比例法
    FIXED_RISK = "fixed_risk"                    # 固定风险法
    KELLY = "kelly"                              # Kelly公式
    HALF_KELLY = "half_kelly"                    # 半Kelly
    VOLATILITY_TARGET = "volatility_target"      # 波动率目标法
    RISK_PARITY = "risk_parity"                  # 风险平价法
    OPTIMAL_F = "optimal_f"                      # 最优f法

@dataclass
class PositionSizeResult:
    """头寸规模结果"""
    symbol: str
    recommended_size: float          # 建议头寸规模（权重）
    max_size: float                 # 最大允许规模
    min_size: float                 # 最小规模
    risk_contribution: float        # 风险贡献度
    expected_return: float          # 预期收益
    confidence: float               # 置信度
    method_used: str               # 使用的方法
    constraints_applied: List[str]  # 应用的约束

class PositionSizer:
    """
    头寸规模管理器
    
    核心职责：
    1. 基于不同方法计算最优头寸规模
    2. 应用风险约束和限制
    3. 动态调整头寸大小
    4. 考虑交易成本影响
    """
    
    def __init__(self,
                 # 基础参数
                 portfolio_value: float = 1e6,           # 组合总价值
                 risk_free_rate: float = 0.03,           # 无风险利率
                 
                 # 头寸限制
                 max_position_size: float = 0.10,        # 最大单个头寸权重
                 min_position_size: float = 0.005,       # 最小头寸权重
                 max_portfolio_leverage: float = 1.0,    # 最大杠杆
                 
                 # 风险控制
                 portfolio_risk_target: float = 0.15,    # 目标组合风险
                 max_individual_risk: float = 0.05,      # 单个头寸最大风险
                 correlation_adjustment: bool = True,     # 是否调整相关性
                 
                 # Kelly参数
                 kelly_lookback: int = 252,              # Kelly公式回望期
                 kelly_fractional: float = 0.25,        # Kelly分数因子
                 
                 # 交易成本
                 transaction_cost: float = 0.002,        # 交易成本比率
                 market_impact_coeff: float = 0.001,     # 市场冲击系数
                 
                 # 其他参数
                 rebalance_threshold: float = 0.05,      # 重新平衡阈值
                 confidence_threshold: float = 0.6       # 置信度阈值
                 ):
        """
        初始化头寸规模管理器
        
        Args:
            portfolio_value: 组合总价值
            risk_free_rate: 无风险利率
            max_position_size: 最大单个头寸权重
            min_position_size: 最小头寸权重
            max_portfolio_leverage: 最大组合杠杆
            portfolio_risk_target: 目标组合风险水平
            max_individual_risk: 单个头寸最大风险贡献
            correlation_adjustment: 是否进行相关性调整
            kelly_lookback: Kelly公式计算的历史回望期
            kelly_fractional: Kelly分数的保守因子
            transaction_cost: 交易成本比率
            market_impact_coeff: 市场冲击成本系数
            rebalance_threshold: 触发再平衡的阈值
            confidence_threshold: 最小信号置信度要求
        """
        self.portfolio_value = portfolio_value
        self.risk_free_rate = risk_free_rate
        
        # 头寸限制
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_portfolio_leverage = max_portfolio_leverage
        
        # 风险控制
        self.portfolio_risk_target = portfolio_risk_target
        self.max_individual_risk = max_individual_risk
        self.correlation_adjustment = correlation_adjustment
        
        # Kelly参数
        self.kelly_lookback = kelly_lookback
        self.kelly_fractional = kelly_fractional
        
        # 成本参数
        self.transaction_cost = transaction_cost
        self.market_impact_coeff = market_impact_coeff
        
        # 其他参数
        self.rebalance_threshold = rebalance_threshold
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"PositionSizer initialized: "
                   f"max_size={max_position_size:.1%}, "
                   f"risk_target={portfolio_risk_target:.1%}")
    
    def calculate_position_sizes(self,
                               signals: pd.DataFrame,
                               market_data: Dict[str, pd.DataFrame],
                               current_positions: Optional[Dict[str, float]] = None,
                               covariance_matrix: Optional[pd.DataFrame] = None,
                               method: SizingMethod = SizingMethod.VOLATILITY_TARGET) -> Dict[str, PositionSizeResult]:
        """
        计算推荐的头寸规模
        
        Args:
            signals: 信号数据DataFrame
            market_data: 历史市场数据
            current_positions: 当前持仓
            covariance_matrix: 协方差矩阵
            method: 头寸规模计算方法
            
        Returns:
            {symbol: PositionSizeResult} 字典
        """
        if signals.empty:
            logger.warning("信号数据为空")
            return {}
        
        logger.info(f"计算头寸规模，方法: {method.value}，信号数量: {len(signals)}")
        
        # 过滤低置信度信号
        if 'prediction_confidence' in signals.columns:
            signals = signals[signals['prediction_confidence'] >= self.confidence_threshold]
        
        if signals.empty:
            logger.warning("过滤后无有效信号")
            return {}
        
        position_sizes = {}
        
        try:
            for _, signal_row in signals.iterrows():
                symbol = signal_row['symbol']
                
                # 计算单个头寸规模
                size_result = self._calculate_single_position_size(
                    signal_row, market_data, covariance_matrix, method
                )
                
                if size_result:
                    position_sizes[symbol] = size_result
            
            # 应用组合层面的约束和调整
            position_sizes = self._apply_portfolio_constraints(
                position_sizes, current_positions, covariance_matrix
            )
            
            logger.info(f"头寸规模计算完成，有效头寸: {len(position_sizes)}")
            
        except Exception as e:
            logger.error(f"头寸规模计算失败: {e}")
        
        return position_sizes
    
    def _calculate_single_position_size(self,
                                       signal: pd.Series,
                                       market_data: Dict[str, pd.DataFrame],
                                       covariance_matrix: Optional[pd.DataFrame],
                                       method: SizingMethod) -> Optional[PositionSizeResult]:
        """计算单个股票的头寸规模"""
        
        symbol = signal['symbol']
        expected_return = signal.get('expected_return', 0)
        risk_adjusted_score = signal.get('risk_adjusted_score', 0)
        prediction_confidence = signal.get('prediction_confidence', 0.5)
        
        if symbol not in market_data:
            return None
        
        try:
            # 获取历史数据计算波动率
            df = market_data[symbol]
            if 'close' not in df.columns:
                return None
            
            returns = df['close'].pct_change().dropna().tail(self.kelly_lookback)
            if len(returns) < 20:  # 数据不足
                return None
            
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            
            # 根据方法计算头寸规模
            if method == SizingMethod.FIXED_FRACTIONAL:
                recommended_size = self._fixed_fractional_sizing(expected_return, volatility)
            elif method == SizingMethod.FIXED_RISK:
                recommended_size = self._fixed_risk_sizing(expected_return, volatility)
            elif method == SizingMethod.KELLY:
                recommended_size = self._kelly_sizing(returns, expected_return)
            elif method == SizingMethod.HALF_KELLY:
                recommended_size = self._kelly_sizing(returns, expected_return) * 0.5
            elif method == SizingMethod.VOLATILITY_TARGET:
                recommended_size = self._volatility_target_sizing(volatility, risk_adjusted_score)
            elif method == SizingMethod.RISK_PARITY:
                recommended_size = self._risk_parity_sizing(volatility)
            else:
                recommended_size = self._volatility_target_sizing(volatility, risk_adjusted_score)
            
            # 应用基础约束
            recommended_size = max(0, min(recommended_size, self.max_position_size))
            
            if recommended_size < self.min_position_size:
                return None
            
            # 计算风险贡献
            risk_contribution = recommended_size * volatility
            
            # 构建结果
            result = PositionSizeResult(
                symbol=symbol,
                recommended_size=recommended_size,
                max_size=self.max_position_size,
                min_size=self.min_position_size,
                risk_contribution=risk_contribution,
                expected_return=expected_return,
                confidence=prediction_confidence,
                method_used=method.value,
                constraints_applied=[]
            )
            
            return result
            
        except Exception as e:
            logger.debug(f"计算 {symbol} 头寸规模失败: {e}")
            return None
    
    def _fixed_fractional_sizing(self, expected_return: float, volatility: float) -> float:
        """固定比例法"""
        # 简单的固定比例，基于期望收益调整
        base_size = 0.02  # 基础2%
        return base_size * (1 + max(0, expected_return * 10))  # 根据预期收益调整
    
    def _fixed_risk_sizing(self, expected_return: float, volatility: float) -> float:
        """固定风险法"""
        if volatility <= 0:
            return 0
        
        # 目标单个头寸风险为组合总风险的一部分
        target_position_risk = self.max_individual_risk
        position_size = target_position_risk / volatility
        
        return min(position_size, self.max_position_size)
    
    def _kelly_sizing(self, returns: pd.Series, expected_return: float) -> float:
        """Kelly公式法"""
        if len(returns) == 0:
            return 0
        
        # 计算胜率和赔率
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) == 0 or len(negative_returns) == 0:
            return self.min_position_size
        
        win_rate = len(positive_returns) / len(returns)
        avg_win = positive_returns.mean()
        avg_loss = abs(negative_returns.mean())
        
        if avg_loss <= 0:
            return self.min_position_size
        
        # Kelly公式: f* = (bp - q) / b
        # 其中 b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        odds_ratio = avg_win / avg_loss
        kelly_fraction = (odds_ratio * win_rate - (1 - win_rate)) / odds_ratio
        
        # 应用保守因子
        kelly_fraction *= self.kelly_fractional
        
        return max(0, min(kelly_fraction, self.max_position_size))
    
    def _volatility_target_sizing(self, volatility: float, signal_strength: float) -> float:
        """波动率目标法"""
        if volatility <= 0:
            return 0
        
        # 基于信号强度和波动率调整头寸
        # 目标是每个头寸贡献相似的风险
        target_risk_per_position = self.portfolio_risk_target / 20  # 假设20个分散头寸
        
        base_size = target_risk_per_position / volatility
        
        # 根据信号强度调整
        signal_multiplier = 1 + max(-0.5, min(1.0, signal_strength))
        adjusted_size = base_size * signal_multiplier
        
        return max(0, min(adjusted_size, self.max_position_size))
    
    def _risk_parity_sizing(self, volatility: float) -> float:
        """风险平价法"""
        if volatility <= 0:
            return 0
        
        # 风险平价：每个资产贡献相等的风险
        # 简化实现：基于逆波动率加权
        inverse_vol_weight = (1.0 / volatility) if volatility > 0 else 0
        
        # 标准化（这里需要所有资产的信息，简化处理）
        normalized_weight = inverse_vol_weight * 0.05  # 临时标准化
        
        return min(normalized_weight, self.max_position_size)
    
    def _apply_portfolio_constraints(self,
                                   position_sizes: Dict[str, PositionSizeResult],
                                   current_positions: Optional[Dict[str, float]],
                                   covariance_matrix: Optional[pd.DataFrame]) -> Dict[str, PositionSizeResult]:
        """应用组合层面的约束"""
        
        if not position_sizes:
            return position_sizes
        
        # 1. 总权重约束
        total_weight = sum(result.recommended_size for result in position_sizes.values())
        
        if total_weight > self.max_portfolio_leverage:
            # 按比例缩减
            scale_factor = self.max_portfolio_leverage / total_weight
            for result in position_sizes.values():
                result.recommended_size *= scale_factor
                result.constraints_applied.append('portfolio_leverage_limit')
        
        # 2. 总体风险约束
        if covariance_matrix is not None:
            position_sizes = self._apply_risk_constraints(position_sizes, covariance_matrix)
        
        # 3. 考虑当前持仓的交易成本
        if current_positions:
            position_sizes = self._adjust_for_transaction_costs(position_sizes, current_positions)
        
        # 4. 过滤过小的头寸
        filtered_sizes = {}
        for symbol, result in position_sizes.items():
            if result.recommended_size >= self.min_position_size:
                filtered_sizes[symbol] = result
            else:
                logger.debug(f"过滤过小头寸: {symbol} ({result.recommended_size:.3%})")
        
        return filtered_sizes
    
    def _apply_risk_constraints(self,
                               position_sizes: Dict[str, PositionSizeResult],
                               covariance_matrix: pd.DataFrame) -> Dict[str, PositionSizeResult]:
        """应用风险约束"""
        
        symbols = list(position_sizes.keys())
        available_symbols = [s for s in symbols if s in covariance_matrix.index]
        
        if not available_symbols:
            return position_sizes
        
        try:
            # 计算组合风险
            weights = np.array([position_sizes[s].recommended_size for s in available_symbols])
            cov_sub = covariance_matrix.loc[available_symbols, available_symbols].values
            
            portfolio_variance = np.dot(weights, np.dot(cov_sub, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # 如果超过目标风险，按比例缩减
            if portfolio_volatility > self.portfolio_risk_target:
                scale_factor = self.portfolio_risk_target / portfolio_volatility
                
                for symbol in available_symbols:
                    result = position_sizes[symbol]
                    result.recommended_size *= scale_factor
                    result.constraints_applied.append('portfolio_risk_limit')
                    
                logger.info(f"应用组合风险约束，缩减因子: {scale_factor:.3f}")
        
        except Exception as e:
            logger.warning(f"风险约束应用失败: {e}")
        
        return position_sizes
    
    def _adjust_for_transaction_costs(self,
                                     position_sizes: Dict[str, PositionSizeResult],
                                     current_positions: Dict[str, float]) -> Dict[str, PositionSizeResult]:
        """调整交易成本影响"""
        
        for symbol, result in position_sizes.items():
            current_weight = current_positions.get(symbol, 0)
            target_weight = result.recommended_size
            
            # 计算交易成本
            weight_change = abs(target_weight - current_weight)
            transaction_cost = weight_change * self.transaction_cost
            
            # 如果交易成本过高，考虑减少调整幅度
            if weight_change > 0:
                cost_ratio = transaction_cost / weight_change
                
                if cost_ratio > 0.01:  # 交易成本超过1%
                    # 减少调整幅度
                    adjusted_change = weight_change * (1 - cost_ratio)
                    result.recommended_size = current_weight + np.sign(target_weight - current_weight) * adjusted_change
                    result.constraints_applied.append('transaction_cost_adjustment')
        
        return position_sizes
    
    def optimize_portfolio_sizes(self,
                                position_sizes: Dict[str, PositionSizeResult],
                                covariance_matrix: pd.DataFrame,
                                expected_returns: Dict[str, float]) -> Dict[str, float]:
        """
        组合层面的头寸规模优化
        
        Args:
            position_sizes: 初始头寸规模
            covariance_matrix: 协方差矩阵
            expected_returns: 预期收益
            
        Returns:
            优化后的权重字典
        """
        symbols = list(position_sizes.keys())
        available_symbols = [s for s in symbols if s in covariance_matrix.index]
        
        if len(available_symbols) < 2:
            return {s: result.recommended_size for s, result in position_sizes.items()}
        
        try:
            n = len(available_symbols)
            
            # 初始权重
            x0 = np.array([position_sizes[s].recommended_size for s in available_symbols])
            
            # 预期收益向量
            mu = np.array([expected_returns.get(s, 0) for s in available_symbols])
            
            # 协方差矩阵
            cov = covariance_matrix.loc[available_symbols, available_symbols].values
            
            # 目标函数：最大化夏普比率
            def objective(weights):
                portfolio_return = np.dot(weights, mu)
                portfolio_variance = np.dot(weights, np.dot(cov, weights))
                
                if portfolio_variance <= 0:
                    return -np.inf
                
                sharpe = (portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
                return -sharpe  # 最小化负夏普比率
            
            # 约束条件
            constraints = []
            
            # 权重和约束
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - min(1.0, self.max_portfolio_leverage)
            })
            
            # 组合风险约束
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: self.portfolio_risk_target**2 - np.dot(w, np.dot(cov, w))
            })
            
            # 边界约束
            bounds = [(0, self.max_position_size) for _ in range(n)]
            
            # 优化
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = {}
                for i, symbol in enumerate(available_symbols):
                    optimized_weights[symbol] = max(result.x[i], 0)
                
                # 添加未优化的股票
                for symbol in symbols:
                    if symbol not in optimized_weights:
                        optimized_weights[symbol] = position_sizes[symbol].recommended_size
                
                logger.info("头寸规模优化完成")
                return optimized_weights
            else:
                logger.warning("头寸规模优化失败，使用原始权重")
                
        except Exception as e:
            logger.warning(f"头寸规模优化失败: {e}")
        
        # 返回原始权重
        return {s: result.recommended_size for s, result in position_sizes.items()}
    
    def get_sizing_summary(self, position_sizes: Dict[str, PositionSizeResult]) -> Dict[str, any]:
        """获取头寸规模总结"""
        
        if not position_sizes:
            return {'total_positions': 0}
        
        sizes = [result.recommended_size for result in position_sizes.values()]
        risks = [result.risk_contribution for result in position_sizes.values()]
        confidences = [result.confidence for result in position_sizes.values()]
        
        summary = {
            'total_positions': len(position_sizes),
            'total_weight': sum(sizes),
            'max_position_size': max(sizes),
            'min_position_size': min(sizes),
            'avg_position_size': np.mean(sizes),
            'position_size_std': np.std(sizes),
            'total_risk_contribution': sum(risks),
            'avg_confidence': np.mean(confidences),
            'concentration_ratio': sum(sorted(sizes, reverse=True)[:5]),  # Top 5集中度
            'size_distribution': {
                'large_positions': len([s for s in sizes if s > 0.05]),    # >5%
                'medium_positions': len([s for s in sizes if 0.02 < s <= 0.05]),  # 2-5%
                'small_positions': len([s for s in sizes if s <= 0.02])     # <=2%
            }
        }
        
        return summary