#!/usr/bin/env python3
"""
组合再平衡器 (Portfolio Rebalancer)
处理组合的动态再平衡和调整逻辑

核心功能：
1. 智能再平衡触发条件判断
2. 渐进式再平衡策略
3. 交易成本优化的再平衡
4. 风险事件触发的紧急再平衡

设计原则：
- 成本效益：避免过度交易
- 风险控制：及时应对风险变化
- 适应性：根据市场条件调整再平衡策略

作者: Kronos Team
日期: 2025-01-26
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RebalanceFrequency(Enum):
    """再平衡频率枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    DYNAMIC = "dynamic"  # 基于条件的动态再平衡

class RebalanceTrigger(Enum):
    """再平衡触发条件"""
    SCHEDULED = "scheduled"        # 定时触发
    DRIFT_THRESHOLD = "drift"      # 权重漂移触发
    RISK_THRESHOLD = "risk"        # 风险阈值触发
    SIGNAL_CHANGE = "signal"       # 信号变化触发
    EMERGENCY = "emergency"        # 紧急情况触发

class Rebalancer:
    """
    投资组合再平衡器
    
    核心职责：
    1. 判断是否需要进行再平衡
    2. 实施渐进式再平衡策略
    3. 优化再平衡的交易成本
    4. 监控和响应风险事件
    """
    
    def __init__(self,
                 # 基础配置
                 rebalance_frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY,
                 
                 # 漂移阈值
                 weight_drift_threshold: float = 0.05,     # 权重漂移阈值
                 total_drift_threshold: float = 0.15,      # 总漂移阈值
                 
                 # 风险阈值
                 volatility_threshold: float = 0.25,       # 组合波动率上限
                 max_position_weight: float = 0.10,        # 单个持仓权重上限
                 
                 # 交易成本控制
                 min_trade_size: float = 0.01,             # 最小交易规模
                 max_turnover_per_rebalance: float = 0.5,  # 单次再平衡最大换手率
                 
                 # 渐进式再平衡
                 enable_gradual_rebalancing: bool = True,  # 启用渐进式再平衡
                 rebalancing_speed: float = 0.5,           # 再平衡速度因子
                 
                 # 信号变化阈值
                 signal_change_threshold: float = 0.3      # 信号变化触发阈值
                 ):
        """
        初始化再平衡器
        
        Args:
            rebalance_frequency: 再平衡频率
            weight_drift_threshold: 单个权重漂移阈值
            total_drift_threshold: 总体漂移阈值
            volatility_threshold: 组合波动率阈值
            max_position_weight: 最大单个持仓权重
            min_trade_size: 最小交易规模
            max_turnover_per_rebalance: 单次最大换手率
            enable_gradual_rebalancing: 是否启用渐进式再平衡
            rebalancing_speed: 再平衡速度（0-1，越大越快）
            signal_change_threshold: 信号变化触发阈值
        """
        self.rebalance_frequency = rebalance_frequency
        
        # 漂移控制
        self.weight_drift_threshold = weight_drift_threshold
        self.total_drift_threshold = total_drift_threshold
        
        # 风险控制
        self.volatility_threshold = volatility_threshold
        self.max_position_weight = max_position_weight
        
        # 交易成本控制
        self.min_trade_size = min_trade_size
        self.max_turnover_per_rebalance = max_turnover_per_rebalance
        
        # 渐进式再平衡
        self.enable_gradual_rebalancing = enable_gradual_rebalancing
        self.rebalancing_speed = max(0.1, min(1.0, rebalancing_speed))
        
        # 信号变化控制
        self.signal_change_threshold = signal_change_threshold
        
        # 状态跟踪
        self._last_rebalance_date = None
        self._rebalance_history = []
        
        logger.info(f"Rebalancer initialized: "
                   f"frequency={rebalance_frequency.value}, "
                   f"drift_threshold={weight_drift_threshold}, "
                   f"gradual={enable_gradual_rebalancing}")
    
    def should_rebalance(self,
                        current_date: str,
                        current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        current_signals: Optional[pd.DataFrame] = None,
                        previous_signals: Optional[pd.DataFrame] = None,
                        portfolio_metrics: Optional[Dict] = None) -> Tuple[bool, RebalanceTrigger, Dict]:
        """
        判断是否需要再平衡
        
        Args:
            current_date: 当前日期
            current_weights: 当前组合权重
            target_weights: 目标权重
            current_signals: 当前信号
            previous_signals: 之前信号
            portfolio_metrics: 组合指标
            
        Returns:
            (是否需要再平衡, 触发原因, 详细信息)
        """
        trigger_info = {
            'date': current_date,
            'reasons': [],
            'metrics': {}
        }
        
        # 1. 检查定时触发
        if self._check_scheduled_rebalance(current_date):
            trigger_info['reasons'].append('scheduled')
            return True, RebalanceTrigger.SCHEDULED, trigger_info
        
        # 2. 检查权重漂移
        drift_needed, drift_metrics = self._check_weight_drift(current_weights, target_weights)
        trigger_info['metrics'].update(drift_metrics)
        
        if drift_needed:
            trigger_info['reasons'].append('weight_drift')
            return True, RebalanceTrigger.DRIFT_THRESHOLD, trigger_info
        
        # 3. 检查风险阈值
        risk_needed, risk_metrics = self._check_risk_threshold(current_weights, portfolio_metrics)
        trigger_info['metrics'].update(risk_metrics)
        
        if risk_needed:
            trigger_info['reasons'].append('risk_threshold')
            return True, RebalanceTrigger.RISK_THRESHOLD, trigger_info
        
        # 4. 检查信号变化
        if current_signals is not None and previous_signals is not None:
            signal_needed, signal_metrics = self._check_signal_change(current_signals, previous_signals)
            trigger_info['metrics'].update(signal_metrics)
            
            if signal_needed:
                trigger_info['reasons'].append('signal_change')
                return True, RebalanceTrigger.SIGNAL_CHANGE, trigger_info
        
        # 5. 如果使用动态频率，基于市场条件判断
        if self.rebalance_frequency == RebalanceFrequency.DYNAMIC:
            dynamic_needed, dynamic_reason = self._check_dynamic_conditions(
                current_weights, target_weights, portfolio_metrics
            )
            if dynamic_needed:
                trigger_info['reasons'].append(f'dynamic_{dynamic_reason}')
                return True, RebalanceTrigger.DRIFT_THRESHOLD, trigger_info
        
        return False, RebalanceTrigger.SCHEDULED, trigger_info
    
    def execute_rebalance(self,
                         current_weights: Dict[str, float],
                         target_weights: Dict[str, float],
                         trigger: RebalanceTrigger,
                         market_data: Optional[Dict] = None) -> Dict[str, any]:
        """
        执行再平衡
        
        Args:
            current_weights: 当前权重
            target_weights: 目标权重
            trigger: 触发原因
            market_data: 市场数据（用于成本估算）
            
        Returns:
            再平衡结果字典
        """
        logger.info(f"执行再平衡，触发原因: {trigger.value}")
        
        # 1. 计算目标权重调整
        adjusted_target = self._adjust_target_weights(
            current_weights, target_weights, trigger
        )
        
        # 2. 生成交易指令
        trade_instructions = self._generate_rebalance_trades(
            current_weights, adjusted_target
        )
        
        # 3. 优化交易序列
        optimized_trades = self._optimize_trade_sequence(trade_instructions, market_data)
        
        # 4. 计算预期成本和影响
        cost_analysis = self._analyze_rebalance_cost(optimized_trades, market_data)
        
        # 5. 记录再平衡历史
        rebalance_record = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'trigger': trigger.value,
            'trade_count': len(optimized_trades),
            'turnover': cost_analysis.get('turnover', 0),
            'expected_cost': cost_analysis.get('total_cost', 0)
        }
        self._rebalance_history.append(rebalance_record)
        
        result = {
            'adjusted_target_weights': adjusted_target,
            'trade_instructions': optimized_trades,
            'cost_analysis': cost_analysis,
            'rebalance_info': rebalance_record
        }
        
        logger.info(f"再平衡完成: {len(optimized_trades)} 笔交易, "
                   f"换手率: {cost_analysis.get('turnover', 0):.2%}")
        
        return result
    
    def _check_scheduled_rebalance(self, current_date: str) -> bool:
        """检查定时再平衡"""
        if self.rebalance_frequency == RebalanceFrequency.DYNAMIC:
            return False
        
        if self._last_rebalance_date is None:
            return True
        
        current = pd.to_datetime(current_date)
        last_rebalance = pd.to_datetime(self._last_rebalance_date)
        
        if self.rebalance_frequency == RebalanceFrequency.DAILY:
            return (current - last_rebalance).days >= 1
        elif self.rebalance_frequency == RebalanceFrequency.WEEKLY:
            return (current - last_rebalance).days >= 7
        elif self.rebalance_frequency == RebalanceFrequency.MONTHLY:
            return (current - last_rebalance).days >= 30
        elif self.rebalance_frequency == RebalanceFrequency.QUARTERLY:
            return (current - last_rebalance).days >= 90
        
        return False
    
    def _check_weight_drift(self, 
                           current_weights: Dict[str, float],
                           target_weights: Dict[str, float]) -> Tuple[bool, Dict]:
        """检查权重漂移"""
        metrics = {}
        
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        total_drift = 0.0
        max_individual_drift = 0.0
        drift_violations = 0
        
        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            
            drift = abs(current_w - target_w)
            total_drift += drift
            max_individual_drift = max(max_individual_drift, drift)
            
            if drift > self.weight_drift_threshold:
                drift_violations += 1
        
        metrics.update({
            'total_drift': total_drift,
            'max_individual_drift': max_individual_drift,
            'drift_violations': drift_violations,
            'avg_drift': total_drift / len(all_symbols) if all_symbols else 0
        })
        
        # 判断是否需要再平衡
        needs_rebalance = (
            total_drift > self.total_drift_threshold or
            max_individual_drift > self.weight_drift_threshold or
            drift_violations > len(all_symbols) * 0.3  # 超过30%的持仓偏离
        )
        
        return needs_rebalance, metrics
    
    def _check_risk_threshold(self, 
                             current_weights: Dict[str, float],
                             portfolio_metrics: Optional[Dict]) -> Tuple[bool, Dict]:
        """检查风险阈值"""
        metrics = {}
        
        # 检查单个持仓权重
        max_weight = max(current_weights.values()) if current_weights else 0
        weight_violation = max_weight > self.max_position_weight
        
        metrics['max_position_weight'] = max_weight
        metrics['weight_violation'] = weight_violation
        
        # 检查组合波动率（如果有）
        volatility_violation = False
        if portfolio_metrics and 'expected_volatility' in portfolio_metrics:
            current_vol = portfolio_metrics['expected_volatility']
            volatility_violation = current_vol > self.volatility_threshold
            metrics['current_volatility'] = current_vol
            metrics['volatility_violation'] = volatility_violation
        
        needs_rebalance = weight_violation or volatility_violation
        
        return needs_rebalance, metrics
    
    def _check_signal_change(self, 
                           current_signals: pd.DataFrame,
                           previous_signals: pd.DataFrame) -> Tuple[bool, Dict]:
        """检查信号变化"""
        metrics = {}
        
        try:
            # 找到共同股票
            current_symbols = set(current_signals['symbol'])
            previous_symbols = set(previous_signals['symbol'])
            common_symbols = current_symbols & previous_symbols
            
            if not common_symbols:
                return False, metrics
            
            # 计算信号变化
            signal_changes = []
            
            for symbol in common_symbols:
                current_signal = current_signals[current_signals['symbol'] == symbol]['risk_adjusted_score'].iloc[0]
                previous_signal = previous_signals[previous_signals['symbol'] == symbol]['risk_adjusted_score'].iloc[0]
                
                # 标准化信号变化
                signal_change = abs(current_signal - previous_signal) / (abs(previous_signal) + 1e-6)
                signal_changes.append(signal_change)
            
            avg_signal_change = np.mean(signal_changes) if signal_changes else 0
            max_signal_change = max(signal_changes) if signal_changes else 0
            
            metrics.update({
                'avg_signal_change': avg_signal_change,
                'max_signal_change': max_signal_change,
                'signals_analyzed': len(common_symbols)
            })
            
            needs_rebalance = avg_signal_change > self.signal_change_threshold
            
        except Exception as e:
            logger.warning(f"信号变化检查失败: {e}")
            needs_rebalance = False
        
        return needs_rebalance, metrics
    
    def _check_dynamic_conditions(self, 
                                 current_weights: Dict[str, float],
                                 target_weights: Dict[str, float],
                                 portfolio_metrics: Optional[Dict]) -> Tuple[bool, str]:
        """检查动态条件"""
        
        # 基于市场波动性的动态调整
        if portfolio_metrics and 'expected_volatility' in portfolio_metrics:
            current_vol = portfolio_metrics['expected_volatility']
            
            # 高波动市场下更频繁再平衡
            if current_vol > 0.20:  # 20%以上波动率
                return True, "high_volatility"
        
        # 基于权重偏离程度
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        total_drift = sum(abs(current_weights.get(s, 0) - target_weights.get(s, 0)) for s in all_symbols)
        
        # 总偏离超过10%就触发
        if total_drift > 0.10:
            return True, "moderate_drift"
        
        return False, ""
    
    def _adjust_target_weights(self,
                              current_weights: Dict[str, float],
                              target_weights: Dict[str, float],
                              trigger: RebalanceTrigger) -> Dict[str, float]:
        """调整目标权重（渐进式再平衡）"""
        
        if not self.enable_gradual_rebalancing or trigger == RebalanceTrigger.EMERGENCY:
            # 紧急情况或禁用渐进式时，直接使用目标权重
            return target_weights.copy()
        
        adjusted_weights = {}
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            
            # 渐进式调整：当前权重 + 调整速度 * (目标权重 - 当前权重)
            weight_diff = target_w - current_w
            adjusted_w = current_w + self.rebalancing_speed * weight_diff
            
            adjusted_weights[symbol] = adjusted_w
        
        return adjusted_weights
    
    def _generate_rebalance_trades(self,
                                  current_weights: Dict[str, float],
                                  target_weights: Dict[str, float]) -> List[Dict]:
        """生成再平衡交易指令"""
        
        trades = []
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            
            weight_change = target_w - current_w
            
            # 过滤小额交易
            if abs(weight_change) < self.min_trade_size:
                continue
            
            trade = {
                'symbol': symbol,
                'current_weight': current_w,
                'target_weight': target_w,
                'weight_change': weight_change,
                'action': 'BUY' if weight_change > 0 else 'SELL',
                'priority': abs(weight_change),  # 权重变化越大，优先级越高
                'trade_type': 'rebalance'
            }
            
            trades.append(trade)
        
        # 按优先级排序
        trades.sort(key=lambda x: x['priority'], reverse=True)
        
        return trades
    
    def _optimize_trade_sequence(self, 
                                trades: List[Dict],
                                market_data: Optional[Dict] = None) -> List[Dict]:
        """优化交易序列"""
        
        if not trades:
            return trades
        
        # 限制单次再平衡的总换手率
        total_turnover = sum(abs(trade['weight_change']) for trade in trades) / 2.0
        
        if total_turnover > self.max_turnover_per_rebalance:
            # 按优先级截断交易
            cumulative_turnover = 0.0
            optimized_trades = []
            
            for trade in trades:
                trade_turnover = abs(trade['weight_change']) / 2.0
                if cumulative_turnover + trade_turnover <= self.max_turnover_per_rebalance:
                    optimized_trades.append(trade)
                    cumulative_turnover += trade_turnover
                else:
                    # 调整最后一笔交易的大小
                    remaining_turnover = self.max_turnover_per_rebalance - cumulative_turnover
                    if remaining_turnover > self.min_trade_size:
                        adjusted_trade = trade.copy()
                        adjusted_trade['weight_change'] = np.sign(trade['weight_change']) * remaining_turnover * 2.0
                        adjusted_trade['target_weight'] = adjusted_trade['current_weight'] + adjusted_trade['weight_change']
                        optimized_trades.append(adjusted_trade)
                    break
            
            logger.info(f"交易序列优化: {len(optimized_trades)}/{len(trades)} 交易保留")
            return optimized_trades
        
        return trades
    
    def _analyze_rebalance_cost(self, 
                               trades: List[Dict],
                               market_data: Optional[Dict] = None) -> Dict[str, any]:
        """分析再平衡成本"""
        
        if not trades:
            return {'total_cost': 0.0, 'turnover': 0.0, 'trade_count': 0}
        
        total_turnover = sum(abs(trade['weight_change']) for trade in trades) / 2.0
        
        # 简化的成本估算
        transaction_cost = total_turnover * 0.002  # 假设0.2%的交易费用
        market_impact = total_turnover * 0.001     # 假设0.1%的市场冲击
        total_cost = transaction_cost + market_impact
        
        cost_analysis = {
            'total_cost': total_cost,
            'transaction_cost': transaction_cost,
            'market_impact': market_impact,
            'turnover': total_turnover,
            'trade_count': len(trades),
            'avg_trade_size': np.mean([abs(t['weight_change']) for t in trades]) if trades else 0
        }
        
        return cost_analysis
    
    def get_rebalance_statistics(self) -> Dict[str, any]:
        """获取再平衡统计信息"""
        
        if not self._rebalance_history:
            return {'total_rebalances': 0}
        
        stats = {
            'total_rebalances': len(self._rebalance_history),
            'avg_turnover': np.mean([r['turnover'] for r in self._rebalance_history]),
            'avg_cost': np.mean([r['expected_cost'] for r in self._rebalance_history]),
            'avg_trades_per_rebalance': np.mean([r['trade_count'] for r in self._rebalance_history]),
            'trigger_distribution': {}
        }
        
        # 统计触发原因分布
        for record in self._rebalance_history:
            trigger = record['trigger']
            stats['trigger_distribution'][trigger] = stats['trigger_distribution'].get(trigger, 0) + 1
        
        return stats