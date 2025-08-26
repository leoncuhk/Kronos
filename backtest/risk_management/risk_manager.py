#!/usr/bin/env python3
"""
风险管理器 (Risk Manager)
量化交易四步法的第四步：风险管理和执行控制

核心功能：
1. 实时风险监控和预警
2. 止损和止盈管理
3. 头寸规模控制
4. 市场异常检测和应对
5. 流动性风险管理

设计理念：
- 安全第一：风险控制优先于收益追求
- 动态适应：根据市场条件动态调整风险参数
- 多层防护：建立多层次的风险防护体系

作者: Kronos Team
日期: 2025-01-26
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """预警类型枚举"""
    POSITION_LIMIT = "position_limit"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CONCENTRATION = "concentration"
    CORRELATION = "correlation"
    MARKET_STRESS = "market_stress"

@dataclass
class RiskAlert:
    """风险预警数据类"""
    alert_type: AlertType
    risk_level: RiskLevel
    message: str
    current_value: float
    threshold: float
    symbol: Optional[str] = None
    timestamp: Optional[str] = None
    suggested_action: Optional[str] = None

@dataclass
class RiskMetrics:
    """风险指标数据类"""
    portfolio_volatility: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float
    sharpe_ratio: float
    max_position_weight: float
    concentration_ratio: float
    liquidity_ratio: float

class RiskManager:
    """
    综合风险管理器
    
    核心职责：
    1. 实时监控组合风险指标
    2. 生成风险预警和建议
    3. 执行风险控制措施
    4. 管理止损止盈策略
    5. 应对市场异常情况
    """
    
    def __init__(self,
                 # 基础风险参数
                 max_portfolio_drawdown: float = 0.08,      # 最大组合回撤
                 max_position_weight: float = 0.10,         # 最大单个头寸权重
                 max_portfolio_volatility: float = 0.20,    # 最大组合波动率
                 
                 # 止损止盈参数
                 stop_loss_threshold: float = -0.05,        # 止损阈值 -5%
                 take_profit_threshold: float = 0.15,       # 止盈阈值 +15%
                 trailing_stop_ratio: float = 0.02,         # 追踪止损比例
                 
                 # 流动性风险参数
                 min_daily_volume: float = 1e6,             # 最小日成交量
                 max_position_vs_volume: float = 0.05,      # 持仓相对成交量的最大比例
                 liquidity_stress_multiplier: float = 2.0,  # 流动性压力测试倍数
                 
                 # 相关性风险参数
                 max_correlation_threshold: float = 0.8,    # 最大相关性阈值
                 correlation_lookback: int = 60,            # 相关性计算回望期
                 
                 # 市场风险参数
                 market_stress_threshold: float = -0.03,    # 市场压力阈值（日收益）
                 volatility_expansion_threshold: float = 2.0, # 波动率扩张阈值
                 
                 # 动态风险调整
                 risk_parity_adjustment: bool = True,       # 启用风险平价调整
                 stress_test_scenarios: int = 1000          # 压力测试场景数
                 ):
        """
        初始化风险管理器
        
        Args:
            max_portfolio_drawdown: 最大允许组合回撤
            max_position_weight: 单个头寸最大权重
            max_portfolio_volatility: 组合最大波动率
            stop_loss_threshold: 止损阈值
            take_profit_threshold: 止盈阈值
            trailing_stop_ratio: 追踪止损比例
            min_daily_volume: 最小日成交量要求
            max_position_vs_volume: 头寸占成交量的最大比例
            liquidity_stress_multiplier: 流动性压力测试倍数
            max_correlation_threshold: 最大相关性阈值
            correlation_lookback: 相关性计算窗口
            market_stress_threshold: 市场压力阈值
            volatility_expansion_threshold: 波动率扩张阈值
            risk_parity_adjustment: 是否启用风险平价调整
            stress_test_scenarios: 压力测试场景数量
        """
        # 基础风险限制
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.max_position_weight = max_position_weight
        self.max_portfolio_volatility = max_portfolio_volatility
        
        # 止损止盈
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.trailing_stop_ratio = trailing_stop_ratio
        
        # 流动性风险
        self.min_daily_volume = min_daily_volume
        self.max_position_vs_volume = max_position_vs_volume
        self.liquidity_stress_multiplier = liquidity_stress_multiplier
        
        # 相关性风险
        self.max_correlation_threshold = max_correlation_threshold
        self.correlation_lookback = correlation_lookback
        
        # 市场风险
        self.market_stress_threshold = market_stress_threshold
        self.volatility_expansion_threshold = volatility_expansion_threshold
        
        # 动态调整
        self.risk_parity_adjustment = risk_parity_adjustment
        self.stress_test_scenarios = stress_test_scenarios
        
        # 内部状态
        self._current_positions = {}
        self._position_entry_prices = {}
        self._trailing_stops = {}
        self._risk_alerts = []
        self._risk_history = []
        
        logger.info(f"RiskManager initialized: "
                   f"max_drawdown={max_portfolio_drawdown:.1%}, "
                   f"max_position={max_position_weight:.1%}, "
                   f"stop_loss={stop_loss_threshold:.1%}")
    
    def monitor_portfolio_risk(self,
                             current_positions: Dict[str, float],
                             market_data: Dict[str, pd.DataFrame],
                             portfolio_value: float,
                             benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        监控投资组合风险
        
        Args:
            current_positions: 当前持仓 {symbol: weight}
            market_data: 市场数据
            portfolio_value: 组合价值
            benchmark_data: 基准数据（可选）
            
        Returns:
            风险监控结果字典
        """
        logger.info(f"监控组合风险，持仓数量: {len(current_positions)}")
        
        # 更新内部状态
        self._current_positions = current_positions
        
        risk_report = {
            'timestamp': datetime.now().isoformat(),
            'risk_metrics': {},
            'risk_alerts': [],
            'position_risks': {},
            'market_conditions': {},
            'recommended_actions': []
        }
        
        try:
            # 1. 计算风险指标
            risk_metrics = self._calculate_risk_metrics(
                current_positions, market_data, portfolio_value
            )
            risk_report['risk_metrics'] = risk_metrics.__dict__
            
            # 2. 检查风险预警
            alerts = self._check_risk_alerts(risk_metrics, current_positions, market_data)
            risk_report['risk_alerts'] = [alert.__dict__ for alert in alerts]
            
            # 3. 分析单个头寸风险
            position_risks = self._analyze_position_risks(current_positions, market_data)
            risk_report['position_risks'] = position_risks
            
            # 4. 评估市场条件
            market_conditions = self._assess_market_conditions(market_data, benchmark_data)
            risk_report['market_conditions'] = market_conditions
            
            # 5. 生成风险建议
            recommendations = self._generate_risk_recommendations(
                risk_metrics, alerts, position_risks, market_conditions
            )
            risk_report['recommended_actions'] = recommendations
            
            # 6. 更新风险历史
            self._update_risk_history(risk_report)
            
            logger.info(f"风险监控完成，发现 {len(alerts)} 个预警")
            
        except Exception as e:
            logger.error(f"风险监控失败: {e}")
            risk_report['error'] = str(e)
        
        return risk_report
    
    def execute_risk_controls(self,
                            risk_report: Dict[str, Any],
                            current_positions: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        执行风险控制措施
        
        Args:
            risk_report: 风险监控报告
            current_positions: 当前持仓
            
        Returns:
            风险控制指令列表
        """
        control_actions = []
        
        try:
            alerts = [RiskAlert(**alert) for alert in risk_report.get('risk_alerts', [])]
            
            for alert in alerts:
                if alert.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    action = self._create_risk_control_action(alert, current_positions)
                    if action:
                        control_actions.append(action)
            
            # 执行止损止盈检查
            stop_actions = self._check_stop_loss_take_profit(current_positions, risk_report)
            control_actions.extend(stop_actions)
            
            logger.info(f"生成 {len(control_actions)} 个风险控制指令")
            
        except Exception as e:
            logger.error(f"风险控制执行失败: {e}")
        
        return control_actions
    
    def _calculate_risk_metrics(self,
                               positions: Dict[str, float],
                               market_data: Dict[str, pd.DataFrame],
                               portfolio_value: float) -> RiskMetrics:
        """计算风险指标"""
        
        if not positions:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        try:
            # 计算收益率序列
            returns_data = self._calculate_portfolio_returns(positions, market_data)
            
            if returns_data is None or len(returns_data) == 0:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
            
            # 组合波动率（年化）
            portfolio_volatility = returns_data.std() * np.sqrt(252)
            
            # 最大回撤
            cumulative_returns = (1 + returns_data).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdowns.min())
            
            # VaR和ES
            var_95 = abs(returns_data.quantile(0.05))
            expected_shortfall = abs(returns_data[returns_data <= returns_data.quantile(0.05)].mean())
            
            # 夏普比率（假设无风险利率3%）
            excess_returns = returns_data.mean() * 252 - 0.03
            sharpe_ratio = excess_returns / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # 头寸集中度指标
            weights = list(positions.values())
            max_position_weight = max(weights) if weights else 0
            concentration_ratio = sum(sorted(weights, reverse=True)[:5])  # Top 5集中度
            
            # 流动性比率（简化计算）
            liquidity_ratio = self._calculate_liquidity_ratio(positions, market_data)
            
            return RiskMetrics(
                portfolio_volatility=portfolio_volatility,
                max_drawdown=max_drawdown,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                sharpe_ratio=sharpe_ratio,
                max_position_weight=max_position_weight,
                concentration_ratio=concentration_ratio,
                liquidity_ratio=liquidity_ratio
            )
            
        except Exception as e:
            logger.warning(f"风险指标计算失败: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_portfolio_returns(self,
                                   positions: Dict[str, float],
                                   market_data: Dict[str, pd.DataFrame],
                                   lookback_days: int = 252) -> Optional[pd.Series]:
        """计算组合收益率序列"""
        
        try:
            # 获取股票收益率
            stock_returns = {}
            
            for symbol, weight in positions.items():
                if symbol in market_data and weight != 0:
                    df = market_data[symbol]
                    if 'close' in df.columns and len(df) > 1:
                        returns = df['close'].pct_change().dropna().tail(lookback_days)
                        if len(returns) > 0:
                            stock_returns[symbol] = returns
            
            if not stock_returns:
                return None
            
            # 对齐时间序列
            returns_df = pd.DataFrame(stock_returns).dropna()
            
            if returns_df.empty:
                return None
            
            # 计算组合收益率
            portfolio_returns = pd.Series(0, index=returns_df.index)
            
            for symbol, weight in positions.items():
                if symbol in returns_df.columns:
                    portfolio_returns += returns_df[symbol] * weight
            
            return portfolio_returns
            
        except Exception as e:
            logger.warning(f"组合收益率计算失败: {e}")
            return None
    
    def _calculate_liquidity_ratio(self,
                                  positions: Dict[str, float],
                                  market_data: Dict[str, pd.DataFrame]) -> float:
        """计算流动性比率"""
        
        try:
            total_liquidity_score = 0.0
            total_weight = 0.0
            
            for symbol, weight in positions.items():
                if symbol in market_data and weight != 0:
                    df = market_data[symbol]
                    if 'vol' in df.columns:
                        # 计算流动性评分（基于成交量稳定性）
                        recent_volume = df['vol'].tail(20)
                        if len(recent_volume) > 1:
                            avg_volume = recent_volume.mean()
                            volume_stability = 1.0 / (1.0 + recent_volume.std() / avg_volume)
                            
                            # 结合绝对成交量
                            volume_score = min(1.0, avg_volume / self.min_daily_volume)
                            liquidity_score = 0.6 * volume_score + 0.4 * volume_stability
                            
                            total_liquidity_score += abs(weight) * liquidity_score
                            total_weight += abs(weight)
            
            return total_liquidity_score / total_weight if total_weight > 0 else 0
            
        except Exception as e:
            logger.warning(f"流动性比率计算失败: {e}")
            return 0
    
    def _check_risk_alerts(self,
                          risk_metrics: RiskMetrics,
                          positions: Dict[str, float],
                          market_data: Dict[str, pd.DataFrame]) -> List[RiskAlert]:
        """检查风险预警"""
        
        alerts = []
        
        # 1. 组合波动率预警
        if risk_metrics.portfolio_volatility > self.max_portfolio_volatility:
            alerts.append(RiskAlert(
                alert_type=AlertType.VOLATILITY,
                risk_level=RiskLevel.HIGH,
                message=f"组合波动率过高: {risk_metrics.portfolio_volatility:.1%}",
                current_value=risk_metrics.portfolio_volatility,
                threshold=self.max_portfolio_volatility,
                suggested_action="减少高波动性持仓"
            ))
        
        # 2. 最大回撤预警
        if risk_metrics.max_drawdown > self.max_portfolio_drawdown:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                risk_level=RiskLevel.CRITICAL,
                message=f"组合回撤超限: {risk_metrics.max_drawdown:.1%}",
                current_value=risk_metrics.max_drawdown,
                threshold=self.max_portfolio_drawdown,
                suggested_action="立即减仓或止损"
            ))
        
        # 3. 头寸集中度预警
        if risk_metrics.max_position_weight > self.max_position_weight:
            max_symbol = max(positions.keys(), key=lambda s: positions[s])
            alerts.append(RiskAlert(
                alert_type=AlertType.POSITION_LIMIT,
                risk_level=RiskLevel.MEDIUM,
                message=f"单个头寸权重过高: {risk_metrics.max_position_weight:.1%}",
                current_value=risk_metrics.max_position_weight,
                threshold=self.max_position_weight,
                symbol=max_symbol,
                suggested_action=f"减少 {max_symbol} 持仓"
            ))
        
        # 4. 流动性预警
        if risk_metrics.liquidity_ratio < 0.5:
            alerts.append(RiskAlert(
                alert_type=AlertType.LIQUIDITY,
                risk_level=RiskLevel.MEDIUM,
                message=f"组合流动性不足: {risk_metrics.liquidity_ratio:.1%}",
                current_value=risk_metrics.liquidity_ratio,
                threshold=0.5,
                suggested_action="增加高流动性股票配置"
            ))
        
        # 5. 集中度预警
        if risk_metrics.concentration_ratio > 0.6:
            alerts.append(RiskAlert(
                alert_type=AlertType.CONCENTRATION,
                risk_level=RiskLevel.MEDIUM,
                message=f"Top5集中度过高: {risk_metrics.concentration_ratio:.1%}",
                current_value=risk_metrics.concentration_ratio,
                threshold=0.6,
                suggested_action="增加组合分散化"
            ))
        
        return alerts
    
    def _analyze_position_risks(self,
                               positions: Dict[str, float],
                               market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """分析单个头寸风险"""
        
        position_risks = {}
        
        for symbol, weight in positions.items():
            if abs(weight) < 0.001:  # 忽略极小权重
                continue
            
            risk_info = {
                'weight': weight,
                'volatility': 0,
                'max_drawdown': 0,
                'liquidity_score': 0,
                'risk_contribution': 0
            }
            
            try:
                if symbol in market_data:
                    df = market_data[symbol]
                    
                    # 个股波动率
                    if 'close' in df.columns:
                        returns = df['close'].pct_change().dropna().tail(60)
                        if len(returns) > 1:
                            risk_info['volatility'] = returns.std() * np.sqrt(252)
                            
                            # 个股最大回撤
                            cumulative = (1 + returns).cumprod()
                            running_max = cumulative.expanding().max()
                            drawdowns = (cumulative - running_max) / running_max
                            risk_info['max_drawdown'] = abs(drawdowns.min())
                    
                    # 流动性评分
                    if 'vol' in df.columns:
                        avg_volume = df['vol'].tail(20).mean()
                        risk_info['liquidity_score'] = min(1.0, avg_volume / self.min_daily_volume)
                
                # 风险贡献（简化计算）
                risk_info['risk_contribution'] = abs(weight) * risk_info['volatility']
                
            except Exception as e:
                logger.debug(f"分析 {symbol} 风险失败: {e}")
            
            position_risks[symbol] = risk_info
        
        return position_risks
    
    def _assess_market_conditions(self,
                                 market_data: Dict[str, pd.DataFrame],
                                 benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """评估市场条件"""
        
        conditions = {
            'market_stress': False,
            'volatility_regime': 'normal',
            'correlation_regime': 'normal',
            'liquidity_conditions': 'normal'
        }
        
        try:
            # 计算市场平均收益率（简化）
            if market_data:
                recent_returns = []
                for symbol, df in market_data.items():
                    if 'close' in df.columns:
                        returns = df['close'].pct_change().dropna().tail(5)
                        if len(returns) > 0:
                            recent_returns.extend(returns.tolist())
                
                if recent_returns:
                    avg_return = np.mean(recent_returns)
                    
                    # 市场压力检测
                    if avg_return < self.market_stress_threshold:
                        conditions['market_stress'] = True
                    
                    # 波动率状态
                    return_std = np.std(recent_returns)
                    if return_std > 0.03:  # 日波动率超过3%
                        conditions['volatility_regime'] = 'high'
                    elif return_std < 0.01:
                        conditions['volatility_regime'] = 'low'
        
        except Exception as e:
            logger.warning(f"市场条件评估失败: {e}")
        
        return conditions
    
    def _generate_risk_recommendations(self,
                                     risk_metrics: RiskMetrics,
                                     alerts: List[RiskAlert],
                                     position_risks: Dict[str, Dict],
                                     market_conditions: Dict[str, Any]) -> List[str]:
        """生成风险管理建议"""
        
        recommendations = []
        
        # 基于预警生成建议
        critical_alerts = [a for a in alerts if a.risk_level == RiskLevel.CRITICAL]
        high_alerts = [a for a in alerts if a.risk_level == RiskLevel.HIGH]
        
        if critical_alerts:
            recommendations.append("⚠️ 检测到严重风险，建议立即减仓或止损")
        
        if high_alerts:
            recommendations.append("⚡ 检测到高风险，建议调整持仓结构")
        
        # 基于风险指标生成建议
        if risk_metrics.sharpe_ratio < 0.5:
            recommendations.append("📊 夏普比率偏低，考虑优化持仓配置")
        
        if risk_metrics.concentration_ratio > 0.5:
            recommendations.append("🎯 持仓集中度较高，建议增加分散化")
        
        if risk_metrics.liquidity_ratio < 0.6:
            recommendations.append("💧 流动性不足，增加高流动性股票配置")
        
        # 基于市场条件生成建议
        if market_conditions.get('market_stress'):
            recommendations.append("🌊 市场处于压力状态，建议降低杠杆和风险敞口")
        
        if market_conditions.get('volatility_regime') == 'high':
            recommendations.append("⚡ 高波动市场环境，建议缩短持仓周期")
        
        return recommendations
    
    def _create_risk_control_action(self,
                                   alert: RiskAlert,
                                   positions: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """创建风险控制指令"""
        
        if alert.alert_type == AlertType.POSITION_LIMIT and alert.symbol:
            # 头寸限制：减少特定股票持仓
            current_weight = positions.get(alert.symbol, 0)
            target_weight = min(current_weight, self.max_position_weight)
            
            return {
                'action_type': 'position_adjustment',
                'symbol': alert.symbol,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'reason': alert.message,
                'priority': 'high' if alert.risk_level == RiskLevel.CRITICAL else 'medium'
            }
        
        elif alert.alert_type == AlertType.DRAWDOWN:
            # 回撤控制：全面减仓
            return {
                'action_type': 'portfolio_scale_down',
                'scale_factor': 0.8,  # 减仓20%
                'reason': alert.message,
                'priority': 'critical'
            }
        
        return None
    
    def _check_stop_loss_take_profit(self,
                                    positions: Dict[str, float],
                                    risk_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检查止损止盈"""
        
        actions = []
        
        try:
            position_risks = risk_report.get('position_risks', {})
            
            for symbol, weight in positions.items():
                if abs(weight) < 0.001:
                    continue
                
                position_info = position_risks.get(symbol, {})
                
                # 简化的止损检查（基于最大回撤）
                max_drawdown = position_info.get('max_drawdown', 0)
                
                if max_drawdown > abs(self.stop_loss_threshold):
                    actions.append({
                        'action_type': 'stop_loss',
                        'symbol': symbol,
                        'current_weight': weight,
                        'target_weight': 0,
                        'trigger_value': max_drawdown,
                        'reason': f'{symbol} 触发止损：回撤 {max_drawdown:.1%}',
                        'priority': 'high'
                    })
        
        except Exception as e:
            logger.warning(f"止损止盈检查失败: {e}")
        
        return actions
    
    def _update_risk_history(self, risk_report: Dict[str, Any]):
        """更新风险历史记录"""
        
        summary = {
            'timestamp': risk_report['timestamp'],
            'portfolio_volatility': risk_report['risk_metrics'].get('portfolio_volatility', 0),
            'max_drawdown': risk_report['risk_metrics'].get('max_drawdown', 0),
            'alert_count': len(risk_report.get('risk_alerts', [])),
            'critical_alert_count': len([
                a for a in risk_report.get('risk_alerts', []) 
                if a.get('risk_level') == 'critical'
            ])
        }
        
        self._risk_history.append(summary)
        
        # 保持历史记录在合理长度
        if len(self._risk_history) > 1000:
            self._risk_history = self._risk_history[-1000:]
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险管理总结"""
        
        if not self._risk_history:
            return {'status': 'no_data'}
        
        recent_records = self._risk_history[-30:] if len(self._risk_history) >= 30 else self._risk_history
        
        summary = {
            'current_status': {
                'portfolio_volatility': recent_records[-1]['portfolio_volatility'],
                'max_drawdown': recent_records[-1]['max_drawdown'],
                'recent_alerts': recent_records[-1]['alert_count']
            },
            'risk_trends': {
                'avg_volatility': np.mean([r['portfolio_volatility'] for r in recent_records]),
                'max_drawdown': max([r['max_drawdown'] for r in recent_records]),
                'alert_frequency': np.mean([r['alert_count'] for r in recent_records])
            },
            'risk_violations': {
                'volatility_breaches': sum([
                    1 for r in recent_records 
                    if r['portfolio_volatility'] > self.max_portfolio_volatility
                ]),
                'drawdown_breaches': sum([
                    1 for r in recent_records 
                    if r['max_drawdown'] > self.max_portfolio_drawdown
                ])
            },
            'records_count': len(self._risk_history)
        }
        
        return summary