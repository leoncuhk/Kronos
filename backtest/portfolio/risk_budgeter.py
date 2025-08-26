#!/usr/bin/env python3
"""
风险预算器 (Risk Budgeter)
处理投资组合的风险分配和约束管理

核心功能：
1. 风险预算分配和监控
2. 风险贡献度分解
3. 风险约束的实时检查
4. 动态风险限额调整

设计理念：
- 风险优先：先确定可承受的风险，再分配资本
- 动态调整：根据市场条件调整风险预算
- 透明监控：提供清晰的风险贡献分解

作者: Kronos Team
日期: 2025-01-26
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskBudgeter:
    """
    投资组合风险预算器
    
    核心功能：
    1. 设定和监控风险预算限制
    2. 计算风险贡献度
    3. 动态调整风险分配
    4. 风险约束优化
    """
    
    def __init__(self,
                 # 总体风险预算
                 portfolio_volatility_limit: float = 0.15,    # 组合波动率上限
                 max_drawdown_limit: float = 0.08,            # 最大回撤限制
                 var_confidence_level: float = 0.95,          # VaR置信度
                 
                 # 个股风险预算
                 max_stock_risk_contribution: float = 0.15,   # 单只股票最大风险贡献
                 max_sector_risk_contribution: float = 0.30,  # 单个行业最大风险贡献
                 
                 # 风险因子预算
                 max_factor_exposure: Dict[str, float] = None, # 风险因子暴露限制
                 
                 # 相关性控制
                 max_correlation_threshold: float = 0.7,      # 最大相关性阈值
                 correlation_cluster_limit: int = 5,          # 高相关股票集群数量限制
                 
                 # 动态风险调整
                 risk_scaling_factor: float = 1.0,            # 风险缩放因子
                 volatility_lookback: int = 60,               # 波动率估计回望期
                 
                 # 风险预警阈值
                 risk_warning_threshold: float = 0.8          # 风险预警阈值（相对于限额）
                 ):
        """
        初始化风险预算器
        
        Args:
            portfolio_volatility_limit: 组合总波动率限制
            max_drawdown_limit: 最大回撤限制
            var_confidence_level: VaR计算置信度
            max_stock_risk_contribution: 单只股票最大风险贡献比例
            max_sector_risk_contribution: 单个行业最大风险贡献比例
            max_factor_exposure: 风险因子暴露限制字典
            max_correlation_threshold: 相关性阈值
            correlation_cluster_limit: 高相关股票数量限制
            risk_scaling_factor: 风险调整因子
            volatility_lookback: 波动率计算窗口
            risk_warning_threshold: 风险预警阈值
        """
        # 总体风险限制
        self.portfolio_volatility_limit = portfolio_volatility_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.var_confidence_level = var_confidence_level
        
        # 个股和行业风险限制
        self.max_stock_risk_contribution = max_stock_risk_contribution
        self.max_sector_risk_contribution = max_sector_risk_contribution
        
        # 风险因子限制
        self.max_factor_exposure = max_factor_exposure or {
            'market_beta': 1.5,
            'size_factor': 0.3,
            'value_factor': 0.3,
            'momentum_factor': 0.3,
            'quality_factor': 0.3
        }
        
        # 相关性控制
        self.max_correlation_threshold = max_correlation_threshold
        self.correlation_cluster_limit = correlation_cluster_limit
        
        # 动态调整参数
        self.risk_scaling_factor = risk_scaling_factor
        self.volatility_lookback = volatility_lookback
        self.risk_warning_threshold = risk_warning_threshold
        
        # 风险监控历史
        self._risk_history = []
        
        logger.info(f"RiskBudgeter initialized: "
                   f"vol_limit={portfolio_volatility_limit:.1%}, "
                   f"max_stock_contrib={max_stock_risk_contribution:.1%}")
    
    def assess_portfolio_risk(self,
                             weights: Dict[str, float],
                             covariance_matrix: pd.DataFrame,
                             expected_returns: Optional[Dict[str, float]] = None,
                             historical_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, any]:
        """
        评估投资组合风险
        
        Args:
            weights: 股票权重字典
            covariance_matrix: 协方差矩阵
            expected_returns: 预期收益字典
            historical_data: 历史数据（用于回撤分析）
            
        Returns:
            风险评估结果字典
        """
        logger.info(f"评估组合风险，持仓数量: {len(weights)}")
        
        risk_assessment = {
            'total_risk': {},
            'risk_contributions': {},
            'risk_violations': [],
            'risk_warnings': [],
            'factor_exposures': {},
            'correlation_analysis': {}
        }
        
        try:
            if not weights:
                return risk_assessment
            
            # 1. 计算总体风险指标
            total_risk = self._calculate_total_risk(weights, covariance_matrix)
            risk_assessment['total_risk'] = total_risk
            
            # 2. 计算风险贡献度
            risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
            risk_assessment['risk_contributions'] = risk_contributions
            
            # 3. 检查风险约束违背
            violations = self._check_risk_violations(weights, total_risk, risk_contributions)
            risk_assessment['risk_violations'] = violations
            
            # 4. 生成风险预警
            warnings = self._generate_risk_warnings(total_risk, risk_contributions)
            risk_assessment['risk_warnings'] = warnings
            
            # 5. 分析相关性结构
            correlation_analysis = self._analyze_correlations(weights, covariance_matrix)
            risk_assessment['correlation_analysis'] = correlation_analysis
            
            # 6. 估计风险因子暴露（简化版本）
            factor_exposures = self._estimate_factor_exposures(weights, historical_data)
            risk_assessment['factor_exposures'] = factor_exposures
            
            # 7. 记录风险历史
            self._update_risk_history(risk_assessment)
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            risk_assessment['error'] = str(e)
        
        return risk_assessment
    
    def apply_risk_constraints(self,
                              target_weights: Dict[str, float],
                              covariance_matrix: pd.DataFrame,
                              expected_returns: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        应用风险约束，调整权重
        
        Args:
            target_weights: 目标权重
            covariance_matrix: 协方差矩阵
            expected_returns: 预期收益
            
        Returns:
            风险约束后的权重
        """
        if not target_weights:
            return target_weights
        
        logger.info("应用风险约束调整权重")
        
        try:
            # 1. 先检查当前权重的风险
            risk_assessment = self.assess_portfolio_risk(target_weights, covariance_matrix, expected_returns)
            
            # 2. 如果没有违规，直接返回
            if not risk_assessment.get('risk_violations'):
                logger.info("权重满足风险约束")
                return target_weights
            
            # 3. 使用优化方法调整权重
            adjusted_weights = self._optimize_weights_with_risk_constraints(
                target_weights, covariance_matrix, expected_returns
            )
            
            logger.info(f"权重调整完成，调整了 {len(adjusted_weights)} 个持仓")
            return adjusted_weights
            
        except Exception as e:
            logger.warning(f"风险约束应用失败: {e}，返回原权重")
            return target_weights
    
    def _calculate_total_risk(self, 
                             weights: Dict[str, float],
                             covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算总体风险指标"""
        
        symbols = list(weights.keys())
        weight_array = np.array([weights[s] for s in symbols])
        
        # 确保协方差矩阵包含所有股票
        available_symbols = [s for s in symbols if s in covariance_matrix.index]
        
        if not available_symbols:
            return {'portfolio_volatility': 0.0, 'var_95': 0.0, 'expected_shortfall': 0.0}
        
        # 过滤权重和协方差矩阵
        filtered_weights = np.array([weights[s] for s in available_symbols])
        cov_sub = covariance_matrix.loc[available_symbols, available_symbols].values
        
        # 组合方差和波动率
        portfolio_variance = np.dot(filtered_weights, np.dot(cov_sub, filtered_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # VaR计算（假设正态分布）
        from scipy import stats
        var_95 = portfolio_volatility * stats.norm.ppf(1 - self.var_confidence_level)
        
        # Expected Shortfall (CVaR)
        expected_shortfall = portfolio_volatility * stats.norm.pdf(stats.norm.ppf(1 - self.var_confidence_level)) / (1 - self.var_confidence_level)
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'portfolio_variance': portfolio_variance,
            'var_95': abs(var_95),  # 转为正数表示损失
            'expected_shortfall': abs(expected_shortfall),
            'risk_budget_utilization': portfolio_volatility / self.portfolio_volatility_limit
        }
    
    def _calculate_risk_contributions(self,
                                    weights: Dict[str, float],
                                    covariance_matrix: pd.DataFrame) -> Dict[str, any]:
        """计算各个持仓的风险贡献度"""
        
        symbols = list(weights.keys())
        available_symbols = [s for s in symbols if s in covariance_matrix.index]
        
        if not available_symbols:
            return {}
        
        filtered_weights = np.array([weights[s] for s in available_symbols])
        cov_sub = covariance_matrix.loc[available_symbols, available_symbols].values
        
        # 组合方差
        portfolio_variance = np.dot(filtered_weights, np.dot(cov_sub, filtered_weights))
        
        if portfolio_variance <= 0:
            return {}
        
        # 边际风险贡献 (Marginal Risk Contribution)
        marginal_contributions = np.dot(cov_sub, filtered_weights)
        
        # 风险贡献 (Risk Contribution) = 权重 × 边际风险贡献
        risk_contributions_abs = filtered_weights * marginal_contributions
        
        # 风险贡献比例
        risk_contributions_pct = risk_contributions_abs / portfolio_variance
        
        # 构建结果
        contributions = {}
        for i, symbol in enumerate(available_symbols):
            contributions[symbol] = {
                'absolute_contribution': risk_contributions_abs[i],
                'percentage_contribution': risk_contributions_pct[i],
                'marginal_contribution': marginal_contributions[i],
                'weight': filtered_weights[i]
            }
        
        return contributions
    
    def _check_risk_violations(self,
                              weights: Dict[str, float],
                              total_risk: Dict[str, float],
                              risk_contributions: Dict[str, any]) -> List[Dict]:
        """检查风险约束违背"""
        
        violations = []
        
        # 1. 检查总体波动率限制
        portfolio_vol = total_risk.get('portfolio_volatility', 0)
        if portfolio_vol > self.portfolio_volatility_limit:
            violations.append({
                'type': 'portfolio_volatility',
                'current_value': portfolio_vol,
                'limit': self.portfolio_volatility_limit,
                'excess': portfolio_vol - self.portfolio_volatility_limit,
                'severity': 'high'
            })
        
        # 2. 检查单个股票风险贡献
        for symbol, contrib in risk_contributions.items():
            contrib_pct = contrib['percentage_contribution']
            if contrib_pct > self.max_stock_risk_contribution:
                violations.append({
                    'type': 'stock_risk_contribution',
                    'symbol': symbol,
                    'current_value': contrib_pct,
                    'limit': self.max_stock_risk_contribution,
                    'excess': contrib_pct - self.max_stock_risk_contribution,
                    'severity': 'medium'
                })
        
        # 3. 检查VaR限制（可选）
        var_95 = total_risk.get('var_95', 0)
        max_var = self.max_drawdown_limit  # 使用最大回撤作为VaR的代理
        if var_95 > max_var:
            violations.append({
                'type': 'value_at_risk',
                'current_value': var_95,
                'limit': max_var,
                'excess': var_95 - max_var,
                'severity': 'high'
            })
        
        return violations
    
    def _generate_risk_warnings(self,
                               total_risk: Dict[str, float],
                               risk_contributions: Dict[str, any]) -> List[Dict]:
        """生成风险预警"""
        
        warnings = []
        
        # 1. 组合风险预警
        portfolio_vol = total_risk.get('portfolio_volatility', 0)
        risk_utilization = portfolio_vol / self.portfolio_volatility_limit
        
        if risk_utilization > self.risk_warning_threshold:
            warnings.append({
                'type': 'portfolio_risk_high',
                'message': f'组合风险使用率过高: {risk_utilization:.1%}',
                'risk_utilization': risk_utilization,
                'severity': 'medium'
            })
        
        # 2. 个股集中度预警
        if risk_contributions:
            max_contrib = max(contrib['percentage_contribution'] for contrib in risk_contributions.values())
            max_contrib_symbol = max(risk_contributions.keys(), 
                                   key=lambda s: risk_contributions[s]['percentage_contribution'])
            
            warning_threshold = self.max_stock_risk_contribution * self.risk_warning_threshold
            if max_contrib > warning_threshold:
                warnings.append({
                    'type': 'concentration_warning',
                    'message': f'股票 {max_contrib_symbol} 风险贡献过高: {max_contrib:.1%}',
                    'symbol': max_contrib_symbol,
                    'contribution': max_contrib,
                    'severity': 'medium'
                })
        
        # 3. 风险预算使用率预警
        budget_utilization = total_risk.get('risk_budget_utilization', 0)
        if budget_utilization > self.risk_warning_threshold:
            warnings.append({
                'type': 'risk_budget_high',
                'message': f'风险预算使用率: {budget_utilization:.1%}',
                'utilization': budget_utilization,
                'severity': 'low'
            })
        
        return warnings
    
    def _analyze_correlations(self,
                             weights: Dict[str, float],
                             covariance_matrix: pd.DataFrame) -> Dict[str, any]:
        """分析相关性结构"""
        
        symbols = [s for s in weights.keys() if s in covariance_matrix.index]
        
        if len(symbols) < 2:
            return {}
        
        try:
            # 计算相关系数矩阵
            cov_sub = covariance_matrix.loc[symbols, symbols]
            volatilities = np.sqrt(np.diag(cov_sub))
            correlation_matrix = cov_sub / np.outer(volatilities, volatilities)
            
            # 找出高相关性股票对
            high_corr_pairs = []
            n = len(symbols)
            
            for i in range(n):
                for j in range(i+1, n):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > self.max_correlation_threshold:
                        high_corr_pairs.append({
                            'stock1': symbols[i],
                            'stock2': symbols[j],
                            'correlation': corr,
                            'combined_weight': weights[symbols[i]] + weights[symbols[j]]
                        })
            
            # 计算加权平均相关性
            total_weight_sq = 0
            weighted_corr_sum = 0
            
            for i, symbol_i in enumerate(symbols):
                for j, symbol_j in enumerate(symbols):
                    if i != j:
                        weight_i = weights[symbol_i]
                        weight_j = weights[symbol_j]
                        corr = correlation_matrix.iloc[i, j]
                        
                        weighted_corr_sum += weight_i * weight_j * corr
                        total_weight_sq += weight_i * weight_j
            
            avg_correlation = weighted_corr_sum / total_weight_sq if total_weight_sq > 0 else 0
            
            return {
                'high_correlation_pairs': high_corr_pairs,
                'high_corr_count': len(high_corr_pairs),
                'weighted_avg_correlation': avg_correlation,
                'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
                'correlation_warnings': len(high_corr_pairs) > self.correlation_cluster_limit
            }
            
        except Exception as e:
            logger.warning(f"相关性分析失败: {e}")
            return {}
    
    def _estimate_factor_exposures(self,
                                  weights: Dict[str, float],
                                  historical_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, float]:
        """估计风险因子暴露（简化版本）"""
        
        # 这是一个简化的实现，实际应该使用更复杂的因子模型
        exposures = {
            'market_beta': 0.0,
            'size_factor': 0.0,
            'value_factor': 0.0,
            'momentum_factor': 0.0,
            'quality_factor': 0.0
        }
        
        if not historical_data or not weights:
            return exposures
        
        try:
            # 简化的Beta计算（假设有市场数据）
            # 实际应该使用多因子模型进行回归
            total_beta = 0.0
            total_weight = 0.0
            
            for symbol, weight in weights.items():
                if symbol in historical_data:
                    # 这里应该计算相对于市场基准的Beta
                    # 简化处理：假设Beta = 1 + 随机噪音
                    estimated_beta = 1.0 + np.random.normal(0, 0.3)  # 临时实现
                    total_beta += weight * estimated_beta
                    total_weight += weight
            
            if total_weight > 0:
                exposures['market_beta'] = total_beta / total_weight
            
        except Exception as e:
            logger.warning(f"因子暴露估计失败: {e}")
        
        return exposures
    
    def _optimize_weights_with_risk_constraints(self,
                                              target_weights: Dict[str, float],
                                              covariance_matrix: pd.DataFrame,
                                              expected_returns: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """使用优化方法应用风险约束"""
        
        symbols = list(target_weights.keys())
        available_symbols = [s for s in symbols if s in covariance_matrix.index]
        
        if not available_symbols:
            return target_weights
        
        try:
            n = len(available_symbols)
            
            # 目标权重向量
            target_array = np.array([target_weights[s] for s in available_symbols])
            
            # 协方差矩阵
            cov_sub = covariance_matrix.loc[available_symbols, available_symbols].values
            
            # 目标函数：最小化与目标权重的偏离，同时控制风险
            def objective(x):
                # 权重偏离惩罚
                weight_deviation = np.sum((x - target_array) ** 2)
                
                # 风险惩罚
                portfolio_variance = np.dot(x, np.dot(cov_sub, x))
                risk_penalty = max(0, np.sqrt(portfolio_variance) - self.portfolio_volatility_limit) ** 2
                
                return weight_deviation + 10 * risk_penalty  # 风险惩罚权重
            
            # 约束条件
            constraints = []
            
            # 权重和约束
            constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            
            # 风险约束
            def risk_constraint(x):
                portfolio_vol = np.sqrt(np.dot(x, np.dot(cov_sub, x)))
                return self.portfolio_volatility_limit - portfolio_vol
            
            constraints.append({'type': 'ineq', 'fun': risk_constraint})
            
            # 边界约束
            bounds = [(0, 0.1) for _ in range(n)]  # 假设最大10%权重限制
            
            # 优化求解
            result = minimize(objective, target_array, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = dict(zip(available_symbols, result.x))
                
                # 添加缺失的股票（权重为0）
                for symbol in symbols:
                    if symbol not in optimized_weights:
                        optimized_weights[symbol] = 0.0
                
                return optimized_weights
            else:
                logger.warning("风险约束优化失败")
                return target_weights
                
        except Exception as e:
            logger.warning(f"风险约束优化失败: {e}")
            return target_weights
    
    def _update_risk_history(self, risk_assessment: Dict[str, any]):
        """更新风险监控历史"""
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_volatility': risk_assessment.get('total_risk', {}).get('portfolio_volatility', 0),
            'risk_violations_count': len(risk_assessment.get('risk_violations', [])),
            'risk_warnings_count': len(risk_assessment.get('risk_warnings', [])),
            'max_stock_risk_contrib': max([
                contrib.get('percentage_contribution', 0) 
                for contrib in risk_assessment.get('risk_contributions', {}).values()
            ], default=0)
        }
        
        self._risk_history.append(record)
        
        # 保持历史记录在合理长度内
        if len(self._risk_history) > 1000:
            self._risk_history = self._risk_history[-1000:]
    
    def get_risk_budget_summary(self) -> Dict[str, any]:
        """获取风险预算使用总结"""
        
        if not self._risk_history:
            return {'status': 'no_data'}
        
        recent_records = self._risk_history[-30:] if len(self._risk_history) >= 30 else self._risk_history
        
        summary = {
            'current_risk_utilization': recent_records[-1]['portfolio_volatility'] / self.portfolio_volatility_limit,
            'avg_risk_utilization': np.mean([r['portfolio_volatility'] / self.portfolio_volatility_limit for r in recent_records]),
            'max_risk_utilization': max([r['portfolio_volatility'] / self.portfolio_volatility_limit for r in recent_records]),
            'violation_frequency': sum([1 for r in recent_records if r['risk_violations_count'] > 0]) / len(recent_records),
            'warning_frequency': sum([1 for r in recent_records if r['risk_warnings_count'] > 0]) / len(recent_records),
            'risk_budget_remaining': max(0, self.portfolio_volatility_limit - recent_records[-1]['portfolio_volatility']),
            'records_count': len(self._risk_history)
        }
        
        return summary