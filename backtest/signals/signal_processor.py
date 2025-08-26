#!/usr/bin/env python3
"""
信号处理器模块
处理和优化从KronosAlphaGenerator生成的原始信号

核心功能：
1. 信号平滑和噪音过滤
2. 信号稳定性检验
3. 多周期信号融合
4. 信号质量评估

作者: Kronos Team
日期: 2025-01-26
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SignalProcessor:
    """
    信号处理器
    
    核心功能：
    1. 信号质量控制和异常值处理
    2. 多时间周期信号融合
    3. 信号稳定性和持续性分析
    4. 信号衰减和时间加权
    """
    
    def __init__(self,
                 # 信号过滤参数
                 outlier_method: str = "iqr",           # 异常值检测方法
                 outlier_threshold: float = 3.0,        # 异常值阈值
                 
                 # 信号平滑参数
                 smoothing_window: int = 5,              # 平滑窗口
                 smoothing_method: str = "ewm",          # 平滑方法
                 alpha: float = 0.3,                     # EWM衰减因子
                 
                 # 信号稳定性参数
                 stability_window: int = 10,             # 稳定性检验窗口
                 min_consistency: float = 0.6,          # 最小一致性要求
                 
                 # 多周期融合参数
                 enable_multi_horizon: bool = True,     # 启用多周期融合
                 horizon_weights: Optional[Dict] = None  # 不同周期权重
                 ):
        """
        初始化信号处理器
        
        Args:
            outlier_method: 异常值检测方法 ('iqr', 'zscore', 'isolation')
            outlier_threshold: 异常值检测阈值
            smoothing_window: 信号平滑窗口大小
            smoothing_method: 平滑方法 ('ma', 'ewm', 'savgol')
            alpha: 指数加权移动平均的衰减因子
            stability_window: 信号稳定性检验窗口
            min_consistency: 信号方向一致性的最小要求
            enable_multi_horizon: 是否启用多时间周期融合
            horizon_weights: 不同预测周期的权重
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        self.smoothing_window = smoothing_window
        self.smoothing_method = smoothing_method
        self.alpha = alpha
        
        self.stability_window = stability_window
        self.min_consistency = min_consistency
        
        self.enable_multi_horizon = enable_multi_horizon
        self.horizon_weights = horizon_weights or {5: 0.4, 10: 0.4, 20: 0.2}
        
        # 信号历史缓存
        self._signal_history = {}
        
        logger.info(f"SignalProcessor initialized: "
                   f"outlier_method={outlier_method}, "
                   f"smoothing={smoothing_method}, "
                   f"stability_window={stability_window}")
    
    def process_signals(self, 
                       raw_signals: pd.DataFrame,
                       date: str,
                       previous_signals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        处理原始Alpha信号
        
        Args:
            raw_signals: 原始信号DataFrame
            date: 信号日期
            previous_signals: 历史信号数据（用于稳定性检验）
            
        Returns:
            处理后的信号DataFrame，包含新列：
            - processed_score: 处理后的评分
            - signal_quality: 信号质量评分
            - stability_score: 信号稳定性评分
            - final_weight: 最终权重
        """
        if raw_signals.empty:
            logger.warning("输入信号为空")
            return raw_signals
        
        logger.info(f"处理 {len(raw_signals)} 个原始信号: {date}")
        
        # 1. 复制原始数据
        processed_df = raw_signals.copy()
        
        # 2. 异常值检测和处理
        processed_df = self._handle_outliers(processed_df)
        
        # 3. 信号标准化
        processed_df = self._normalize_signals(processed_df)
        
        # 4. 信号平滑处理
        if previous_signals is not None:
            processed_df = self._smooth_signals(processed_df, previous_signals, date)
        
        # 5. 信号质量评估
        processed_df = self._assess_signal_quality(processed_df)
        
        # 6. 信号稳定性检验
        if previous_signals is not None:
            processed_df = self._assess_signal_stability(processed_df, previous_signals)
        else:
            processed_df['stability_score'] = 1.0  # 首次信号默认稳定性为1
        
        # 7. 计算最终权重
        processed_df = self._calculate_final_weights(processed_df)
        
        # 8. 更新信号历史
        self._update_signal_history(processed_df, date)
        
        logger.info(f"信号处理完成，最终有效信号: {len(processed_df)}")
        return processed_df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """异常值检测和处理"""
        
        processed_df = df.copy()
        
        # 对主要信号列进行异常值检测
        signal_cols = ['expected_return', 'risk_adjusted_score', 'prediction_confidence']
        
        for col in signal_cols:
            if col not in df.columns:
                continue
                
            try:
                if self.outlier_method == "iqr":
                    # IQR方法
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - self.outlier_threshold * IQR
                    upper_bound = Q3 + self.outlier_threshold * IQR
                    
                    # 将异常值限制在边界内
                    processed_df[col] = df[col].clip(lower_bound, upper_bound)
                    
                elif self.outlier_method == "zscore":
                    # Z-Score方法
                    z_scores = np.abs(stats.zscore(df[col]))
                    outlier_mask = z_scores > self.outlier_threshold
                    
                    # 将异常值替换为中位数
                    median_val = df[col].median()
                    processed_df.loc[outlier_mask, col] = median_val
                    
                outlier_count = (df[col] != processed_df[col]).sum()
                if outlier_count > 0:
                    logger.info(f"处理 {col} 列异常值: {outlier_count} 个")
                    
            except Exception as e:
                logger.warning(f"异常值处理失败 {col}: {e}")
                continue
        
        return processed_df
    
    def _normalize_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """信号标准化处理"""
        
        processed_df = df.copy()
        
        # 对主要信号列进行标准化
        signal_cols = ['expected_return', 'risk_adjusted_score']
        
        for col in signal_cols:
            if col not in df.columns:
                continue
                
            try:
                # 使用RobustScaler减少异常值影响
                scaler = RobustScaler()
                processed_df[f'{col}_normalized'] = scaler.fit_transform(
                    df[[col]].values
                ).flatten()
                
            except Exception as e:
                logger.warning(f"信号标准化失败 {col}: {e}")
                processed_df[f'{col}_normalized'] = df[col]
        
        return processed_df
    
    def _smooth_signals(self, 
                       current_df: pd.DataFrame,
                       previous_df: pd.DataFrame,
                       date: str) -> pd.DataFrame:
        """信号平滑处理"""
        
        processed_df = current_df.copy()
        
        # 找到共同股票
        common_symbols = set(current_df['symbol']) & set(previous_df['symbol'])
        
        if not common_symbols:
            logger.info("没有共同股票，跳过信号平滑")
            return processed_df
        
        # 对共同股票的信号进行平滑
        signal_cols = ['risk_adjusted_score', 'expected_return']
        
        for col in signal_cols:
            if col not in current_df.columns or col not in previous_df.columns:
                continue
                
            try:
                if self.smoothing_method == "ewm":
                    # 指数加权移动平均
                    for symbol in common_symbols:
                        current_mask = processed_df['symbol'] == symbol
                        previous_mask = previous_df['symbol'] == symbol
                        
                        if current_mask.any() and previous_mask.any():
                            current_val = processed_df.loc[current_mask, col].iloc[0]
                            previous_val = previous_df.loc[previous_mask, col].iloc[0]
                            
                            # EWM平滑
                            smoothed_val = self.alpha * current_val + (1 - self.alpha) * previous_val
                            processed_df.loc[current_mask, f'{col}_smoothed'] = smoothed_val
                        else:
                            # 如果没有历史数据，使用当前值
                            processed_df.loc[current_mask, f'{col}_smoothed'] = processed_df.loc[current_mask, col]
                
                # 对新股票，使用原始值
                new_symbols = set(current_df['symbol']) - common_symbols
                for symbol in new_symbols:
                    mask = processed_df['symbol'] == symbol
                    processed_df.loc[mask, f'{col}_smoothed'] = processed_df.loc[mask, col]
                    
            except Exception as e:
                logger.warning(f"信号平滑失败 {col}: {e}")
                processed_df[f'{col}_smoothed'] = processed_df[col]
        
        return processed_df
    
    def _assess_signal_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """信号质量评估"""
        
        processed_df = df.copy()
        
        try:
            # 信号质量指标
            quality_scores = []
            
            for idx, row in df.iterrows():
                quality_score = 0.0
                
                # 1. 预测置信度权重 (30%)
                confidence_score = row.get('prediction_confidence', 0.5)
                quality_score += 0.3 * confidence_score
                
                # 2. 信号强度权重 (25%)
                signal_strength = row.get('signal_strength', 1.0)
                # 标准化信号强度到0-1范围
                normalized_strength = min(signal_strength / df['signal_strength'].quantile(0.9), 1.0)
                quality_score += 0.25 * normalized_strength
                
                # 3. 预测样本数权重 (20%)
                sample_count = row.get('sample_count', 1)
                # 样本数越多，质量越高
                sample_score = min(sample_count / 20.0, 1.0)  # 20个样本为满分
                quality_score += 0.2 * sample_score
                
                # 4. 置信区间宽度权重 (25%) - 宽度越小越好
                conf_upper = row.get('confidence_upper', row.get('expected_return', 0) + 0.1)
                conf_lower = row.get('confidence_lower', row.get('expected_return', 0) - 0.1)
                conf_width = abs(conf_upper - conf_lower)
                
                # 将置信区间宽度转换为质量评分（宽度越小，评分越高）
                width_score = 1.0 / (1.0 + conf_width * 10)  # 调整系数
                quality_score += 0.25 * width_score
                
                quality_scores.append(max(0.0, min(1.0, quality_score)))
            
            processed_df['signal_quality'] = quality_scores
            
            logger.info(f"信号质量评估完成，平均质量: {np.mean(quality_scores):.3f}")
            
        except Exception as e:
            logger.warning(f"信号质量评估失败: {e}")
            processed_df['signal_quality'] = 1.0
        
        return processed_df
    
    def _assess_signal_stability(self, 
                                current_df: pd.DataFrame,
                                previous_df: pd.DataFrame) -> pd.DataFrame:
        """信号稳定性评估"""
        
        processed_df = current_df.copy()
        stability_scores = []
        
        for idx, row in current_df.iterrows():
            symbol = row['symbol']
            stability_score = 1.0  # 默认稳定性评分
            
            try:
                # 查找历史信号
                historical_signals = self._get_historical_signals(symbol)
                
                if len(historical_signals) >= 2:
                    # 计算信号方向一致性
                    current_signal = row.get('risk_adjusted_score', 0)
                    
                    # 检查最近几期信号的方向一致性
                    recent_signals = historical_signals[-self.stability_window:]
                    
                    if len(recent_signals) > 1:
                        # 计算信号方向的一致性
                        current_direction = 1 if current_signal > 0 else -1
                        historical_directions = [1 if s > 0 else -1 for s in recent_signals]
                        
                        # 计算与历史方向的一致性
                        consistency = sum(1 for d in historical_directions if d == current_direction) / len(historical_directions)
                        
                        # 信号强度的变化稳定性
                        signal_changes = np.diff(recent_signals + [current_signal])
                        volatility_penalty = np.std(signal_changes) if len(signal_changes) > 1 else 0
                        
                        # 综合稳定性评分
                        stability_score = (
                            0.7 * consistency +  # 方向一致性权重
                            0.3 * max(0, 1 - volatility_penalty)  # 波动性惩罚
                        )
                
                stability_scores.append(max(0.0, min(1.0, stability_score)))
                
            except Exception as e:
                logger.debug(f"计算 {symbol} 稳定性失败: {e}")
                stability_scores.append(1.0)
        
        processed_df['stability_score'] = stability_scores
        
        avg_stability = np.mean(stability_scores)
        logger.info(f"信号稳定性评估完成，平均稳定性: {avg_stability:.3f}")
        
        return processed_df
    
    def _calculate_final_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算最终信号权重"""
        
        processed_df = df.copy()
        
        try:
            # 综合权重计算
            final_weights = []
            
            for idx, row in df.iterrows():
                # 基础权重来自风险调整评分
                base_weight = row.get('risk_adjusted_score', 0)
                
                # 质量调整
                quality_score = row.get('signal_quality', 1.0)
                
                # 稳定性调整
                stability_score = row.get('stability_score', 1.0)
                
                # 最终权重 = 基础权重 × 质量因子 × 稳定性因子
                final_weight = base_weight * quality_score * stability_score
                
                final_weights.append(final_weight)
            
            processed_df['final_weight'] = final_weights
            
            # 标准化权重到合理范围
            if len(final_weights) > 0:
                weights_array = np.array(final_weights)
                processed_df['final_weight_normalized'] = (
                    weights_array / (np.abs(weights_array).max() + 1e-8)
                )
            
            logger.info("最终权重计算完成")
            
        except Exception as e:
            logger.warning(f"权重计算失败: {e}")
            processed_df['final_weight'] = processed_df.get('risk_adjusted_score', 1.0)
            processed_df['final_weight_normalized'] = processed_df['final_weight']
        
        return processed_df
    
    def _get_historical_signals(self, symbol: str) -> List[float]:
        """获取股票的历史信号"""
        return self._signal_history.get(symbol, [])
    
    def _update_signal_history(self, df: pd.DataFrame, date: str):
        """更新信号历史记录"""
        for idx, row in df.iterrows():
            symbol = row['symbol']
            signal_value = row.get('risk_adjusted_score', 0)
            
            if symbol not in self._signal_history:
                self._signal_history[symbol] = []
            
            self._signal_history[symbol].append(signal_value)
            
            # 保持历史记录在合理长度内
            if len(self._signal_history[symbol]) > 50:  # 保留最近50期
                self._signal_history[symbol] = self._signal_history[symbol][-50:]
    
    def get_signal_statistics(self, df: pd.DataFrame) -> Dict:
        """获取信号统计信息"""
        if df.empty:
            return {}
        
        try:
            stats_dict = {
                'total_signals': len(df),
                'avg_expected_return': df['expected_return'].mean() if 'expected_return' in df.columns else 0,
                'avg_risk_adjusted_score': df['risk_adjusted_score'].mean() if 'risk_adjusted_score' in df.columns else 0,
                'avg_signal_quality': df['signal_quality'].mean() if 'signal_quality' in df.columns else 0,
                'avg_stability_score': df['stability_score'].mean() if 'stability_score' in df.columns else 0,
                'positive_signals_ratio': (df['expected_return'] > 0).mean() if 'expected_return' in df.columns else 0,
                'high_quality_signals': (df['signal_quality'] > 0.7).sum() if 'signal_quality' in df.columns else 0,
                'stable_signals': (df['stability_score'] > 0.7).sum() if 'stability_score' in df.columns else 0
            }
            
            return stats_dict
            
        except Exception as e:
            logger.warning(f"统计信息计算失败: {e}")
            return {'total_signals': len(df)}
    
    def filter_signals(self, 
                      df: pd.DataFrame,
                      min_quality: float = 0.5,
                      min_stability: float = 0.5,
                      min_confidence: float = 0.6) -> pd.DataFrame:
        """
        根据质量标准过滤信号
        
        Args:
            df: 信号DataFrame
            min_quality: 最小信号质量要求
            min_stability: 最小稳定性要求
            min_confidence: 最小预测置信度要求
            
        Returns:
            过滤后的信号DataFrame
        """
        if df.empty:
            return df
        
        original_count = len(df)
        
        # 应用过滤条件
        filtered_df = df.copy()
        
        if 'signal_quality' in df.columns:
            filtered_df = filtered_df[filtered_df['signal_quality'] >= min_quality]
        
        if 'stability_score' in df.columns:
            filtered_df = filtered_df[filtered_df['stability_score'] >= min_stability]
        
        if 'prediction_confidence' in df.columns:
            filtered_df = filtered_df[filtered_df['prediction_confidence'] >= min_confidence]
        
        filtered_count = len(filtered_df)
        
        logger.info(f"信号过滤: {filtered_count}/{original_count} 信号通过质量检验")
        
        return filtered_df.reset_index(drop=True)