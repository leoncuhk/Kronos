#!/usr/bin/env python3
"""
Kronos Alpha因子生成器
量化交易四步法的第二步：生成核心Alpha因子

核心理念：
将Kronos模型定位为"概率优势排序引擎"，而非"单点预测神器"
通过多路径采样和不确定性量化，生成风险调整后的Alpha信号

主要功能：
1. 基于Kronos模型的多路径预测采样
2. 计算"H天平均预期收益率"作为基础信号  
3. 预测未来H天的已实现波动率
4. 构建风险调整信号：预期收益率 / 预期波动率
5. 不确定性量化和置信区间计算

作者: Kronos Team
日期: 2025-01-26
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    # 导入主项目的模型
    from model import Kronos, KronosTokenizer
    from finetune.config import BaseConfig
except ImportError as e:
    raise ImportError(f"无法导入Kronos模型: {e}. 请确保在项目根目录运行并已完成模型训练")

logger = logging.getLogger(__name__)

class KronosAlphaGenerator:
    """
    Kronos Alpha因子生成器
    
    核心职责：
    1. 将Kronos定位为概率排序引擎
    2. 通过Monte Carlo采样获得预测分布
    3. 计算风险调整后的Alpha信号
    4. 提供不确定性量化指标
    """
    
    def __init__(self,
                 tokenizer_path: Optional[str] = None,
                 predictor_path: Optional[str] = None,
                 config: Optional[BaseConfig] = None,
                 device: str = "auto",
                 
                 # 预测参数
                 lookback_window: int = 90,        # 历史数据回望期
                 prediction_horizon: int = 10,    # 预测天数 H
                 sample_count: int = 20,           # Monte Carlo采样次数
                 
                 # 信号参数  
                 temperature: float = 0.8,         # 采样温度
                 top_k: int = 50,                  # Top-k采样
                 top_p: float = 0.9,               # Nucleus采样
                 
                 # 风险调整参数
                 risk_adjustment: bool = True,     # 是否进行风险调整
                 volatility_lookback: int = 60,    # 波动率计算回望期
                 ):
        """
        初始化Alpha因子生成器
        
        Args:
            tokenizer_path: Tokenizer模型路径
            predictor_path: Predictor模型路径  
            config: 配置对象
            device: 计算设备
            lookback_window: 历史数据窗口
            prediction_horizon: 预测时间范围
            sample_count: 蒙特卡洛采样数量
            temperature: 采样温度参数
            top_k: Top-k采样参数
            top_p: Nucleus采样参数
            risk_adjustment: 是否启用风险调整
            volatility_lookback: 波动率计算窗口
        """
        self.config = config if config is not None else BaseConfig()
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu")
        
        # 模型参数
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.sample_count = sample_count
        
        # 采样参数
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # 风险调整参数
        self.risk_adjustment = risk_adjustment
        self.volatility_lookback = volatility_lookback
        
        # 加载模型
        self.tokenizer = None
        self.predictor = None
        self._load_models(tokenizer_path, predictor_path)
        
        # 缓存计算结果
        self._signal_cache = {}
        
        logger.info(f"KronosAlphaGenerator initialized: "
                   f"device={self.device}, "
                   f"lookback={lookback_window}, "
                   f"horizon={prediction_horizon}, "
                   f"samples={sample_count}")
    
    def _load_models(self, tokenizer_path: Optional[str], predictor_path: Optional[str]):
        """加载Kronos模型"""
        try:
            # 确定模型路径
            if tokenizer_path is None:
                # 尝试使用微调后的模型，如果不存在则使用预训练模型
                finetuned_path = self.config.finetuned_tokenizer_path
                if os.path.exists(finetuned_path):
                    tokenizer_path = finetuned_path
                else:
                    tokenizer_path = self.config.pretrained_tokenizer_path
            
            if predictor_path is None:
                finetuned_path = self.config.finetuned_predictor_path
                if os.path.exists(finetuned_path):
                    predictor_path = finetuned_path
                else:
                    predictor_path = self.config.pretrained_predictor_path
            
            logger.info(f"加载Tokenizer: {tokenizer_path}")
            self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.to(self.device)
            self.tokenizer.eval()
            
            logger.info(f"加载Predictor: {predictor_path}")
            self.predictor = Kronos.from_pretrained(predictor_path)
            self.predictor.to(self.device) 
            self.predictor.eval()
            
            logger.info("✅ Kronos模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate_alpha_signals(self,
                             data: Dict[str, pd.DataFrame],
                             symbols: List[str],
                             as_of_date: str) -> pd.DataFrame:
        """
        为股票池生成Alpha信号
        
        Args:
            data: 历史股票数据
            symbols: 股票代码列表
            as_of_date: 信号生成日期
            
        Returns:
            包含Alpha信号的DataFrame，包含列：
            - symbol: 股票代码
            - expected_return: 预期收益率（H天平均）
            - expected_volatility: 预期波动率
            - risk_adjusted_score: 风险调整后评分
            - prediction_confidence: 预测置信度
            - signal_strength: 信号强度
        """
        cache_key = f"{'-'.join(sorted(symbols))}_{as_of_date}"
        
        # 检查缓存
        if cache_key in self._signal_cache:
            logger.info(f"从缓存返回Alpha信号: {as_of_date}")
            return self._signal_cache[cache_key]
        
        logger.info(f"为 {len(symbols)} 只股票生成Alpha信号: {as_of_date}")
        
        signals = []
        eval_date = pd.to_datetime(as_of_date)
        
        for i, symbol in enumerate(symbols):
            if i % 20 == 0:
                logger.info(f"处理进度: {i+1}/{len(symbols)}")
            
            try:
                # 生成单只股票的信号
                signal = self._generate_single_signal(data, symbol, eval_date)
                if signal is not None:
                    signals.append(signal)
                    
            except Exception as e:
                logger.warning(f"生成 {symbol} 信号失败: {e}")
                continue
        
        if not signals:
            logger.warning("未生成任何有效信号")
            return pd.DataFrame()
        
        # 转换为DataFrame
        signals_df = pd.DataFrame(signals)
        
        # 缓存结果
        self._signal_cache[cache_key] = signals_df
        
        logger.info(f"成功生成 {len(signals_df)} 个Alpha信号")
        return signals_df
    
    def _generate_single_signal(self,
                               data: Dict[str, pd.DataFrame],
                               symbol: str,
                               eval_date: pd.Timestamp) -> Optional[Dict]:
        """为单只股票生成Alpha信号"""
        
        if symbol not in data:
            return None
        
        df = data[symbol]
        
        # 获取用于预测的历史数据
        df_before = df[df.index <= eval_date].tail(self.lookback_window)
        
        if len(df_before) < self.lookback_window:
            logger.debug(f"{symbol}: 历史数据不足 ({len(df_before)} < {self.lookback_window})")
            return None
        
        try:
            # 准备输入数据
            input_data = self._prepare_input_data(df_before)
            
            # Monte Carlo预测采样
            predictions = self._monte_carlo_prediction(input_data)
            
            if predictions is None or len(predictions) == 0:
                return None
            
            # 计算Alpha信号指标
            signal_metrics = self._calculate_signal_metrics(
                predictions=predictions,
                historical_data=df_before,
                symbol=symbol
            )
            
            signal_metrics['symbol'] = symbol
            signal_metrics['signal_date'] = eval_date.strftime('%Y-%m-%d')
            
            return signal_metrics
            
        except Exception as e:
            logger.debug(f"计算 {symbol} 信号失败: {e}")
            return None
    
    def _prepare_input_data(self, df: pd.DataFrame) -> torch.Tensor:
        """准备模型输入数据"""
        
        # 使用配置中的特征列
        feature_cols = [col for col in self.config.feature_list if col in df.columns]
        
        if not feature_cols:
            raise ValueError("没有找到有效的特征列")
        
        # 提取特征数据
        features = df[feature_cols].values
        
        # 简单的标准化处理
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # 转换为tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        return input_tensor
    
    def _monte_carlo_prediction(self, input_data: torch.Tensor) -> Optional[np.ndarray]:
        """Monte Carlo预测采样"""
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.sample_count):
                try:
                    # Tokenizer编码
                    if hasattr(self.tokenizer, 'encode'):
                        encoded = self.tokenizer.encode(input_data)
                    else:
                        # 如果没有encode方法，直接使用forward
                        encoded = self.tokenizer(input_data)
                        if isinstance(encoded, tuple):
                            encoded = encoded[-1]  # 通常取最后一个输出作为token indices
                    
                    # Predictor预测，使用采样生成
                    if hasattr(self.predictor, 'generate'):
                        # 如果有generate方法，使用采样生成
                        pred = self.predictor.generate(
                            encoded,
                            max_length=encoded.shape[1] + self.prediction_horizon,
                            temperature=self.temperature,
                            top_k=self.top_k,
                            top_p=self.top_p,
                            do_sample=True
                        )
                    else:
                        # 如果没有generate方法，使用forward然后采样
                        logits = self.predictor(encoded)
                        
                        # 从logits中采样
                        if isinstance(logits, (list, tuple)):
                            logits = logits[0]  # 取第一个输出
                        
                        # 应用温度采样
                        scaled_logits = logits / self.temperature
                        probs = F.softmax(scaled_logits, dim=-1)
                        
                        # 简单采样（这里需要根据实际模型输出格式调整）
                        pred = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
                    
                    # 将预测结果转换为收益率（这里是简化处理）
                    pred_numpy = pred.cpu().numpy()
                    
                    # 将token预测转换为收益率预测（简化映射）
                    # 实际应用中需要根据tokenizer的具体实现来正确解码
                    returns = self._tokens_to_returns(pred_numpy)
                    
                    if returns is not None and len(returns) >= self.prediction_horizon:
                        predictions.append(returns[:self.prediction_horizon])
                        
                except Exception as e:
                    logger.debug(f"单次预测失败: {e}")
                    continue
        
        if len(predictions) < self.sample_count // 2:  # 至少成功一半
            logger.warning(f"预测成功率过低: {len(predictions)}/{self.sample_count}")
            return None
        
        return np.array(predictions)
    
    def _tokens_to_returns(self, tokens: np.ndarray) -> Optional[np.ndarray]:
        """
        将token预测转换为收益率预测
        这是一个简化的实现，实际需要根据tokenizer的编码方式来解码
        """
        try:
            # 简化处理：将token值映射到收益率范围
            # 实际应用中需要使用tokenizer的decode方法
            
            # 假设token值在0-1000范围，映射到-10%到10%的收益率
            normalized_tokens = tokens / 1000.0  # 标准化到0-1
            returns = (normalized_tokens - 0.5) * 0.2  # 映射到-10%到10%
            
            # 取平均并调整
            if len(returns.shape) > 1:
                returns = returns.mean(axis=0)
            
            return returns.flatten()
            
        except Exception as e:
            logger.debug(f"Token转换失败: {e}")
            return None
    
    def _calculate_signal_metrics(self,
                                predictions: np.ndarray,
                                historical_data: pd.DataFrame,
                                symbol: str) -> Dict:
        """计算Alpha信号指标"""
        
        # 1. 基础信号：H天平均预期收益率
        expected_return = predictions.mean()
        
        # 2. 预测不确定性：跨样本标准差
        prediction_std = predictions.std()
        
        # 3. 计算置信区间
        confidence_95 = np.percentile(predictions, [2.5, 97.5])
        
        # 4. 预期波动率估算
        if self.risk_adjustment:
            expected_volatility = self._estimate_volatility(predictions, historical_data)
        else:
            expected_volatility = prediction_std
        
        # 5. 风险调整评分
        if expected_volatility > 1e-6:
            risk_adjusted_score = expected_return / expected_volatility
        else:
            risk_adjusted_score = expected_return
        
        # 6. 信号强度（基于置信区间宽度的倒数）
        confidence_width = confidence_95[1] - confidence_95[0]
        signal_strength = 1.0 / (confidence_width + 1e-6)
        
        # 7. 预测置信度（正收益预测的比例）
        positive_predictions = (predictions > 0).mean()
        prediction_confidence = max(positive_predictions, 1 - positive_predictions)
        
        return {
            'expected_return': expected_return,
            'expected_volatility': expected_volatility,
            'risk_adjusted_score': risk_adjusted_score,
            'prediction_confidence': prediction_confidence,
            'signal_strength': signal_strength,
            'prediction_std': prediction_std,
            'confidence_lower': confidence_95[0],
            'confidence_upper': confidence_95[1],
            'sample_count': len(predictions)
        }
    
    def _estimate_volatility(self, 
                           predictions: np.ndarray,
                           historical_data: pd.DataFrame) -> float:
        """估算预期波动率"""
        
        try:
            # 方法1：基于预测分布的波动率
            pred_volatility = predictions.std()
            
            # 方法2：基于历史波动率的调整
            if 'close' in historical_data.columns:
                hist_returns = historical_data['close'].pct_change().dropna()
                
                if len(hist_returns) >= self.volatility_lookback:
                    hist_volatility = hist_returns.tail(self.volatility_lookback).std()
                    
                    # 结合预测和历史波动率
                    estimated_volatility = 0.6 * pred_volatility + 0.4 * hist_volatility
                else:
                    estimated_volatility = pred_volatility
            else:
                estimated_volatility = pred_volatility
            
            return estimated_volatility
            
        except Exception as e:
            logger.debug(f"波动率估算失败: {e}")
            return predictions.std()
    
    def rank_signals(self, signals_df: pd.DataFrame, 
                    ranking_method: str = "risk_adjusted") -> pd.DataFrame:
        """
        对Alpha信号进行排序
        
        Args:
            signals_df: 信号DataFrame
            ranking_method: 排序方法 ('expected_return', 'risk_adjusted', 'composite')
            
        Returns:
            包含排名的DataFrame
        """
        if signals_df.empty:
            return signals_df
        
        df = signals_df.copy()
        
        if ranking_method == "expected_return":
            # 按预期收益率排序
            df['rank'] = df['expected_return'].rank(ascending=False)
            df['percentile'] = df['expected_return'].rank(pct=True, ascending=False)
            
        elif ranking_method == "risk_adjusted":
            # 按风险调整评分排序
            df['rank'] = df['risk_adjusted_score'].rank(ascending=False)
            df['percentile'] = df['risk_adjusted_score'].rank(pct=True, ascending=False)
            
        elif ranking_method == "composite":
            # 综合评分：结合多个指标
            # 标准化各个指标
            df['norm_expected_return'] = (df['expected_return'] - df['expected_return'].mean()) / df['expected_return'].std()
            df['norm_risk_adjusted'] = (df['risk_adjusted_score'] - df['risk_adjusted_score'].mean()) / df['risk_adjusted_score'].std()
            df['norm_confidence'] = (df['prediction_confidence'] - df['prediction_confidence'].mean()) / df['prediction_confidence'].std()
            
            # 综合评分
            df['composite_score'] = (
                0.4 * df['norm_risk_adjusted'] +
                0.3 * df['norm_expected_return'] +
                0.3 * df['norm_confidence']
            )
            
            df['rank'] = df['composite_score'].rank(ascending=False)
            df['percentile'] = df['composite_score'].rank(pct=True, ascending=False)
        
        else:
            raise ValueError(f"不支持的排序方法: {ranking_method}")
        
        # 按排名排序
        df = df.sort_values('rank').reset_index(drop=True)
        
        logger.info(f"使用 {ranking_method} 方法完成信号排序")
        return df
    
    def get_top_signals(self, 
                       signals_df: pd.DataFrame,
                       top_n: int,
                       ranking_method: str = "risk_adjusted") -> pd.DataFrame:
        """
        获取Top N信号
        
        Args:
            signals_df: 信号DataFrame
            top_n: 选择的股票数量
            ranking_method: 排序方法
            
        Returns:
            Top N信号DataFrame
        """
        if signals_df.empty:
            return signals_df
        
        # 排序
        ranked_df = self.rank_signals(signals_df, ranking_method)
        
        # 选择Top N
        top_signals = ranked_df.head(top_n)
        
        logger.info(f"选择Top {len(top_signals)} 信号 (方法: {ranking_method})")
        return top_signals