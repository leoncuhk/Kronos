#!/usr/bin/env python3
"""
执行管理器 (Execution Manager)
负责交易指令的执行、订单管理和执行分析

核心功能：
1. 交易指令的智能执行
2. 订单分拆和时间分散
3. 市场冲击成本控制
4. 执行绩效分析
5. 滑点和执行成本监控

设计原则：
- 成本最小化：最小化市场冲击和交易成本
- 执行效率：在合理时间内完成交易
- 风险控制：避免执行过程中的额外风险

作者: Kronos Team
日期: 2025-01-26
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"           # 市价单
    LIMIT = "limit"             # 限价单
    STOP = "stop"               # 止损单
    STOP_LIMIT = "stop_limit"   # 止损限价单
    TWAP = "twap"               # 时间加权平均价格
    VWAP = "vwap"               # 成交量加权平均价格

class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"         # 待执行
    PARTIAL = "partial"         # 部分成交
    FILLED = "filled"           # 完全成交
    CANCELLED = "cancelled"     # 已取消
    REJECTED = "rejected"       # 已拒绝

class ExecutionAlgo(Enum):
    """执行算法枚举"""
    SIMPLE = "simple"           # 简单执行
    TWAP = "twap"              # 时间加权平均
    VWAP = "vwap"              # 成交量加权平均
    IS = "implementation_shortfall"  # 执行缺口算法
    POV = "percentage_of_volume"     # 成交量比例算法

@dataclass
class Order:
    """交易订单数据类"""
    order_id: str
    symbol: str
    side: str                   # 'BUY' 或 'SELL'
    quantity: float            # 数量（股数）
    order_type: OrderType
    price: Optional[float] = None              # 限价单价格
    stop_price: Optional[float] = None         # 止损价格
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0               # 已成交数量
    avg_fill_price: float = 0.0                # 平均成交价格
    commission: float = 0.0                    # 佣金费用
    timestamp: datetime = field(default_factory=datetime.now)
    parent_instruction_id: Optional[str] = None  # 父指令ID
    algo_params: Dict[str, Any] = field(default_factory=dict)  # 算法参数
    
    @property
    def remaining_quantity(self) -> float:
        """剩余数量"""
        return self.quantity - self.filled_quantity
    
    @property
    def fill_ratio(self) -> float:
        """成交比例"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0
    
    @property
    def is_complete(self) -> bool:
        """是否完全成交"""
        return self.status == OrderStatus.FILLED or self.remaining_quantity <= 0

@dataclass
class TradeInstruction:
    """交易指令数据类"""
    instruction_id: str
    symbol: str
    target_weight: float        # 目标权重
    current_weight: float       # 当前权重
    portfolio_value: float      # 组合价值
    urgency: str = "normal"     # 紧急程度 ('low', 'normal', 'high', 'urgent')
    max_participation_rate: float = 0.1    # 最大市场参与率
    time_horizon: int = 5       # 执行时间窗口（分钟）
    price_limit: Optional[float] = None     # 价格限制
    
    @property
    def target_value(self) -> float:
        """目标头寸价值"""
        return self.target_weight * self.portfolio_value
    
    @property
    def current_value(self) -> float:
        """当前头寸价值"""
        return self.current_weight * self.portfolio_value
    
    @property
    def trade_value(self) -> float:
        """交易价值"""
        return self.target_value - self.current_value
    
    @property
    def side(self) -> str:
        """交易方向"""
        return 'BUY' if self.trade_value > 0 else 'SELL'

@dataclass
class ExecutionReport:
    """执行报告数据类"""
    instruction_id: str
    symbol: str
    total_quantity: float
    executed_quantity: float
    avg_execution_price: float
    benchmark_price: float      # 基准价格（如开始价格）
    total_commission: float
    market_impact: float
    slippage: float            # 滑点
    execution_time: float      # 执行时间（分钟）
    participation_rate: float  # 实际参与率
    completion_ratio: float    # 完成比例

class ExecutionManager:
    """
    交易执行管理器
    
    核心职责：
    1. 接收交易指令并制定执行计划
    2. 管理订单生命周期
    3. 监控执行绩效和成本
    4. 提供执行分析和报告
    """
    
    def __init__(self,
                 # 基础参数
                 default_commission_rate: float = 0.001,      # 默认佣金费率
                 market_impact_model: str = "linear",         # 市场冲击模型
                 
                 # 执行参数
                 default_participation_rate: float = 0.05,    # 默认市场参与率
                 max_order_size: float = 0.02,               # 最大单笔订单规模
                 min_order_size: float = 100,                # 最小订单规模（股数）
                 
                 # 时间参数
                 default_execution_window: int = 30,          # 默认执行窗口（分钟）
                 order_interval: int = 1,                    # 订单间隔（分钟）
                 
                 # 风险控制
                 max_slippage_tolerance: float = 0.01,       # 最大滑点容忍度
                 price_deviation_limit: float = 0.02,        # 价格偏离限制
                 
                 # 算法参数
                 enable_smart_routing: bool = True,          # 启用智能路由
                 adaptive_participation: bool = True,        # 自适应参与率
                 
                 # 监控参数
                 execution_monitoring: bool = True           # 启用执行监控
                 ):
        """
        初始化执行管理器
        
        Args:
            default_commission_rate: 默认佣金费率
            market_impact_model: 市场冲击成本模型
            default_participation_rate: 默认市场参与率
            max_order_size: 最大单笔订单权重
            min_order_size: 最小订单规模
            default_execution_window: 默认执行时间窗口
            order_interval: 订单提交间隔
            max_slippage_tolerance: 最大滑点容忍度
            price_deviation_limit: 价格偏离限制
            enable_smart_routing: 是否启用智能订单路由
            adaptive_participation: 是否使用自适应参与率
            execution_monitoring: 是否启用实时执行监控
        """
        # 基础参数
        self.default_commission_rate = default_commission_rate
        self.market_impact_model = market_impact_model
        
        # 执行参数
        self.default_participation_rate = default_participation_rate
        self.max_order_size = max_order_size
        self.min_order_size = min_order_size
        
        # 时间参数
        self.default_execution_window = default_execution_window
        self.order_interval = order_interval
        
        # 风险控制
        self.max_slippage_tolerance = max_slippage_tolerance
        self.price_deviation_limit = price_deviation_limit
        
        # 算法参数
        self.enable_smart_routing = enable_smart_routing
        self.adaptive_participation = adaptive_participation
        self.execution_monitoring = execution_monitoring
        
        # 内部状态
        self._active_orders: Dict[str, Order] = {}
        self._completed_orders: List[Order] = []
        self._execution_history: List[ExecutionReport] = []
        self._order_counter = 0
        
        logger.info(f"ExecutionManager initialized: "
                   f"participation={default_participation_rate:.1%}, "
                   f"window={default_execution_window}min")
    
    def submit_instruction(self,
                          instruction: TradeInstruction,
                          market_data: Dict[str, pd.DataFrame],
                          execution_algo: ExecutionAlgo = ExecutionAlgo.TWAP) -> List[Order]:
        """
        提交交易指令
        
        Args:
            instruction: 交易指令
            market_data: 市场数据
            execution_algo: 执行算法
            
        Returns:
            生成的订单列表
        """
        logger.info(f"提交交易指令: {instruction.symbol} "
                   f"{instruction.side} {abs(instruction.trade_value):.0f}")
        
        try:
            # 1. 验证指令有效性
            if not self._validate_instruction(instruction, market_data):
                logger.warning(f"指令验证失败: {instruction.instruction_id}")
                return []
            
            # 2. 制定执行计划
            execution_plan = self._create_execution_plan(instruction, market_data, execution_algo)
            
            # 3. 生成订单
            orders = self._generate_orders(instruction, execution_plan, market_data)
            
            # 4. 注册订单
            for order in orders:
                self._active_orders[order.order_id] = order
            
            logger.info(f"生成 {len(orders)} 个子订单")
            return orders
            
        except Exception as e:
            logger.error(f"提交交易指令失败: {e}")
            return []
    
    def execute_orders(self,
                      market_data: Dict[str, pd.DataFrame],
                      current_time: datetime = None) -> List[ExecutionReport]:
        """
        执行待处理订单
        
        Args:
            market_data: 实时市场数据
            current_time: 当前时间
            
        Returns:
            执行报告列表
        """
        if not self._active_orders:
            return []
        
        if current_time is None:
            current_time = datetime.now()
        
        execution_reports = []
        completed_orders = []
        
        for order_id, order in list(self._active_orders.items()):
            try:
                # 检查订单是否应该执行
                if self._should_execute_order(order, market_data, current_time):
                    
                    # 模拟订单执行
                    execution_result = self._simulate_order_execution(order, market_data)
                    
                    if execution_result:
                        # 更新订单状态
                        order.filled_quantity += execution_result['filled_quantity']
                        order.avg_fill_price = (
                            (order.avg_fill_price * (order.filled_quantity - execution_result['filled_quantity']) + 
                             execution_result['fill_price'] * execution_result['filled_quantity']) / 
                            order.filled_quantity
                        )
                        order.commission += execution_result['commission']
                        
                        # 检查订单是否完成
                        if order.is_complete:
                            order.status = OrderStatus.FILLED
                            completed_orders.append(order_id)
                            
                            # 生成执行报告
                            report = self._generate_execution_report(order)
                            execution_reports.append(report)
                
            except Exception as e:
                logger.error(f"执行订单 {order_id} 失败: {e}")
                order.status = OrderStatus.REJECTED
                completed_orders.append(order_id)
        
        # 移除已完成的订单
        for order_id in completed_orders:
            completed_order = self._active_orders.pop(order_id)
            self._completed_orders.append(completed_order)
        
        if execution_reports:
            logger.info(f"完成 {len(execution_reports)} 个订单执行")
        
        return execution_reports
    
    def _validate_instruction(self,
                             instruction: TradeInstruction,
                             market_data: Dict[str, pd.DataFrame]) -> bool:
        """验证交易指令有效性"""
        
        # 检查股票数据是否存在
        if instruction.symbol not in market_data:
            logger.warning(f"股票 {instruction.symbol} 数据不存在")
            return False
        
        # 检查交易规模是否合理
        if abs(instruction.trade_value) < self.min_order_size * 10:  # 假设股价10元
            logger.debug(f"交易规模过小: {instruction.trade_value}")
            return False
        
        # 检查权重变化是否显著
        weight_change = abs(instruction.target_weight - instruction.current_weight)
        if weight_change < 0.001:  # 0.1%
            logger.debug(f"权重变化过小: {weight_change}")
            return False
        
        return True
    
    def _create_execution_plan(self,
                              instruction: TradeInstruction,
                              market_data: Dict[str, pd.DataFrame],
                              execution_algo: ExecutionAlgo) -> Dict[str, Any]:
        """制定执行计划"""
        
        symbol = instruction.symbol
        df = market_data[symbol]
        
        # 获取市场特征
        recent_data = df.tail(20)
        avg_volume = recent_data['vol'].mean() if 'vol' in recent_data.columns else 1e6
        price = recent_data['close'].iloc[-1] if 'close' in recent_data.columns else 100
        volatility = recent_data['close'].pct_change().std() if len(recent_data) > 1 else 0.02
        
        # 计算目标股数
        target_shares = abs(instruction.trade_value) / price
        
        # 确定参与率
        participation_rate = min(
            instruction.max_participation_rate,
            self.default_participation_rate
        )
        
        # 自适应参与率调整
        if self.adaptive_participation:
            if instruction.urgency == "urgent":
                participation_rate *= 2.0
            elif instruction.urgency == "low":
                participation_rate *= 0.5
        
        # 计算执行参数
        expected_volume = avg_volume * participation_rate
        execution_shares_per_interval = min(target_shares * 0.2, expected_volume)  # 每次最多20%
        
        if execution_shares_per_interval > 0:
            estimated_intervals = max(1, int(target_shares / execution_shares_per_interval))
            execution_time = estimated_intervals * self.order_interval
        else:
            estimated_intervals = 1
            execution_time = self.order_interval
        
        plan = {
            'algorithm': execution_algo,
            'target_shares': target_shares,
            'participation_rate': participation_rate,
            'execution_shares_per_interval': execution_shares_per_interval,
            'estimated_intervals': estimated_intervals,
            'execution_time_minutes': execution_time,
            'reference_price': price,
            'expected_volume': expected_volume,
            'volatility': volatility
        }
        
        return plan
    
    def _generate_orders(self,
                        instruction: TradeInstruction,
                        execution_plan: Dict[str, Any],
                        market_data: Dict[str, pd.DataFrame]) -> List[Order]:
        """根据执行计划生成订单"""
        
        orders = []
        algorithm = execution_plan['algorithm']
        
        if algorithm == ExecutionAlgo.SIMPLE:
            # 简单执行：一次性订单
            orders.append(self._create_single_order(instruction, execution_plan))
            
        elif algorithm == ExecutionAlgo.TWAP:
            # TWAP：时间均匀分割
            orders = self._create_twap_orders(instruction, execution_plan)
            
        elif algorithm == ExecutionAlgo.VWAP:
            # VWAP：基于成交量分割
            orders = self._create_vwap_orders(instruction, execution_plan, market_data)
            
        else:
            # 默认使用TWAP
            orders = self._create_twap_orders(instruction, execution_plan)
        
        return orders
    
    def _create_single_order(self,
                           instruction: TradeInstruction,
                           execution_plan: Dict[str, Any]) -> Order:
        """创建单一订单"""
        
        self._order_counter += 1
        order_id = f"ORDER_{instruction.symbol}_{self._order_counter:06d}"
        
        return Order(
            order_id=order_id,
            symbol=instruction.symbol,
            side=instruction.side,
            quantity=execution_plan['target_shares'],
            order_type=OrderType.MARKET,
            parent_instruction_id=instruction.instruction_id,
            algo_params={'execution_plan': execution_plan}
        )
    
    def _create_twap_orders(self,
                           instruction: TradeInstruction,
                           execution_plan: Dict[str, Any]) -> List[Order]:
        """创建TWAP订单序列"""
        
        orders = []
        total_shares = execution_plan['target_shares']
        num_orders = execution_plan['estimated_intervals']
        shares_per_order = total_shares / num_orders
        
        for i in range(num_orders):
            self._order_counter += 1
            order_id = f"TWAP_{instruction.symbol}_{self._order_counter:06d}"
            
            # 最后一个订单包含剩余股数
            if i == num_orders - 1:
                quantity = total_shares - (shares_per_order * i)
            else:
                quantity = shares_per_order
            
            order = Order(
                order_id=order_id,
                symbol=instruction.symbol,
                side=instruction.side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                parent_instruction_id=instruction.instruction_id,
                algo_params={
                    'execution_plan': execution_plan,
                    'order_index': i,
                    'total_orders': num_orders,
                    'scheduled_time': datetime.now() + timedelta(minutes=i * self.order_interval)
                }
            )
            
            orders.append(order)
        
        return orders
    
    def _create_vwap_orders(self,
                          instruction: TradeInstruction,
                          execution_plan: Dict[str, Any],
                          market_data: Dict[str, pd.DataFrame]) -> List[Order]:
        """创建VWAP订单序列"""
        
        # 简化的VWAP实现，实际应该基于历史成交量模式
        # 这里使用均匀分布作为近似
        return self._create_twap_orders(instruction, execution_plan)
    
    def _should_execute_order(self,
                             order: Order,
                             market_data: Dict[str, pd.DataFrame],
                             current_time: datetime) -> bool:
        """判断订单是否应该执行"""
        
        # 检查订单状态
        if order.status != OrderStatus.PENDING:
            return False
        
        # 检查时间条件（对于TWAP等算法）
        if 'scheduled_time' in order.algo_params:
            scheduled_time = order.algo_params['scheduled_time']
            if current_time < scheduled_time:
                return False
        
        # 检查市场条件
        if order.symbol not in market_data:
            return False
        
        df = market_data[order.symbol]
        if df.empty or 'close' not in df.columns:
            return False
        
        return True
    
    def _simulate_order_execution(self,
                                 order: Order,
                                 market_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """模拟订单执行"""
        
        try:
            df = market_data[order.symbol]
            current_price = df['close'].iloc[-1]
            
            # 简化的执行模拟
            filled_quantity = order.remaining_quantity
            
            # 模拟滑点
            slippage_factor = np.random.normal(0, 0.001)  # 简化的滑点模型
            if order.side == 'BUY':
                fill_price = current_price * (1 + abs(slippage_factor))
            else:
                fill_price = current_price * (1 - abs(slippage_factor))
            
            # 计算佣金
            commission = filled_quantity * fill_price * self.default_commission_rate
            
            return {
                'filled_quantity': filled_quantity,
                'fill_price': fill_price,
                'commission': commission,
                'execution_time': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"模拟订单执行失败: {e}")
            return None
    
    def _generate_execution_report(self, order: Order) -> ExecutionReport:
        """生成执行报告"""
        
        # 获取基准价格（简化处理）
        benchmark_price = order.algo_params.get('execution_plan', {}).get('reference_price', order.avg_fill_price)
        
        # 计算滑点
        if order.side == 'BUY':
            slippage = (order.avg_fill_price - benchmark_price) / benchmark_price
        else:
            slippage = (benchmark_price - order.avg_fill_price) / benchmark_price
        
        # 计算市场冲击（简化）
        market_impact = abs(slippage) * 0.5  # 假设一半的滑点来自市场冲击
        
        report = ExecutionReport(
            instruction_id=order.parent_instruction_id or "",
            symbol=order.symbol,
            total_quantity=order.quantity,
            executed_quantity=order.filled_quantity,
            avg_execution_price=order.avg_fill_price,
            benchmark_price=benchmark_price,
            total_commission=order.commission,
            market_impact=market_impact,
            slippage=slippage,
            execution_time=0,  # 简化处理
            participation_rate=0,  # 简化处理
            completion_ratio=order.fill_ratio
        )
        
        self._execution_history.append(report)
        return report
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        
        if not self._execution_history:
            return {'total_executions': 0}
        
        reports = self._execution_history
        
        stats = {
            'total_executions': len(reports),
            'avg_slippage': np.mean([r.slippage for r in reports]),
            'avg_commission_rate': np.mean([
                r.total_commission / (r.executed_quantity * r.avg_execution_price)
                for r in reports if r.executed_quantity > 0 and r.avg_execution_price > 0
            ]),
            'avg_completion_ratio': np.mean([r.completion_ratio for r in reports]),
            'total_market_impact': sum([r.market_impact for r in reports]),
            'execution_efficiency': {
                'low_slippage_trades': len([r for r in reports if abs(r.slippage) < 0.005]),
                'high_slippage_trades': len([r for r in reports if abs(r.slippage) > 0.02]),
                'complete_fills': len([r for r in reports if r.completion_ratio > 0.95])
            }
        }
        
        # 按股票分组统计
        symbol_stats = {}
        for report in reports:
            if report.symbol not in symbol_stats:
                symbol_stats[report.symbol] = []
            symbol_stats[report.symbol].append(report)
        
        stats['by_symbol'] = {
            symbol: {
                'executions': len(symbol_reports),
                'avg_slippage': np.mean([r.slippage for r in symbol_reports]),
                'total_volume': sum([r.executed_quantity * r.avg_execution_price for r in symbol_reports])
            }
            for symbol, symbol_reports in symbol_stats.items()
        }
        
        return stats
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        if order_id in self._active_orders:
            order = self._active_orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                cancelled_order = self._active_orders.pop(order_id)
                self._completed_orders.append(cancelled_order)
                logger.info(f"订单 {order_id} 已取消")
                return True
        return False
    
    def get_active_orders(self) -> Dict[str, Order]:
        """获取活跃订单"""
        return self._active_orders.copy()
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """获取订单状态"""
        if order_id in self._active_orders:
            return self._active_orders[order_id].status
        
        for order in self._completed_orders:
            if order.order_id == order_id:
                return order.status
        
        return None