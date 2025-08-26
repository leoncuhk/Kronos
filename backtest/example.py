#!/usr/bin/env python3
"""
Kronos量化回测框架使用示例

本示例演示如何使用完整的四步量化交易回测系统：
1. 配置回测参数
2. 初始化回测引擎
3. 运行回测
4. 分析结果

作者: Kronos Team
日期: 2025-01-26
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# 添加项目根目录路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backtest import BacktestEngine, BacktestConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_simple_backtest():
    """运行简单回测示例"""
    
    print("🚀 Kronos量化回测框架示例")
    print("="*50)
    
    # 1. 创建回测配置
    config = BacktestConfig(
        # 基础配置
        start_date="2023-01-01",
        end_date="2023-12-31", 
        initial_capital=1_000_000,  # 100万初始资金
        benchmark="CSI300",
        
        # 交易域配置
        universe_config={
            'market': 'csi300',
            'min_market_cap': 5e8,      # 最小市值5亿
            'min_avg_volume': 5e5,      # 最小日均成交量50万
            'rebalance_freq': 'monthly'
        },
        
        # 信号配置 - Kronos Alpha引擎
        signal_config={
            'lookback_window': 60,      # 60天历史数据
            'prediction_horizon': 5,    # 预测5天收益
            'sample_count': 15,         # Monte Carlo 15次采样
            'temperature': 0.7,         # 采样温度
            'risk_adjustment': True     # 启用风险调整
        },
        
        # 组合配置
        portfolio_config={
            'max_positions': 30,        # 最多30只股票
            'max_weight_per_stock': 0.08,  # 单只股票最大8%权重
            'construction_method': 'signal_weighted',  # 信号加权
            'long_only': True           # 仅做多
        },
        
        # 风险配置
        risk_config={
            'max_portfolio_volatility': 0.18,   # 最大18%年化波动率
            'max_drawdown': 0.06,              # 最大6%回撤
            'stop_loss_threshold': -0.04,      # 4%止损
            'position_sizing_method': 'volatility_target'
        },
        
        # 执行配置
        execution_config={
            'commission_rate': 0.0008,         # 0.08%交易费用
            'market_impact_coeff': 0.0005,     # 市场冲击成本
            'execution_algo': 'twap'           # TWAP执行算法
        }
    )
    
    print("✅ 回测配置完成")
    
    # 2. 初始化回测引擎
    print("🔧 初始化回测引擎...")
    
    try:
        engine = BacktestEngine(config=config)
        print("✅ 回测引擎初始化成功")
    except Exception as e:
        print(f"❌ 回测引擎初始化失败: {e}")
        return
    
    # 3. 运行回测
    print("🚀 开始运行回测...")
    print("   这可能需要几分钟时间，请耐心等待...")
    
    try:
        start_time = datetime.now()
        result = engine.run_backtest()
        end_time = datetime.now()
        
        print(f"✅ 回测完成！耗时: {(end_time - start_time).total_seconds():.1f}秒")
        
    except Exception as e:
        print(f"❌ 回测运行失败: {e}")
        return
    
    # 4. 显示结果
    engine.print_summary(result)
    
    # 5. 保存结果
    try:
        output_path = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        engine.save_results(result, output_path)
        print(f"📁 回测结果已保存至: {output_path}")
    except Exception as e:
        print(f"⚠️  结果保存失败: {e}")
    
    # 6. 策略分析提示
    print("\n📊 策略分析建议:")
    
    if result.sharpe_ratio > 1.5:
        print("   🎯 夏普比率优秀，策略风险调整收益良好")
    elif result.sharpe_ratio > 1.0:
        print("   👍 夏普比率良好，可考虑进一步优化")
    else:
        print("   ⚠️  夏普比率偏低，建议调整策略参数")
    
    if result.max_drawdown < 0.05:
        print("   🛡️ 回撤控制良好，风险管理有效")
    elif result.max_drawdown < 0.10:
        print("   📊 回撤适中，可接受范围")
    else:
        print("   🚨 回撤较大，建议加强风险控制")
    
    if result.information_ratio > 0.5:
        print("   📈 相对基准表现优异，Alpha创造能力强")
    else:
        print("   📊 可考虑优化Alpha信号质量")
    
    print("\n🔧 参数优化建议:")
    print("   1. 调整信号参数：lookback_window, prediction_horizon")
    print("   2. 优化组合构建：max_positions, construction_method") 
    print("   3. 强化风险管理：volatility限制, 止损参数")
    print("   4. 改进执行策略：参与率, 执行算法")
    
    return result

def run_advanced_backtest():
    """运行高级回测示例（包含更多功能）"""
    
    print("\n🎯 高级回测示例 - 多策略对比")
    print("="*50)
    
    strategies = [
        ("保守策略", {
            'max_positions': 20,
            'max_weight_per_stock': 0.06,
            'max_portfolio_volatility': 0.12,
            'construction_method': 'risk_parity'
        }),
        ("均衡策略", {
            'max_positions': 30,
            'max_weight_per_stock': 0.08,
            'max_portfolio_volatility': 0.18,
            'construction_method': 'signal_weighted'
        }),
        ("积极策略", {
            'max_positions': 50,
            'max_weight_per_stock': 0.10,
            'max_portfolio_volatility': 0.25,
            'construction_method': 'mean_variance'
        })
    ]
    
    results = {}
    
    for strategy_name, strategy_params in strategies:
        print(f"\n🔄 运行 {strategy_name}...")
        
        # 创建策略配置
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            initial_capital=1_000_000
        )
        
        # 更新策略参数
        config.portfolio_config.update(strategy_params)
        
        try:
            engine = BacktestEngine(config=config)
            result = engine.run_backtest()
            results[strategy_name] = result
            
            print(f"   ✅ {strategy_name} 完成")
            print(f"      年化收益: {result.annualized_return:.2%}")
            print(f"      夏普比率: {result.sharpe_ratio:.3f}")
            print(f"      最大回撤: {result.max_drawdown:.2%}")
            
        except Exception as e:
            print(f"   ❌ {strategy_name} 失败: {e}")
    
    # 策略对比
    if results:
        print(f"\n📊 策略对比摘要:")
        print(f"{'策略':<12} {'年化收益':<10} {'夏普比率':<10} {'最大回撤':<10} {'信息比率':<10}")
        print("-" * 55)
        
        for name, result in results.items():
            print(f"{name:<12} {result.annualized_return:>9.2%} {result.sharpe_ratio:>9.3f} "
                  f"{result.max_drawdown:>9.2%} {result.information_ratio:>9.3f}")
        
        # 推荐最佳策略
        best_strategy = max(results.items(), 
                          key=lambda x: x[1].sharpe_ratio * (1 - x[1].max_drawdown))
        print(f"\n🏆 综合表现最佳策略: {best_strategy[0]}")
    
    return results

def main():
    """主函数"""
    
    print("\n🌟 欢迎使用 Kronos 量化回测框架！")
    print("   基于四步量化交易法的专业回测系统")
    print("   将Kronos定位为概率优势排序引擎\n")
    
    try:
        # 运行简单回测
        simple_result = run_simple_backtest()
        
        if simple_result is None:
            print("❌ 简单回测失败，跳过高级示例")
            return
        
        # 询问是否运行高级示例
        print(f"\n❓ 是否运行多策略对比示例？(y/n)")
        choice = input("   请输入选择: ").strip().lower()
        
        if choice in ['y', 'yes', '是']:
            advanced_results = run_advanced_backtest()
            
            if advanced_results:
                print(f"\n📋 完整结果已生成，可用于进一步分析")
        
        print(f"\n🎉 Kronos回测示例完成！")
        print(f"   感谢使用 Kronos 量化交易框架")
        print(f"   如有问题，请联系 Kronos Team\n")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()