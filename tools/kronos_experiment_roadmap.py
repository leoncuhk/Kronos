#!/usr/bin/env python3
"""
🚀 Kronos实验路线图 - 完整实验方案

基于项目代码分析，提供两套完整的实验方案：
方案A: 无Qlib依赖 - 专注预测和数据获取
方案B: 完整方案 - 包含微调训练功能

作者: AI Assistant
时间: 2024年
"""

from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

class KronosExperimentConfig:
    """实验配置管理"""
    
    # === 基础配置 ===
    PROJECT_ROOT = project_root
    CONDA_ENV = "quant"
    
    # === GPU配置 ===
    USE_GPU = True  # 基于报告显示RTX 4060可获得11.1x加速
    DEVICE = "cuda" if USE_GPU else "cpu"
    
    # === 数据源配置 ===
    DATA_SOURCES = {
        "yfinance": {
            "symbols": ["BTC-USD", "ETH-USD", "AAPL", "TSLA", "SPY"],
            "intervals": ["1h", "1d"],
            "description": "通过yfinance获取全球股票和加密货币数据"
        },
        "binance": {
            "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            "intervals": ["1h", "4h", "1d"],
            "description": "通过Binance API获取加密货币数据"
        },
        "local_csv": {
            "path": project_root / "examples" / "data",
            "files": ["XSHG_5min_600977.csv"],
            "description": "使用项目提供的本地示例数据"
        }
    }
    
    # === 预测配置 ===
    PREDICTION_CONFIGS = {
        "quick_test": {
            "lookback": 100,
            "pred_len": 24,
            "sample_count": 5,
            "description": "快速验证测试"
        },
        "standard": {
            "lookback": 400,
            "pred_len": 120,
            "sample_count": 30,
            "description": "标准预测配置（与examples一致）"
        },
        "advanced": {
            "lookback": 500,
            "pred_len": 200,
            "sample_count": 50,
            "description": "高级概率预测"
        }
    }

# ============================================================================
# 🎯 方案A: 无Qlib依赖方案 - 专注预测和数据获取
# ============================================================================

PLAN_A_ROADMAP = {
    "title": "🎯 方案A: 无Qlib依赖 - 专注预测实验",
    "description": "基于YFinance和Binance API，专注于预测功能验证和优化",
    "advantages": [
        "✅ 快速上手，无需复杂数据准备",
        "✅ 实时数据，贴近实际应用",
        "✅ 多数据源支持",
        "✅ GPU加速优化"
    ],
    "phases": {
        "phase_1": {
            "title": "🔧 环境验证与基础测试",
            "duration": "30分钟",
            "tasks": [
                "验证conda环境和GPU配置",
                "运行examples/prediction_example.py验证基础功能",
                "测试模型加载和推理速度"
            ],
            "expected_results": [
                "模型成功加载并预测",
                "GPU加速生效（约11x提速）",
                "预测结果可视化正常"
            ]
        },
        "phase_2": {
            "title": "📊 多源数据获取实现",
            "duration": "1小时",
            "tasks": [
                "实现YFinance数据获取模块",
                "实现Binance API数据获取模块",
                "统一数据格式和预处理流程",
                "添加数据验证和清洗功能"
            ],
            "expected_results": [
                "支持多种金融资产数据获取",
                "数据格式兼容Kronos输入要求",
                "自动处理缺失值和异常数据"
            ]
        },
        "phase_3": {
            "title": "🧠 概率预测与不确定性量化",
            "duration": "1.5小时",
            "tasks": [
                "实现蒙特卡洛采样预测",
                "计算预测置信区间",
                "量化上涨概率和波动率放大",
                "专业级结果可视化"
            ],
            "expected_results": [
                "概率分布预测而非点预测",
                "业务友好的指标输出",
                "专业级图表展示"
            ]
        },
        "phase_4": {
            "title": "⚡ 性能优化与自动化",
            "duration": "1小时",
            "tasks": [
                "批量预测优化",
                "结果缓存机制",
                "定时预测调度",
                "Web仪表板（可选）"
            ],
            "expected_results": [
                "预测速度进一步提升",
                "支持批量资产预测",
                "自动化预测流程"
            ]
        }
    },
    "total_time": "4小时",
    "difficulty": "中等",
    "recommended_for": "想快速体验Kronos预测能力的用户"
}

# ============================================================================
# 🏗️ 方案B: 完整方案 - 包含微调训练功能
# ============================================================================

PLAN_B_ROADMAP = {
    "title": "🏗️ 方案B: 完整方案 - 包含微调训练",
    "description": "完整体验包括数据准备、微调训练、回测验证在内的所有功能",
    "advantages": [
        "✅ 完整功能体验",
        "✅ 自定义数据训练",
        "✅ 专业级回测验证",
        "✅ 完整的量化投资流程"
    ],
    "prerequisites": [
        "安装Qlib: pip install pyqlib",
        "准备训练数据（推荐CSI300或美股数据）",
        "足够的训练时间（数小时到数天）"
    ],
    "phases": {
        "phase_1": {
            "title": "🔧 Qlib环境搭建",
            "duration": "1小时",
            "tasks": [
                "安装pyqlib依赖",
                "下载和配置Qlib数据",
                "验证数据加载功能",
                "配置训练参数"
            ],
            "expected_results": [
                "Qlib环境正常运行",
                "训练数据准备完成",
                "配置文件正确设置"
            ]
        },
        "phase_2": {
            "title": "🎓 模型微调训练",
            "duration": "2-8小时（取决于数据规模）",
            "tasks": [
                "数据预处理（qlib_data_preprocess.py）",
                "Tokenizer微调训练",
                "Predictor微调训练",
                "训练过程监控"
            ],
            "expected_results": [
                "微调后的Tokenizer模型",
                "微调后的Predictor模型",
                "训练曲线和指标记录"
            ]
        },
        "phase_3": {
            "title": "📈 回测验证",
            "duration": "30分钟",
            "tasks": [
                "运行回测脚本（qlib_test.py）",
                "分析回测结果",
                "对比基准表现",
                "风险指标计算"
            ],
            "expected_results": [
                "完整的回测报告",
                "投资策略表现分析",
                "风险收益指标"
            ]
        },
        "phase_4": {
            "title": "🔬 模型评估与优化",
            "duration": "1-2小时",
            "tasks": [
                "不同数据源效果对比",
                "超参数优化实验",
                "集成多个模型预测",
                "部署生产环境"
            ],
            "expected_results": [
                "最优模型配置",
                "生产级预测系统",
                "完整的实验报告"
            ]
        }
    },
    "total_time": "4.5-11.5小时",
    "difficulty": "高",
    "recommended_for": "想要完整体验和深入研究的用户"
}

# ============================================================================
# 🛠️ 实用工具函数
# ============================================================================

def print_roadmap(plan):
    """打印实验路线图"""
    print("\n" + "="*80)
    print(f"{plan['title']}")
    print("="*80)
    print(f"📝 {plan['description']}\n")
    
    if 'advantages' in plan:
        print("💡 优势:")
        for adv in plan['advantages']:
            print(f"   {adv}")
        print()
    
    if 'prerequisites' in plan:
        print("⚠️  前置条件:")
        for req in plan['prerequisites']:
            print(f"   • {req}")
        print()
    
    print("📋 实施阶段:")
    for phase_key, phase in plan['phases'].items():
        print(f"\n{phase['title']} ({phase['duration']})")
        print("   任务:")
        for task in phase['tasks']:
            print(f"     • {task}")
        print("   预期结果:")
        for result in phase['expected_results']:
            print(f"     ✓ {result}")
    
    print(f"\n⏱️  总耗时: {plan['total_time']}")
    print(f"🎯 难度: {plan['difficulty']}")
    print(f"👥 推荐对象: {plan['recommended_for']}")
    print("="*80)

def get_recommended_plan():
    """根据用户需求推荐方案"""
    print("\n🤔 如何选择实验方案？\n")
    
    print("选择 方案A 如果您：")
    print("✓ 想快速验证Kronos的预测能力")
    print("✓ 主要关注实时数据预测")
    print("✓ 不需要自定义训练")
    print("✓ 希望快速上手和验证")
    
    print("\n选择 方案B 如果您：")
    print("✓ 想要完整的量化投资体验")
    print("✓ 需要在自己的数据上微调模型")
    print("✓ 关注投资策略回测")
    print("✓ 有充足的时间和计算资源")
    
    print("\n💡 建议：如果是首次使用，推荐先尝试方案A，验证基础功能后再考虑方案B")

if __name__ == "__main__":
    print("🚀 Kronos金融时序预测 - 完整实验指南")
    print("基于项目代码深度分析，为您提供两套完整实验方案")
    
    # 显示推荐选择
    get_recommended_plan()
    
    # 显示两套方案详情
    print_roadmap(PLAN_A_ROADMAP)
    print_roadmap(PLAN_B_ROADMAP)
    
    # 环境信息
    print(f"\n🔧 当前环境配置:")
    print(f"   项目路径: {KronosExperimentConfig.PROJECT_ROOT}")
    print(f"   Conda环境: {KronosExperimentConfig.CONDA_ENV}")
    print(f"   推荐设备: {KronosExperimentConfig.DEVICE}")
    print(f"   GPU加速: {'启用' if KronosExperimentConfig.USE_GPU else '禁用'}")
