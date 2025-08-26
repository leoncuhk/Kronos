# Kronos Finetune Pipeline - Complete Guide

专业级的Kronos模型微调管道，提供开发、基础、生产三种配置模式。

## 🚀 快速开始

### 推荐方式：智能启动
```bash
cd finetune/scripts
python quick_start.py
```
系统会自动检测GPU并推荐最适合的配置模式。

### 手动指定配置
```bash
# 开发模式 - 快速测试 (~3分钟)
python run_complete_finetune.py --config dev

# 基础模式 - 标准训练 (~30分钟，推荐)
python run_complete_finetune.py --config base

# 生产模式 - 完整训练 (~2小时)
python run_complete_finetune.py --config production
```

### 高级选项
```bash
# 跳过数据准备 (使用已有数据)
python run_complete_finetune.py --skip-data-prep

# 仅训练Predictor (使用预训练Tokenizer)  
python run_complete_finetune.py --skip-tokenizer

# 干运行 - 查看执行计划
python run_complete_finetune.py --dry-run --verbose

# 获取完整帮助
python run_complete_finetune.py --help
```

## 📐 架构设计

### 核心组件

#### UnifiedTrainer - 统一训练接口
```python
# 支持tokenizer和predictor训练
trainer = UnifiedTrainer(config, model_type="tokenizer")  
results = trainer.train()
```

#### DataManager - 智能数据管理
```python
# 自动检测数据源，生成高质量训练数据
data_manager = DataManager(config)
success = data_manager.create_and_save_datasets()
```

#### 三层配置系统
```python
# 基于实际代码的配置层次
from finetune.config import BaseConfig, ProductionConfig, DevelopmentConfig

config = BaseConfig()          # 标准配置 (推荐)
config = ProductionConfig()    # 生产配置 (高级)  
config = DevelopmentConfig()   # 开发配置 (快速)
```

### 训练流程
```
数据准备 → Tokenizer训练 → Predictor训练 → 验证
    ↓           ↓             ↓         ↓
  DataManager  UnifiedTrainer UnifiedTrainer 健康检查
  智能数据源   混合精度训练   学习率调度    性能验证
```

## ⚙️ 配置详情

### Development Config (dev)
```python
epochs = 1                       # 快速验证
n_train_iter = 50 * batch_size   # 最小训练量
use_amp = False                  # 简化设置
# 适用：代码调试、功能验证 (~3分钟)
```

### Base Config (推荐)
```python
epochs = 2                       # 标准训练轮数
n_train_iter = 100 * batch_size  # 适中训练量
tokenizer_lr = 2e-4              # 优化的学习率
predictor_lr = 4e-5
use_amp = True                   # 混合精度训练
# 适用：日常使用、实际微调 (~30分钟)
```

### Production Config
```python
epochs = 10                      # 充分训练
n_train_iter = 2000 * batch_size # 大量训练数据
use_amp = True                   # 混合精度
gradient_checkpointing = True    # 内存优化
early_stopping_patience = 5     # 防止过拟合
# 适用：生产部署、追求极致效果 (~2小时)
```

## 📊 数据处理

### 智能数据源
系统按以下优先级处理数据：
1. **真实数据**: 自动检测qlib数据源
2. **模拟数据**: 生成高质量的中国A股模拟数据
3. **数据验证**: 完整性检查、异常值检测

### 数据集分割 (防止数据泄露)
- **训练集**: 2011-01-01 至 2022-08-31 (~11.7年)
- **验证集**: 2022-09-01 至 2023-12-31 (16个月)  
- **测试集**: 2024-01-01 至 2024-12-31 (12个月)

### 支持的股票池
默认使用10只代表性中国A股：
- 银行: 平安银行(000001.SZ), 浦发银行(600000.SH), 招商银行(600036.SH)
- 消费: 贵州茅台(600519.SH), 五粮液(000858.SZ)
- 科技: 海康威视(002415.SZ), 东方财富(300059.SZ), 宁德时代(300750.SZ)
- 地产: 万科A(000002.SZ)
- 新能源: 比亚迪(002594.SZ)

## 🤖 训练详情

### Tokenizer训练
- **目标**: 学习OHLCV数据的离散化表示
- **损失函数**: 重建损失 + BSQ量化损失
- **输出**: 微调后的tokenizer模型

### Predictor训练  
- **目标**: 基于tokenized数据进行时序预测
- **依赖**: 自动检测并使用微调后的tokenizer
- **损失函数**: 交叉熵损失 (多头预测)
- **输出**: 微调后的predictor模型

### 模型输出结构
```
outputs/models/
├── tokenizer/best_model/          # HuggingFace格式
├── predictor/best_model/          # HuggingFace格式
├── tokenizer/checkpoint_epoch_X.pt # PyTorch检查点
└── predictor/checkpoint_epoch_X.pt # PyTorch检查点
```

## 🛠️ 代码质量保证

### 已修复的关键问题 ✅
1. **数据泄露风险**: 严格的时序分割，无重叠时间范围
2. **pandas兼容性**: 支持最新pandas版本
3. **数据分割逻辑**: 基于时间范围的正确分割
4. **Predictor训练**: 正确的tokenizer输出处理
5. **配置传递**: 完整的参数传递机制
6. **构造函数**: QlibDataset的灵活配置
7. **错误处理**: 友好的错误信息和调试提示
8. **时间特征**: 正确的datetime格式处理

### 质量指标
- ✅ **逻辑正确性**: 所有核心算法经过验证
- ✅ **时序安全**: 无数据泄露风险
- ✅ **错误处理**: 完善的异常捕获
- ✅ **兼容性**: 支持多种环境和数据格式

## 📈 性能基准

### 硬件要求
- **CPU模式**: 自动推荐dev配置
- **GPU模式**: 推荐base或production配置  
- **内存要求**: 最少8GB RAM，推荐16GB+

### 训练时间基准
| 配置 | CPU时间 | GPU时间 | 适用场景 |
|------|---------|---------|----------|
| dev | ~3分钟 | ~1分钟 | 代码验证 |
| base | ~30分钟 | ~10分钟 | 日常使用 ⭐️ |
| production | ~2小时 | ~30分钟 | 生产部署 |

## 🔧 故障排除

### 常见问题

#### 数据相关
```bash
# 问题: Dataset file not found
# 解决: 重新运行数据准备
python run_complete_finetune.py --config dev

# 问题: 数据质量问题
# 解决: 检查数据验证报告
cat outputs/data/data_info_real.json
```

#### 训练相关
```bash
# 问题: GPU内存不足
# 解决: 降低batch_size或使用dev配置
export BATCH_SIZE=32
python run_complete_finetune.py --config dev

# 问题: 训练中断
# 解决: 检查保存的检查点
ls outputs/models/*/checkpoint_*.pt
```

#### 模型相关
```bash
# 问题: 模型加载失败
# 解决: 检查预训练模型路径和网络连接
python -c "from transformers import AutoModel; print('HF连接正常')"
```

## 🚀 扩展开发

### 自定义配置
```python
from finetune.config import BaseConfig

class MyConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.my_custom_param = "value"
        self.epochs = 5  # 自定义训练轮数
```

### 新增数据源
```python
from finetune.data import DataManager

class CustomDataManager(DataManager):
    def load_custom_data_source(self):
        # 实现自定义数据加载逻辑
        pass
```

### 自定义训练器
```python
from finetune.training import UnifiedTrainer

class CustomTrainer(UnifiedTrainer):
    def compute_custom_loss(self, batch, model_output):
        # 实现自定义损失函数
        pass
```

## 🎯 最佳实践

### 推荐工作流程
1. **开发阶段**: 使用 `--config dev` 快速验证
2. **测试阶段**: 使用 `--config base` 进行实际训练
3. **生产部署**: 使用 `--config production` 获得最佳效果

### 性能优化
- 使用混合精度训练(AMP)减少GPU内存
- 调整batch_size匹配GPU内存
- 生产模式启用梯度检查点节省内存
- 使用多GPU时可扩展为分布式训练

### 日志和监控
- **训练日志**: `outputs/logs/tokenizer_training.log`, `predictor_training.log`
- **模型检查点**: `outputs/models/tokenizer/`, `outputs/models/predictor/`
- **数据验证报告**: `outputs/data/data_info_real.json`

## 🔬 技术规格

### 系统要求
- **Python**: 3.8+
- **PyTorch**: 1.12+
- **依赖**: pandas, numpy, transformers, qlib (可选)
- **硬件**: CPU (最小) / GPU (推荐)

### 存储需求
- **代码**: <50MB
- **模型**: ~500MB-2GB (取决于配置)
- **数据**: ~100MB-1GB (取决于股票数量)
- **日志**: ~10MB-100MB

---

## 📞 支持

如遇问题，请检查：
1. **训练日志**: 详细的错误信息和调试提示
2. **GitHub Issues**: 社区支持和问题讨论
3. **配置文档**: 确保配置参数正确

Kronos Finetune Pipeline是一个**工业级、用户友好、高度可配置**的机器学习训练系统，提供从快速原型到生产部署的完整解决方案。