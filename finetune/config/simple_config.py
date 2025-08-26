import os
from pathlib import Path


class SimpleConfig:
    """
    简化配置 - 专注实用性，去除复杂分层
    适合"微调完有效就应用"的直接使用场景
    """

    def __init__(self, 
                 epochs: int = None, 
                 batch_size: int = None,
                 quick_mode: bool = False):
        """
        初始化配置
        
        Args:
            epochs: 训练轮数 (默认根据quick_mode自动选择)
            batch_size: 批次大小 (默认50)
            quick_mode: 快速模式 - 用于代码测试 (默认False)
        """
        # =================================================================
        # 路径配置 - 使用相对路径，便携性好
        # =================================================================
        self.project_root = Path(__file__).parent.parent.parent
        self.outputs_dir = self.project_root / "outputs"
        
        # 自动创建必要目录
        self.outputs_dir.mkdir(exist_ok=True)
        (self.outputs_dir / "models").mkdir(exist_ok=True)
        (self.outputs_dir / "data").mkdir(exist_ok=True)
        (self.outputs_dir / "results").mkdir(exist_ok=True)
        (self.outputs_dir / "logs").mkdir(exist_ok=True)

        # =================================================================
        # 数据配置
        # =================================================================
        self.qlib_data_path = str(self.outputs_dir / "qlib_data")
        self.dataset_path = str(self.outputs_dir / "data" / "processed_datasets")
        
        # 数据参数 - 基于原项目经验优化
        self.lookback_window = 90
        self.predict_window = 10
        self.max_context = 512
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        
        # 数据集时间范围
        self.dataset_begin_time = "2011-01-01"
        self.dataset_end_time = '2025-06-05'
        self.train_time_range = ["2011-01-01", "2022-12-31"]
        self.val_time_range = ["2022-09-01", "2024-06-30"]
        self.test_time_range = ["2024-04-01", "2025-06-05"]

        # =================================================================
        # 训练配置 - 智能默认值
        # =================================================================
        self.batch_size = batch_size or 50
        
        if quick_mode:
            # 快速模式 - 用于代码测试和调试
            self.epochs = epochs or 1
            self.n_train_iter = 20 * self.batch_size   # 20个batch快速测试
            self.n_val_iter = 5 * self.batch_size     # 5个batch验证
            print("🚀 快速模式: 用于代码测试，不追求训练效果")
        else:
            # 标准模式 - 平衡效率与效果
            self.epochs = epochs or 5  # 默认5轮，通常足够看到效果
            self.n_train_iter = 500 * self.batch_size  # 适中的训练量
            self.n_val_iter = 50 * self.batch_size    # 适中的验证量
            print("📈 标准模式: 平衡训练时间与模型效果")
        
        # 学习率 - 基于原项目验证的有效值
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 4e-5
        
        # 优化器参数
        self.accumulation_steps = 1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1
        
        # 其他参数
        self.clip = 5.0
        self.log_interval = 50  # 更频繁的日志，便于观察进度
        self.seed = 100

        # =================================================================
        # 模型路径配置
        # =================================================================
        # 预训练模型 - 使用HuggingFace Hub
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-small"
        
        # 输出路径
        self.save_path = str(self.outputs_dir / "models")
        self.tokenizer_save_folder_name = 'finetuned_tokenizer'
        self.predictor_save_folder_name = 'finetuned_predictor'
        
        # 微调后模型路径
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/best_model"
        
        # 结果路径
        self.backtest_result_path = str(self.outputs_dir / "results")
        
        # =================================================================
        # 实验配置 - 简化
        # =================================================================
        self.use_comet = False  # 默认关闭复杂的实验跟踪
        
        # GPU配置
        self.use_amp = True if not quick_mode else False  # 快速模式关闭混合精度避免复杂性
        self.gradient_checkpointing = False  # 简化内存管理
        
    def to_dict(self):
        """转为字典格式，便于序列化"""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items() 
                if not k.startswith('_')}
    
    def summary(self):
        """打印配置摘要"""
        print("🔧 配置摘要:")
        print(f"   训练轮数: {self.epochs}")
        print(f"   批次大小: {self.batch_size}")
        print(f"   训练迭代: {self.n_train_iter // self.batch_size} batches")
        print(f"   验证迭代: {self.n_val_iter // self.batch_size} batches")
        print(f"   混合精度: {'开启' if self.use_amp else '关闭'}")
        print(f"   输出目录: {self.outputs_dir}")


# 便捷的预设配置
def get_quick_config(**kwargs):
    """快速测试配置 - 用于代码调试"""
    return SimpleConfig(quick_mode=True, **kwargs)

def get_standard_config(**kwargs):
    """标准配置 - 用于实际微调"""
    return SimpleConfig(quick_mode=False, **kwargs)

def get_intensive_config(**kwargs):
    """强化配置 - 用于追求最佳效果"""
    kwargs.setdefault('epochs', 10)
    config = SimpleConfig(quick_mode=False, **kwargs)
    config.n_train_iter = 1000 * config.batch_size  # 更多训练
    config.n_val_iter = 100 * config.batch_size     # 更多验证
    return config