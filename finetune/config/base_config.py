import os
from pathlib import Path


class BaseConfig:
    """
    Base configuration class for Kronos finetune pipeline.
    Uses relative paths and environment variables for better portability.
    """

    def __init__(self):
        # =================================================================
        # Project paths (relative to project root)
        # =================================================================
        self.project_root = Path(__file__).parent.parent.parent
        self.outputs_dir = self.project_root / "outputs"
        self.finetune_dir = self.project_root / "finetune"
        
        # Ensure directories exist
        self.outputs_dir.mkdir(exist_ok=True)
        (self.outputs_dir / "models").mkdir(exist_ok=True)
        (self.outputs_dir / "data").mkdir(exist_ok=True)
        (self.outputs_dir / "results").mkdir(exist_ok=True)
        (self.outputs_dir / "logs").mkdir(exist_ok=True)

        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        self.qlib_data_path = os.getenv(
            'QLIB_DATA_PATH', 
            str(self.outputs_dir / "qlib_data")
        )
        self.instrument = 'csi300'

        # Overall time range for data loading from Qlib
        self.dataset_begin_time = "2011-01-01"
        self.dataset_end_time = '2025-06-05'

        # Sliding window parameters for creating samples
        self.lookback_window = 90
        self.predict_window = 10
        self.max_context = 512

        # Features to be used from the raw data
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # Time-based features to be generated
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        self.train_time_range = ["2011-01-01", "2022-12-31"]
        self.val_time_range = ["2022-09-01", "2024-06-30"]
        self.test_time_range = ["2024-04-01", "2025-06-05"]
        self.backtest_time_range = ["2024-07-01", "2025-06-05"]

        # Output paths
        self.dataset_path = str(self.outputs_dir / "data" / "processed_datasets")
        self.save_path = str(self.outputs_dir / "models")
        self.backtest_result_path = str(self.outputs_dir / "results" / "backtest_results")
        
        # Create dataset directory
        Path(self.dataset_path).mkdir(parents=True, exist_ok=True)
        Path(self.backtest_result_path).mkdir(parents=True, exist_ok=True)

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0
        self.epochs = int(os.getenv('TRAIN_EPOCHS', 2))
        self.log_interval = 100
        self.batch_size = int(os.getenv('BATCH_SIZE', 50))

        # Training/validation iterations
        self.n_train_iter = int(os.getenv('TRAIN_ITER', 100)) * self.batch_size
        self.n_val_iter = int(os.getenv('VAL_ITER', 20)) * self.batch_size

        # Learning rates
        self.tokenizer_learning_rate = float(os.getenv('TOKENIZER_LR', 2e-4))
        self.predictor_learning_rate = float(os.getenv('PREDICTOR_LR', 4e-5))

        # Optimizer parameters
        self.accumulation_steps = 1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Miscellaneous
        self.seed = int(os.getenv('RANDOM_SEED', 100))

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = os.getenv('USE_COMET', 'False').lower() == 'true'
        self.comet_config = {
            "api_key": os.getenv("COMET_API_KEY", ""),
            "project_name": os.getenv("COMET_PROJECT", "Kronos-Finetune"),
            "workspace": os.getenv("COMET_WORKSPACE", "")
        }
        self.comet_tag = 'finetune_production'
        self.comet_name = 'kronos_finetune'

        # Model save folders
        self.tokenizer_save_folder_name = 'tokenizer'
        self.predictor_save_folder_name = 'predictor'
        self.backtest_save_folder_name = 'backtest'

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # Pretrained model paths (can be local or HuggingFace Hub)
        self.pretrained_tokenizer_path = os.getenv(
            'PRETRAINED_TOKENIZER_PATH', 
            "NeoQuasar/Kronos-Tokenizer-base"
        )
        self.pretrained_predictor_path = os.getenv(
            'PRETRAINED_PREDICTOR_PATH',
            "NeoQuasar/Kronos-small"
        )

        # Finetuned model paths
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/best_model"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50
        self.backtest_n_symbol_drop = 5
        self.backtest_hold_thresh = 5
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        """Set benchmark based on instrument."""
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852", 
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")

    def to_dict(self):
        """Convert config to dictionary for serialization."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                # Convert Path objects to strings
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict

    def __str__(self):
        """String representation of config."""
        lines = ["Configuration:"]
        lines.append("-" * 50)
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)