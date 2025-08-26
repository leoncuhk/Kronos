import os
from .base_config import BaseConfig


class ProductionConfig(BaseConfig):
    """
    Production configuration with optimized settings for serious training.
    """
    
    def __init__(self):
        super().__init__()
        
        # Override training parameters for production
        self.epochs = int(os.getenv('TRAIN_EPOCHS', 10))  # More epochs for production
        self.n_train_iter = int(os.getenv('TRAIN_ITER', 2000)) * self.batch_size
        self.n_val_iter = int(os.getenv('VAL_ITER', 400)) * self.batch_size
        
        # Production-specific settings
        self.use_amp = True  # Automatic Mixed Precision
        self.gradient_checkpointing = True  # Save memory
        self.early_stopping_patience = 5
        
        # Enhanced logging for production
        self.save_checkpoint_every = 1  # Save every epoch
        self.eval_every = 500  # Evaluate every N steps
        
        # Production model folders
        self.tokenizer_save_folder_name = 'production_tokenizer'
        self.predictor_save_folder_name = 'production_predictor'
        
        # Comet tags for production runs
        self.comet_tag = 'production_finetune'
        self.comet_name = 'kronos_production'


class DevelopmentConfig(BaseConfig):
    """
    Development configuration for quick testing and debugging.
    """
    
    def __init__(self):
        super().__init__()
        
        # Fast training for development
        self.epochs = 1
        self.n_train_iter = 50 * self.batch_size
        self.n_val_iter = 10 * self.batch_size
        
        # Development settings
        self.use_amp = False
        self.gradient_checkpointing = False
        
        # Dev model folders
        self.tokenizer_save_folder_name = 'dev_tokenizer'
        self.predictor_save_folder_name = 'dev_predictor'
        
        # Comet tags for dev runs
        self.comet_tag = 'development_finetune'
        self.comet_name = 'kronos_dev'