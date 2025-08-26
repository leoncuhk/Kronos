# Tokenizer-specific training logic
# Currently consolidated into UnifiedTrainer
# This file exists for compatibility and future extensions

from .trainer import UnifiedTrainer

class TokenizerTrainer(UnifiedTrainer):
    """Tokenizer trainer - currently just inherits from UnifiedTrainer."""
    
    def __init__(self, config):
        super().__init__(config, model_type="tokenizer")