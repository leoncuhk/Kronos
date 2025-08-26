#!/usr/bin/env python3
"""
Unified Training Interface for Kronos Finetune Pipeline

This module provides a clean, unified interface for training both
Tokenizer and Predictor models, consolidating functionality from
the various scattered training scripts.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Optional, Union, Any
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from model import Kronos, KronosTokenizer
from finetune.data import QlibDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedTrainer:
    """
    Unified trainer for both Tokenizer and Predictor models.
    Provides a clean interface with proper error handling and logging.
    """
    
    def __init__(self, config, model_type: str = "tokenizer"):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            model_type: Either "tokenizer" or "predictor"
        """
        self.config = config
        self.model_type = model_type.lower()
        
        if self.model_type not in ["tokenizer", "predictor"]:
            raise ValueError("model_type must be either 'tokenizer' or 'predictor'")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision training
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        
        # Setup logging
        self.setup_logging()
        
        logger.info(f"Initialized {self.model_type} trainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {getattr(config, 'use_amp', False)}")
        
    def setup_logging(self):
        """Setup logging directory and files."""
        log_dir = Path(self.config.outputs_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create model-specific log file
        log_file = log_dir / f"{self.model_type}_training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    def load_model(self) -> nn.Module:
        """Load the appropriate pre-trained model."""
        logger.info(f"Loading pre-trained {self.model_type} model...")
        
        try:
            if self.model_type == "tokenizer":
                model = KronosTokenizer.from_pretrained(
                    self.config.pretrained_tokenizer_path,
                    cache_dir=getattr(self.config, 'model_cache_dir', None)
                )
            else:  # predictor
                model = Kronos.from_pretrained(
                    self.config.pretrained_predictor_path,
                    cache_dir=getattr(self.config, 'model_cache_dir', None)
                )
                
            model = model.to(self.device)
            model.train()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
            self.model = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        if self.model is None:
            raise RuntimeError("Model must be loaded before setting up optimizer")
        
        # Select learning rate based on model type
        if self.model_type == "tokenizer":
            lr = self.config.tokenizer_learning_rate
        else:
            lr = self.config.predictor_learning_rate
            
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay
        )
        
        logger.info(f"Optimizer setup: AdamW with lr={lr}")
        
        # Setup mixed precision scaler if enabled
        if getattr(self.config, 'use_amp', False):
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
    
    def setup_scheduler(self, steps_per_epoch: int):
        """Setup learning rate scheduler."""
        total_steps = steps_per_epoch * self.config.epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        logger.info(f"Scheduler setup: OneCycleLR with {total_steps} total steps")
    
    def prepare_data(self) -> tuple:
        """Prepare training and validation data loaders."""
        logger.info("Preparing data loaders...")
        
        try:
            # Load datasets
            train_dataset = QlibDataset('train')
            val_dataset = QlibDataset('val')
            
            if len(train_dataset) == 0:
                raise RuntimeError("Training dataset is empty. Please run data preparation first.")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            logger.info(f"Training samples: {len(train_dataset)}")
            logger.info(f"Validation samples: {len(val_dataset)}")
            logger.info(f"Batch size: {self.config.batch_size}")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise
    
    def compute_loss(self, batch, model_output) -> Dict[str, torch.Tensor]:
        """Compute loss based on model type."""
        if self.model_type == "tokenizer":
            return self.compute_tokenizer_loss(batch, model_output)
        else:
            return self.compute_predictor_loss(batch, model_output)
    
    def compute_tokenizer_loss(self, batch, model_output) -> Dict[str, torch.Tensor]:
        """Compute loss for tokenizer training."""
        if isinstance(batch, tuple) and len(batch) >= 1:
            batch_x = batch[0]
        else:
            batch_x = batch
            
        batch_x = batch_x.to(self.device)
        
        # Forward pass
        (z_pre, z), bsq_loss, quantized, z_indices = model_output
        
        # Reconstruction loss
        recon_loss_pre = nn.MSELoss()(z_pre, batch_x)
        recon_loss_full = nn.MSELoss()(z, batch_x)
        
        # Total loss (BSQ loss is already computed in the model)
        total_loss = recon_loss_pre + recon_loss_full + bsq_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss_pre': recon_loss_pre,
            'recon_loss_full': recon_loss_full,
            'bsq_loss': bsq_loss
        }
    
    def compute_predictor_loss(self, batch, model_output) -> Dict[str, torch.Tensor]:
        """Compute loss for predictor training."""
        if isinstance(batch, tuple):
            batch_x, batch_x_stamp = batch
            batch_x_stamp = batch_x_stamp.to(self.device)
        else:
            batch_x = batch
            batch_x_stamp = None
            
        batch_x = batch_x.to(self.device)
        
        # Model output is logits for each prediction step
        logits = model_output
        
        # Create targets (next tokens) - this is a simplified version
        # In practice, you'd need proper tokenization first
        targets = self.create_targets(batch_x)
        
        # Compute cross-entropy loss
        if isinstance(logits, (list, tuple)):
            # Multiple prediction heads
            total_loss = 0
            losses = {}
            for i, logit in enumerate(logits):
                loss = nn.CrossEntropyLoss()(
                    logit.reshape(-1, logit.shape[-1]), 
                    targets.reshape(-1)
                )
                losses[f's{i+1}_loss'] = loss
                total_loss += loss
            losses['total_loss'] = total_loss
            return losses
        else:
            # Single prediction head
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )
            return {'total_loss': loss}
    
    def create_targets(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Create targets for predictor training."""
        # This is a simplified target creation
        # In practice, you'd use the tokenizer to create proper targets
        batch_size, seq_len, _ = batch_x.shape
        # Create dummy targets for now - replace with proper tokenization
        targets = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
        return targets
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {'total_loss': 0.0, 'num_batches': 0}
        
        logger.info(f"Starting epoch {epoch}")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Limit number of batches if specified
            if hasattr(self.config, 'n_train_iter'):
                if batch_idx * self.config.batch_size >= self.config.n_train_iter:
                    break
            
            try:
                # Forward pass
                if self.scaler is not None:  # Mixed precision
                    with torch.cuda.amp.autocast():
                        model_output = self.model(batch[0].to(self.device) if isinstance(batch, tuple) else batch.to(self.device))
                        losses = self.compute_loss(batch, model_output)
                        loss = losses['total_loss']
                    
                    # Backward pass
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.config.accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        if self.scheduler is not None:
                            self.scheduler.step()
                else:  # Normal precision
                    model_output = self.model(batch[0].to(self.device) if isinstance(batch, tuple) else batch.to(self.device))
                    losses = self.compute_loss(batch, model_output)
                    loss = losses['total_loss']
                    
                    # Backward pass
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config.accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        if self.scheduler is not None:
                            self.scheduler.step()
                
                # Update metrics
                epoch_metrics['total_loss'] += loss.item()
                epoch_metrics['num_batches'] += 1
                
                # Add other loss components
                for key, value in losses.items():
                    if key != 'total_loss':
                        if key not in epoch_metrics:
                            epoch_metrics[key] = 0.0
                        epoch_metrics[key] += value.item()
                
                # Log progress
                if batch_idx % self.config.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                        f"Loss = {loss.item():.4f}, LR = {current_lr:.2e}"
                    )
                
                self.global_step += 1
                
            except Exception as e:
                logger.error(f"Error in training step {batch_idx}: {e}")
                continue
        
        # Average metrics
        if epoch_metrics['num_batches'] > 0:
            for key in epoch_metrics:
                if key != 'num_batches':
                    epoch_metrics[key] /= epoch_metrics['num_batches']
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        logger.info(f"Average training loss: {epoch_metrics['total_loss']:.4f}")
        
        return epoch_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {'total_loss': 0.0, 'num_batches': 0}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Limit validation batches if specified
                if hasattr(self.config, 'n_val_iter'):
                    if batch_idx * self.config.batch_size >= self.config.n_val_iter:
                        break
                
                try:
                    model_output = self.model(batch[0].to(self.device) if isinstance(batch, tuple) else batch.to(self.device))
                    losses = self.compute_loss(batch, model_output)
                    
                    # Update metrics
                    val_metrics['total_loss'] += losses['total_loss'].item()
                    val_metrics['num_batches'] += 1
                    
                    # Add other loss components
                    for key, value in losses.items():
                        if key != 'total_loss':
                            if key not in val_metrics:
                                val_metrics[key] = 0.0
                            val_metrics[key] += value.item()
                            
                except Exception as e:
                    logger.error(f"Error in validation step {batch_idx}: {e}")
                    continue
        
        # Average metrics
        if val_metrics['num_batches'] > 0:
            for key in val_metrics:
                if key != 'num_batches':
                    val_metrics[key] /= val_metrics['num_batches']
        
        logger.info(f"Validation loss: {val_metrics['total_loss']:.4f}")
        return val_metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        save_dir = Path(self.config.save_path) / getattr(
            self.config, 
            f"{self.model_type}_save_folder_name",
            self.model_type
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config),
            'training_history': self.training_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = save_dir / "best_model"
            best_path.mkdir(exist_ok=True)
            
            # Save in HuggingFace format for easy loading
            self.model.save_pretrained(str(best_path))
            logger.info(f"Saved best model to {best_path}")
            
            # Also save a PyTorch checkpoint for the best model
            best_checkpoint = save_dir / f"best_{self.model_type}_epoch_{epoch}.pt"
            torch.save(checkpoint, best_checkpoint)
            logger.info(f"Saved best checkpoint to {best_checkpoint}")
    
    def train(self) -> Dict[str, Any]:
        """Complete training pipeline."""
        logger.info(f"Starting {self.model_type} training pipeline")
        
        try:
            # Setup
            self.load_model()
            self.setup_optimizer()
            
            # Prepare data
            train_loader, val_loader = self.prepare_data()
            self.setup_scheduler(len(train_loader))
            
            # Training loop
            for epoch in range(1, self.config.epochs + 1):
                self.current_epoch = epoch
                
                # Train
                train_metrics = self.train_epoch(train_loader, epoch)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Save metrics
                epoch_summary = {
                    'epoch': epoch,
                    'train': train_metrics,
                    'val': val_metrics
                }
                self.training_history.append(epoch_summary)
                
                # Check if this is the best model
                val_loss = val_metrics['total_loss']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    logger.info(f"New best validation loss: {val_loss:.4f}")
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_loss, is_best)
                
            logger.info("✅ Training completed successfully!")
            
            return {
                'success': True,
                'best_val_loss': self.best_val_loss,
                'training_history': self.training_history,
                'final_epoch': self.current_epoch
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_history': self.training_history
            }