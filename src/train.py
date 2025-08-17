"""
Training script for hand landmark detection model.
"""
import os
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from model import create_model, HandLandmarkLoss
from dataset import SyntheticHandDataset, create_dataloader
from utils import save_checkpoint, ensure_dir


class Trainer:
    """Training class for hand landmark detection models."""
    
    def __init__(self, config: Dict):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        self.model = create_model(
            model_type=config['model_type'],
            num_landmarks=config['num_landmarks'],
            input_channels=config['input_channels']
        ).to(self.device)
        
        # Loss function
        self.criterion = HandLandmarkLoss(use_wing_loss=config.get('use_wing_loss', True))
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('lr_patience', 10),
            verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup directories
        self.checkpoint_dir = config['checkpoint_dir']
        self.log_dir = config['log_dir']
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model: {config['model_type']}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation dataloaders."""
        # For demonstration, using synthetic dataset
        # In practice, replace with your actual dataset
        train_dataset = SyntheticHandDataset(
            num_samples=self.config['train_samples'],
            image_size=self.config['image_size']
        )
        
        val_dataset = SyntheticHandDataset(
            num_samples=self.config['val_samples'],
            image_size=self.config['image_size']
        )
        
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
        
        val_loader = create_dataloader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, landmarks) in enumerate(pbar):
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, landmarks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log batch metrics
            if batch_idx % self.config.get('log_interval', 100) == 0:
                step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_distance_error = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, landmarks in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                loss = self.criterion(predictions, landmarks)
                
                # Calculate distance error (in pixels)
                # Denormalize landmarks for meaningful distance calculation
                pred_denorm = predictions * torch.tensor([self.config['image_size'][1], 
                                                        self.config['image_size'][0]], 
                                                       device=self.device)
                target_denorm = landmarks * torch.tensor([self.config['image_size'][1], 
                                                        self.config['image_size'][0]], 
                                                        device=self.device)
                
                distances = torch.norm(pred_denorm - target_denorm, dim=2)  # L2 distance per landmark
                mean_distance = distances.mean()
                
                total_loss += loss.item()
                total_distance_error += mean_distance.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_distance_error = total_distance_error / num_batches
        
        return avg_loss, avg_distance_error
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_data = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pth')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint_data, best_path)
            print(f"New best model saved (val_loss: {self.best_val_loss:.4f})")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self, keep_last: int = 5):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
        
        if len(checkpoints) > keep_last:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            
            # Remove oldest checkpoints
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
                os.remove(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        # Load checkpoint if resuming
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data()
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_distance_error = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log metrics
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_distance_error', val_distance_error, epoch)
            self.writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Distance Error: {val_distance_error:.2f} pixels")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Time: {elapsed_time:.2f}s")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config.get('early_stopping', None):
                patience = self.config['early_stopping']
                if len(self.val_losses) > patience:
                    recent_losses = self.val_losses[-patience:]
                    if all(loss >= self.best_val_loss for loss in recent_losses):
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set."""
        self.model.eval()
        metrics = {
            'total_loss': 0.0,
            'distance_errors': [],
            'per_landmark_errors': [[] for _ in range(self.config['num_landmarks'])]
        }
        
        with torch.no_grad():
            for images, landmarks in tqdm(test_loader, desc='Evaluation'):
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                
                predictions = self.model(images)
                loss = self.criterion(predictions, landmarks)
                
                # Calculate per-sample and per-landmark errors
                pred_denorm = predictions * torch.tensor([self.config['image_size'][1], 
                                                        self.config['image_size'][0]], 
                                                       device=self.device)
                target_denorm = landmarks * torch.tensor([self.config['image_size'][1], 
                                                        self.config['image_size'][0]], 
                                                        device=self.device)
                
                distances = torch.norm(pred_denorm - target_denorm, dim=2)
                
                metrics['total_loss'] += loss.item()
                metrics['distance_errors'].extend(distances.mean(dim=1).cpu().numpy())
                
                # Per-landmark errors
                for landmark_idx in range(self.config['num_landmarks']):
                    landmark_errors = distances[:, landmark_idx].cpu().numpy()
                    metrics['per_landmark_errors'][landmark_idx].extend(landmark_errors)
        
        # Calculate final metrics
        metrics['mean_loss'] = metrics['total_loss'] / len(test_loader)
        metrics['mean_distance_error'] = np.mean(metrics['distance_errors'])
        metrics['std_distance_error'] = np.std(metrics['distance_errors'])
        
        # Per-landmark statistics
        metrics['per_landmark_stats'] = []
        for landmark_idx in range(self.config['num_landmarks']):
            errors = metrics['per_landmark_errors'][landmark_idx]
            stats = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'median': np.median(errors),
                'max': np.max(errors)
            }
            metrics['per_landmark_stats'].append(stats)
        
        return metrics


def get_default_config() -> Dict:
    """Get default training configuration."""
    return {
        'model_type': 'lightweight',
        'num_landmarks': 21,
        'input_channels': 3,
        'image_size': (224, 224),
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'train_samples': 8000,
        'val_samples': 2000,
        'use_wing_loss': True,
        'grad_clip': 1.0,
        'lr_patience': 10,
        'early_stopping': 20,
        'log_interval': 50,
        'num_workers': 4,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'runs'
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train hand landmark detection model')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--model-type', type=str, choices=['cnn', 'resnet', 'lightweight'], 
                       default='lightweight', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.model_type:
        config['model_type'] = args.model_type
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.lr:
        config['learning_rate'] = args.lr
    
    # Add timestamp to directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['checkpoint_dir'] = f"checkpoints/{config['model_type']}_{timestamp}"
    config['log_dir'] = f"runs/{config['model_type']}_{timestamp}"
    
    # Save config
    ensure_dir(config['checkpoint_dir'])
    config_path = os.path.join(config['checkpoint_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train(config['num_epochs'], resume_from=args.resume)


if __name__ == "__main__":
    main()
