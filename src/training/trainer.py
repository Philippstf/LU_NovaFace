import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from ..models.gan import LipEnhancementGAN
from ..data.dataset import LipEnhancementDataset, create_data_loaders

class GANTrainer:
    """Trainer for Lip Enhancement GAN."""
    
    def __init__(self,
                 model: LipEnhancementGAN,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 lr_g: float = 2e-4,
                 lr_d: float = 2e-4,
                 beta1: float = 0.5,
                 beta2: float = 0.999,
                 device: str = 'cuda',
                 checkpoint_dir: str = './checkpoints',
                 log_dir: str = './logs',
                 use_wandb: bool = True,
                 wandb_project: str = 'nuva-face-mvp'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.model.generator.parameters(),
            lr=lr_g, 
            betas=(beta1, beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2)
        )
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.StepLR(
            self.optimizer_g, step_size=50, gamma=0.5
        )
        
        self.scheduler_d = optim.lr_scheduler.StepLR(
            self.optimizer_d, step_size=50, gamma=0.5
        )
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Wandb
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    'lr_g': lr_g,
                    'lr_d': lr_d,
                    'beta1': beta1,
                    'beta2': beta2,
                    'lambda_l1': model.lambda_l1,
                    'lambda_perceptual': model.lambda_perceptual,
                    'lambda_style': model.lambda_style,
                    'generator_type': type(model.generator).__name__,
                    'discriminator_type': type(model.discriminator).__name__
                }
            )
            wandb.watch(self.model)
            
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        running_losses = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'g_gan': 0.0,
            'g_l1': 0.0,
            'g_perceptual': 0.0,
            'g_style': 0.0,
            'd_real': 0.0,
            'd_fake': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
                    
            # Train discriminator
            self.optimizer_d.zero_grad()
            
            d_losses, g_losses = self.model.training_step(batch)
            
            d_losses['discriminator_loss'].backward()
            self.optimizer_d.step()
            
            # Train generator (every other iteration to balance)
            if batch_idx % 2 == 0:
                self.optimizer_g.zero_grad()
                
                # Recompute generator losses (discriminator has updated)
                fake_after_images = self.model.generator(
                    batch['before_image'], 
                    batch['condition']
                )
                
                g_losses = self.model.generator_loss(
                    batch['before_image'],
                    batch['after_image'],
                    fake_after_images,
                    batch['condition']
                )
                
                g_losses['generator_loss'].backward()
                self.optimizer_g.step()
            
            # Update running losses
            running_losses['g_loss'] += g_losses['generator_loss'].item()
            running_losses['d_loss'] += d_losses['discriminator_loss'].item()
            running_losses['g_gan'] += g_losses['loss_gan'].item()
            running_losses['g_l1'] += g_losses['loss_l1'].item()
            running_losses['g_perceptual'] += g_losses['loss_perceptual'].item()
            running_losses['g_style'] += g_losses['loss_style'].item()
            running_losses['d_real'] += d_losses['loss_real'].item()
            running_losses['d_fake'] += d_losses['loss_fake'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'G': f"{g_losses['generator_loss'].item():.3f}",
                'D': f"{d_losses['discriminator_loss'].item():.3f}",
                'L1': f"{g_losses['loss_l1'].item():.3f}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    'train/g_loss': g_losses['generator_loss'].item(),
                    'train/d_loss': d_losses['discriminator_loss'].item(),
                    'train/g_l1': g_losses['loss_l1'].item(),
                    'train/step': self.global_step
                })
                
            self.global_step += 1
            
        # Average losses
        num_batches = len(self.train_loader)
        for key in running_losses:
            running_losses[key] /= num_batches
            
        return running_losses
        
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        running_losses = {
            'g_loss': 0.0,
            'd_loss': 0.0,
            'g_l1': 0.0,
            'g_perceptual': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                        
                # Forward pass
                d_losses, g_losses = self.model.training_step(batch)
                
                # Accumulate losses
                running_losses['g_loss'] += g_losses['generator_loss'].item()
                running_losses['d_loss'] += d_losses['discriminator_loss'].item()
                running_losses['g_l1'] += g_losses['loss_l1'].item()
                running_losses['g_perceptual'] += g_losses['loss_perceptual'].item()
                
        # Average losses
        num_batches = len(self.val_loader)
        for key in running_losses:
            running_losses[key] /= num_batches
            
        return running_losses
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            
        # Save epoch checkpoint
        if epoch % 10 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{epoch}.pth')
            
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return epoch."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch']
        
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate and save sample images."""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch from validation set
            batch = next(iter(self.val_loader))
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
                    
            # Select first few samples
            before_images = batch['before_image'][:num_samples]
            real_after_images = batch['after_image'][:num_samples]
            conditions = batch['condition'][:num_samples]
            
            # Generate fake images
            fake_after_images = self.model.generator(before_images, conditions)
            
            # Denormalize images for visualization
            def denorm(x):
                return (x + 1) / 2
                
            # Create comparison grid
            comparison = torch.cat([
                denorm(before_images),
                denorm(real_after_images),
                denorm(fake_after_images)
            ], dim=0)
            
            # Save grid
            grid = vutils.make_grid(
                comparison, 
                nrow=num_samples, 
                padding=2, 
                normalize=False
            )
            
            save_path = self.log_dir / f'samples_epoch_{epoch}.png'
            vutils.save_image(grid, save_path)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'samples': wandb.Image(grid),
                    'epoch': epoch
                })
                
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """Main training loop."""
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            start_epoch = self.load_checkpoint(resume_from) + 1
            self.logger.info(f"Resumed training from epoch {start_epoch}")
            
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            val_losses = self.validate()
            
            # Learning rate scheduling
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train G: {train_losses['g_loss']:.4f}, "
                f"Train D: {train_losses['d_loss']:.4f}, "
                f"Val G: {val_losses['g_loss']:.4f}, "
                f"Val L1: {val_losses['g_l1']:.4f}"
            )
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/generator_loss': train_losses['g_loss'],
                    'train/discriminator_loss': train_losses['d_loss'],
                    'train/l1_loss': train_losses['g_l1'],
                    'val/generator_loss': val_losses['g_loss'],
                    'val/discriminator_loss': val_losses['d_loss'],
                    'val/l1_loss': val_losses['g_l1'],
                    'lr_g': self.optimizer_g.param_groups[0]['lr'],
                    'lr_d': self.optimizer_d.param_groups[0]['lr']
                })
                
            # Save checkpoint
            is_best = val_losses['g_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['g_loss']
                
            self.save_checkpoint(epoch, is_best)
            
            # Generate samples
            if epoch % 5 == 0:
                self.generate_samples(epoch)
                
        self.logger.info("Training completed!")
        
        if self.use_wandb:
            wandb.finish()


def main():
    """Main training function."""
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Train Lip Enhancement GAN')
    parser.add_argument('--metadata-path', type=str, default='./data/raw/metadata.csv',
                       help='Path to metadata CSV file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr-g', type=float, default=2e-4,
                       help='Learning rate for generator')
    parser.add_argument('--lr-d', type=float, default=2e-4,
                       help='Learning rate for discriminator')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size')
    parser.add_argument('--generator', type=str, default='conditional',
                       choices=['conditional', 'unet'],
                       help='Generator architecture')
    parser.add_argument('--discriminator', type=str, default='patch',
                       choices=['conditional', 'patch', 'multiscale'],
                       help='Discriminator architecture')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        metadata_path=args.metadata_path,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Get condition info from dataset
    condition_info = train_loader.dataset.get_condition_info()
    condition_dim = condition_info['condition_dim']
    
    # Create model
    model = LipEnhancementGAN(
        generator_type=args.generator,
        discriminator_type=args.discriminator,
        condition_dim=condition_dim
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = GANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        device=device,
        use_wandb=not args.no_wandb
    )
    
    # Start training
    trainer.train(num_epochs=args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()