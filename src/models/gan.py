import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import torchvision.models as models
from .generator import ConditionalGenerator, UNetGenerator
from .discriminator import ConditionalDiscriminator, PatchDiscriminator, MultiScaleDiscriminator

class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, layers: list = [2, 7, 12, 21, 30]):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between x and y."""
        # Normalize inputs
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0.0
        x_features = self._extract_features(x)
        y_features = self._extract_features(y)
        
        for feat_x, feat_y in zip(x_features, y_features):
            loss += F.mse_loss(feat_x, feat_y)
            
        return loss / len(self.layers)
        
    def _extract_features(self, x: torch.Tensor) -> list:
        """Extract features from specified layers."""
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features


class LipEnhancementGAN(nn.Module):
    """
    Complete GAN model for lip enhancement.
    Combines generator, discriminator, and loss functions.
    """
    
    def __init__(self,
                 generator_type: str = 'conditional',  # 'conditional' or 'unet'
                 discriminator_type: str = 'patch',     # 'conditional', 'patch', 'multiscale'
                 condition_dim: int = 5,
                 ngf: int = 64,
                 ndf: int = 64,
                 lambda_l1: float = 100.0,
                 lambda_perceptual: float = 10.0,
                 lambda_style: float = 50.0):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual 
        self.lambda_style = lambda_style
        
        # Initialize generator
        if generator_type == 'conditional':
            self.generator = ConditionalGenerator(
                condition_dim=condition_dim,
                ngf=ngf
            )
        elif generator_type == 'unet':
            self.generator = UNetGenerator(
                condition_dim=condition_dim,
                ngf=ngf
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
            
        # Initialize discriminator
        if discriminator_type == 'conditional':
            self.discriminator = ConditionalDiscriminator(
                condition_dim=condition_dim,
                ndf=ndf
            )
        elif discriminator_type == 'patch':
            self.discriminator = PatchDiscriminator(
                condition_dim=condition_dim,
                ndf=ndf
            )
        elif discriminator_type == 'multiscale':
            self.discriminator = MultiScaleDiscriminator(
                condition_dim=condition_dim,
                ndf=ndf
            )
        else:
            raise ValueError(f"Unknown discriminator type: {discriminator_type}")
            
        # Loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
        
        self.discriminator_type = discriminator_type
        
    def forward(self, before_images: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Generate after images from before images and conditions."""
        return self.generator(before_images, conditions)
        
    def discriminator_loss(self, 
                          before_images: torch.Tensor,
                          real_after_images: torch.Tensor,
                          fake_after_images: torch.Tensor,
                          conditions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute discriminator loss."""
        
        # Real pairs (before + real after)
        real_pairs = torch.cat([before_images, real_after_images], dim=1)
        
        # Fake pairs (before + fake after)
        fake_pairs = torch.cat([before_images, fake_after_images.detach()], dim=1)
        
        if self.discriminator_type == 'multiscale':
            # Multi-scale discriminator returns list of outputs
            real_outputs = self.discriminator(real_pairs, conditions)
            fake_outputs = self.discriminator(fake_pairs, conditions)
            
            loss_real = 0
            loss_fake = 0
            
            for real_out, fake_out in zip(real_outputs, fake_outputs):
                # Real should be classified as 1
                target_real = torch.ones_like(real_out)
                loss_real += self.criterion_gan(real_out, target_real)
                
                # Fake should be classified as 0
                target_fake = torch.zeros_like(fake_out)
                loss_fake += self.criterion_gan(fake_out, target_fake)
                
            loss_real /= len(real_outputs)
            loss_fake /= len(fake_outputs)
            
        else:
            # Single discriminator output
            real_output = self.discriminator(real_pairs, conditions)
            fake_output = self.discriminator(fake_pairs, conditions)
            
            # Real should be classified as 1
            target_real = torch.ones_like(real_output)
            loss_real = self.criterion_gan(real_output, target_real)
            
            # Fake should be classified as 0
            target_fake = torch.zeros_like(fake_output)
            loss_fake = self.criterion_gan(fake_output, target_fake)
            
        total_loss = (loss_real + loss_fake) * 0.5
        
        return {
            'discriminator_loss': total_loss,
            'loss_real': loss_real,
            'loss_fake': loss_fake
        }
        
    def generator_loss(self,
                      before_images: torch.Tensor,
                      real_after_images: torch.Tensor,
                      fake_after_images: torch.Tensor,
                      conditions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute generator loss."""
        
        # Adversarial loss
        fake_pairs = torch.cat([before_images, fake_after_images], dim=1)
        
        if self.discriminator_type == 'multiscale':
            fake_outputs = self.discriminator(fake_pairs, conditions)
            loss_gan = 0
            
            for fake_out in fake_outputs:
                target = torch.ones_like(fake_out)
                loss_gan += self.criterion_gan(fake_out, target)
                
            loss_gan /= len(fake_outputs)
        else:
            fake_output = self.discriminator(fake_pairs, conditions)
            target = torch.ones_like(fake_output)
            loss_gan = self.criterion_gan(fake_output, target)
        
        # L1 reconstruction loss
        loss_l1 = self.criterion_l1(fake_after_images, real_after_images)
        
        # Perceptual loss
        loss_perceptual = self.perceptual_loss(fake_after_images, real_after_images)
        
        # Style loss (Gram matrix)
        loss_style = self._style_loss(fake_after_images, real_after_images)
        
        # Total generator loss
        total_loss = (loss_gan + 
                     self.lambda_l1 * loss_l1 + 
                     self.lambda_perceptual * loss_perceptual +
                     self.lambda_style * loss_style)
        
        return {
            'generator_loss': total_loss,
            'loss_gan': loss_gan,
            'loss_l1': loss_l1,
            'loss_perceptual': loss_perceptual,
            'loss_style': loss_style
        }
        
    def _style_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute style loss using Gram matrices."""
        def gram_matrix(tensor):
            B, C, H, W = tensor.shape
            features = tensor.view(B, C, H * W)
            gram = torch.bmm(features, features.transpose(1, 2))
            return gram / (C * H * W)
            
        # Extract features using first few layers of perceptual loss VGG
        x_features = []
        y_features = []
        
        for i, layer in enumerate(self.perceptual_loss.vgg[:8]):  # Use early layers
            x = layer(x)
            y = layer(y)
            if i in [1, 3, 6]:  # Select specific layers for style
                x_features.append(x)
                y_features.append(y)
                
        style_loss = 0
        for feat_x, feat_y in zip(x_features, y_features):
            gram_x = gram_matrix(feat_x)
            gram_y = gram_matrix(feat_y)
            style_loss += F.mse_loss(gram_x, gram_y)
            
        return style_loss / len(x_features)
        
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Perform one training step.
        
        Returns:
            Tuple of (discriminator_losses, generator_losses)
        """
        before_images = batch['before_image']
        real_after_images = batch['after_image'] 
        conditions = batch['condition']
        
        # Generate fake images
        fake_after_images = self.generator(before_images, conditions)
        
        # Compute losses
        d_losses = self.discriminator_loss(
            before_images, real_after_images, fake_after_images, conditions
        )
        
        g_losses = self.generator_loss(
            before_images, real_after_images, fake_after_images, conditions
        )
        
        return d_losses, g_losses


def test_gan():
    """Test GAN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test parameters
    batch_size = 2
    image_size = 256
    condition_dim = 5
    
    # Create test batch
    batch = {
        'before_image': torch.randn(batch_size, 3, image_size, image_size).to(device),
        'after_image': torch.randn(batch_size, 3, image_size, image_size).to(device),
        'condition': torch.randn(batch_size, condition_dim).to(device)
    }
    
    print("Testing LipEnhancementGAN...")
    
    # Test different configurations
    configs = [
        ('conditional', 'patch'),
        ('unet', 'patch'), 
        ('conditional', 'multiscale')
    ]
    
    for gen_type, disc_type in configs:
        print(f"\nTesting {gen_type} generator + {disc_type} discriminator...")
        
        gan = LipEnhancementGAN(
            generator_type=gen_type,
            discriminator_type=disc_type,
            condition_dim=condition_dim
        ).to(device)
        
        with torch.no_grad():
            # Test forward pass
            fake_images = gan(batch['before_image'], batch['condition'])
            print(f"Generated image shape: {fake_images.shape}")
            
            # Test training step
            d_losses, g_losses = gan.training_step(batch)
            
            print(f"Discriminator loss: {d_losses['discriminator_loss'].item():.4f}")
            print(f"Generator loss: {g_losses['generator_loss'].item():.4f}")
            print(f"L1 loss: {g_losses['loss_l1'].item():.4f}")
            print(f"Perceptual loss: {g_losses['loss_perceptual'].item():.4f}")
            
        print(f"Total parameters: {sum(p.numel() for p in gan.parameters()):,}")


if __name__ == "__main__":
    test_gan()