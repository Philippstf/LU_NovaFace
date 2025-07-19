import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualBlock(nn.Module):
    """Residual block for generator."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return F.relu(out + residual)


class ConditionEncoder(nn.Module):
    """Encode treatment conditions into feature maps."""
    
    def __init__(self, condition_dim: int = 5, embed_dim: int = 256):
        super().__init__()
        self.condition_dim = condition_dim
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, embed_dim)
        )
        
    def forward(self, condition: torch.Tensor, spatial_size: Tuple[int, int]) -> torch.Tensor:
        """
        Encode condition and broadcast to spatial dimensions.
        
        Args:
            condition: [B, condition_dim] treatment parameters
            spatial_size: (H, W) for broadcasting
            
        Returns:
            [B, embed_dim, H, W] condition feature maps
        """
        # Encode condition
        embedded = self.encoder(condition)  # [B, embed_dim]
        
        # Broadcast to spatial dimensions
        H, W = spatial_size
        embedded = embedded.unsqueeze(-1).unsqueeze(-1)  # [B, embed_dim, 1, 1]
        embedded = embedded.expand(-1, -1, H, W)  # [B, embed_dim, H, W]
        
        return embedded


class ConditionalGenerator(nn.Module):
    """
    Conditional generator for lip enhancement.
    Takes before image + treatment conditions and generates after image.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 condition_dim: int = 5,
                 ngf: int = 64,
                 n_residual_blocks: int = 6):
        super().__init__()
        
        self.condition_encoder = ConditionEncoder(condition_dim, embed_dim=ngf)
        
        # Encoder (downsampling)
        self.encoder = nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels + ngf, ngf, 7, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            # Downsampling
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True),
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(ngf * 4) for _ in range(n_residual_blocks)]
        )
        
        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            # Upsampling
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            
            # Final convolution
            nn.Conv2d(ngf, input_channels, 7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Generate after image from before image and conditions.
        
        Args:
            image: [B, 3, H, W] before images
            condition: [B, condition_dim] treatment parameters
            
        Returns:
            [B, 3, H, W] generated after images
        """
        B, C, H, W = image.shape
        
        # Encode conditions
        condition_maps = self.condition_encoder(condition, (H, W))
        
        # Concatenate image with condition maps
        x = torch.cat([image, condition_maps], dim=1)
        
        # Encoder
        x = self.encoder(x)
        
        # Residual blocks
        x = self.residual_blocks(x)
        
        # Decoder
        output = self.decoder(x)
        
        return output


class UNetGenerator(nn.Module):
    """
    U-Net style generator with skip connections.
    Alternative architecture for better detail preservation.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 condition_dim: int = 5,
                 ngf: int = 64):
        super().__init__()
        
        self.condition_encoder = ConditionEncoder(condition_dim, embed_dim=ngf//2)
        
        # Encoder
        self.enc1 = self._make_encoder_block(input_channels + ngf//2, ngf, normalize=False)
        self.enc2 = self._make_encoder_block(ngf, ngf * 2)
        self.enc3 = self._make_encoder_block(ngf * 2, ngf * 4)
        self.enc4 = self._make_encoder_block(ngf * 4, ngf * 8)
        self.enc5 = self._make_encoder_block(ngf * 8, ngf * 8)
        
        # Decoder with skip connections
        self.dec1 = self._make_decoder_block(ngf * 8, ngf * 8, dropout=True)
        self.dec2 = self._make_decoder_block(ngf * 16, ngf * 4, dropout=True)  # 16 = 8 + 8 (skip)
        self.dec3 = self._make_decoder_block(ngf * 8, ngf * 2, dropout=True)   # 8 = 4 + 4 (skip)
        self.dec4 = self._make_decoder_block(ngf * 4, ngf)                     # 4 = 2 + 2 (skip)
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, input_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def _make_encoder_block(self, in_channels: int, out_channels: int, normalize: bool = True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, True))
        return nn.Sequential(*layers)
        
    def _make_decoder_block(self, in_channels: int, out_channels: int, dropout: bool = False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)
        
    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Generate enhanced image with U-Net architecture."""
        B, C, H, W = image.shape
        
        # Encode conditions
        condition_maps = self.condition_encoder(condition, (H, W))
        
        # Concatenate image with condition maps
        x = torch.cat([image, condition_maps], dim=1)
        
        # Encoder with skip connections
        enc1 = self.enc1(x)      # [B, 64, H/2, W/2]
        enc2 = self.enc2(enc1)   # [B, 128, H/4, W/4]
        enc3 = self.enc3(enc2)   # [B, 256, H/8, W/8]
        enc4 = self.enc4(enc3)   # [B, 512, H/16, W/16]
        enc5 = self.enc5(enc4)   # [B, 512, H/32, W/32]
        
        # Decoder with skip connections
        dec1 = self.dec1(enc5)                           # [B, 512, H/16, W/16]
        dec2 = self.dec2(torch.cat([dec1, enc4], 1))     # [B, 256, H/8, W/8]
        dec3 = self.dec3(torch.cat([dec2, enc3], 1))     # [B, 128, H/4, W/4]
        dec4 = self.dec4(torch.cat([dec3, enc2], 1))     # [B, 64, H/2, W/2]
        
        # Final layer
        output = self.final(torch.cat([dec4, enc1], 1))  # [B, 3, H, W]
        
        return output


def test_generators():
    """Test generator architectures."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 2
    image_size = 256
    condition_dim = 5
    
    before_images = torch.randn(batch_size, 3, image_size, image_size).to(device)
    conditions = torch.randn(batch_size, condition_dim).to(device)
    
    print("Testing ConditionalGenerator...")
    gen1 = ConditionalGenerator(condition_dim=condition_dim).to(device)
    with torch.no_grad():
        output1 = gen1(before_images, conditions)
    print(f"Input shape: {before_images.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters: {sum(p.numel() for p in gen1.parameters()):,}")
    
    print("\nTesting UNetGenerator...")
    gen2 = UNetGenerator(condition_dim=condition_dim).to(device)
    with torch.no_grad():
        output2 = gen2(before_images, conditions)
    print(f"Input shape: {before_images.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters: {sum(p.numel() for p in gen2.parameters()):,}")


if __name__ == "__main__":
    test_generators()