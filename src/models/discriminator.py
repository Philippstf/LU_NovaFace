import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SpectralNorm:
    """Spectral normalization wrapper."""
    
    def __init__(self, module, name='weight', power_iterations=1):
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def __call__(self, module, inputs):
        self._update_u_v()
        return inputs

    @staticmethod
    def apply(module, name, power_iterations):
        fn = SpectralNorm(module, name, power_iterations)
        weight = getattr(module, name)
        delattr(module, name)
        module.register_parameter(name + "_bar", nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', power_iterations=1):
    """Apply spectral normalization to module."""
    SpectralNorm.apply(module, name, power_iterations)
    return module


class ConditionalDiscriminator(nn.Module):
    """
    Conditional discriminator for lip enhancement GAN.
    Takes concatenated before+after images and treatment conditions.
    """
    
    def __init__(self, 
                 input_channels: int = 6,  # 3 (before) + 3 (after)
                 condition_dim: int = 5,
                 ndf: int = 64,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.condition_dim = condition_dim
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512)
        )
        
        # Convolutional layers
        sequence = [
            # First layer (no normalization)
            nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Intermediate layers
        nf_mult = 1
        for i in range(1, 4):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8)
            conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1)
            if use_spectral_norm:
                conv = spectral_norm(conv)
            sequence += [
                conv,
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** 4, 8)
        conv = nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1)
        if use_spectral_norm:
            conv = spectral_norm(conv)
        sequence += [
            conv,
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        self.conv_layers = nn.Sequential(*sequence)
        
        # Final classification layer with condition fusion
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * nf_mult + 512, 1, 4, stride=1, padding=1),
        )
        
    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs fake image pairs.
        
        Args:
            images: [B, 6, H, W] concatenated before+after images
            condition: [B, condition_dim] treatment parameters
            
        Returns:
            [B, 1, H_out, W_out] discriminator output maps
        """
        # Process images through conv layers
        x = self.conv_layers(images)  # [B, ndf*8, H/16, W/16]
        
        # Encode condition
        cond_encoded = self.condition_encoder(condition)  # [B, 512]
        
        # Broadcast condition to spatial dimensions
        B, C, H, W = x.shape
        cond_maps = cond_encoded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # Concatenate features with condition
        x = torch.cat([x, cond_maps], dim=1)
        
        # Final classification
        output = self.classifier(x)
        
        return output


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies overlapping patches.
    More efficient and can handle arbitrary input sizes.
    """
    
    def __init__(self, 
                 input_channels: int = 6,
                 condition_dim: int = 5,
                 ndf: int = 64,
                 n_layers: int = 3):
        super().__init__()
        
        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, ndf),
            nn.ReLU(True)
        )
        
        # First layer
        sequence = [
            nn.Conv2d(input_channels + ndf, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # Intermediate layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 6, H, W] concatenated before+after images
            condition: [B, condition_dim] treatment parameters
            
        Returns:
            [B, 1, H_out, W_out] patch classifications
        """
        B, C, H, W = images.shape
        
        # Encode condition and broadcast
        cond_encoded = self.condition_encoder(condition)  # [B, ndf]
        cond_maps = cond_encoded.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # Concatenate images with condition
        x = torch.cat([images, cond_maps], dim=1)
        
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator that operates at different resolutions.
    Helps capture both fine and coarse details.
    """
    
    def __init__(self, 
                 input_channels: int = 6,
                 condition_dim: int = 5,
                 ndf: int = 64,
                 num_scales: int = 3):
        super().__init__()
        
        self.num_scales = num_scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        
        # Create discriminators for each scale
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            netD = PatchDiscriminator(
                input_channels=input_channels,
                condition_dim=condition_dim,
                ndf=ndf,
                n_layers=3 if i == 0 else 2  # More layers for finest scale
            )
            self.discriminators.append(netD)
            
    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> list:
        """
        Returns list of discriminator outputs at different scales.
        """
        results = []
        x = images
        
        for i, discriminator in enumerate(self.discriminators):
            results.append(discriminator(x, condition))
            if i < self.num_scales - 1:
                x = self.downsample(x)
                
        return results


def test_discriminators():
    """Test discriminator architectures."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    batch_size = 2
    image_size = 256
    condition_dim = 5
    
    # Concatenated before+after images
    images = torch.randn(batch_size, 6, image_size, image_size).to(device)
    conditions = torch.randn(batch_size, condition_dim).to(device)
    
    print("Testing ConditionalDiscriminator...")
    disc1 = ConditionalDiscriminator(condition_dim=condition_dim).to(device)
    with torch.no_grad():
        output1 = disc1(images, conditions)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters: {sum(p.numel() for p in disc1.parameters()):,}")
    
    print("\nTesting PatchDiscriminator...")
    disc2 = PatchDiscriminator(condition_dim=condition_dim).to(device)
    with torch.no_grad():
        output2 = disc2(images, conditions)
    print(f"Input shape: {images.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters: {sum(p.numel() for p in disc2.parameters()):,}")
    
    print("\nTesting MultiScaleDiscriminator...")
    disc3 = MultiScaleDiscriminator(condition_dim=condition_dim).to(device)
    with torch.no_grad():
        outputs3 = disc3(images, conditions)
    print(f"Input shape: {images.shape}")
    print(f"Number of scales: {len(outputs3)}")
    for i, out in enumerate(outputs3):
        print(f"Scale {i} output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in disc3.parameters()):,}")


if __name__ == "__main__":
    test_discriminators()