import torch
import torch.nn as nn

from torchvision.ops import SqueezeExcitation

class SE(nn.Module):
    """Squeeze-and-Excitation block. Use torchvision.ops.SqueezeExcitation, check more info at https://arxiv.org/abs/1709.01507
        If no squeeze_channels is provided, it will be calculated as input_channels // reduction_ratio.
    """
    def __init__(self, input_channels: int, squeeze_channels: int = None, reduction_ratio: int = 16):
        super().__init__()
        squeeze_channels = squeeze_channels or input_channels // reduction_ratio
        self.se = SqueezeExcitation(
            input_channels=input_channels,
            squeeze_channels=squeeze_channels,
            activation=nn.ReLU,
            scale_activation=nn.Sigmoid
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(x)
    

class CBAM(nn.Module):
    """Convolutional Block Attention Module for both channel and spatial attention. 
    Use SE and add Channel Attention to the original SE.
    """
    def __init__(self, input_channels: int, squeeze_channels: int = None):
        super().__init__()
        self.channel_attention = SE(input_channels, reduction_ratio=8)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_weights


