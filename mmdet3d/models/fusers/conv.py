from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser", "MultiScaleConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: int, use_depth_attention: bool = True) -> None:
        """ConvFuser with optional depth-guided attention for small object detection.
        
        Args:
            in_channels: List of input channel numbers for each modality.
            out_channels: Output channel number after fusion.
            use_depth_attention: Whether to use depth-guided channel attention.
                Default: True.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_depth_attention = use_depth_attention
        
        # Base fusion layers
        self.fusion_conv = nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(out_channels)
        self.fusion_relu = nn.ReLU(True)
        
        # Depth-guided channel attention for small object enhancement
        if use_depth_attention:
            self.attention_pool = nn.AdaptiveAvgPool2d(1)
            self.attention_reduce = nn.Conv2d(out_channels, out_channels // 16, 1, bias=False)
            self.attention_relu = nn.ReLU(True)
            self.attention_expand = nn.Conv2d(out_channels // 16, out_channels, 1, bias=False)
            self.attention_sigmoid = nn.Sigmoid()
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        fused = torch.cat(inputs, dim=1)
        # Apply base fusion layers
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused)
        fused = self.fusion_relu(fused)
        
        # Apply depth-guided attention if enabled
        if self.use_depth_attention:
            # Calculate attention weights
            attention_weights = self.attention_pool(fused)
            attention_weights = self.attention_reduce(attention_weights)
            attention_weights = self.attention_relu(attention_weights)
            attention_weights = self.attention_expand(attention_weights)
            attention_weights = self.attention_sigmoid(attention_weights)
            # Clamp attention weights for numerical stability
            attention_weights = torch.clamp(attention_weights, 0.1, 0.9)
            # Apply attention weights to features
            fused = fused * attention_weights
        
        return fused


@FUSERS.register_module()
class MultiScaleConvFuser(nn.Module):
    """Multi-scale Convolutional Fuser with attention mechanism.
    
    This fuser processes multi-scale features from camera and LiDAR,
    applying channel attention to improve feature fusion quality.
    
    Args:
        in_channels (int): List of input channel numbers for each modality.
        out_channels (int): Output channel number after fusion.
        attention_ratio (int): Reduction ratio for channel attention.
            Default: 16.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        attention_ratio: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Base fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # Channel attention for fused features
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // attention_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // attention_ratio, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        
        # Refinement convolution
        self.refine_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Ensure all inputs have the same spatial dimensions
        if len(inputs) < 2:
            raise ValueError(f"Expected at least 2 input tensors, got {len(inputs)}")
        
        # Resize all inputs to the first input's spatial size
        target_size = inputs[0].shape[-2:]
        resized_inputs = []
        for inp in inputs:
            if inp.shape[-2:] != target_size:
                inp = F.interpolate(
                    inp, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            resized_inputs.append(inp)
        
        # Concatenate and fuse
        fused = torch.cat(resized_inputs, dim=1)
        fused = self.fusion_conv(fused)
        
        # Apply channel attention with numerical stability
        attention_weights = self.channel_attention(fused)
        # Clamp attention weights to prevent extreme values
        attention_weights = torch.clamp(attention_weights, 0.1, 0.9)
        fused = fused * attention_weights
        
        # Refine fused features with residual connection and numerical stability
        refined = self.refine_conv(fused)
        # Clamp refined features to prevent explosion
        refined = torch.clamp(refined, -100.0, 100.0)
        fused = fused + refined
        
        return fused
