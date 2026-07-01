# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "CBAM",
    "CoordAtt",
    "ECA",
    "SE",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "HetConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    'FCM',
    'Pzconv',
    'Down',
    'MCSPF',
    'MSDAF',
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

## SE
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

## ECA
class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel,2)+  b)/gamma))
        kernel_size = kernel_size if kernel_size % 2  else kernel_size+1
        padding = kernel_size//2
        self.avg_pool =nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

## CoordAtt
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

#####FBRT-YOLO: Faster and better for real-time aerial image detection#####
'''
Channel interaction (semantic information guides spatial information):
w1 = σ(AvgPool(Conv_dw(x)))
'''
class Channel(nn.Module): 
    def __init__(self, dim):
        super().__init__()
        self.dwconv = self.dconv = nn.Conv2d(
            dim, dim, 3,
            1, 1, groups=dim
        )
        self.Apt = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2 = self.dwconv(x)
        x5 = self.Apt(x2)
        x6 = self.sigmoid(x5)

        return x6

'''
Spatial interaction (spatial information guides semantic information):
w2 = σ(BN(Conv_1x1(x)))
'''
class Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x5 = self.bn(x1)
        x6 = self.sigmoid(x5)

        return x6

class FCM(nn.Module):
    # def __init__(self, dim,dim_out, beta=0.75,k1=3,k2=5,k3=7):
    def __init__(self, dim,dim_out, beta=0.75):
        super().__init__()
        # self.k1 = k1
        # self.k2 = k2
        # self.k3 = k3
        self.one = dim - int(dim*beta)
        self.two = int(dim*beta)
        self.conv1 = Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = Conv(self.one, dim, 1, 1)

        self.conv2 = Conv(self.two, dim, 1, 1)
        self.conv3 = Conv(dim, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)
        # self.pzconv = Pzconv(dim, k1, k2, k3)
        self.residual_proj = nn.Identity() if dim == dim_out else Conv(dim, dim_out, k=1, s=1)
        # self.output_conv = Conv(dim_out, dim_out, k=3, s=1, p=1)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        aggregated = x33 + x44
        # w1 = self.spatial(x4)
        # w2 = self.channel(x3)
        # x5 = self.pzconv(x)
        # aggregated = w1 * x5 + w2 * x5
        residual = self.residual_proj(x)
        aggregated = aggregated + residual
        # output = self.output_conv(aggregated)
        return aggregated


class FCM_3(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class FCM_2(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44

        return x5


class FCM_1(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()

        self.one = dim // 4
        self.two = dim - dim // 4
        self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

#     def forward(self, x):
#         x1, x2 = torch.split(x, [self.one, self.two], dim=1)
#         x3 = self.conv1(x1)
#         x3 = self.conv12(x3)
#         x3 = self.conv123(x3)
#         x4 = self.conv2(x2)
#         x33 = self.spatial(x4) * x3
#         x44 = self.channel(x3) * x4
#         x5 = x33 + x44

#         return x5


# class FCM(nn.Module):
#     def __init__(self, dim,dim_out):
#         super().__init__()
#         self.one = dim // 4
#         self.two = dim - dim // 4
#         self.conv1 = Conv(dim // 4, dim // 4, 3, 1, 1)
#         self.conv12 = Conv(dim // 4, dim // 4, 3, 1, 1)
#         self.conv123 = Conv(dim // 4, dim, 1, 1)

#         self.conv2 = Conv(dim - dim // 4, dim, 1, 1)
#         self.conv3 = Conv(dim, dim, 1, 1)
#         self.spatial = Spatial(dim)
#         self.channel = Channel(dim)

#     def forward(self, x):
#         x1, x2 = torch.split(x, [self.one, self.two], dim=1)
#         x3 = self.conv1(x1)
#         x3 = self.conv12(x3)
#         x3 = self.conv123(x3)
#         x4 = self.conv2(x2)
#         x33 = self.spatial(x4) * x3
#         x44 = self.channel(x3) * x4
#         x5 = x33 + x44
#         x5 = self.conv3(x5)
#         return x5

class Pzconv(nn.Module): # Multi-Kernel Perception Unit - MKP
    def __init__(self, dim, k1=3, k2=5, k3=7):
        super().__init__()
        self.conv1 = nn.Conv2d(
            dim, dim, k1,
            1, k1//2, groups=dim
        )
        self.conv2 = Conv(dim, dim, k=1, s=1, )
        self.conv3 = nn.Conv2d(
            dim, dim, k2,
            1, k2//2, groups=dim
        )
        self.conv4 = Conv(dim, dim, k=1, s=1, )
        self.conv5 = nn.Conv2d(
            dim, dim, k3,
            1, k3//2, groups=dim
        )

    def forward(self, x):
        # x1 = self.conv1(x)
        # x2 = self.conv2(x1)
        # x3 = self.conv3(x2)
        # x4 = self.conv4(x3)
        # x5 = self.conv5(x4)
        x1 = self.conv1(x)
        x3 = self.conv3(x1)
        x5 = self.conv5(x3)
        # x6 = x5 + x
        return x5

# First use grouped convolution for spatial downsampling, then use pointwise convolution for channel expansion.
class Down(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.conv2 = Conv(dim, dim, 3, 2, 1, g=dim // 2, act=False)
        self.conv4 = Conv(dim, dim_out, 1, 1)

    def forward(self, x):
        x2 = self.conv2(x)
        x2 = self.conv4(x2)
        return x2

## Greenhouse blueberry ##
class InvertedResidualBlock(nn.Module):
    """Lightweight inverted residual block for MCSPF."""

    def __init__(self, c: int, expand: float = 2.0, act: nn.Module | None = None):
        super().__init__()
        hidden = int(c * expand)
        self.act = act or nn.SiLU()
        self.pw1 = Conv(c, hidden, k=1, s=1)
        self.dw = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.pw2 = Conv(hidden, c, k=1, s=1)

    def forward(self, x):
        y = self.pw1(x)
        y = self.act(self.bn(self.dw(y)))
        y = self.pw2(y)
        return x + y


class SimpleSPP(nn.Module):
    """Spatial pyramid pooling with three kernel sizes."""

    def __init__(self, c: int, k: tuple[int, int, int] = (3, 5, 7)):
        super().__init__()
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in k])
        self.conv = Conv(c * (len(k) + 1), c, k=1, s=1)

    def forward(self, x):
        pools = [m(x) for m in self.m]
        return self.conv(torch.cat([x, *pools], 1))


class MCSPF(nn.Module):
    """
    Multi-branch Cross-Stage Pyramid Fusion.
    1) 1x1 squeeze + channel split; 2) branch1 keeps shallow info; 3) branch2 uses IRB + SPP; 4) concat + fuse.
    """

    def __init__(self, c1: int, c2: int, expand: float = 2.0, irb_repeats: int = 1):
        super().__init__()
        self.expand = expand
        self.irb_repeats = irb_repeats
        self.out_channels = c2
        self._build(c1, c2)

    def _build(self, c1: int, c2: int):
        """(Re)build internal layers to align with current input/output channels (handles different width scales)."""
        self.in_channels = c1
        c_half = max(1, c2 // 2)
        self.reduce = Conv(c1, c2, k=1, s=1)
        self.branch1 = Conv(c_half, c_half, k=1, s=1)
        self.branch2_irb = nn.Sequential(
            *[InvertedResidualBlock(c_half, expand=self.expand) for _ in range(max(1, self.irb_repeats))]
        )
        self.branch2_spp = SimpleSPP(c_half)
        self.fuse = Conv(c_half * 2, c2, k=1, s=1)

    def forward(self, x):
        # Dynamically rebuild branches when input channels differ from the build-time value.
        if x.shape[1] != getattr(self, "in_channels", None):
            self._build(x.shape[1], self.out_channels)
            self.to(x.device)

        x = self.reduce(x)
        x1, x2 = torch.split(x, [x.shape[1] // 2, x.shape[1] - x.shape[1] // 2], dim=1)
        b1 = self.branch1(x1)
        b2 = self.branch2_spp(self.branch2_irb(x2))
        out = torch.cat((b1, b2), dim=1)
        return self.fuse(out)
## Greenhouse blueberry ##

## MSDAF: Multi-Scale Dual-Attention Fusion Module ##
class MSDAF(nn.Module):
    """
    Multi-Scale Dual-Attention Fusion Module (MSDAF).
    
    This module addresses important associations for small-object detection in deep neural networks.
    It combines multi-scale convolutional feature extraction with channel and spatial attention
    mechanisms to improve sensitivity to fine-grained details.
    
    The module mainly consists of three stages:
    A. Segmentation: splits the input into channel-attention and spatial-attention branches
       according to the split ratio beta, and evenly divides it into four parts for multi-scale
       feature extraction.
    B. Directional Conversion: extracts dual-attention weights and multi-scale features.
    C. Feature Aggregation: performs weighted fusion and outputs the final features.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        beta (float): Split ratio that controls the channel proportion assigned to the
            channel-attention branch. Valid range is [0, 1].
            - beta=0.1: 10% of channels are used for channel attention.
            - beta=0.25: 25% of channels are used for channel attention.
            - beta=0.5: 50% of channels are used for channel attention (balanced).
            - beta=0.75: 75% of channels are used for channel attention.
            - beta=0.9: 90% of channels are used for channel attention.
    """
    
    def __init__(self, c1, c2, beta=0.5):
        super().__init__()
        
        # Validate parameters.
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in range [0, 1], but got {beta}")
        if c1 < 4:
            raise ValueError(f"MSDAF requires at least 4 input channels for multi-scale split, but received {c1}.")
        
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        
        # A. Segmentation stage - split branches according to beta.
        # X^1: channel-attention branch (beta*C).
        # X^2: spatial-attention branch ((1-beta)*C).
        self.channel_split_channels = max(1, int(c1 * beta))  # Ensure at least one channel.
        self.spatial_split_channels = c1 - self.channel_split_channels
        
        # Evenly split into four parts for multi-scale feature extraction (X_3, X_4, X_5, X_6).
        self.multiscale_channels = c1 // 4
        self.multiscale_remainder = c1 % 4
        
        # B. Directional conversion stage.
        # 1. Channel-attention weight w_1 (what features are important).
        # ω_1 = Sigmoid(AvgPool(PWConv(DWConv(X_1))))
        self.channel_attention_dwconv = DWConv(self.channel_split_channels, self.channel_split_channels, k=3, s=1)
        self.channel_attention_pwconv = Conv(self.channel_split_channels, c2, k=1, s=1)
        self.channel_attention_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_sigmoid = nn.Sigmoid()
        
        # 2. Spatial-attention weight w_2 (where the target region is).
        # ω_2 = Sigmoid(PWConv(X_2))
        self.spatial_attention_pwconv = Conv(self.spatial_split_channels, c2, k=1, s=1)
        self.spatial_attention_sigmoid = nn.Sigmoid()
        
        # 3. Multi-scale convolutional extraction (7x7, 5x5, 3x3).
        # x_3 = Conv_{k=7}(X_3)
        # x_4 = Conv_{k=5}(X_4)  
        # x_5 = Conv_{k=3}(X_5)
        # X_6 remains unchanged.
        self.conv_7x7 = Conv(self.multiscale_channels, self.multiscale_channels, k=7, s=1, p=3)
        self.conv_5x5 = Conv(self.multiscale_channels, self.multiscale_channels, k=5, s=1, p=2)
        self.conv_3x3 = Conv(self.multiscale_channels, self.multiscale_channels, k=3, s=1, p=1)
        
        # 4. Feature concatenation and fusion.
        # X_{multiscale} = PWConv(Concat([x_3, x_4, x_5, X_6], dim=1))
        multiscale_concat_channels = self.multiscale_channels * 3 + (self.multiscale_channels + self.multiscale_remainder)
        self.multiscale_fusion = Conv(multiscale_concat_channels, c2, k=1, s=1)
        
        # C. Feature aggregation stage.
        # X_{output} = Conv_{k=3}(ω_1 ⊗ X_{multiscale} + ω_2 ⊗ X_{multiscale} + X_{input})
        self.residual_proj = nn.Identity() if c1 == c2 else Conv(c1, c2, k=1, s=1)
        self.output_conv = Conv(c2, c2, k=3, s=1, p=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features X_{input} ∈ R^{C×H×W}.
            
        Returns:
            torch.Tensor: Output features X_{output} ∈ R^{C×H×W}.
        """
        # A. Segmentation stage.
        # 1. Channel split: split into two branches according to beta.
        x1, x2 = torch.split(x, [self.channel_split_channels, self.spatial_split_channels], dim=1)
        
        # 2. Even split: split into four parts for multi-scale processing.
        # Split scheme: the first three parts use multiscale_channels, and the last part contains the remaining channels.
        split_sizes = [self.multiscale_channels] * 3 + [self.multiscale_channels + self.multiscale_remainder]
        x3, x4, x5, x6 = torch.split(x, split_sizes, dim=1)
        
        # B. Directional conversion stage.
        # 1. Channel-attention weight w_1.
        w1 = self.channel_attention_dwconv(x1)
        w1 = self.channel_attention_pwconv(w1)
        w1 = self.channel_attention_pool(w1)
        w1 = self.channel_attention_sigmoid(w1)  # ω_1 ∈ R^{C×1×1}
        
        # 2. Spatial-attention weight w_2.
        w2 = self.spatial_attention_pwconv(x2)
        w2 = self.spatial_attention_sigmoid(w2)  # ω_2 ∈ R^{1×H×W}
        
        # 3. Multi-scale convolutional feature extraction.
        x3_out = self.conv_7x7(x3)  # 7x7 convolution.
        x4_out = self.conv_5x5(x4)  # 5x5 convolution.
        x5_out = self.conv_3x3(x5)  # 3x3 convolution.
        # x6 remains unchanged and unprocessed.
        
        # 4. Feature concatenation and fusion.
        x_multiscale = torch.cat([x3_out, x4_out, x5_out, x6], dim=1)
        x_multiscale = self.multiscale_fusion(x_multiscale)  # X_{multiscale} ∈ R^{C×H×W'}
        
        # C. Feature aggregation stage.
        # Weighted fusion: ω_1 ⊗ X_{multiscale} + ω_2 ⊗ X_{multiscale}.
        aggregated = w1 * x_multiscale + w2 * x_multiscale
        
        # Residual connection: + X_{input}.
        residual = self.residual_proj(x)
        aggregated = aggregated + residual
        
        # Final output convolution.
        output = self.output_conv(aggregated)
        
        return output
##MSDAF##

class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))

## Custom HetConv ##
class HetConv(nn.Module):
    """
    HetConv (Heterogeneous Kernel-Based Convolution) for lightweight detection heads.
    
    This module implements heterogeneous convolution that mixes different kernel sizes within
    the same layer to reduce parameters and computational cost while maintaining performance.
    
    Attributes:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the larger kernels (e.g., 3 for 3x3).
        p (int): Control parameter - 1/p fraction uses larger kernels, (1-1/p) uses 1x1 kernels.
        stride (int): Stride for convolution.
        padding (int): Padding for convolution.
        groups (int): Number of groups for convolution.
        hetconv_3x3 (nn.Conv2d): 3x3 convolution for spatial feature extraction.
        hetconv_1x1 (nn.Conv2d): 1x1 convolution for channel mixing.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function.
        
    Examples:
        Create a HetConv layer
        >>> hetconv = HetConv(256, 256, 3, p=4)
        >>> x = torch.randn(1, 256, 40, 40)
        >>> output = hetconv(x)
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, p: int = 4, 
                 stride: int = 1, padding: int = None, groups: int = 1, bias: bool = False, act: bool = True):
        """
        Initialize HetConv layer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the larger kernels.
            p (int): Control parameter for kernel distribution.
            stride (int): Stride for convolution.
            padding (int): Padding for convolution.
            groups (int): Number of groups for convolution.
            bias (bool): Whether to use bias.
            act (bool): Whether to use activation.
        """
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p = p
        
        # Calculate channels for different kernel types
        # 1/p of channels use kernel_size x kernel_size convolution
        self.out_channels_3x3 = out_channels // p
        # (1 - 1/p) of channels use 1x1 convolution  
        self.out_channels_1x1 = out_channels - self.out_channels_3x3
        
        # 3x3 convolution for spatial feature extraction
        self.hetconv_3x3 = nn.Conv2d(
            in_channels, self.out_channels_3x3, kernel_size, stride, padding, 
            groups=min(groups, self.out_channels_3x3), bias=bias
        )
        
        # 1x1 convolution for efficient channel mixing
        self.hetconv_1x1 = nn.Conv2d(
            in_channels, self.out_channels_1x1, 1, stride, 0, 
            groups=min(groups, self.out_channels_1x1), bias=bias
        )
        
        # Batch normalization and activation
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HetConv layer."""
        # Apply both convolution types
        out_3x3 = self.hetconv_3x3(x)
        out_1x1 = self.hetconv_1x1(x)
        
        # Concatenate outputs from different kernel sizes
        out = torch.cat([out_3x3, out_1x1], dim=1)
        
        # Apply batch normalization and activation
        out = self.act(self.bn(out))
        
        return out

class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, W, H) and output shape is (B, 4C, W/2, H/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """
        Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]
