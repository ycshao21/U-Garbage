import torch
import torch.nn as nn


class _ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(_ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
        )

    def forward(self, x):
        max_out = self.max_pool(x)
        avg_out = self.avg_pool(x)

        max_out = self.shared_mlp(max_out)
        avg_out = self.shared_mlp(avg_out)

        out = max_out + avg_out
        out = torch.sigmoid(out)
        return out


class _SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(_SpatialAttention, self).__init__()
        if kernel_size not in (3, 7):
            raise ValueError("Kernel size must be 3 or 7")

        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)

        out = torch.cat([max_out, avg_out], dim=1)
        out = self.conv(out)
        out = torch.sigmoid(out)
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = _ChannelAttention(in_channels, ratio)
        self.spatial_attention = _SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x