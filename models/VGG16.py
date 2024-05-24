import torch
import torch.nn as nn
import torch.nn.functional as F

from .CBAM import CBAM


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer, use_cbam=True):
        super(_ConvBlock, self).__init__()

        conv_block = []
        for i in range(num_layer):
            conv_block.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
            )
            conv_block.append(nn.BatchNorm2d(out_channels))

            if i < num_layer - 1:
                conv_block.append(nn.ReLU(inplace=True))

            in_channels = out_channels

        self.conv_block = nn.Sequential(*conv_block)

        if use_cbam:
            self.cbam = CBAM(out_channels)
        else:
            self.cbam = None

    def forward(self, x):
        x = self.conv_block(x)
        if self.cbam:
            x = self.cbam(x)
        return x


class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, use_cbam=True):
        super(_ResBlock, self).__init__()

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

        self.convblock = _ConvBlock(in_channels, out_channels, num_layers, use_cbam)

    def forward(self, x):
        residual = self.downsample(x)
        x = self.convblock(x)
        x = x + residual
        return x


class _Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_layers, use_cbam=True, use_residual=True
    ):
        super(_Block, self).__init__()

        if use_residual:
            self.block = _ResBlock(in_channels, out_channels, num_layers, use_cbam)
        else:
            self.block = _ConvBlock(in_channels, out_channels, num_layers, use_cbam)

    def forward(self, x):
        x = self.block(x)
        x = F.relu(x)
        return x


class VGG16(nn.Module):
    """VGG16 model with CBAM and residual connections"""

    def __init__(self, in_channels, num_classes, use_cbam=True, use_residual=True):
        super(VGG16, self).__init__()

        self.block1 = _Block(
            in_channels, 64, num_layers=2, use_cbam=use_cbam, use_residual=use_residual
        )
        self.block2 = _Block(
            64, 128, num_layers=2, use_cbam=use_cbam, use_residual=use_residual
        )
        self.block3 = _Block(
            128, 256, num_layers=3, use_cbam=use_cbam, use_residual=use_residual
        )
        self.block4 = _Block(
            256, 512, num_layers=3, use_cbam=use_cbam, use_residual=use_residual
        )
        self.block5 = _Block(
            512, 512, num_layers=3, use_cbam=use_cbam, use_residual=use_residual
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)

        x = self.block2(x)
        x = self.maxpool(x)

        x = self.block3(x)
        x = self.maxpool(x)

        x = self.block4(x)
        x = self.maxpool(x)

        x = self.block5(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


# if __name__ == "__main__":
#     img_shape = (3, 224, 224)
#     num_classes = 12

#     model = VGG16(in_channels=img_shape[0], num_classes=num_classes, use_cbam=True)

#     input_tensor = torch.randn(1, *img_shape)
#     output = model(input_tensor)
#     print(output.shape)

#     print(model)
