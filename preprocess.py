import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block with two convolutional layers
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

class Preprocessor(nn.Module):
    def __init__(self, input_channels=1, output_size=256, input_height=None, input_width=None):
        super(Preprocessor, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Residual blocks
        self.layer1 = self._make_layer(ResidualBlock, 64, 128, blocks=2, stride=2)
        self.layer2 = self._make_layer(ResidualBlock, 128, 256, blocks=2, stride=2)
        self.layer3 = self._make_layer(ResidualBlock, 256, 512, blocks=2, stride=2)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Bottleneck embedding
        self.fc = nn.Linear(512, output_size)

        # Decoder
        self.decoder_fc = nn.Linear(output_size, 512 * 30 * 40)  # Adjusted for spatial dimensions
        self.decoder_layer1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.decoder_layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.decoder_layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                64, input_channels, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),  # To get outputs between 0 and 1
        )

        # Store input dimensions
        if input_height is None or input_width is None:
            raise ValueError("Input image height and width must be provided")
        self.input_height = input_height
        self.input_width = input_width

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """
        Create layers consisting of multiple residual blocks
        """
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # Adjust the input to match the output dimensions
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x, reconstruct=True):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)  # Shape: [B, 64, H/2, W/2]
        x = self.layer1(x)  # Shape: [B, 128, H/4, W/4]
        x = self.layer2(x)  # Shape: [B, 256, H/8, W/8]
        x = self.layer3(x)  # Shape: [B, 512, H/16, W/16]
        x = self.avgpool(x)  # Shape: [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # Shape: [B, 512]
        h = self.activation(self.fc(x))  # Bottleneck embedding

        if reconstruct:
            # Decoder
            x = self.decoder_fc(h)  # Shape: [B, 512 * 30 * 40]
            x = self.activation(x)
            x = x.view(-1, 512, 30, 40)  # Reshape to [B, 512, 30, 40]
            x = self.decoder_layer1(x)  # [B, 256, 60, 80]
            x = self.decoder_layer2(x)  # [B, 128, 120, 160]
            x = self.decoder_layer3(x)  # [B, 64, 240, 320]
            x_recon = self.decoder_layer4(x)  # [B, 1, 480, 640]
            return h, x_recon
        else:
            return h
