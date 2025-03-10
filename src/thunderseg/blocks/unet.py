import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g_conv = self.W_g(g)
        x_conv = self.W_x(x)
        activation = self.relu(g_conv + x_conv)
        attention = self.sigmoid(self.psi(activation))
        return x * attention

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        
        # Encoder
        self.encoder1 = self._block(in_channels, features, "enc1")
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._block(features, features*2, "enc2")
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self._block(features*2, features*4, "enc3")
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self._block(features*4, features*8, "enc4")
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(features*8, features*16, "bottleneck")

        # Decoder with attention
        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, 2, 2)
        self.att4 = AttentionGate(features*8, features*8, features*4)
        self.decoder4 = self._block(features*16, features*8, "dec4")

        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, 2, 2)
        self.att3 = AttentionGate(features*4, features*4, features*2)
        self.decoder3 = self._block(features*8, features*4, "dec3")

        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, 2, 2)
        self.att2 = AttentionGate(features*2, features*2, features)
        self.decoder2 = self._block(features*4, features*2, "dec2")

        self.upconv1 = nn.ConvTranspose2d(features*2, features, 2, 2)
        self.att1 = AttentionGate(features, features, features//2)
        self.decoder1 = self._block(features*2, features, "dec1")

        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with attention
        dec4 = self.upconv4(bottleneck)
        att4 = self.att4(enc4, dec4)
        dec4 = torch.cat((att4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        att3 = self.att3(enc3, dec3)
        dec3 = torch.cat((att3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        att2 = self.att2(enc2, dec2)
        dec2 = torch.cat((att2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        att1 = self.att1(enc1, dec1)
        dec1 = torch.cat((att1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final(dec1)