import torch
from torch import nn


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
 
        self.pool = nn.MaxPool2d(2)
 
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
 
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
 
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
 
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
 
        # Output
        self.final = nn.Conv2d(64, 3, kernel_size=1)
 
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)            # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
 
        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # [B, 512, H/8, W/8]
 
        # Decoder
        d3 = self.up3(b)           # [B, 256, H/4, W/4]
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
 
        d2 = self.up2(d3)          # [B, 128, H/2, W/2]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
 
        d1 = self.up1(d2)          # [B, 64, H, W]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
 
        return self.final(d1)
    
