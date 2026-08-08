import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# Standard DeepLabV3+ ASPP module

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3x3_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3x3_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.con3x3_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_6(x)
        x3 = self.conv3x3_12(x)
        x4 = self.con3x3_18(x)
        x5 = F.interpolate(self.image_pool(x), size=(h, w), mode='bilinear', align_corners=False)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(out)


# Standard DeepLabV3+ Decoder

class Decoder(nn.Module):
    def __init__(self, low_channels=256, in_channels=256):
        super().__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + 48, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

    def forward(self, x, low_feat):
        low = self.low_conv(low_feat)
        x = F.interpolate(x, size=low.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low], dim=1)
        return self.fuse(x)


# Standard DeepLabV3+ (OS=16)

class DeepLabV3PlusSCF(nn.Module):
    def __init__(self, in_channels=9, out_channels=1):
        super().__init__()
        
     
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
      
        pretrained_weight = self.backbone.conv1.weight.data
        new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.data[:, :3] = pretrained_weight
            new_conv.weight.data[:, 3:] = pretrained_weight[:, :1].repeat(1, in_channels-3, 1, 1)
        self.backbone.conv1 = new_conv

      
        self.backbone.layer3[0].conv2.stride = (1, 1)
        self.backbone.layer3[0].downsample[0].stride = (1, 1)
        for block in self.backbone.layer3:
            block.conv2.dilation = (2, 2)
            block.conv2.padding = (2, 2)

        self.backbone.layer4[0].conv2.stride = (1, 1)
        self.backbone.layer4[0].downsample[0].stride = (1, 1)
        for block in self.backbone.layer4:
            block.conv2.dilation = (4, 4)
            block.conv2.padding = (4, 4)

       
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        self.aspp = ASPP(in_channels=2048, out_channels=256)
        self.decoder = Decoder(low_channels=256, in_channels=256)
        self.out = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        x0 = self.layer0(x)
        c1 = self.layer1(x0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        aspp_out = self.aspp(c4)
        dec_out = self.decoder(aspp_out, c1)
        out = self.out(dec_out)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        out = torch.sigmoid(out)
        return out