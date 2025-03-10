import torch.nn as nn

from torchvision.models.resnet import Bottleneck

from thunderseg.blocks.fpn import CBAM_FPN
from thunderseg.utils.tool import BackboneWithGivenFPN

class HR_ResNet(nn.Module):
    """A high resolution shallow ResNet focus on small object to only keep C2, and C3.
    Ideal to use as a backbone for UAS detections for small objects. 
    """
    def __init__(self):
        super().__init__()
        
        # -------
        # C1 stage (H/2, W/2), 
        # remove maxpool to keep C1 as H/2, W/2 Caution: This would increase 300% memory usage
        # -------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
    
        # -------
        # C2 stage (H/2, W/2)
        # ------- 
        self.layer1 = self._make_layer(
            block=Bottleneck, 
            inplanes=64, 
            planes=64, 
            block_numbers=3, 
            stride=1 # No downsample in C2
        )

        # -------
        # C3 stage (H/4, W/4)
        # -------
        self.layer2 = self._make_layer(
            block=Bottleneck, 
            inplanes=256, 
            planes=128, 
            block_numbers=4, 
            stride=2
        )

        # -------
        # C4 stage (H/8, W/8)
        # -------
        self.layer3 = self._make_layer(
            block=Bottleneck, 
            inplanes=512, 
            planes=256, 
            block_numbers=6, 
            stride=2
        )

        # -------
        # Reduce C4 channels to C4_reduce (H/8, W/8)
        # -------
        self.layer3_reduce = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1),  # reduce channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        c1 = self.relu(x) # C1: [B, 64, H/2, W/2]
        c2 = self.layer1(c1) # C2: [B, 256, H/2, W/2] (stride=1)
        c3 = self.layer2(c2) # C3: [B, 512, H/4, W/4] (stride=2)
        c4 = self.layer3(c3) # C4: [B, 1024, H/8, W/8] (stride=2)
        c4_reduce = self.layer3_reduce(c4) # C4_reduce: [B, 256, H/8, W/8] Reduce channel becuase C4 stage does not capture small objects

        return {"C2":c2, "C3":c3, "C4":c4_reduce}
    
    def _make_layer(self, block, inplanes, planes, block_numbers, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, block_numbers):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

def hr_resnet_cbamfpn():
    backbone = HR_ResNet()
    in_channels_list = [256, 512, 256] #C2:256, C3:512, C4:256
    out_channels = 128
    fpn = CBAM_FPN(in_channels_list, out_channels=out_channels) # Reduce channels to 128 to increase speed
    return BackboneWithGivenFPN(backbone=backbone, 
                                fpn=fpn, 
                                in_channels_list=in_channels_list, 
                                out_channels=out_channels, 
                                return_layers={"layer1":"P2", "layer2":"P3", "layer3_reduce":"P4"})
    