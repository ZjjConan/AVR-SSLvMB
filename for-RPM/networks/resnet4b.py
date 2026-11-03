import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResBlock, ConvBNAct

class ResNet(nn.Module):

    def __init__(self, in_channels=1, block_drop=0.0, big=False):
        super().__init__()

        if big:
            channels = [32, 64, 128, 256]
        else:
            channels = [32, 64, 96, 128]
        strides = [2, 2, 2, 1]

        # -------------------------------------------------------------------
        # frame encoder 
        self.in_channels = in_channels
        self.ou_channels = channels[-1]
        self.num_stages = len(strides)

        for l in range(len(strides)):
            setattr(
                self, "res"+str(l), 
                self._make_layer(
                    channels[l], 
                    stride=strides[l], 
                    block=ResBlock, 
                    dropout=block_drop
                )
            )
        # -------------------------------------------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, channels, stride, block, dropout):
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
            ConvBNAct(self.in_channels, channels, use_act=False, kernel_size=1, bias=False)
        )
        stage = block(
            self.in_channels, channels, 
            downsample, stride=stride, 
            dropout=dropout
        )
        self.in_channels = channels
        return stage


    def forward(self, x):
        for l in range(self.num_stages):
            x = getattr(self, "res"+str(l))(x)

        return x
      

def resnet4b(args):
    return ResNet(args.in_channels, args.block_drop)

def resnet4b_big(args):
    return ResNet(args.in_channels, args.block_drop, big=True)
