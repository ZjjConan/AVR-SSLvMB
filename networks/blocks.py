import torch.nn as nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, use_act=True, conv_dim=2, **kwargs):
        super(ConvBNAct, self).__init__()

        layer = []
        layer.append(
            getattr(nn, 'Conv{}d'.format(conv_dim))(in_channels, out_channels, **kwargs)
        )

        layer.append(
            getattr(nn, 'BatchNorm{}d'.format(conv_dim))(out_channels)
        )
        if use_act:
            layer.append(nn.ReLU())

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class ResBlock(nn.Module):

    def __init__(self, in_channels, ou_channels, downsample, stride=1, dropout=0.0):
        super().__init__()

        md_channels = ou_channels

        self.conv1 = ConvBNAct(
            in_channels, md_channels, use_act=True, conv_dim=2,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.conv2 = ConvBNAct(
            md_channels, md_channels, use_act=True, conv_dim=2,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.conv3 = ConvBNAct(
            md_channels, ou_channels, use_act=True, conv_dim=2,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.downsample = downsample
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.drop(out)
        out = out + identity
        return out


class BasicBlock(nn.Module):

    def __init__(self, in_channels, ou_channels, downsample, stride=1, dropout=0.0):
        super().__init__()

        md_channels = ou_channels

        self.conv1 = ConvBNAct(
            in_channels, md_channels, use_act=True, conv_dim=2,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.conv2 = ConvBNAct(
            md_channels, md_channels, use_act=True, conv_dim=2,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.downsample = downsample
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.drop(out)
        out = out + identity
        return out


# Convolutional Concept Block
class ConvConceptBlock(nn.Module):

    def __init__(self, in_channels, ou_channels, md_channels=None, downsample=None, stride=1, dropout=0.0, permute=False):
        
        super().__init__()

        self.permute = permute
        if md_channels == None:
            md_channels = ou_channels

        self.conv1 = ConvBNAct(
            in_channels, md_channels, use_act=True, conv_dim=2,
            kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.conv2 = ConvBNAct(
            md_channels, ou_channels, use_act=True, conv_dim=2,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.downsample = downsample
        
    def forward(self, x):
        if self.permute:
            x = x.permute(0, 2, 1, 3)
        identity = self.downsample(x)
        out = self.conv1(x)        
        out = self.conv2(out)
        out = self.drop(out)
        out = out + identity
        return out


class Classifier(nn.Module):

    def __init__(self, in_channels, ou_channels, norm_layer=nn.BatchNorm1d, dropout=0.0, classifier_hidreduce=1.0):
        super().__init__()

        md_channels = int(in_channels // classifier_hidreduce)
        self.fc1 = nn.Linear(in_channels, md_channels, bias=False)
        self.bn1 = norm_layer(md_channels)
        self.fc2 = nn.Linear(md_channels, ou_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out