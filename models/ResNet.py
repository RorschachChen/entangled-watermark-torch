# https://github.com/ZJZAC/Passport-aware-Normalization/blob/main/Image_cls/Ours/models/resnet_normal.py
import torch.nn as nn
import torch.nn.functional as F
from models.conv2d import ConvBlock


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(BasicBlock, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_2 = ConvBlock(planes, planes, 3, 1, 1, bn=norm_type, relu=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes,
                                      1, stride, 0, bn=norm_type, relu=False)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbn_2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(Bottleneck, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 1, 1, 0, bn=norm_type, relu=True)
        self.convbnrelu_2 = ConvBlock(planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_3 = ConvBlock(planes, self.expansion * planes, 1, 1, 0, bn=norm_type, relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes, 1, stride, 0, bn=norm_type, relu=False)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbnrelu_2(out)
        out = self.convbn_3(out) + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_type='bn'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.norm_type = norm_type

        self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1, bn=norm_type, relu=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.convbnrelu_1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet18(**model_kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)


def ResNet34(**model_kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **model_kwargs)


def ResNet50(**model_kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **model_kwargs)


def ResNet101(**model_kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **model_kwargs)


def ResNet152(**model_kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **model_kwargs)