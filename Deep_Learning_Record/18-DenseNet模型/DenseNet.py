import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            layers.append(layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.trans(x)

class DenseNet(nn.Module):
    def __init__(self, block_config, growth_rate=32, num_init_features=64, bn_size=4, num_classes=1000):

        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each denseblock
        num_features = num_init_features
        layers = []
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate)
            layers.append(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                layers.append(trans)
                num_features = num_features // 2
        layers.append(nn.BatchNorm2d(num_features))
        self.denseblock = nn.Sequential(*layers)

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        features = self.denseblock(features)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

# 封装函数
def densenet121(num_classes=102):
    return DenseNet(block_config=(6, 12, 24, 16), growth_rate=32, num_init_features=64, num_classes=num_classes)

def densenet161(num_classes=102):
    return DenseNet(block_config=(6, 12, 36, 24), growth_rate=48, num_init_features=96, num_classes=num_classes)

def densenet169(num_classes=102):
    return DenseNet(block_config=(6, 12, 32, 32), growth_rate=32, num_init_features=64, num_classes=num_classes)

def densenet201(num_classes=102):
    return DenseNet(block_config=(6, 12, 48, 32), growth_rate=32, num_init_features=64, num_classes=num_classes)

# 查看模型结构以及参数量，input_size表示示例输入数据的维度信息
# summary(densenet121(), input_size=(1,3,224,224))