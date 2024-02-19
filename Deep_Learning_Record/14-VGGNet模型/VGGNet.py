import torch
import torch.nn as nn
from torchinfo import summary

# 定义VGGNet的网络结构
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 定义相关配置项，其中M表示池化层
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 根据配置项拼接卷积层
def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# 封装函数
def vgg11(num_classes=102):
    return VGG(make_layers(cfgs['vgg11']), num_classes=num_classes)

def vgg13(num_classes=102):
    return VGG(make_layers(cfgs['vgg13']), num_classes=num_classes)

def vgg16(num_classes=102):
    return VGG(make_layers(cfgs['vgg16']), num_classes=num_classes)

def vgg19(num_classes=102):
    return VGG(make_layers(cfgs['vgg19']), num_classes=num_classes)

# 查看模型结构以及参数量，input_size表示示例输入数据的维度信息
# summary(vgg11(), input_size=(1,3,224,224))
