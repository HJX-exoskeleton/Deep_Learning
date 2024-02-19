import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# 定义一个基础卷积结构BasicConv2d, 改进型
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        # 卷积+BN层
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    # 定义前向传播函数
    def forward(self, x):
        # 依次经过卷积和BN层，最后激活函数ReLU
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# 定义Inception结构
class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()

        # 定义四个分支路径
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = F.relu(self.branch1(x), inplace=True)
        branch2 = F.relu(self.branch2(x), inplace=True)
        branch3 = F.relu(self.branch3(x), inplace=True)
        branch4 = F.relu(self.branch4(x), inplace=True)

        # 连结输出
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


# 定义GoogLeNet的网络结构
class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()

        self.part1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.part2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.part3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.part4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 144, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.part5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.part6 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = self.part4(x)
        x = self.part5(x)
        x = self.part6(x)
        return x

# 查看模型结构以及参数量，input_size表示示例输入数据的维度信息
summary(GoogLeNet(), input_size=(1,3,224,224))