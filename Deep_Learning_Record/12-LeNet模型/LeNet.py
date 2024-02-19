import torch
import torch.nn as nn
from torchinfo import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层1：输入1个通道，输出6个通道，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 卷积层2：输入6个通道，输出16个通道，卷积核大小为5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层1：输入16x5x5=400个节点，输出120个节点
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层2：输入120个节点，输出84个节点
        self.fc2 = nn.Linear(120, 84)
        # 输出层：输入84个节点，输出10个节点
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 使用Sigmoid激活函数，并进行最大池化
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        # 使用Sigmoid激活函数，并进行最大池化
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        # 将多维张量展平为一维张量
        x = x.view(-1, 16 * 4 * 4)
        # 全连接层
        x = torch.relu(self.fc1(x))
        # 全连接层
        x = torch.relu(self.fc2(x))
        # 全连接层
        x = self.fc3(x)
        return x

# 查看模型结构以及参数量，input_size表示示例输入数据的维度信息
summary(LeNet(), input_size=(1,1,28,28))
