import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from LeNet_Batch_Normalization import LeNet

# 设置随机种子
torch.manual_seed(0)

# 定义模型、优化器、损失函数
model = LeNet()
optimizer = optim.SGD(model.parameters(), lr=0.02)
criterion = nn.CrossEntropyLoss()

# 设置数据变换和数据加载器
transform = transforms.Compose([
    transforms.ToTensor(),  # 将数据转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 对数据进行归一化
])

train_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=True, download=False, transform=transform)  # 加载训练数据
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)  # 实例化训练数据加载器
test_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=False, download=False, transform=transform)  # 加载测试数据
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)  # 实例化测试数据加载器

# 设置epoch数并开始训练
num_epochs = 10  # 设置epoch数
loss_history = []  # 创建损失历史记录列表
acc_history = []  # 创建准确率历史记录列表

for epoch in tqdm(range(num_epochs)):
    # 批量训练
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 测试模型
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        total_test_correct = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            total_test_loss += criterion(outputs, labels).item()
            total_test_correct += (outputs.argmax(1) == labels).sum().item()

    # 输出测试集上的损失和准确率
    loss_history.append(np.log10(total_test_loss))  # 将损失加入损失历史记录列表，由于数值较大，这里取对数
    acc_history.append(total_test_correct / len(test_dataset))  # 将准确率加入准确率历史记录列表

# 使用图表库（例如Matplotlib）绘制损失和准确率的曲线图
plt.plot(loss_history, label='loss')
plt.plot(acc_history, label='accuracy')
plt.legend()
plt.show()

# 输出准确率
print("Accuracy:", acc_history[-1])