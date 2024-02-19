import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import *
import numpy as np
import matplotlib.pyplot as plt
import sys
from VGGNet import vgg11

# 设备检测,若未检测到cuda设备则在CPU上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
torch.manual_seed(0)

# 定义模型、优化器、损失函数
model = vgg11().to(device)  # 这里vgg11()里面的参数？
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# 设置训练集的数据变换，进行数据增强
transform_train = transforms.Compose([
    transforms.RandomRotation(30),  # 随机旋转 -30度到30度之间
    transforms.RandomResizedCrop((224, 224)),  # 随机比例裁剪并进行resize
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
    transforms.ToTensor(),  # 将数据转换为张量
    # 对三通道数据进行归一化（均值，标准差）， 数值是从ImageNet数据集上的百万张图片中随机抽样计算得到
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 对数据进行归一化
])

# 设置测试集的数据变换，进行数据增强
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # resize
    transforms.ToTensor(),  # 将数据转化为张量
    # 对三通道数据进行归一化（均值，标准差），数值是从ImageNet数据集上的百万张图片中随机抽样计算得到
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载训练数据，需要特别注意的是Flowers102数据集，test簇的数据量较多些，所以这里使用"test"作为训练集
train_dataset = datasets.Flowers102(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/flowers102', split="test",
                                    download=False, transform=transform_train)
# 实例化训练数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# 加载测试数据，使用“train”作为测试集
test_dataset = datasets.Flowers102(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/flowers102', split="train",
                                   download=False, transform=transform_test)
# 实例化测试数据加载器
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

# 设置epoch数并开始训练
num_epochs = 50  # 设置epoch数
loss_history = []  # 创建损失历史记录列表
acc_history = []  # 创建准确率历史记录列表

# tqdm用于显示进度条并评估任务时间开销
for epoch in tqdm(range(num_epochs), file=sys.stdout):
    # 记录损失和预测正确数
    total_loss = 0
    total_correct = 0

    # 批量训练
    model.train()
    for inputs, labels in train_loader:

        # 将数据转换到指定计算资源设备上
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 预测、损失函数、反向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 记录训练集loss
        total_loss += loss.item()

    # 测试模型，不计算梯度
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            # 将数据转换到指定计算资源设备上
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 预测
            outputs = model(inputs)
            # 记录测试集预测正确数
            total_correct += (outputs.argmax(1) == labels).sum().item()

    # 记录训练集损失和测试集准确率
    loss_history.append(np.log10(total_loss))  # 将损失加入损失历史记录列表，由于数值有时较大，这里取对数
    acc_history.append(total_correct / len(test_dataset))  # 将准确率加入准确率历史记录列表

    # 打印中间值
    # 每5个epoch打印一次中间值
    if epoch % 5 == 0:
        tqdm.write("Epoch: {0} Loss: {1} Acc: {2}".format(epoch, loss_history[-1], acc_history[-1]))

# 使用Matplotlib绘制损失和准确率的曲线图
plt.plot(loss_history, label='loss')
plt.plot(acc_history, label='accuracy')
plt.legend()
plt.show()

# 输出准确率
print("Accuracy:", acc_history[-1])

