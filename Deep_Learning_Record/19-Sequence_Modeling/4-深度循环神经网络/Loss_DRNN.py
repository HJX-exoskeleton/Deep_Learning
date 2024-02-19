import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义超参数
input_size = 28
hidden_size = 64
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=True, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=False, transform=transforms.ToTensor())

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 定义 DRNN 模型
class DRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(DRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # 通过 LSTM 计算输出和最终隐藏状态
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出作为最终输出
        out = self.fc(out[:, -1, :])
        return out


model = DRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 记录损失
loss_list = []
acc_list = []

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 转换成序列
        images = images.reshape(-1, 28, 28).to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        loss_list.append(loss.item())

        if (i + 1) % 100 == 0:

            # 测试模型
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.reshape(-1, 28, 28).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc_list.append(correct / total)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, num_epochs, i + 1,
                                                                                  total_step, loss.item(),
                                                                                  correct / total))

# 绘制损失曲线
plt.plot(loss_list, label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.show()

print('Test Accuracy of the model on the 10000 test images: {:.2f} %'.format(100 * acc_list[-1]))