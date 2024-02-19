import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 定义一个简单的网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# 随机生成训练数据
x = torch.randn(100, 1)
y = x.pow(2) + 0.1 * torch.randn(100, 1)

# 实例化网络
net = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adadelta(net.parameters())

# 记录训练损失
losses = []

# 开始训练
for epoch in range(100):
    # 前向传播 + 反向传播 + 优化
    output = net(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录损失
    losses.append(loss.item())

# 绘制训练损失图
plt.plot(losses)
plt.show()