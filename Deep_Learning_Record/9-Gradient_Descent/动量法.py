import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *

# 定义模型和损失函数
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        return self.output(x)
loss_fn = nn.MSELoss()

# 生成随机数据
np.random.seed(0)
n_samples = 200
x = np.linspace(-5, 5, n_samples)
y = 0.3 * (x ** 2) + np.random.randn(n_samples)

# 转换为Tensor
x = torch.unsqueeze(torch.from_numpy(x).float(), 1)
y = torch.unsqueeze(torch.from_numpy(y).float(), 1)

names = ["momentum", "sgd"] # 一个使用动量法，一个不使用
losses = [[], []]

# 超参数
learning_rate = 0.0005
n_epochs = 500

# 分别训练
for i in range(len(names)):
    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9 if i == 0 else 0) # 一个使用动量法，一个不使用
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        losses[i].append(loss.item())

# 绘制损失值的变化趋势
plt.figure()
plt.plot(losses[0], 'r-', label='momentum')
plt.plot(losses[1], 'b-', label='sgd')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss')
plt.ylim((0, 12))
plt.legend()
plt.show()