# 借助模型自动更新参数

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 设置随机数种子，使得每次运行代码生成的数据相同
np.random.seed(42)

# 生成随机数据
x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)

# 将数据转换为 pytorch tensor
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# 设置超参数
learning_rate = 0.1
num_epochs = 1000

# 定义输入数据的维度和输出数据的维度
input_dim = 1
output_dim = 1

# 定义模型
model = nn.Linear(input_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(num_epochs):
    # 将输入数据喂给模型
    y_pred = model(x_tensor)

    # 计算损失
    loss = criterion(y_pred, y_tensor)

    # 清空梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

# 输出训练后的参数
print('w:', model.weight.data)
print('b:', model.bias.data)

# 可视化
plt.plot(x,y,'o')
plt.plot(x_tensor.numpy(), y_pred.detach().numpy())
plt.show()