import matplotlib.pyplot as plt
import numpy as np
import torch

# 首先，我们定义一个随机训练数据
np.random.seed(0)
x = np.random.uniform(0, 2, 100)
y = x * 3 + 1 + np.random.normal(0, 0.5, 100)

# 将训练数据转换为 PyTorch Tensor
x = torch.from_numpy(x).float().view(-1, 1)
y = torch.from_numpy(y).float().view(-1, 1)

# 然后，我们定义一个线性模型和损失函数
model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()

# 接下来，我们使用 Adam 优化器来训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 初始化用于可视化训练过程的列表
losses = []

# 开始训练循环
for i in range(100):
    # 进行前向传递，计算损失
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    # 将损失存储到列表中，以便我们可视化
    losses.append(loss.item())

    # 进行反向传递，更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 可视化训练过程
plt.plot(losses)
plt.ylim((0, 15))
plt.show()