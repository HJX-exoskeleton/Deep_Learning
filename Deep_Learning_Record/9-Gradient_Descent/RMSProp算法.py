import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt

# 假设我们有一个简单的线性回归模型
# y = w * x + b
# 其中 w 和 b 是需要学习的参数

# 定义超参数
learning_rate = 0.01
num_epochs = 100

# 随机生成训练数据
X = torch.randn(100, 1)
y = 2 * X + 3 + torch.randn(100, 1)

# 初始化参数
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 创建 RMSProp optimizer
optimizer = torch.optim.RMSprop([w, b], lr=learning_rate)

# 记录每次迭代的 loss
losses = []

# 训练模型
for epoch in range(num_epochs):
  # 计算预测值
  y_pred = w * X + b

  # 计算 loss
  loss = torch.mean((y_pred - y) ** 2)

  # 记录 loss
  losses.append(loss.item())

  # 清空上一步的梯度
  optimizer.zero_grad()

  # 计算梯度
  loss.backward()

  # 更新参数
  optimizer.step()

# 可视化训练过程
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()