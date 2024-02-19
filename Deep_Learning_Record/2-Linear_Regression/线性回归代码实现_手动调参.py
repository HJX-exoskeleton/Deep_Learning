# 手动更新参数

import numpy as np
import torch
import matplotlib.pyplot as plt
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 设置随机数种子，使得每次运行代码生成的数据相同
np.random.seed(42)

# 生成随机数据
x = np.random.rand(100,1)
y = 1 + 2 * x + 0.1 * np.random.randn(100,1)

# 将数据转换为pytorch tensor   tensor是张量的意思
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# 设置超参数
learning_rate = 0.1
num_epochs = 1000

# 初始化参数
w = torch.randn(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# 开始训练
for epoch in range(num_epochs):
    # 计算预测值
    y_pred = x_tensor * w + b

    # 计算损失
    loss = ((y_pred - y_tensor)**2).mean()

    # 反向传播
    loss.backward()

    # 更新参数
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # 清空梯度
        w.grad.zero_()
        b.grad.zero_()

# 输入训练后的参数
print('w:', w)
print('b:', b)

# 可视化
plt.plot(x,y,'o')
plt.plot(x_tensor.numpy(), y_pred.detach().numpy())
plt.show()