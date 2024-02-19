import torch
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 定义函数
def f(x, y):
    return x ** 2 + 2* y ** 2

# 定义初始值
x = torch.tensor(-10., requires_grad=True)
y = torch.tensor(-10., requires_grad=True)

# 记录每一步的值
xs = []
ys = []
zs = []

# 迭代更新参数
learning_rate = 0.1
for i in range(100):
    # 计算预测值和损失
    z = f(x, y)

    # 记录参数和损失
    xs.append(x.item())
    ys.append(y.item())
    zs.append(z.item())

    # 反向传播
    z.backward()

    # 更新参数
    x.data -= learning_rate * x.grad
    y.data -= learning_rate * y.grad

    # 清空梯度
    x.grad.zero_()
    y.grad.zero_()

# 绘制图像
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(xs, ys, zs, 'r-')
ax.scatter(xs, ys, zs, s=50, c='r')  # 圆点大小为 50，颜色为红色

plt.show()

# 打印结果
print(f'最终参数值：x={x.item()}, y={y.item()}')


# 绘制原始的二维函数图像
X, Y = torch.meshgrid(torch.arange(-10, 10, 0.1), torch.arange(-10, 10, 0.1))
Z = f(X, Y)
plt.contour(X, Y, Z, levels=30)

# 绘制搜索过程曲线
plt.plot(xs, ys, 'r-')
plt.scatter(xs, ys, s=50, c='r')  # 圆点大小为 50，颜色为红色
plt.show()