import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch


# 定义函数
def f(x):
    return x ** 2 + 4 * x + 1


# 定义初始值
x = torch.tensor(-10., requires_grad=True)

# 记录每一步的值
xs = []
ys = []

# 迭代更新参数
learning_rate = 0.1
for i in range(100):
    # 计算预测值和损失
    y = f(x)

    # 记录参数和损失
    xs.append(x.item())
    ys.append(y.item())

    # 反向传播求梯度
    y.backward()

    # 更新参数
    with torch.no_grad():
        x -= learning_rate * x.grad

        # 梯度清零
        x.grad.zero_()

# 显示真实的函数曲线
x_origin = torch.arange(-10, 10, 0.1)
y_origin = f(x_origin)
plt.plot(x_origin, y_origin, 'b-')

# 绘制搜索过程
plt.plot(xs, ys, 'r--')
plt.scatter(xs, ys, s=50, c='r')  # 圆点大小为 50，颜色为红色
plt.xlabel('x')
plt.ylabel('y')

plt.show()

# 打印结果
print(f'最终参数值：{x.item()}')