import torch
import matplotlib.pyplot as plt
from torch import nn
from matplotlib import ticker
import warnings
warnings.filterwarnings("ignore")
from Attention_Heat_Map import show_attention


# 注意力池化
# 数据集合生成
# 定义一个映射函数
def func(x):
    return x + torch.sin(x)  # 映射函数: y = x + sin(x)

n = 100  # 样本个数
x, _ = torch.sort(torch.rand(n) * 10)  # 生成0-10的随机样本并排序
y = func(x) + torch.normal(0.0, 1, (n,))  # 生成训练样本对应的y值，增加均值为0，标准差为1的扰动
# print(x)
# print(y)

# 绘制曲线上的点
x_curve = torch.arange(0, 10, 0.1)
y_curve = func(x_curve)
plt.plot(x_curve, y_curve)
plt.plot(x, y, 'o')
plt.show()


# 非参数注意力池化
# 平均池化
# y_hat = torch.repeat_interleave(y.mean(), n)  # 将y_train中的元素进行复制，输入张量为y.mean，重复次数为n
# plt.plot(x_curve, y_curve)
# plt.plot(x, y, 'o')
# plt.plot(x_curve, y_hat)
# plt.show()


# nadaraya-watson 核回归
x_nw = x_curve.repeat_interleave(n).reshape((-1, n))
# print(x_nw.shape)
# print(x_nw)
  
# 带入公式得到注意力权重矩阵
attention_weights = nn.functional.softmax(-(x_nw - x)**2 / 2, dim=1)
# print(attention_weights.shape)
# print(attention_weights)


# y_hat为注意力权重和y值的乘积， 是加权平均值
# y_hat = torch.matmul(attention_weights, y)
# plt.plot(x_curve, y_curve)
# plt.plot(x, y, 'o')
# plt.plot(x_curve, y_hat)
# plt.show()

# 展示注意力热图(自注意力)
show_attention(None, attention_weights)

