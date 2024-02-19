import torch

# 创建训练数据和标签
data = torch.zeros(1, 1, 5, 5)
label = torch.zeros(1, 1, 3, 3)

# 将训练数据和标签转换为 PyTorch 张量
data = data.float()
label = label.float()

import torch.nn as nn

# 创建卷积层，输入通道数为 1，输出通道数为 1
conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(conv.parameters(), lr=0.001)

# 训练模型
for i in range(1000):
    # 计算预测值
    output = conv(data)

    # 计算损失
    loss = loss_fn(output, label)

    # 清空梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 优化模型参数
    optimizer.step()

# 访问卷积层的卷积核
conv_kernel = conv.weight.data
print(conv_kernel)

# 访问卷积层的偏置项
conv_bias = conv.bias.data
print(conv_bias)
