import torch
import torch.nn as nn

# 定义数据
input_tensor = torch.randn(1, 1, 28, 28)
target_tensor = torch.randn(1, 1, 28, 28)

# 定义卷积层
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

# 计算卷积
output_tensor = conv(input_tensor)
print(output_tensor.shape)  # 输出: torch.Size([1, 1, 28, 28])

# 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(conv.parameters(), lr=0.001)

# 定义训练循环
for epoch in range(100):
    # 计算损失
    output_tensor = conv(input_tensor)
    loss = loss_fn(output_tensor, target_tensor)

    # 清空梯度
    optimizer.zero_grad()

    # 计算梯度并更新权重
    loss.backward()
    optimizer.step()

print(conv.state_dict())