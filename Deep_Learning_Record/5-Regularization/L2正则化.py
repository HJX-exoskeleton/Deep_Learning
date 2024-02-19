import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt

# 生成数据
X_train = torch.randn(1000, 30)
y_train = torch.sin(X_train) + torch.randn(1000, 30) * 0.1
X_test = torch.randn(100, 30)
y_test = torch.sin(X_test) + torch.randn(100, 30) * 0.1

# 假设我们有一个包含两个隐藏层的神经网络
model = torch.nn.Sequential(
    torch.nn.Linear(30, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 30)
)

# 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 不使用 L2 正则化的情况
train_losses = []
test_losses = []

for epoch in range(50):
    # 计算训练损失
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    train_losses.append(loss.item())

    # 使用优化器更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算测试损失
    with torch.no_grad():
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        test_losses.append(loss.item())

# 绘制训练损失和测试损失的曲线
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.show()

# 使用 L2 正则化的情况
model = torch.nn.Sequential(
    torch.nn.Linear(30, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 30)
)

# 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)  # 加入 L2 正则化

train_losses = []
test_losses = []

for epoch in range(50):
    # 计算训练损失
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    train_losses.append(loss.item())

    # 使用优化器更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算测试损失
    with torch.no_grad():
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
        test_losses.append(loss.item())

# 绘制训练损失和测试损失的曲线
plt.plot(train_losses, label='L2-train')
plt.plot(test_losses, label='L2-test')
plt.legend()
plt.show()