import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 随机数种子
torch.manual_seed(2333)

# 定义超参数
num_samples = 20 # 样本数
hidden_size = 200 # 隐藏层大小
num_epochs = 500  # 训练轮数

# 生成训练集和测试集
x_train = torch.unsqueeze(torch.linspace(-1, 1, num_samples), 1)
y_train = x_train + 0.3 * torch.randn(num_samples, 1)
x_test = torch.unsqueeze(torch.linspace(-1, 1, num_samples), 1)
y_test = x_test + 0.3 *  torch.randn(num_samples, 1)

# 定义一个可能会过拟合的网络
net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 1),
)

# 定义一个包含 Dropout 的网络
net_dropout = torch.nn.Sequential(
    torch.nn.Linear(1, hidden_size),
    torch.nn.Dropout(0.5),  # p=0.5
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.Dropout(0.5),  # p=0.5
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 1),
)

# 定义优化器和损失函数
optimizer_overfitting = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_dropout = torch.optim.Adam(net_dropout.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 训练网络
for i in range(num_epochs):
    pred_overfitting = net_overfitting(x_train)
    pred_dropout = net_dropout(x_train)

    loss_overfitting = criterion(pred_overfitting, y_train)
    loss_dropout = criterion(pred_dropout, y_train)

    optimizer_overfitting.zero_grad()
    optimizer_dropout.zero_grad()

    loss_overfitting.backward()
    loss_dropout.backward()

    optimizer_overfitting.step()
    optimizer_dropout.step()

# 在测试过程中不使用 Dropout
net_overfitting.eval()
net_dropout.eval()

# 预测
test_pred_overfitting = net_overfitting(x_test)
test_pred_dropout = net_dropout(x_test)

# 绘制拟合效果
plt.scatter(x_train.data.numpy(), y_train.data.numpy(), c='r', alpha=0.3, label='train')
plt.scatter(x_test.data.numpy(), y_test.data.numpy(), c='b', alpha=0.3, label='test')
plt.plot(x_test.data.numpy(), test_pred_overfitting.data.numpy(), 'r-', lw=2, label='overfitting')
plt.plot(x_test.data.numpy(), test_pred_dropout.data.numpy(), 'b--', lw=2, label='dropout')
plt.legend(loc='upper left')
plt.ylim((-2, 2))
plt.show()

