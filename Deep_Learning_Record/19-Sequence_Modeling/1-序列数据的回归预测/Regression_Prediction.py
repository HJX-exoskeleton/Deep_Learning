import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from torch import nn
from tqdm import *

gs10 = pdr.get_data_fred('GS10')
plt.plot(gs10)
plt.show()

# 构造数据集
num = len(gs10)  # 总数据量
time = torch.arange(1, num + 1, dtype=torch.float32)
x = torch.tensor(gs10['GS10'].to_list())  # 股价列表
n = 6  # 预测序列长度
features = torch.zeros((num - n, n))
for i in range(n):
    features[:, i] = x[i: num - n + i]

labels = x[n:].reshape((-1, 1))  # 真实结果列表
train_loader = DataLoader(TensorDataset(features[:num - n], labels[:num - n]), 4, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# 建立模型
lr = 0.001
model = Model()
criterion = nn.MSELoss(reduction='none')
trainer = torch.optim.Adam(model.parameters(), lr)

num_epochs = 50
loss_history = []

for epoch in tqdm(range(num_epochs)):
    # 批量训练
    for X, y in train_loader:
        trainer.zero_grad()
        l = criterion(model(X), y)
        l.sum().backward()
        trainer.step()
    # 输出损失
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for X, y in train_loader:
            outputs = model(X)
            l = criterion(outputs, y)
            total_loss += l.sum() / l.numel()
        avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 2}: Validation loss = {avg_loss:.4f}')
    loss_history.append(avg_loss)


# 使用图表库（例如Matplotlib）绘制损失和准确率的曲线图
plt.plot(loss_history, label='loss')
plt.legend()
plt.show()

# 模型预测
preds = model(features)

plt.plot(time[:num-6], gs10['GS10'].to_list()[6:num], label='gs10')
plt.plot(time[:num-6], preds.detach().numpy(), label='preds')
plt.legend()
plt.show()


# 未来预测(在原始代码中后续增添的一部分)
# 假设n=，使用最后6个观察值作为特征

# last_features = features[-1].reshape(1, n)
#
# future_preds = []
# current_features = last_features
#
# for _ in range(10):  # 预测未来10步
#     # 使用当前特征进行预测
#     with torch.no_grad():  # 不需要计算梯度
#         current_pred = model(current_features)
#     future_preds.append(current_pred.item())
#
#     # 更新特征以包含最新的预测值
#     current_features = torch.cat((current_features[:, 1:], current_pred.reshape(1, 1)), dim=1)
#
# # 将预测值转换为Tensor以便绘图
# future_preds_tensor = torch.tensor(future_preds)
#
#
# # 绘制现有的GS10数据和模型预测
# plt.plot(time[:num-6], gs10['GS10'].to_list()[6:num], label='GS10 Actual')
# plt.plot(time[:num-6], preds.detach().numpy(), label='Model Predictions')
#
# # 为未来预测添加时间点
# future_time = torch.arange(num-6, num+4)  # 调整未来时间点的计算方法以匹配实际情况(对应于range(10), 4-(-6)=10)
# plt.plot(future_time, future_preds_tensor, label='Future Predictions', linestyle='--')
# plt.legend()
# plt.show()

