import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn


transformation = torchvision.transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=True, download=False, transform=transformation)
test_dataset = torchvision.datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=False, download=False, transform=transformation)

# 数据加载器
batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 打印数据形状
# for images, labels in train_dataloader:
#     print(images.shape, labels.shape)
#
#     plt.imshow(images[0][0], cmap='gray')
#     plt.show()
#
#     print(labels[0]) # 打印标签


# 构建网络
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        logits = self.linear(x)
        return logits

input_size = 28*28
output_size = 10
model = Model(input_size, output_size)

# 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 模型评估函数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.view(-1, input_size)
            logits = model(x)
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# 模型训练
for epoch in range(10):
    for images, labels in train_dataloader:
        # 将图像和标签转换成张量
        images = images.view(-1, 28 * 28)
        labels = labels.long()

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = evaluate(model, test_dataloader)
    print(f'Epoch {epoch + 1}: test accuracy = {accuracy:.2f}')
