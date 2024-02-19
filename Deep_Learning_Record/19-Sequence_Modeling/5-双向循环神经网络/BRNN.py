import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 定义 BRNN 模型
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向 LSTM，因此有 2 个隐藏层

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 双向 LSTM，因此有 2 倍的层数
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# 设置超参数
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=False, transform=transforms.ToTensor())

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 可视化训练过程
train_losses = []
test_losses = []

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    model.train()  # 确保在训练开始时模型处于训练模式
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28, 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    train_losses.append(train_loss / total_step)

    # 测试模型
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(-1, 28, 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))

        print('Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(test_loss / len(test_loader), 100 * correct / total))

# 可视化损失
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'model.ckpt')
