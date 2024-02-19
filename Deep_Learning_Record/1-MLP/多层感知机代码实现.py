import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim


train_data = datasets.MNIST(root="D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist", train=True, transform=transforms.ToTensor(), download=False)
test_data = datasets.MNIST(root="D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist", train=False, transform=transforms.ToTensor(), download=False)

batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)


# 定义MLP网络  继承nn.Module
class MLP(nn.Module):

    # 初始化方法
    # input_size输入数据的维度
    # hidden_size隐藏层的大小
    # num_classes输出分类的数量
    def __init__(self, input_size, hidden_size, num_classes):
        # 调用父类的初始化方法
        super(MLP, self).__init__()
        # 定义第1个全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 定义激活函数
        self.relu = nn.ReLU()
        # 定义第2个全连接层
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 定义第3个全连接层
        self.fc3 = nn.Linear(hidden_size, num_classes)

    # 定义forward函数
    # x 输入的数据
    def forward(self, x):
        # 第一次运算
        out = self.fc1(x)
        # 将上一步结果送给激活函数
        out = self.relu(out)
        # 将上一步结果送给fc2
        out = self.fc2(out)
        # 同样将结果送给激活函数
        out = self.relu(out)
        # 将上一步结果传递给fc3
        out = self.fc3(out)
        # 返回结果
        return out

# 定义参数
input_size = 28 * 28   # 输入大小
hidden_size = 512   # 隐藏层大小
num_classes = 10   # 输出大小（类别数）

# 初始化MLP
model = MLP(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001   # 学习率
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练网络
num_epochs = 10   # 训练轮数
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将images转成向量
        images = images.reshape(-1, 28 * 28)
        # 将数据送到网络中
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)

        # 首先将梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch[{epoch+1}/{num_epochs}], Step[{i+1}/{len(train_loader)}], Loss:{loss.item():.4f}')

# 测试网络
with torch.no_grad():
    correct = 0
    total = 0
    # 从test_loader中循环读取测试数据
    for images, labels in test_loader:
        # 将images转化成向量
        images = images.reshape(-1, 28 * 28)
        # 将数据送给网络
        outputs = model(images)
        # 取出最大值对应的索引 即预测值
        _, predicted = torch.max(outputs.data, 1)
        # 累加label数
        total += labels.size(0)
        # 预测值与label值比对 获取预测正确的数量
        correct += (predicted == labels).sum().item()

    # 打印最终的正确率
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

torch.save(model, "mnist_mlp_model.pkl")