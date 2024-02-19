import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据预处理方法，将数据转换为Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载MNIST数据
mnist_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=True, download=False, transform=transform)
# 加载数据，并使用 DataLoader 进行分批处理，batch_size 设置为 64
train_loader = torch.utils.data.DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True, num_workers=4)

# 设置随机数种子，以便在多次运行代码时得到相同的结果
torch.manual_seed(42)

# 定义要显示的样本数量
num_samples = 12

# 创建一个matplotlib绘图窗口，并显示指定数量的MNIST样本
fig, axs = plt.subplots(1, num_samples, figsize=(10, 10))
for i in range(num_samples):
    # 从MNIST数据集中随机选择一个样本
    idx = torch.randint(len(mnist_dataset), size=(1,)).item()
    # 获取该样本的图像信息
    img, _ = mnist_dataset[idx]
    # 在绘图窗口中显示该样本的图像
    axs[i].imshow(img.squeeze(), cmap='gray')
    # 不显示坐标轴
    axs[i].axis('off')
plt.show()

# 定义参数
input_dim = 28  # MNIST数据集的图像长宽
latent_dim = 2  # 隐变量为维度
n_epochs = 10  # 定义训练轮数


# 定义VAE的网络结构
class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        # 编码器部分， 都使用全连接
        self.fc1 = nn.Linear(x_dim, h_dim1)  # 输入x_dim, 输出h_dim1
        self.fc2 = nn.Linear(h_dim1, h_dim2)  # 输入h_dim1, 输出h_dim2
        self.fc31 = nn.Linear(h_dim2, z_dim)  # 输入h_dim2, 输出z_dim, 输出mu, 均值
        self.fc32 = nn.Linear(h_dim2, z_dim)  # 输入h_dim2, 输出z_dim, 输出log_var, 方差的对数

        # 解码器部分， 都使用全连接
        self.fc4 = nn.Linear(z_dim, h_dim2)  # 输入z_dim, 输出h_dim2
        self.fc5 = nn.Linear(h_dim2, h_dim1)  # 输入h_dim2, 输出h_dim1
        self.fc6 = nn.Linear(h_dim1, x_dim)  # 输入h_dim1, 输出x_dim

    # 编码器处理部分
    def encoder(self, x):
        # 全连接 + ReLU
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        # 返回mu和log_var
        return self.fc31(h), self.fc32(h)

    # 解码器处理部分
    def decoder(self, z):
        # 全连接 + ReLU
        h = torch.relu(self.fc4(z))
        h = torch.relu(self.fc5(h))
        # 接sigmoid激活函数，输出重建后的x
        return torch.sigmoid(self.fc6(h))

    # 重参数化技巧
    def sampling(self, mu, log_var):
        # 计算标准差
        std = torch.exp(0.5 * log_var)
        # 从标准正态分布中随机采样eps
        eps = torch.randn_like(std)
        # 返回z
        return mu + eps * std

    # 定义前向传播函数
    def forward(self, x):
        # 编码器，输出mu和llog_var
        mu, log_var = self.encoder(x.view(-1, input_dim * input_dim))
        # 重参数化
        z = self.sampling(mu, log_var)
        # 返回解码器输出、mu、log_var和z
        return self.decoder(z), mu, log_var, z


def main():
    # 实例化VAE模型
    vae= VAE(x_dim=input_dim * input_dim, h_dim1= 512, h_dim2=512, z_dim=latent_dim).to(device)

    # 定义Adam优化器，用于优化VAE模型的参数，学习率为0.001
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    # 定义VAE的损失函数，其中包含重构误差和KL散度
    def loss_function(recon_x, x, mu, log_var):
        # 重构误差，使用二元交叉熵损失函数
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim * input_dim), reduction='sum')
        # KL散度，计算高斯分布之间的散度
        # 详见VAE论文中的Appendix B
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 将重构误差和KL散度相加作为总损失
        return BCE + KLD

    # 循环开始训练
    for epoch in range(n_epochs):
        # 进入训练模式
        vae.train()
        train_loss = 0

        # 遍历训练数据集
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # 前向传播，计算重构误差和KL散度
            recon_batch, mu, log_var, z = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)

            # 反向传播，记录损失值，更新模型参数
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        # 输出平均损失
        print('===> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        # 进入评估模式
        vae.eval()

        # 生成新样本
        with torch.no_grad():
            # 随机生成正态分布，并使用解码器将采样结果转换为新的样本
            z = torch.randn(num_samples, latent_dim).to(device)
            sample = vae.decoder(z).cpu()
            images = sample.view(num_samples, input_dim, input_dim).numpy()

            # 可视化生成的样本
            fig, axs = plt.subplots(1, num_samples, figsize=(10, 10))
            for i in range(num_samples):
                axs[i].imshow(images[i], cmap='gray')
                axs[i].axis('off')
            plt.show()


if __name__ == '__main__':
    main()



