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

# 定义数据预处理方法，将数据转换为Tensor，并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转为张量
    transforms.Normalize([0.5], [0.5])  # 进行归一化，均值为0.5， 标准差为0.5
])

# 加载MNIST数据
mnist_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=True, download=False, transform=transform)
# 加载数据，并使用 DataLoader 进行分批处理，batch_size 设置为 64
train_loader = torch.utils.data.DataLoader(dataset=mnist_dataset, batch_size=64, shuffle=True, num_workers=4)

# 设置随机数种子，以便在多次运行代码时得到相同的结果
torch.manual_seed(42)

# 定义要显示的样本数量
num_samples = 12

# # 创建一个matplotlib绘图窗口，并显示指定数量的MNIST样本
# fig, axs = plt.subplots(1, num_samples, figsize=(10, 10))
# for i in range(num_samples):
#     # 从MNIST数据集中随机选择一个样本
#     idx = torch.randint(len(mnist_dataset), size=(1,)).item()
#     # 获取该样本的图像信息
#     img, _ = mnist_dataset[idx]
#     # 在绘图窗口中显示该样本的图像
#     axs[i].imshow(img.squeeze(), cmap='gray')
#     # 不显示坐标轴
#     axs[i].axis('off')
# plt.show()

# 定义参数
input_dim = 28  # MNIST数据集的图像长宽
n_epochs = 10  # 定义训练轮数
noise_dim = 100  # 随机噪声维度


# 定义GAN的网络结构
# 定义生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义每个block的结构，全连接 + BN + ReLU
        def block(in_feat, out_feat, normalize=True):
              layers = [nn.Linear(in_feat, out_feat)]  # 全连接
              if normalize:
                  layers.append(nn.BatchNorm1d(out_feat))  # BN层
              layers.append(nn.ReLU(inplace=True))  # ReLU激活函数
              return layers

        # 定义生成器的网络结构，4个block
        self.model = nn.Sequential(
            *block(noise_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, input_dim * input_dim),  # 全连接层输出
            nn.Tanh()  # Tanh激活函数，将输出映射到[-1,1]之间
        )

    # 定义前向传播函数
    def forward(self, z):
        # 经过生成器模型
        img = self.model(z)
        # 调整输出维度
        img = img.view(-1, 1, input_dim, input_dim)
        return img

# 定义判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义判别器的网络模型，3个全连接层
        self.model = nn.Sequential(
            # 全连接层，输入维度为28*28，输出维度为512
            nn.Linear(input_dim * input_dim, 512),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 全连接层，输入维度为512， 输出维度为256
            nn.Linear(512, 256),
            # ReLU激活函数
            nn.ReLU(inplace=True),
            # 全连接层，输入维度为256，输出维度为1
            nn.Linear(256, 1),
            # sigmoid激活函数，将输出映射到(0,1)之间
            nn.Sigmoid(),
        )

    def forward(self, x):
        # flatten将x的维度降为一维
        x = torch.flatten(x, 1)
        # 输入x并计算输出
        x = self.model(x)
        return x

# 定义二元交叉熵损失函数
adversarial_loss = torch.nn.BCELoss().to(device)

# 实例化生成器和判别器
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# 定义生成器和判别器的优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

# 如果GPU可用，则使用cuda.FloatTensor, 否则使用FloatTensor
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def main():
    # 循环开始训练
    for epoch in range(n_epochs):
        # 分别记录每轮生成器和判别器的loss
        generator_loss, discriminator_loss = 0, 0
        # 遍历训练数据集
        for batch_idx, (imgs, _) in enumerate(train_loader):
            # 定义真实标签的Tensor，数值全为1.0，不需要计算梯度
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            # 定义假实标签的Tensor，数值全为0.0，不需要计算梯度
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # 将真实图片转化为Tensor
            real_imgs = Variable(imgs.type(Tensor))

            # 开始训练生成器
            optimizer_G.zero_grad()

            # 随机生成一个满足正态分布均值为0，方差为1的z
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], noise_dim))))

            # 通过生成器生成图片
            gen_imgs = generator(z)

            # 计算生成器的损失并记录
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            generator_loss += g_loss.item()

            # 反向传播并更新参数
            g_loss.backward()
            optimizer_G.step()

            # 开始训练判别器
            optimizer_D.zero_grad()

            # 计算判别器在真实图片上的损失
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            # 计算判别器在生成图片上的损失
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

            # 计算判别器的总损失并记录
            d_loss = (real_loss + fake_loss) / 2
            discriminator_loss += d_loss.item()

            # 反向传播并更新参数
            d_loss.backward()
            optimizer_D.step()

        # 输出每一轮的生成器和判别器的平均损失
        print('===> Epoch: {} Generator loss: {:.4f} Discriminator loss: {:.4f}'.format(
            epoch, generator_loss / len(train_loader.dataset), discriminator_loss / len(train_loader.dataset)))

        # 将最后生成的图片转换为numpy数组
        images = gen_imgs.view(-1, 28, 28).detach().cpu().numpy()
        # 可视化生成的样本
        fig, axs = plt.subplots(1, num_samples, figsize=(10, 10))
        for i in range(num_samples):
            axs[i].imshow(images[i], cmap='gray')
            axs[i].axis('off')
        plt.show()


if __name__ == '__main__':
    main()

