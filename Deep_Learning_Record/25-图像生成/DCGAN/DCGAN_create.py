import torch
import torch.nn as nn
from torchvision import utils
import matplotlib.pyplot as plt

import os
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


# 您的生成器定义保持不变
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        # 生成器网络定义保持不变...
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# 初始化生成器
nz = 100  # 潜在向量的维度
ngf = 64  # 生成器特征图深度
nc = 3    # 输出图像通道数
generator = Generator(nz, ngf, nc)

# 加载预训练的模型权重
generator.load_state_dict(torch.load('D:/Python_file/HJX_file/DCGAN_hjx/models/gen_8.pth'))

# 将生成器设置为评估模式
generator.eval()

# 生成潜在向量
fixed_noise = torch.randn(1, nz, 1, 1)  # 生成一个潜在向量

# 生成图像
with torch.no_grad():  # 不计算梯度
    fake_image = generator(fixed_noise).detach().cpu()

# 展示图像
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Image")
plt.imshow(utils.make_grid(fake_image, padding=2, normalize=True).permute(1, 2, 0))
plt.show()
