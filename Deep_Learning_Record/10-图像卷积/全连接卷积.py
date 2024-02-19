import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import numpy as np

# 加载图像
from skimage import io
image = io.imread("D:/Python_file/Deep_Learning/pycharm_pytorch/10-图像卷积/images/HJX.png")

# 将图像转换为 PyTorch 张量
image = torch.tensor(image).float()

# 将图像的通道数转换为单通道
image = image.mean(dim=2)

# 添加批量维度
image = image.unsqueeze(0).unsqueeze(0)

# 创建卷积层，输出通道数为 3
conv = torch.nn.Conv2d(1, 3, kernel_size=3)

# 对输入图像进行卷积
output = conv(image)

# 将输出转换为 NumPy 数组
output_image = output.squeeze(0).detach().numpy()

# 将输出图像的 3 个通道可视化
plt.subplot(1, 3, 1)
plt.imshow(output_image[0], cmap='gray')
plt.title("Output channel 1")
plt.subplot(1, 3, 2)
plt.imshow(output_image[1], cmap='gray')
plt.title("Output channel 2")
plt.subplot(1, 3, 3)
plt.imshow(output_image[2], cmap='gray')
plt.title("Output channel 3")
plt.show()