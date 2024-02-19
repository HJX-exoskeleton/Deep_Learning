import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 创建 8x8 张量
image = torch.zeros(8, 8)
image[::2, ::2] = 1
# image[1::2, 1::2] = 0

# 将 image 张量转换为 numpy 数组
image_np = image.numpy()

# 使用 matplotlib 的 imshow 函数将图像显示出来
plt.imshow(image_np, cmap='gray')
plt.show()

# 创建卷积核
kernel = torch.tensor([[-1.0, 1.0]])

# 使用卷积核对 image 进行卷积
output = F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0))

print(output)
