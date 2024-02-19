import torch
import torch.nn.functional as F

# 定义输入图像和卷积核
image = torch.zeros(7, 7, dtype=torch.float32)

# 将中心位置设置为 1
image[3][3] = 1.0

kernel = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]],dtype=torch.float32)

# 计算卷积
result = F.conv2d(image.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding = 1)
print(result)