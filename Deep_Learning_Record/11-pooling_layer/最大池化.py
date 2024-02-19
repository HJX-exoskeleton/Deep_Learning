import torch
import matplotlib.pyplot as plt

# 创建一个 4x4 的二维张量，每一行代表一个图像的一行像素
input_tensor = torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# 将输入张量转化为浮点数类型的张量
input_tensor = input_tensor.to(torch.float)

# 将输入张量转化为四维张量，因为 MaxPool2d 需要接受四维的输入
input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

# 将输入张量传入最大池化层，指定池化窗口的大小为 2x2，步长为 2
max_pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
output = max_pooling_layer(input_tensor)

# 使用 Matplotlib 绘制输入张量和输出张量
fig, axs = plt.subplots(1, 2)

# 绘制输入张量
axs[0].axis('off')
axs[0].imshow(input_tensor[0, 0].detach().numpy(),cmap="gray")

# 在图像上为每个像素添加像素值的标签，使用红色显示
for i in range(input_tensor.shape[2]):
    for j in range(input_tensor.shape[3]):
        axs[0].text(j, i, input_tensor[0, 0, i, j].item(), ha="center", va="center", color="r")

# 绘制输出张量
axs[1].axis('off')
axs[1].imshow(output[0, 0].detach().numpy(),cmap="gray")

# 在图像上为每个像素添加像素值的标签，使用红色显示
for i in range(output.shape[2]):
    for j in range(output.shape[3]):
        axs[1].text(j, i, output[0, 0, i, j].item(), ha="center", va="center", color="r")

plt.show()