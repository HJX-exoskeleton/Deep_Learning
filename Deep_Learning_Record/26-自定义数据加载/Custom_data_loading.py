import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.datasets import ImageFolder
from PIL import Image  # pip install Pillow
from torch.utils.data import Dataset
from  torch.utils.data import Subset
import pandas as pd

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 设置随机数种子，以便在多次运行代码时得到相同的结果
torch.manual_seed(42)

# # MNIST
# # 定义数据预处理方法，将数据转换为Tensor
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# train_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=True, download=False, transform=transform)  # 加载训练数据
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)  # 实例化训练数据加载器
# test_dataset = datasets.MNIST(root='D:/Python_file/Deep_Learning/pycharm_pytorch/data/mnist', train=False, download=False, transform=transform)  # 加载测试数据
# test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)  # 实例化测试数据加载器


# 简单案例
def print_dir_tree(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)

        print(f'{indent}{os.path.basename(root)}/')

        # subindent = ' ' * 4 * (level + 1)
        # for f in files:
        #     print(f'{subindent}{f}')

# startpath = 'D:/Python_file/Deep_Learning/pycharm_pytorch/data/fruit_101'
startpath = 'D:/Python_file/Deep_Learning/pycharm_pytorch/data/flower_color_images'
print_dir_tree(startpath)


# 定义数据预处理方法
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小为128*128
    transforms.ToTensor(),  # 将数据转换为Tensor张量
])

# 创建图像数据集
# ImageFolder类会自动遍历指定目录下的所有子目录
# 并将每个子目录中的图像文件视为同一类别的数据
dataset = ImageFolder('D:/Python_file/Deep_Learning/pycharm_pytorch/data/fruit_101', transform=transform)
# print(len(dataset))
# print(dataset.classes)
# print(dataset.class_to_idx)

# 定义绘图函数，传入dataset即可
def plot(dataset, shuffle=True):
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=16, shuffle=shuffle)

    # 取出一组数据
    images, labels = next(iter(dataloader))

    # 将通道维度(C)移到最后一组维度，方便使用matplotlib绘图
    images = np.transpose(images, (0, 2, 3, 1))

    # 创建4x4的子图对象
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))

    # 遍历每个子图，绘制图像并添加子图标题
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')  # 隐藏坐标轴

        if hasattr(dataset, 'classes'):  # 如果数据集有预定义的类别名称，使用该名称作为子图标题
            ax.set_title(dataset.classes[labels[i]], fontsize=12)
        else:  # 否则使用类别索引作为子图标题
            ax.set_title(labels[i], fontsize=12)

    plt.show()

# plot(dataset)


# 自定义数据集1
class Flowers(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # 遍历数据集目录，获取所有图像文件的路径和标签
        for filename in sorted(os.listdir(data_dir)):
            image_path = os.path.join(data_dir, filename)
            label = int(filename.split('_')[0])
            self.image_paths.append(image_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 加载图像数据和标签
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        # 对图像数据进行转换
        if self.transform:
            image = self.transform(image)

        # 将标签转换为PyTorch张量
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# 定义数据转换方法
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小为128*128
    transforms.ToTensor(),  # 将数据转换为张量
])

# dataset = Flowers('D:/Python_file/Deep_Learning/pycharm_pytorch/data/flower_color_images/flowers/flowers', transform=transform)
# plot(dataset)


# 取子集操作
# dataset = Flowers('D:/Python_file/Deep_Learning/pycharm_pytorch/data/flower_color_images/flowers/flowers', transform=transform)
# subset = Subset(dataset, [i for i in range(16)])
# plot(subset, False)


# 处理csv文件
csv = pd.read_csv('D:/Python_file/Deep_Learning/pycharm_pytorch/data/flower_color_images/flower_images/flower_images/flower_labels.csv')
# print(csv)

# 判断某个文件属于哪一个类别
# print(csv.loc[csv.file == '0206.png', 'label'].iloc[0])

# 将label去重之后转为numpy数组
# print(csv.iloc[:, 1].drop_duplicates().to_numpy())


# 自定义数据集2
class FlowersImages(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # 读取CSV文件
        csv = pd.read_csv(os.path.join(data_dir, csv_file))

        # 遍历数据目录下的所有PNG文件，并将其路径和标签添加到列表中
        for filename in sorted(os.listdir(data_dir)):
            if filename.endswith('.png'):
                self.image_paths.append(os.path.join(data_dir, filename))
                lable = csv.loc[csv['file'] == filename, 'label'].iloc[0]
                self.labels.append(lable)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 加载图像数据和标签
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]

        # 对图像数据进行转换
        if self.transform:
            image = self.transform(image)

        # 将标签转换为PyTorch张量
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# 定义数据转换方法
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小为128*128
    transforms.ToTensor(),  # 将数据转换为张量
])

dataset = FlowersImages('D:/Python_file/Deep_Learning/pycharm_pytorch/data/flower_color_images/flower_images/flower_images/', 'flower_labels.csv', transform=transform)
plot(dataset)
