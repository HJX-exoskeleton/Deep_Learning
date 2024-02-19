import torch
import matplotlib.pyplot as plt
from torch import nn
from matplotlib import ticker
import warnings
warnings.filterwarnings("ignore")


# 绘制注意力热图
def show_attention(axis, attention):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attention, cmap='bone')
    if axis is not None:
        ax.set_xticklabels(axis[0])
        ax.set_yticklabels(axis[1])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


# 生成一个样例
sentence = "I love deep learning more than maching learning"
tokens = sentence.split(' ')

attention_weights = torch.eye(8).reshape((8, 8)) + torch.randn((8, 8)) * 0.1  # 生成注意力权重矩阵
# print(attention_weights)

# 展示自注意力热图
show_attention([tokens, tokens], attention_weights)

