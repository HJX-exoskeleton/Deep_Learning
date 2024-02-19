# 模型参数一般都是张量形式的，虽然单个张量的保存和加载非常简单，但整个模型中包含着大大小小的若干张量，单独保存这些张量会很困难
# 为了解决这个问题，pytorch贴心的为我们准备了内置函数来保存加载整个模型参数
# 我们以5.2节的多层感知机为例，来看一下如何保存
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 定义 MLP 网络
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 定义超参数
input_size = 28 * 28  # 输入大小
hidden_size = 512  # 隐藏层大小
num_classes = 10  # 输出大小（类别数）

# 然后我们实例化一个MLP网络，并随机生成一个输入X，并计算出模型的输出Y
# 实例化 MLP 网络
model = MLP(input_size, hidden_size, num_classes)
X = torch.randn(size=(2, 28*28))

# 然后同样是调用save方法，我们把模型存储到model文件夹里，取名叫做mlp.params
torch.save(model.state_dict(), 'model/mlp_state_dict.pth')

# 接下来，我们来读取保存好的模型参数，重新加载我们的模型
# 我们先把模型params参数读取出来，然后实例化一个模型，然后直接调用load_state_dict方法，传入模型参数params
mlp_state_dict = torch.load('model/mlp_state_dict.pth')
model_load = MLP(input_size, hidden_size, num_classes)
model_load.load_state_dict(mlp_state_dict)

# 此时两个模型model和model_load具有相同的参数，我们给他输入相同的X，看一下输出结果
output1 = model(X)
print(output1)
output2 = model_load(X)
print(output2)


# # 方式2：checkpoint
# # 保存参数
# torch.save(
#     {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         }, 'model/ckpt')
#
# # 加载参数
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
#
# checkpoint = torch.load('model/ckpt')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
#
# model.eval()