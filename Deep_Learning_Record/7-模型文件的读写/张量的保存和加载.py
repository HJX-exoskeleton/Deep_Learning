import torch

# 在深度学习中，模型的参数一般是张量形式的。对于单个的张量，pytorch为我们提供了方便直接的函数来进行读写
# 比如我们定义如下的一个张量a
a = torch.rand(10)
# print(a)

# 可以简单的用一个save函数去存储这个张量a，这里需要我们给他起一个名字，我们就叫它tensor-a,把它放在model文件夹里
torch.save(a, 'model/tensor-a')

# 读取同样简单，只需要用一个load函数就可以完成张量的加载，传入的参数是文件的路径
# tensor_a = torch.load('model/tensor-a')
# print(tensor_a)

# 如果我们要存储的不止一个张量，也没有问题，save和load函数同样支持保存张量列表。先把张量数据存储起来
a = torch.rand(10)
b = torch.rand(10)
c = torch.rand(10)
torch.save([a, b, c], 'model/tensor-abc')

# 然后再把它读取出来
# tensor_abc = torch.load('model/tensor-abc')
# print(tensor_abc)

# 对于多个张量，pytorch同样支持以字典的形式来进行存储。比如我们建立一个字典tensor_dict,然后把它存起来
a = torch.rand(10)
b = torch.rand(10)
c = torch.rand(10)
tensor_dict = {'a': a, 'b': b, 'c': c}
torch.save(tensor_dict, 'model/tensor_dict')

# 然后是读取
# tensor_dict = torch.load('model/tensor_dict')
# print(tensor_dict)

