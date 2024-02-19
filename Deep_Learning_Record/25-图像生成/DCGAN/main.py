from data import ReadData
from model import Discriminator, Generator, weights_init
from net import DCGAN
import torch

import os
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


# 参数定义
ngpu = 1
ngf = 64
ndf = 64
nc = 3
nz = 100
lr = 0.003
beta1 = 0.5
batch_size = 100
num_showimage = 100

data_path="D:/Python_file/HJX_file/DCGAN_hjx/data/faces"
model_save_path="./models/"
figure_save_path="./figures/"

device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

dataset=ReadData(data_path)
dataloader=dataset.getdataloader(batch_size=batch_size)

G = Generator(nz,ngf,nc).apply(weights_init)
print(G)
D = Discriminator(ndf,nc).apply(weights_init)
print(D)

dcgan=DCGAN( lr,beta1,nz,batch_size,num_showimage,device, model_save_path,figure_save_path,G, D, dataloader)

dcgan.train(num_epochs=200)
# dcgan.test()



