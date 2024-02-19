from tqdm import tqdm
import torch
import torchvision as tv
from torch.utils.data import DataLoader
import torch.nn as nn

# config类中定义超参数，
class Config(object):
    """
    定义一个配置类
    """
    # 0.参数调整
    data_path = 'D:/Python_file/HJX_file/DCGAN_hjx/data/faces'
    virs = "result"
    num_workers = 0  # 多线程
    img_size = 96  # 剪切图片的像素大小
    batch_size = 256  # 批处理数量
    max_epoch = 600   # 最大轮次
    lr1 = 2e-4  # 生成器学习率
    lr2 = 2e-4  # 判别器学习率
    beta1 = 0.5  # 正则化系数，Adam优化器参数
    gpu = True  # 是否使用GPU运算（建议使用）
    nz = 100  # 噪声维度
    ngf = 64  # 生成器的卷积核个数
    ndf = 64  # 判别器的卷积核个数

    # 1.模型保存路径
    save_path = 'imgs2_600/'  # opt.netg_path生成图片的保存路径
    # 判别模型的更新频率要高于生成模型
    d_every = 1  # 每一个batch 训练一次判别器
    g_every = 5  # 每1个batch训练一次生成模型
    save_every = 5  # 每save_every次保存一次模型
    netd_path = None
    netg_path = None

    # 测试数据
    gen_img = "result_600.png"
    # 选择保存的照片
    # 一次生成保存64张图片
    gen_num = 64
    gen_search_num = 512
    gen_mean = 0    # 生成模型的噪声均值
    gen_std = 1     # 噪声方差

# 实例化Config类，设定超参数，并设置为全局参数
opt = Config()

# 定义Generation生成模型,通过输入噪声向量来生成图片
class NetG(nn.Module):
    # 构建初始化函数，传入opt类
    def __init__(self, opt):
        super(NetG, self).__init__()
        # self.ngf生成器特征图数目
        self.ngf = opt.ngf
        self.Gene = nn.Sequential(
            # 假定输入为opt.nz*1*1维的数据，opt.nz维的向量
            # output = (input - 1)*stride + output_padding - 2*padding + kernel_size
            # 把一个像素点扩充卷积，让机器自己学会去理解噪声的每个元素是什么意思。
            nn.ConvTranspose2d(in_channels=opt.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0, bias =False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(inplace=True),

            # 输入一个4*4*ngf*8
            nn.ConvTranspose2d(in_channels=self.ngf * 8, out_channels=self.ngf * 4, kernel_size=4, stride=2, padding=1, bias =False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),

            # 输入一个8*8*ngf*4
            nn.ConvTranspose2d(in_channels=self.ngf * 4, out_channels=self.ngf * 2, kernel_size=4, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(inplace=True),

            # 输入一个16*16*ngf*2
            nn.ConvTranspose2d(in_channels=self.ngf * 2, out_channels=self.ngf, kernel_size=4, stride=2, padding=1, bias =False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(inplace=True),

            # 输入一张32*32*ngf
            nn.ConvTranspose2d(in_channels=self.ngf, out_channels=3, kernel_size=5, stride=3, padding=1, bias =False),

            # Tanh收敛速度快于sigmoid,远慢于relu,输出范围为[-1,1]，输出均值为0
            nn.Tanh(),

        )# 输出一张96*96*3

    def forward(self, x):
        return self.Gene(x)

# 构建Discriminator判别器
class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()

        self.ndf = opt.ndf
        # DCGAN定义的判别器，生成器没有池化层
        self.Discrim = nn.Sequential(
            # 卷积层
            # 输入通道数in_channels，输出通道数(即卷积核的通道数)out_channels，此处设定filer过滤器有64个，输出通道自然也就是64。
            # 因为对图片作了灰度处理，此处通道数为1，
            # 卷积核大小kernel_size，步长stride，对称填0行列数padding
            # input:(bitch_size, 3, 96, 96),bitch_size = 单次训练的样本量
            # output:(bitch_size, ndf, 32, 32), (96 - 5 +2 *1)/3 + 1 =32
            # LeakyReLu= x if x>0 else nx (n为第一个函数参数)，开启inplace（覆盖）可以节省内存，取消反复申请内存的过程
            # LeakyReLu取消了Relu的负数硬饱和问题，是否对模型优化有效有待考证
            nn.Conv2d(in_channels=3, out_channels= self.ndf, kernel_size= 5, stride= 3, padding= 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace= True),

            # input:(ndf, 32, 32)
            nn.Conv2d(in_channels= self.ndf, out_channels= self.ndf * 2, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *2, 16, 16)
            nn.Conv2d(in_channels= self.ndf * 2, out_channels= self.ndf *4, kernel_size= 4, stride= 2, padding= 1,bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *4, 8, 8)
            nn.Conv2d(in_channels= self.ndf *4, out_channels= self.ndf *8, kernel_size= 4, stride= 2, padding= 1, bias=False),
            nn.BatchNorm2d(self.ndf *8),
            nn.LeakyReLU(0.2, True),

            # input:(ndf *8, 4, 4)
            # output:(1, 1, 1)
            nn.Conv2d(in_channels= self.ndf *8, out_channels= 1, kernel_size= 4, stride= 1, padding= 0, bias=True),

            # 调用sigmoid函数解决分类问题
            # 因为判别模型要做的是二分类，故用sigmoid即可，因为sigmoid返回值区间为[0,1]，
            # 可作判别模型的打分标准
            nn.Sigmoid()
        )

    def forward(self, x):
        # 展平后返回
        return self.Discrim(x).view(-1)


def train(**kwargs):

    # 配置属性
    # 如果函数无字典输入则使用opt中设定好的默认超参数
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # device(设备)，分配设备
    if opt.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    # 数据预处理1
    # transforms 模块提供一般图像转换操作类的功能，最后转成floatTensor
    # tv.transforms.Compose用于组合多个tv.transforms操作,定义好transforms组合操作后，直接传入图片即可进行处理
    # tv.transforms.Resize，对PIL Image对象作resize运算， 数值保存类型为float64
    # tv.transforms.CenterCrop, 中心裁剪
    # tv.transforms.ToTensor，将opencv读到的图片转为torch image类型（通道，像素，像素）,且把像素范围转为[0，1]
    # tv.transforms.Normalize,执行image = (image - mean)/std 数据归一化操作，一参数是mean,二参数std
    # 因为是三通道，所以mean = (0.5, 0.5, 0.5),从而转成[-1, 1]范围
    transforms = tv.transforms.Compose([
        # 3*96*96
        tv.transforms.Resize(opt.img_size),   # 缩放到 img_size* img_size
        # 中心裁剪成96*96的图片。因为本实验数据已满足96*96尺寸，可省略
        tv.transforms.CenterCrop(opt.img_size),

        # ToTensor 和 Normalize 搭配使用
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据并使用定义好的transforms对图片进行预处理,这里用的是直接定义法
    # dataset是一个包装类，将数据包装成Dataset类，方便之后传入DataLoader中
    # 写法2：
    # 定义类Dataset（Datasets）包装类，重写__getitem__（进行transforms系列操作）、__len__方法（获取样本个数）
    # ### 两种写法有什么区别
    dataset = tv.datasets.ImageFolder(root=opt.data_path, transform=transforms)

    # 数据预处理2
    # 查看drop_last操作,
    dataloader = DataLoader(
        dataset,      # 数据加载
        batch_size=opt.batch_size,    # 批处理大小设置
        shuffle=True,     # 是否进行洗牌操作
        #num_workers=opt.num_workers,     # 是否进行多线程加载数据设置
        drop_last=True           # 为True时，如果数据集大小不能被批处理大小整除，则设置为删除最后一个不完整的批处理。
    )

    # 初始化网络
    netg, netd = NetG(opt), NetD(opt)
    # 判断网络是否有权重数值
    # ### storage存储
    map_location = lambda storage, loc: storage


    # torch.load模型加载，即有模型加载模型在该模型基础上进行训练，没有模型则从头开始
    # f:类文件对象，如果有模型对象路径，则加载返回
    # map_location：一个函数或字典规定如何remap存储位置
    # net.load_state_dict将加载出来的模型数据加载到构建好的net网络中去
    if opt.netg_path:
        netg.load_state_dict(torch.load(f=opt.netg_path, map_location=map_location))
    if opt.netd_path:
        netd.load_state_dict(torch.load(f=opt.netd_path, map_location=map_location))

    # 搬移模型到之前指定设备，本文采用的是cpu,分配设备
    netd.to(device)
    netg.to(device)

    # 定义优化策略
    # torch.optim包内有多种优化算法，
    # Adam优化算法，是带动量的惯性梯度下降算法
    optimize_g = torch.optim.Adam(netg.parameters(), lr=opt.lr1, betas=(opt.beta1, 0.999))
    optimize_d = torch.optim.Adam(netd.parameters(), lr=opt.lr2, betas=(opt.beta1, 0.999))

    # 计算目标值和预测值之间的交叉熵损失函数
    # BCEloss:-w(ylog x +(1 - y)log(1 - x))
    # y为真实标签，x为判别器打分（sigmiod，1为真0为假），加上负号，等效于求对应标签下的最大得分
    # to(device),用于指定CPU/GPU
    criterions = nn.BCELoss().to(device)

    # 定义标签，并且开始注入生成器的输入noise
    true_labels = torch.ones(opt.batch_size).to(device)
    fake_labels = torch.zeros(opt.batch_size).to(device)

    # 生成满足N(1,1)标准正态分布，opt.nz维（100维），opt.batch_size个数的随机噪声
    noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    # 用于保存模型时作生成图像示例
    fix_noises = torch.randn(opt.batch_size, opt.nz, 1, 1).to(device)

    # 训练网络
    # 设置迭代
    for epoch in range(opt.max_epoch):
        # tqdm(iterator())，函数内嵌迭代器，用作循环的进度条显示
        for ii_, (img, _) in tqdm((enumerate(dataloader))):
            # 将处理好的图片赋值
            real_img = img.to(device)

            # 开始训练生成器和判别器
            # 注意要使得生成的训练次数小于一些
            # 每一轮更新一次判别器
            if ii_ % opt.d_every == 0:
                # 优化器梯度清零
                optimize_d.zero_grad()

                # 训练判别器
                # 把判别器的目标函数分成两段分别进行反向求导，再统一优化
                # 真图
                # 把所有的真样本传进netd进行训练，
                output = netd(real_img)
                # 用之前定义好的交叉熵损失函数计算损失
                error_d_real = criterions(output, true_labels)
                # 误差反向计算
                error_d_real.backward()

                # 随机生成的假图
                # .detach() 返回相同数据的 tensor ,且 requires_grad=False
                #  .detach()做截断操作，生成器不记录判别器采用噪声的梯度
                noises = noises.detach()
                # 通过生成模型将随机噪声生成为图片矩阵数据
                fake_image = netg(noises).detach()
                # 将生成的图片交给判别模型进行判别
                output = netd(fake_image)
                # 再次计算损失函数的计算损失
                error_d_fake = criterions(output, fake_labels)
                # 误差反向计算
                # 求导和优化（权重更新）是两个独立的过程，只不过优化时一定需要对应的已求取的梯度值。
                # 所以求得梯度值很关键，而且，经常会累积多种loss对某网络参数造成的梯度，一并更新网络。
                error_d_fake.backward()

                """
                关于为什么要分两步计算loss：
                我们已经知道，BCEloss相当于计算对应标签下的得分，那么我们
                把真样本传入时，因为标签恒为1，BCE此时只有第一项，即真样本得分项
                要补齐成前文提到的判别器目标函数，需要再添置假样本得分项，故两次分开计算梯度，各自最大化各自的得分（假样本得分是log（1-D（x）））
                再统一进行梯度下降即可
                """
                # 计算一次Adam算法，完成判别模型的参数迭代
                # 多个不同loss的backward()来累积同一个网络的grad,计算一次Adam即可
                optimize_d.step()

            # 训练判别器
            if ii_ % opt.g_every == 0:
                optimize_g.zero_grad()
                # 用于netd作判别训练和用于netg作生成训练两组噪声需不同
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_image = netg(noises)
                output = netd(fake_image)
                # 此时判别器已经固定住了，BCE的一项为定值，再求最小化相当于求二项即G得分的最大化
                error_g = criterions(output, true_labels)
                error_g.backward()

                # 计算一次Adam算法，完成判别模型的参数迭代
                optimize_g.step()

        # 保存模型
        if (epoch + 1) % opt.save_every == 0:
            fix_fake_image = netg(fix_noises)
            tv.utils.save_image(fix_fake_image.data[:64], "%s/%s.png" % (opt.save_path, epoch), normalize=True)

            torch.save(netd.state_dict(),  './imgs2_600/' + 'netd_{}.pth'.format(epoch))
            torch.save(netg.state_dict(),  './imgs2_600/' + 'netg_{}.pth'.format(epoch))
            print('保存模型')



# @torch.no_grad():数据不需要计算梯度，也不会进行反向传播
@torch.no_grad()
def generate(**kwargs):
    # 用训练好的模型来生成图片

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device("cuda") if opt.gpu else torch.device("cpu")

    # 加载训练好的权重数据
    netg, netd = NetG(opt).eval(), NetD(opt).eval()
    #  两个参数返回第一个
    map_location = lambda storage, loc: storage

    # opt.netd_path等参数有待修改
    netd.load_state_dict(torch.load('./imgs2_600/netd_599.pth', map_location=map_location), False)
    netg.load_state_dict(torch.load('./imgs2_600/netg_599.pth', map_location=map_location), False)
    netd.to(device)
    netg.to(device)

    # 生成训练好的图片
    # 初始化512组噪声，选其中好的拿来保存输出。
    noise = torch.randn(opt.gen_search_num, opt.nz, 1, 1).normal_(opt.gen_mean, opt.gen_std).to(device)

    fake_image = netg(noise)
    score = netd(fake_image).detach()

    # 挑选出合适的图片
    # 取出得分最高的图片
    indexs = score.topk(opt.gen_num)[1]

    result = []

    for ii in indexs:
        result.append(fake_image.data[ii])

    # 以opt.gen_img为文件名保存生成图片
    tv.utils.save_image(torch.stack(result), opt.gen_img, normalize=True, range=(-1, 1))

def main():
    # 训练模型
    train()
    # 生成图片
    generate()

if __name__ == '__main__':
    main()
