import torch
import torch.nn as nn
import re
import string
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter  # 计数类
from torch.utils.data import DataLoader, TensorDataset
from tqdm import *
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


# 只读取有效内容
with open("有英语-中文普通话对应句 - 2024-02-12.tsv", encoding='utf-8') as f:
    data = []
    for line in f.readlines():
        data.append(line.strip().split('\t')[1] + '\t' + line.strip().split('\t')[3])
# print(data[:5])


# 找出特殊字符
content = ''.join(data)
special_char = re.sub(r'[\u4e00-\u9fa5]', ' ', content)  # 匹配中文，将中文替换掉
# print(set(special_char) - set(string.ascii_letters) - set(string.digits))


# 数据清洗
def cleaning(data):
    for i in range(len(data)):
        # 替换特殊字符
        data[i] = data[i].replace('\u2000b', '')
        data[i] = data[i].replace('\u2000f', '')
        data[i] = data[i].replace('\xad', '')
        data[i] = data[i].replace('\u3000', '')
        eng_mark = [',', '.', '!', '?']  # 因为标点前加空格
        for mark in eng_mark:
            data[i] = data[i].replace(mark, ' '+mark)
        data[i] = data[i].lower()  # 统一替换为小写
    return data
# print(cleaning(data))


def tokenize(data):
    #  分别存储源语言和目标语言的词元
    src_tokens, tgt_tokens = [], []
    for line in data:
        pair = line.split('\t')
        src = pair[0].split(' ')
        tgt = list(pair[1])
        src_tokens.append(src)
        tgt_tokens.append(tgt)
    return src_tokens, tgt_tokens

src_tokens, tgt_tokens = tokenize(data)
# print("src:", src_tokens[:6])
# print("tgt:", tgt_tokens[:6])


def statistics(tokens):
    max_len = 80  # 只统计长度80以下的
    len_list = range(max_len)  # 长度值
    freq_list = np.zeros(max_len)  # 频率值
    for token in tokens:
        if len(token) < max_len:
            freq_list[len_list.index(len(token))] += 1
    return len_list, freq_list

s1, s2 = statistics(src_tokens)
t1, t2 = statistics(tgt_tokens)

plt.plot(s1,s2)
plt.plot(t1,t2)
# plt.show()


flatten = lambda l: [item for sublist in l for item in sublist]  # 展平数组

# 构建词表
class Vocab:
    def __init__(self, tokens):
        self.tokens = tokens  # 传入的tokens是二维列表
        self.token2index = {'<bos>': 0, '<eos>': 1, '<unk>': 2, '<pad>': 3}  # 先存好特殊词元
        # 将词元按词频排序后生成列表
        self.token2index.update({
            token: index + 4
            for index, (token, freq) in enumerate(
                sorted(Counter(flatten(self.tokens)).items(), key=lambda x: x[1], reverse=True))
        })
        # 构建id到词元字典
        self.index2token = {index: token for token, index in self.token2index.items()}

    def __getitem__(self, query):
        # 单一索引
        if isinstance(query, (str, int)):
            if isinstance(query, str):
                return self.token2index.get(query, 0)
            elif isinstance(query, int):
                return self.index2token.get(query, '<unk>')
        # 数组索引
        elif isinstance(query, (list, tuple)):
            return [self.__getitem__(item) for item in query]

    def __len__(self):
        return len(self.index2token)


# 构建数据集
seq_len = 48  # 序列最大长度

# 对数据按照最大长度进行截断和填充
def padding(tokens, seq_len):
    # 该函数针对单个句子进行处理
    # 传入的句子是词元形式
    return tokens[:seq_len] if len(tokens) > seq_len else tokens + ['<pad>'] * (seq_len - len(tokens))

# 实例化source和target词表
src_vocab, tgt_vocab = Vocab(src_tokens), Vocab(tgt_tokens)

# 增加结尾标识<eos>
src_data = torch.tensor([src_vocab[padding(line + ['<eos>'], seq_len)] for line in src_tokens])
tgt_data = torch.tensor([tgt_vocab[padding(line + ['<eos>'], seq_len)] for line in tgt_tokens])

# 训练集和测试集比例8比2， batch_size = 16
train_size = int(len(src_data) * 0.8)
test_size = len(src_data) - train_size
batch_size = 16

train_loader = DataLoader(TensorDataset(src_data[:train_size], tgt_data[:train_size]), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(src_data[-test_size:], tgt_data[-test_size:]), batch_size=1)


# 模型定义
# 定义编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, ebd_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, ebd_size, padding_idx=3)  # 将token表示为embedding
        self.gru = nn.GRU(ebd_size, hidden_size, num_layers=num_layers)

    def forward(self, encoder_inputs):
        encoder_inputs = self.embedding(encoder_inputs).permute(1, 0, 2)
        output, hidden = self.gru(encoder_inputs)
        # hidden的形状为(num_layers, batch_size, hidden_size)
        # 最后时刻的最后一个隐层的输出的隐状态即为上下文向量
        return hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size),
                torch.zeros(1, batch_size, self.lstm.hidden_size))

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, ebd_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, ebd_size, padding_idx=3)
        # 拼接维度ebd_size + hidden_size
        self.gru = nn.GRU(ebd_size + hidden_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_inputs, encoder_states):
        decoder_inputs = self.embedding(decoder_inputs).permute(1, 0, 2)
        content = encoder_states[-1]  # 上下文向量取编码器的最后一个隐层的输出
        content = content.repeat(decoder_inputs.shape[0] ,1 ,1)
        output, hidden = self.gru(torch.cat((decoder_inputs, content), -1), encoder_states)
        logits = self.linear(output)
        return logits, hidden

# seq2seq模型
class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_inputs, decoder_inputs):
        return self.decoder(decoder_inputs, self.encoder(encoder_inputs))


# 模型训练
# 设备检测,若未检测到cuda设备则在CPU上运行
device = "cuda" if torch.cuda.is_available() else "cpu"

# 设置超参数
lr = 0.01
num_epochs = 50
hidden_size = 256

# 建立模型
encoder = Encoder(len(src_vocab), len(src_vocab), hidden_size, num_layers=2)
decoder = Decoder(len(tgt_vocab), len(tgt_vocab), hidden_size, num_layers=2)
model = Seq2Seq(encoder, decoder)
model.to(device)

# 交叉熵损失以及Adam优化器
criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=3)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 记录损失变化
loss_history = []

# 开始训练
model.train()
for epoch in tqdm(range(num_epochs)):
    for encoder_inputs, decoder_targets in train_loader:
        encoder_inputs, decoder_targets = encoder_inputs.to(device), decoder_targets.to(device)
        # 偏移一位作为decoder的输入
        # decoder的输入第一位是<bos>
        bos_column = torch.tensor([tgt_vocab['<bos>']] * decoder_targets.shape[0]).reshape(-1, 1).to(device)
        decoder_inputs = torch.cat((bos_column, decoder_targets[:, :-1]), dim=1)
        pred, _ = model(encoder_inputs, decoder_inputs)
        loss = criterion(pred.permute(1, 2, 0), decoder_targets).mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())


plt.plot(loss_history)
plt.ylabel('train loss')
plt.show()


# 模型保存
torch.save(model.state_dict(), 'seq2seq_params.pt')

