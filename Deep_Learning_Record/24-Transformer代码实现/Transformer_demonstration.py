import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader, TensorDataset
import random
import math
from tqdm import *
import matplotlib.pyplot as plt
from collections import Counter  # 计数类

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据集生成
soundmark = ['ei', 'bi', 'si', 'di:', 'i:', 'ef', 'd3i', 'eit∫', 'ai', 'd3ei', 'kei',
             'el', 'em', 'en', 'a:', 'pi:', 'q:', 'r:', 'es', 'ti:', 'ju:', 'vi:', 'd^blju:', 'eks', 'wai', 'zi:']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

t = 1000  # 总条数
r = 0.9   # 扰动项
seq_len = 6
src_tokens, tgt_tokens = [], []  # 原始序列，目标序列列表

for i in range(t):
    src, tgt = [], []
    for j in range(seq_len):
        ind = random.randint(0, 25)
        src.append(soundmark[ind])
        if random.random() < r:
            tgt.append(alphabet[ind])
        else:
            tgt.append(alphabet[random.randint(0, 25)])
    src_tokens.append(src)
    tgt_tokens.append(tgt)
# print(src_tokens[:2])
# print(tgt_tokens[:2])


flatten = lambda l: [item for sublist in l for item in sublist]  # 展平数组

# 构建词表
class Vocab:
    def __init__(self, tokens):
        self.tokens = tokens  # 传入的tokens是二维列表
        self.token2index = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}  # 先存好特殊词元
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

# 实例化source和target词表
src_vocab, tgt_vocab = Vocab(src_tokens), Vocab(tgt_tokens)
src_vocab_size = len(src_vocab)  # 源语言词表大小
tgt_vocab_size = len(tgt_vocab)  # 目标语言词表大小

# 增加结尾标识<eos>
encoder_input = torch.tensor([src_vocab[line + ['<pad>']] for line in src_tokens])
decoder_input = torch.tensor([src_vocab[['<bos>'] + line] for line in src_tokens])
decoder_output = torch.tensor([tgt_vocab[line + ['<eos>']] for line in tgt_tokens])

# 训练集和测试集比例8比2， batch_size = 16(原始值)
train_size = int(len(encoder_input) * 0.8)
test_size = len(encoder_input) - train_size
batch_size = 16

# 自定义数据集函数
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

train_dataset = MyDataSet(encoder_input[:train_size], decoder_input[:train_size], decoder_output[:train_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_dataset = MyDataSet(encoder_input[-test_size:], decoder_input[-test_size:], decoder_output[-test_size:])
test_loader = DataLoader(test_dataset, batch_size=1)


# 模型构建
# 位置编码
def get_sinsoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # 偶数位用正弦函数
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # 奇数位用余弦函数
    return torch.FloatTensor(sinusoid_table)

# print(get_sinsoid_encoding_table(30, 512))


# 掩码操作
# mask掉没有意义的占位符
def get_attn_pad_mask(seq_q, seq_k):                       # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)          # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)

# mask掉未来信息
def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)          # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
    return subsequence_mask


# Attention计算函数
# 缩放点积注意力计算
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # 如果时停用词P就等于 0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

# 多头注意力计算
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):  # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)  # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  n_heads * d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn

# 构建前馈网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


# 编码器模块
# encoder层
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()  # 多头注意力
        self.pos_ffn = PoswiseFeedForwardNet()  # 前馈网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

# encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)  # 把字转换字向量
        self.pos_emb = nn.Embedding.from_pretrained(get_sinsoid_encoding_table(src_vocab_size, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        word_emb = self.src_emb(enc_inputs)
        pos_emb = self.pos_emb(enc_inputs)
        enc_outputs = word_emb + pos_emb
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model],
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# decoder层
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        return dec_outputs, dec_self_attn, dec_enc_attn

# decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinsoid_encoding_table(tgt_vocab_size, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):                               # dec_inputs: [batch_size, tgt_len]
                                                                                          # enc_intpus: [batch_size, src_len]
                                                                                          # enc_outputs: [batsh_size, src_len, d_model]
        word_emb = self.tgt_emb(dec_inputs)
        pos_emb = self.pos_emb(dec_inputs)
        dec_outputs = word_emb + pos_emb
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)         # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(device)     # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0).to(device)         # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)                     # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:                             # dec_outputs: [batch_size, tgt_len, d_model]
                                                              # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                              # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# Transformer模型
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder().to(device)
        self.Decoder = Decoder().to(device)
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)
    def forward(self, enc_inputs, dec_inputs):                         # enc_inputs: [batch_size, src_len]
                                                                       # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)         # enc_outputs: [batch_size, src_len, d_model],
                                                                       # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                       # dec_outpus    : [batch_size, tgt_len, d_model],
                                                                       # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
                                                                       # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                      # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


# 模型训练
# 参数设置
d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8
num_epochs = 20  # 训练轮数
# 记录损失变化
loss_history = []

model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)     #忽略 占位符 索引为0.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)  # SGD优化器
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器(这里使用效果并不好)

for epoch in tqdm(range(num_epochs)):
    for enc_inputs, dec_inputs, dec_outputs in train_loader:  # enc_inputs : [batch_size, src_len]
        # dec_inputs : [batch_size, tgt_len]
        # dec_outputs: [batch_size, tgt_len]

        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        loss = criterion(outputs, dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())  # 添加损失到列表中

# 训练损失的可视化
plt.plot(loss_history)
plt.xlabel('Batch')  # 每个批次记录一次损失，横坐标表示为"Batch"，因为每个点代表了每次批量处理后的损失
plt.ylabel('Train Loss')
plt.title('Training Loss Over Time')
plt.show()


# # 模型预测(原始代码段)
# model.eval()
# translation_results = []
# correct = 0
# error = 0
#
# for enc_inputs, dec_inputs, dec_outputs in test_loader:
#     enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
#     outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
#
#     outputs = outputs.squeeze()
#     pred_seq = []
#
#     for output in outputs:
#         next_token_index = output.argmax().item()
#         if next_token_index == tgt_vocab['<eos>']:
#             break
#         pred_seq.append(next_token_index)
#
#     pred_seq = tgt_vocab[pred_seq]
#     tgt_seq = dec_outputs.squeeze().tolist()
#
#     # 需要注意在<eos>之前截断
#     if tgt_vocab['<eos>'] in tgt_seq:
#         eos_idx = tgt_seq.index(tgt_vocab['<eos>'])
#         tgt_seq = tgt_vocab[tgt_seq[:eos_idx]]
#     else:
#         tgt_seq = tgt_vocab[tgt_seq]
#     translation_results.append((' '.join(tgt_seq), ' '.join(pred_seq)))
#
#     for i in range(len(tgt_seq)):
#         if i >= len(pred_seq) or pred_seq[i] != tgt_seq[i]:
#             error += 1
#         else:
#             correct += 1
#
# print(correct/(correct + error))
# print(translation_results)


# 模型预测(优化后的代码段--GPT4)
model.eval()  # 确保模型处于评估模式
translation_results = []
correct = 0
total_tokens = 0  # 用于更精确地计算准确率

for enc_inputs, dec_inputs, dec_outputs in test_loader:
    enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
    with torch.no_grad():  # 不计算梯度
        outputs = model(enc_inputs, dec_inputs)[0]
        outputs = outputs.argmax(-1)  # 获取预测的最大索引

    # 检查outputs是否为预期的维度
    if outputs.dim() == 1:  # 如果是1维张量，将其扩展为2维
        outputs = outputs.unsqueeze(0)

    for i in range(outputs.size(0)):  # 遍历批次中的每个样本
        pred_indices = outputs[i].tolist()  # 将张量转换为列表
        actual_indices = dec_outputs[i].tolist()

        # 将索引转换为词汇
        pred_seq = [tgt_vocab.index2token[idx] for idx in pred_indices if
                    idx not in (tgt_vocab.token2index['<pad>'], tgt_vocab.token2index['<eos>'])]
        actual_seq = [tgt_vocab.index2token[idx] for idx in actual_indices if
                      idx not in (tgt_vocab.token2index['<pad>'], tgt_vocab.token2index['<eos>'])]

        translation_results.append((' '.join(actual_seq), ' '.join(pred_seq)))

        # 计算准确性
        min_len = min(len(pred_seq), len(actual_seq))
        correct += sum(1 for j in range(min_len) if pred_seq[j] == actual_seq[j])
        total_tokens += len(actual_seq)

accuracy = correct / total_tokens if total_tokens > 0 else 0
print(f"准确率: {accuracy:.4f}")

for actual, pred in translation_results[:5]:  # 打印前5个翻译结果
    print(f"实际: {actual}\n预测: {pred}\n")