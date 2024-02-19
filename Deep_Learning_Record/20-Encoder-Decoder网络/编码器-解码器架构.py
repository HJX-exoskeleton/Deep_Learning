import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        return x, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size),
                torch.zeros(1, batch_size, self.lstm.hidden_size))


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.linear(x)
        return x, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm.hidden_size),
                torch.zeros(1, batch_size, self.lstm.hidden_size))

# 编码器-解码器模型：将输入数据转化成另外一种输出数据

"""
这里定义了两个网络：编码器 Encoder 和解码器 Decoder。
其中编码器接受一个输入序列和隐藏状态，并返回输出序列和新的隐藏状态。
解码器接受一个输入序列和隐藏状态，并返回输出序列和新的隐藏状态。
"""