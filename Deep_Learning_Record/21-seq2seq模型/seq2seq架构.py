import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device, max_len=5):
        """
        :param encoder: 编码器模块
        :param decoder: 解码器模块
        :param device: 训练设备类型
        :param max_len: 预测序列的最大长度
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_len = max_len
        self.device = device

        en_layers = self.encoder.n_layers
        de_layers = self.decoder.n_layers

        en_hid = self.encoder.hidden_size
        de_hid = self.decoder.hidden_size

        en_direction = self.encoder.bidirections
        de_direction = self.decoder.bidirections

        # 编码器和解码器需要保证以下参数都是一致的
        assert en_layers == de_layers, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert en_hid == de_hid, \
            "Encoder and decoder must have equal number of layers!"
        assert en_direction == de_direction, \
            "If decoder is bidirectional, encoder must be bidirectional either!"

    def forward(self, src):
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        # src [seq_len, batch]
        en_out, hidden = self.encoder(src)
        # en_out [seq_len, batch, direction*hidden_size]
        # hidden [n_layers*directions, batch, hidden_size]
        batch_size = en_out.shape[1]

        #设置一个张量来存储所有的解码器输出
        all_decoder_outputs = torch.zeros((self.max_len, batch_size, self.decoder.out_dim), device=self.device)

        token = torch.tensor([0, 0], device=device)  # decoder的初始输入
        for i in range(self.max_len):
            de_out, hidden = decoder(token, hidden)
            all_decoder_outputs[i] = de_out
            # de_out [batch, out_dim]
            topv, topi = de_out.topk(1) # 获取每个batch预测的最大概率值以及索引，索引对应的目标序列就是预测的值
            token = topi.flatten()  # 解码器下一次的输入

        # 最后返回输出的所有概率矩阵来计算交叉熵损失
        return all_decoder_outputs
