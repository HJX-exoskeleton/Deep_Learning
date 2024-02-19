import random
import torch
import string
from zhon.hanzi import punctuation
import matplotlib.pyplot as plt
import math
import collections


with open('../2-文本序列数据/越女剑.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
exclude = set(punctuation)
lines = [ ''.join(ch for ch in line if ch not in exclude).replace('\n','') for line in lines]
tokens = [list(line) for line in lines ]


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

vocab = Vocab(tokens)

corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus)
print(vocab.token_freqs[:10])


freqs = [freq for token, freq in vocab.token_freqs]
plt.plot(freqs)
plt.show()


# 齐夫定律
fig, ax = plt.subplots()
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
plt.plot(freqs)
plt.show()


# 二元语法词频
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = Vocab(bigram_tokens)
print("Top 10 Bigram Frequencies:")
print(bigram_vocab.token_freqs[:10])


# 三元语法词频
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
print("\nTop 10 Trigram Frequencies:")
print(trigram_vocab.token_freqs[:10])


# 查看不同语法分布的规律
unigram_freqs = [freq for token, freq in vocab.token_freqs]
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
fig, ax = plt.subplots()
ax.set_xscale('log', base=10)
ax.set_yscale('log', base=10)
plt.plot(unigram_freqs, label='unigram')
plt.plot(bigram_freqs, label='bigram')
plt.plot(trigram_freqs, label='trigram')
plt.legend()
plt.plot()
plt.show()
