#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding


# 常用于构建多个相同结构的网络层或模块的情况，避免了手动复制和创建多个对象的繁琐过程。
# 函数的作用是将传入的 module 对象进行深拷贝，并创建一个包含 N 个该拷贝对象的 nn.ModuleList（PyTorch中的模型容器）返回。
def clones(module, N):  # 编码器中有n个，拷贝N个模型
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 注意力机制，传入Q,K,V
def attention(query, key, value, mask=None, dropout=None):
    '''
    注意力机制：QK相乘得到相似度A，AV相乘得到注意力值Z
    '''
    d_k = query.size(-1)
    # 除dk防止跨度太大，出现异常情况  QK转置/dk
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    # mask做掩码注意力机制，在解码时会用
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # 得到注意力的概率，softmax
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):  # 用于计算源点集（src）和目标点集（dst）之间的最近邻距离和索引
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)  # topk() 函数，在每个源点对应的距离中选择最小的距离，并返回该距离和其对应的索引。
    return distances, indices


# DGCNN中的knn算法

# 编码器-解码器 结构
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    '''
    包含linear和softmax层
    '''

    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


# 编码器
class Encoder(nn.Module):  # 构建多层编码器
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# 解码器
class Decoder(nn.Module):  # 构建多层解码器
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):  # 译码器堆叠n=6层
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


'''LayerNorm实现标准化
LayerNorm的作用：对x归一化，使x的均值为0，方差为1
'''


class LayerNorm(nn.Module):  # 两个子层中的每一个都采用残差连接（cite），然后进行层归一化
    '''
    eps是一个平滑的过程，取值通常在（10^-4~10^-8 之间）
        其含义是，对于每个参数，随着其更新的总距离增多，其学习速率也随之变慢。
        防止出现除以0的情况
    '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        '''
         nn.Parameter将一个不可训练的类型Tensor转换成可以训练的类型parameter，
        并将这个parameter绑定到这个module里面。
        使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        '''

    def forward(self, x):  # 输入的东西放在forward中，初始化参数一般在init中
        mean = x.mean(-1, keepdim=True)  # 平均值
        std = x.std(-1, keepdim=True)  # 求标准差
        # LayerNorm的计算公式
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):  # 为了促进这些残差连接，模型中的所有子层以及 作为嵌入层，生成维度的输出d=512
    '''SublayerConnection做的事残差和layernorm
    子层的连接: layer_norm(x + sublayer(x))
        上述可以理解为一个残差网络加上一个LayerNorm归一化
    '''

    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        # 做layernorm标准化
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        '''标准化是此处表示的是：初始的x与attention层的输出相加
          x:是上一层self-attention的输入
          sublayer：self-attention层
        '''
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):  # 每个图层有两个子图层。首先是多头自我关注 机制，第二个是简单的、位置上的全连接
    '''
    attn实例化的是一个多头注意力 attn=MultiHeadAttention(n_heads,d_model,dropout)
    '''

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward  # 实例化feed_forward层
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 克隆两次残差连接
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 除了每个编码器层中的两个子层外，解码器还插入一个 第三子层，对输出执行多头注意力 编码器堆栈。
# 与编码器类似，我们在周围采用残余连接 每个子层，然后是层归一化。
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn  # d多头注意力
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    '''
        d_model:输入维度
        head：头数，默认是8头
    '''

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # 多头注意力平均分
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 取整
        self.h = h  # head参数
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    # query, key, value表示Q,K,V
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):  # 前馈神经网络FFN
    "Implements FFN equation."
    '''   w2(relu(w1x+b1))+b2    需要设置两个参数w1,w2 '''

    def __init__(self, d_model, d_ff, dropout=0.1):  # init中存放的是参数
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)标准化
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):  # forward存放的是输入
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class Transformer(nn.Module):
    def __init__(self, args):  # 初始化类中结构
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):  # transformer前向传播
        src = input[0]  # 源输入数据 src 和目标输入数据 tgt
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


'''位置编码器
class PositionalEncoding(nn.Module): 正弦位置编码，即通过三角函数构建位置编码
    "Implement the PE function."
    param dim: 位置向量的向量维度，一般与词向量维度相同，即d_model
    :param dropout: Dropout层的比率
    :param max_len: 句子的最大长度
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})

        pe = torch.zeros(max_len, d_model)  max_len最长的长度
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        偶数用sin,奇数用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
'''