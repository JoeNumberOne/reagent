#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import copy
import math
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.metrics import r2_score
# from util import transform_point_cloud, npmat2euler
# //////////
import itertools
# import dcputil
import csv
#
# def getTri(pcd,pairs):
#     """
#     Function: get triangles
#     Param:
#         pcd:    point cloud coordinates [B, Number_points, 3]
#         pairs:  get pairs to form triangles [B, Number_points, Number_pairs, 3]
#     Return:
#         result: return the length of the three sides of each triangle, and sort from small to large
#                 [B, Number_points, Number_pairs, 3]
#     """
#     B,N,N_p,_ = pairs.shape
#     result = torch.zeros((B*N, N_p, 3), dtype=torch.float32)
#     temp = (torch.arange(B) * N).reshape(B,1,1,1).repeat(1,N,N_p,3).cuda()
#     pairs = pairs + temp
#     pcd = pcd.reshape(-1,3)
#     pairs = pairs.reshape(-1,N_p,3)
#     result[:,:,0] = (torch.sum(((pcd[pairs[:,:,0],:]-pcd[pairs[:,:,1],:])**2),dim=-1))
#     result[:,:,1] = (torch.sum(((pcd[pairs[:,:,1],:]-pcd[pairs[:,:,2],:])**2),dim=-1))
#     result[:,:,2] = (torch.sum(((pcd[pairs[:,:,0],:]-pcd[pairs[:,:,2],:])**2),dim=-1))
#     result = result.reshape(B,N,N_p,3)
#     result, _ = torch.sort(result,dim=-1,descending=False)
#     return result
#     return result
#
# def knn_tri(x,k):
#     """
#     Function: find the k nearest points outside DISTANCE_THRESHOLD
#     Param:
#         x:  point clouds [B, 3, Number_points]
#         k:  The number of points
#     Return:
#         idx: the index of k nearest points
#     """
#     DISTANCE_THRESHOLD = -0.1
#     x = x.transpose(1,2)
#     distance = -(torch.sum((x.unsqueeze(1) - x.unsqueeze(2)).pow(2), -1) + 1e-7)
#     mask = distance > DISTANCE_THRESHOLD
#     distance[mask] = float('-inf')
#     idx = distance.topk(k=k,dim=-1)[1]
#     return idx
#
def clones(module, N):
    """
    Function: clone the module N times
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    Function: attention mechanism
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def knn(x, k):
    """
    Function: find the k nearest points
    Param:
        x:  point cloud [B, 3, Number_points]
        k:  The number of points
    Return:
        idx: the index of k nearest points
    """
    x = x.transpose(1,2)
    distance = -(torch.sum((x.unsqueeze(1) - x.unsqueeze(2)).pow(2), -1) + 1e-7)
    idx = distance.topk(k=k, dim=-1)[1]
    return idx


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
    def __init__(self, n_emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(n_emb_dims, n_emb_dims//2),
                                nn.LayerNorm(n_emb_dims//2),
                                nn.LeakyReLU(),
                                nn.Linear(n_emb_dims//2, n_emb_dims//4),
                                nn.LayerNorm(n_emb_dims//4),
                                nn.LeakyReLU(),
                                nn.Linear(n_emb_dims//4, n_emb_dims//8),
                                nn.LayerNorm(n_emb_dims//8),
                                nn.LeakyReLU())
        self.proj_rot = nn.Linear(n_emb_dims//8, 4)
        self.proj_trans = nn.Linear(n_emb_dims//8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
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
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x), negative_slope=0.2)))

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.n_emb_dims = 1024
        self.N = 12
        self.dropout = 0.2
        self.n_ff_dims = 1024
        self.n_heads = 4
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0] #[2 512 800]
        tgt = input[1]
        # print(src.shape)
        # print(tgt.shape)
        src = src.transpose(2, 1).contiguous() #[2 800 512]
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous() #[2 512 800]
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous() #[2 512 800]
        return src_embedding, tgt_embedding

class Position_encoding(nn.Module):
    def __init__(self,len,ratio):
        super(Position_encoding,self).__init__()
        self.PE = nn.Sequential(
            nn.Linear(3,len * ratio),
            nn.Sigmoid(),
            nn.Linear(len * ratio,len),
            nn.ReLU()
        )
    def forward(self,x): #[ 2 800 3]
        x=self.PE(x)
        return x
#
# class SE_Block(nn.Module):
#     def __init__(self,ch_in,reduction=16):
#         super(SE_Block,self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(ch_in,ch_in//reduction,bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(ch_in//reduction,ch_in,bias=False),
#             nn.Sigmoid()
#         )
#     def forward(self,x):
#         b,c,_ = x.size()
#         y = self.avg_pool(x).view(b,c)
#         y = self.fc(y).view(b,c,1)
#         return x*y.expand_as(x)
#
# class T_prediction(nn.Module):
#     def __init__(self, args):
#         super(T_prediction, self).__init__()
#         self.n_emb_dims = args.n_emb_dims
#         self.Position_encoding = Position_encoding(args.n_emb_dims,8)
#         self.SE_Block = SE_Block(ch_in=args.n_emb_dims)
#         self.emb_nn = PSE_module(embed_dim=args.n_emb_dims,token_dim=args.token_dim)
#         self.attention = Transformer(args=args)
#         self.temp_net = TemperatureNet(args)
#         self.head = SVDHead(args=args)
#
#     def forward(self, *input):
#         src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity, is_corr = self.predict_embedding(*input)
#         #src[2 3 800] src_embedding[2 512 800] feature_disparity[2 512 ]
#         rotation_ab, translation_ab, corres_ab, weight_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature,is_corr)
#         rotation_ba, translation_ba, corres_ba, weight_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature,is_corr)
#         return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity, corres_ab, weight_ab
#
#     def predict_embedding(self, *input):
#         src = input[0]  #src[2 3 800]
#         tgt = input[1]  #[2 3 800]
#         is_corr = input[2] #=1
#         src_embedding = self.emb_nn(src) #[2 512 800]
#         tgt_embedding = self.emb_nn(tgt) #[2 512 800]
#         src_encoding = self.Position_encoding(src.transpose(1,2)).transpose(1,2).contiguous() #[2 512 800]
#         tgt_encoding = self.Position_encoding(tgt.transpose(1,2)).transpose(1,2).contiguous() #[2 512 800]
#
#         src_embedding_p, tgt_embedding_p = self.attention(src_embedding+src_encoding, tgt_embedding+tgt_encoding)  #src_embedding_p[2 512 800]
#         #src_embedding+src_encoding=[2 512 800]+[2 512 800]
#         src_embedding = self.SE_Block(src_embedding+src_embedding_p)
#         tgt_embedding = self.SE_Block(tgt_embedding+tgt_embedding_p)
#         temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding) #[2 512]
#
#
#         return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity, is_corr
#
#     def predict_keypoint_correspondence(self, *input):
#         src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
#         batch_size, num_dims, num_points = src.size()
#         d_k = src_embedding.size(1)
#         scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
#         scores = scores.view(batch_size*num_points, num_points)
#         temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
#         scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
#         scores = scores.view(batch_size, num_points, num_points)
#         return src, tgt, scores
#