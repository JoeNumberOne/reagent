import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np
from mmcv.ops import DeformConv2dPack as DCN

from config import *


class Agent(nn.Module):

    def __init__(self):
        super().__init__()

        self.state_emb = StateEmbed()
        self.actor_critic = ActorCriticHead()

    def forward(self, src, tgt):
        # O(src, tgt) -> S
        state, emb_tgt = self.state_emb(src, tgt)
        # S -> a, v
        action, value = self.actor_critic(state)

        # reshape a to B x axis x [step, sign]
        action = (action[0].view(-1, 3, 2 * NUM_STEPSIZES + 1),
                  action[1].view(-1, 3, 2 * NUM_STEPSIZES + 1))
        value = value.view(-1, 1, 1)

        return state, action, value, emb_tgt


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k) #使用 knn() 函数找到每个点的最近邻索引，k 是最近邻数量
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class StateEmbed(nn.Module):

    def __init__(self):
        super().__init__()

        # self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        # self.conv5 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        self.conv1 = DCN(in_channels=6, out_channels=64, kernel_size=1, bias=False)
        self.conv2 = DCN(64, 64, kernel_size=1, bias=False)
        self.conv3 = DCN(64, 128, kernel_size=1, bias=False)
        self.conv4 = DCN(128, 256, kernel_size=1, bias=False)
        self.conv5 = DCN(512, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(1024)

    def forward(self, src, tgt):
        B, N, D = src.shape

        # O=(src,tgt) -> S=[Phi(src), Phi(tgt)]
        emb_src = self.embed(src.transpose(2, 1))
        if BENCHMARK and len(tgt.shape) != 3:
            emb_tgt = tgt  # re-use target embedding from first step
        else:
            emb_tgt = self.embed(tgt.transpose(2, 1))
        state = torch.cat((emb_src, emb_tgt), dim=-1)
        state = state.view(B, -1)

        return state, emb_tgt

    def embed(self, x):
        B, N, C = x.size()
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)  # torch.Size([32, 6, 1024, 20])
        x = x.contiguous()
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x1 = x.max(dim=-1, keepdim=True)[0]  # 最大池化

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]
        # 每个x纬度：torch.Size([32, 256, 1024, 20])
        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]
        # print("--------------------")
        # print(x.shape)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        x_pooled = torch.max(x, 2, keepdim=True)[0]
        return x_pooled.view(B, -1)


class ActorCriticHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.activation = nn.ReLU()

        self.emb_r = nn.Sequential(
            nn.Linear(STATE_DIM, HEAD_DIM * 2),
            self.activation,
            nn.Linear(HEAD_DIM * 2, HEAD_DIM),
            self.activation
        )
        self.action_r = nn.Linear(HEAD_DIM, NUM_ACTIONS * NUM_STEPSIZES + NUM_NOPS)

        self.emb_t = nn.Sequential(
            nn.Linear(STATE_DIM, HEAD_DIM * 2),
            self.activation,
            nn.Linear(HEAD_DIM * 2, HEAD_DIM),
            self.activation
        )
        self.action_t = nn.Linear(HEAD_DIM, NUM_ACTIONS * NUM_STEPSIZES + NUM_NOPS)

        self.emb_v = nn.Sequential(
            nn.Linear(HEAD_DIM * 2, HEAD_DIM),
            self.activation
        )
        self.value = nn.Linear(HEAD_DIM, 1)

    def forward(self, state):
        # S -> S'
        emb_t = self.emb_t(state)
        emb_r = self.emb_r(state)
        # S' -> pi
        action_logits_t = self.action_t(emb_t)
        action_logits_r = self.action_r(emb_r)

        # S' -> v
        state_action = torch.cat([emb_t, emb_r], dim=1)
        emb_v = self.emb_v(state_action)
        value = self.value(emb_v)

        return [action_logits_t, action_logits_r], value


# -- action helpers
def action_from_logits(logits, deterministic=True):
    distributions = _get_distributions(*logits)
    actions = _get_actions(*(distributions + (deterministic,)))

    return torch.stack(actions).transpose(1, 0)


def action_stats(logits, action):
    distributions = _get_distributions(*logits)
    logprobs, entropies = _get_logprob_entropy(*(distributions + (action[:, 0], action[:, 1])))

    return torch.stack(logprobs).transpose(1, 0), torch.stack(entropies).transpose(1, 0)


def _get_distributions(action_logits_t, action_logits_r):
    distribution_t = Categorical(logits=action_logits_t)
    distribution_r = Categorical(logits=action_logits_r)

    return distribution_t, distribution_r


def _get_actions(distribution_t, distribution_r, deterministic=True):
    if deterministic:
        action_t = torch.argmax(distribution_t.probs, dim=-1)
        action_r = torch.argmax(distribution_r.probs, dim=-1)
    else:
        action_t = distribution_t.sample()
        action_r = distribution_r.sample()
    return action_t, action_r


def _get_logprob_entropy(distribution_t, distribution_r, action_t, action_r):
    logprob_t = distribution_t.log_prob(action_t)
    logprob_r = distribution_r.log_prob(action_r)

    entropy_t = distribution_t.entropy()
    entropy_r = distribution_r.entropy()

    return [logprob_t, logprob_r], [entropy_t, entropy_r]


# --- model helpers
def load(model, path):
    infos = torch.load(path)
    model.load_state_dict(infos['model_state_dict'])
    return infos


def save(model, path, infos={}):
    infos['model_state_dict'] = model.state_dict()
    torch.save(infos, path)


def plot_grad_flow(model):
    """
    via https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                print(f"no grad for {n}")
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, -1, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=torch.max(torch.stack(max_grads)).cpu())
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
