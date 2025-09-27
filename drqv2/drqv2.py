# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from torch.utils.data import DataLoader, TensorDataset
from NormedConv import *
from cnsn import *
from functools import partial
from refinedBN import CrossBN, RBN1d
from collections import deque


class TimeWindowQueue:
    def __init__(self, window_size):
        self.window_size = window_size
        self.queue = deque()

    def clear(self):
        self.queue.clear()

    def push(self, value):
        self.queue.append(value)
        if len(self.queue) > self.window_size:
            self.queue.popleft()

    def cosine_similarity(self, vectorA, vectorB):
        scale = 100000
        vectorA = vectorA * scale
        vectorB = vectorB * scale
        sizeA = np.linalg.norm(vectorA)
        sizeB = np.linalg.norm(vectorB)
        similarity = np.dot(vectorA, vectorB) / ((sizeA * sizeB) + 1e-8)
        return similarity

    def sim_cal(self):
        if len(self.queue) > 1:
            arr = np.array(self.queue)
            mean_grad = np.mean(arr, axis=0)
            sim_li = []
            for i in range(len(self.queue)):
                sim_li.append(self.cosine_similarity(arr[i], mean_grad))
            mean_sim = np.mean(sim_li)
            std_sim = np.std(sim_li)
            norm_grad = np.linalg.norm(arr, axis=1)
            return mean_sim, std_sim, np.mean(norm_grad)
        else:
            return 0.0, 0.0, 0.0


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5

        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class EncoderBN(nn.Module):
    def __init__(self, obs_shape, momentum_value):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.BatchNorm2d(32, momentum=momentum_value),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.BatchNorm2d(32, momentum=momentum_value),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.BatchNorm2d(32, momentum=momentum_value),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.BatchNorm2d(32, momentum=momentum_value),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class EncoderCNSN(nn.Module):
    def __init__(self, obs_shape, active_num=2, device='cuda'):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35
        self.active_num = active_num
        self.device = device
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     )
        self.activation = nn.Sequential(nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU())

        self.cnsn_modules = nn.Sequential()
        for i in range(4):
            cn = CrossNorm("style", 1)
            sn = SelfNorm(32)
            cnsn = CNSN(cn, sn)
            self.cnsn_modules.append(cnsn)

        self.apply(utils.weight_init)

    def _enable_cross_norm(self):
        active_cn_idxs = np.random.choice(4, self.active_num, replace=False).tolist()
        assert len(set(active_cn_idxs)) == self.active_num
        for idx in active_cn_idxs:
            self.cn_modules[idx].active = True

    def forward(self, obs):
        h = obs / 255.0 - 0.5
        for i in range(4):
            h = self.convnet[i](h)
            h = self.cnsn_modules[i](h)
            h = self.activation[i](h)

        h = h.view(h.shape[0], -1)
        return h


class EncoderONI(nn.Module):
    def __init__(self, obs_shape, oni_t=2, oni_nscale=1.414):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), ONI_Conv2d(32, 32, 3, stride=1, T=oni_t, NScale=oni_nscale),
                                     nn.ReLU(), ONI_Conv2d(32, 32, 3, stride=1, T=oni_t, NScale=oni_nscale),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, hidden_len=2):
        super().__init__()

        self.hidden_len = hidden_len
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True))

        for i in range(self.hidden_len - 1):
            self.policy.append(nn.Linear(hidden_dim, hidden_dim))
            self.policy.append(nn.ReLU(inplace=True))
        self.outlayer = nn.Linear(hidden_dim, action_shape[0])

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = self.outlayer(mu)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class ActorBN(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, hidden_len=2, momentum_value=0.1):
        super().__init__()

        self.hidden_len = hidden_len
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.linears = nn.Sequential(nn.Linear(feature_dim, hidden_dim))
        self.bns = nn.Sequential(nn.BatchNorm1d(hidden_dim, momentum=momentum_value))
        self.activations = nn.Sequential(nn.ReLU(inplace=True))

        for i in range(self.hidden_len - 1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim, momentum=momentum_value))
            self.activations.append(nn.ReLU(inplace=True))
        self.outlayer = nn.Linear(hidden_dim, action_shape[0])

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        for i in range(self.hidden_len):
            h = self.linears[i](h)
            h = self.bns[i](h)
            h = self.activations[i](h)
        mu = self.outlayer(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class ActorLN(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, hidden_len=2):
        super().__init__()

        self.hidden_len = hidden_len
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )

        self.linears = nn.Sequential(nn.Linear(feature_dim, hidden_dim))
        self.norms = nn.Sequential(nn.LayerNorm(hidden_dim))
        self.activations = nn.Sequential(nn.ReLU(inplace=True))

        for i in range(self.hidden_len - 1):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.activations.append(nn.ReLU(inplace=True))

        self.outlayer = nn.Linear(hidden_dim, action_shape[0])

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        for i in range(self.hidden_len):
            h = self.linears[i](h)
            h = self.norms[i](h)
            h = self.activations[i](h)
        mu = self.outlayer(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class ActorOni(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, t=2, nscale=1.414):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(ONI_Linear(feature_dim, hidden_dim, T=t, NScale=nscale),
                                    nn.ReLU(inplace=True),
                                    ONI_Linear(hidden_dim, hidden_dim, T=t, NScale=nscale),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, hidden_len=2):
        super().__init__()
        self.hidden_len = hidden_len
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True)
        )

        for i in range(self.hidden_len - 1):
            self.Q1.append(nn.Linear(hidden_dim, hidden_dim))
            self.Q1.append(nn.ReLU(inplace=True))
            self.Q2.append(nn.Linear(hidden_dim, hidden_dim))
            self.Q2.append(nn.ReLU(inplace=True))

        self.Q1.append(nn.Linear(hidden_dim, 1))
        self.Q2.append(nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class CriticOni(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, t=2, nscale=1.414):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), ONI_Linear(hidden_dim, hidden_dim, T=t, NScale=nscale),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), ONI_Linear(hidden_dim, hidden_dim, T=t, NScale=nscale),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class LNQnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, len):
        super().__init__()
        self.len = len

        self.linears = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.lns = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.activations = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True)
        )
        for _ in range(max(self.len - 2, 0)):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.lns.append(nn.LayerNorm(hidden_dim))
            self.activations.append(nn.ReLU(inplace=True))

        self.outlayer = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        h = input
        for i in range(self.len):
            h = self.linears[i](h)
            h = self.lns[i](h)
            h = self.activations[i](h)

        output = self.outlayer(h)
        return output


class BNQnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, len, momentum_value):
        super().__init__()
        self.len = len
        self.momentum_value = momentum_value
        self.linears = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.bns = nn.Sequential(
            nn.BatchNorm1d(hidden_dim, momentum=momentum_value),
            nn.BatchNorm1d(hidden_dim, momentum=momentum_value)
        )
        self.activations = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True)
        )
        for i in range(max(self.len - 2, 0)):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim, momentum=momentum_value))
            self.activations.append(nn.ReLU(inplace=True))

        self.outlayer = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        h = input
        for i in range(self.len):
            h = self.linears[i](h)
            h = self.bns[i](h)
            h = self.activations[i](h)

        output = self.outlayer(h)
        return output


class CriticBN(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, hidden_len=2, momentum_value=0.1):
        super().__init__()
        self.hidden_len = hidden_len
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = BNQnet(feature_dim + action_shape[0], hidden_dim, self.hidden_len, momentum_value)
        self.Q2 = BNQnet(feature_dim + action_shape[0], hidden_dim, self.hidden_len, momentum_value)

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class CriticLN(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, hidden_len=2):
        super().__init__()
        self.hidden_len = hidden_len
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )

        self.Q1 = LNQnet(feature_dim + action_shape[0], hidden_dim, self.hidden_len)
        self.Q2 = LNQnet(feature_dim + action_shape[0], hidden_dim, self.hidden_len)

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)

        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class CriticCBN(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, cbn_alpha=0.5):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            CrossBN(hidden_dim, cbn_alpha), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            CrossBN(hidden_dim, cbn_alpha), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            CrossBN(hidden_dim, cbn_alpha), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            CrossBN(hidden_dim, cbn_alpha), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2


class RBNQnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, len):
        super().__init__()
        self.len = len
        self.linears = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.rbns = nn.Sequential(
            RBN1d(hidden_dim),
            RBN1d(hidden_dim)
        )
        self.activations = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True)
        )
        for i in range(max(self.len - 2, 0)):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.rbns.append(RBN1d(hidden_dim))
            self.activations.append(nn.LeakyReLU(inplace=True))

        self.outlayer = nn.Linear(hidden_dim, 1)

    def forward(self, input, bonus=None):
        h = input
        for i in range(self.len):
            h = self.linears[i](h)
            if bonus is not None:
                bonus = self.linears[i](bonus)
            h, bonus = self.rbns[i](h, bonus)
            h = self.activations[i](h)
            if bonus is not None:
                bonus = self.activations[i](bonus)
        output = self.outlayer(h)
        return output


class CriticRBN(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, hidden_len=2):
        super().__init__()
        self.hidden_len = hidden_len
        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = RBNQnet(feature_dim + action_shape[0], hidden_dim, self.hidden_len)
        self.Q2 = RBNQnet(feature_dim + action_shape[0], hidden_dim, self.hidden_len)
        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs=None, next_action=None):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        bonus = None
        if next_obs is not None:
            b = self.trunk(next_obs)
            bonus = torch.cat([b, next_action], dim=-1)

        q1 = self.Q1(h_action, bonus)
        q2 = self.Q2(h_action, bonus)
        return q1, q2


# DRQv2ONI agent
class DrQV2AgentOni:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, encode, oni_range,
                 oni_t, oni_nscale, actor_norm=0, actor_hidden_len=2, critic_norm=0, critic_hidden_len=2,
                 update_mode='TTETT', momentum_value=0.1, mix_mode="b", mix_rate=1):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.actor_norm = actor_norm
        self.actor_hidden_len = actor_hidden_len
        self.critic_norm = critic_norm
        self.critic_hidden_len = critic_hidden_len
        self.update_mode = update_mode
        self.momentum_value = momentum_value

        self.bn_stats = {}
        self.temp_bn_stats = {}

        # models
        if encode == 0:
            self.encoder = Encoder(obs_shape).to(device)
        elif encode == 1:
            self.encoder = EncoderONI(obs_shape, oni_t, oni_nscale).to(device)
        else:
            self.encoder = EncoderBN(obs_shape, momentum_value).to(device)

        if oni_range == 2:
            self.actor = ActorOni(self.encoder.repr_dim, action_shape, feature_dim,
                                  hidden_dim, t=oni_t, nscale=oni_nscale).to(device)
        elif actor_norm == 1:
            self.actor = ActorBN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, actor_hidden_len,
                                 momentum_value).to(
                device)
        elif actor_norm == 0:
            self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, actor_hidden_len).to(
                device)
        elif actor_norm == 2:
            self.actor = ActorLN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, actor_hidden_len).to(
                device)
        else:
            raise NotImplementedError

        if self.critic_norm == 0:
            self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, critic_hidden_len).to(
                device)
            self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim,
                                        critic_hidden_len).to(device)
        elif self.critic_norm == 1:
            self.critic = CriticOni(self.encoder.repr_dim, action_shape, feature_dim,
                                    hidden_dim, t=oni_t, nscale=oni_nscale).to(device)
            self.critic_target = CriticOni(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim,
                                           t=oni_t, nscale=oni_nscale).to(device)
        elif self.critic_norm == 2:
            self.critic = CriticBN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, critic_hidden_len,
                                   momentum_value).to(device)
            self.critic_target = CriticBN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim,
                                          critic_hidden_len, momentum_value).to(device)
        elif self.critic_norm == 3:
            self.critic = CriticRBN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, critic_hidden_len).to(
                device)
            self.critic_target = CriticRBN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim,
                                           critic_hidden_len).to(device)
        elif self.critic_norm == 4:
            self.critic = CriticCBN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
            self.critic_target = CriticCBN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim,
                                           cbn_alpha=0.5).to(device)
        elif self.critic_norm == 5:
            self.critic = CriticLN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, critic_hidden_len).to(
                device)
            self.critic_target = CriticLN(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim,
                                          critic_hidden_len).to(device)
        else:
            raise IndexError("critic_norm index error")
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.pert_a2q = None
        self.pert_o2q = None
        self.pert_oa_rate = None
        self.est_bias_mean = None
        self.est_bias_std = None
        self.mc_batch_size = 256
        self.window_size = 10
        self.selected_indices = [2]
        self.actor_grad_rec = TimeWindowQueue(self.window_size)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        self.encoder.eval()
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        self.actor.eval()

        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            if self.update_mode[1] == 'T':
                self.actor.train()
            else:
                self.actor.eval()
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_action = next_action.to(self.device)
            if self.update_mode[4] == 'T':
                self.critic_target.train()
            else:
                self.critic_target.eval()
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)

            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        if self.update_mode[3] == 'T':
            self.critic.train()
        else:
            self.critic.eval()
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, step, data, ori_action, replay_iter=None):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        if self.update_mode[0] == 'T':
            self.actor.train()
        else:
            self.actor.eval()
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        if self.update_mode[2] == 'T':
            self.critic.train()
            training = True
        else:
            self.critic.eval()
            training = False

        self.bn_stats = {}

        def hook(module, input, output, name):
            if isinstance(module, torch.nn.BatchNorm1d):
                mean_val = input[0].mean([0])
                var_val = input[0].var([0], unbiased=False)
                running_mean = module.running_mean.clone()
                running_var = module.running_var.clone()
                self.bn_stats[name] = (mean_val, var_val, running_mean, running_var)

        if self.critic_norm == 2:
            hooks = []
            for name, module in self.critic.Q1.bns.named_children():
                if isinstance(module, torch.nn.BatchNorm1d):
                    hooks.append(module.register_forward_hook(partial(hook, name=f"q1_{name}")))
            for name, module in self.critic.Q2.bns.named_children():
                if isinstance(module, torch.nn.BatchNorm1d):
                    hooks.append(module.register_forward_hook(partial(hook, name=f"q2_{name}")))

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        if self.critic_norm == 2:
            for h in hooks:
                h.remove()
        if data is not None:
            (self.est_bias_mean, self.est_bias_std) = self.cal_est_bias(data, training)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        selected_grads = []
        for idx in self.selected_indices:
            param = list(self.actor.parameters())[idx]
            if param.grad is not None:
                selected_grads.append(param.grad.clone().view(-1))
        if selected_grads:
            current_grad = torch.cat(selected_grads).detach().cpu().numpy()
            self.actor_grad_rec.push(current_grad)

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step, data):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        self.encoder.train()
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        if self.critic_norm == 4:
            metrics.update(
                self.update_criticCBN(obs, action, reward, discount, next_obs, step))
        elif self.critic_norm == 3:
            metrics.update(
                self.update_criticRBN(obs, action, reward, discount, next_obs, step))
        else:
            metrics.update(
                self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step, data, action, replay_iter))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        # if self.critic_norm == 2:
        #     self.soft_update_bn_stats(self.critic_target_tau)

        return metrics

    # def soft_update_bn_stats(self, tau):
    #     for i in range(self.critic.Q1.len):
    #         self.critic_target.Q1.bns[i].running_mean = self.critic.Q1.bns[i].running_mean.clone() * tau + (1 - tau) * self.critic_target.Q1.bns[i].running_mean
    #         self.critic_target.Q1.bns[i].running_var = self.critic.Q1.bns[i].running_var.clone() * tau + (1 - tau) * self.critic_target.Q1.bns[i].running_var
    #         self.critic_target.Q2.bns[i].running_mean = self.critic.Q2.bns[i].running_mean.clone() * tau + (1 - tau) * self.critic_target.Q2.bns[i].running_mean
    #         self.critic_target.Q2.bns[i].running_var = self.critic.Q2.bns[i].running_var.clone() * tau + (1 - tau) * self.critic_target.Q2.bns[i].running_var

    def update_criticRBN(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_action = next_action.to(self.device)
            self.critic_target.train()
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        self.critic.train()
        Q1, Q2 = self.critic(obs, action, next_obs, next_action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        return metrics

    def update_criticCBN(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_action = next_action.to(self.device)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        batch_size = (int)(obs.shape[0])
        obs_off_and_on = torch.cat([obs, next_obs], dim=0)
        actions_off_and_on = torch.cat([action, next_action], dim=0)
        Qs1, Qs2 = self.critic(obs_off_and_on, actions_off_and_on)
        Qs1_split = torch.split(Qs1, batch_size, dim=0)
        Q1 = Qs1_split[0]
        Qs2_split = torch.split(Qs2, batch_size, dim=0)
        Q2 = Qs2_split[0]

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def cal_est_bias(self, data, training):
        def custom_bn_forward(module, input_tensor, layer_name, training, num, bn_stats):
            if num == "q1":
                layer_name = "q1_" + layer_name
            elif num == "q2":
                layer_name = "q2_" + layer_name
            else:
                raise NotImplementedError
            batch_mean, batch_var, running_mean, running_var = bn_stats[layer_name]

            if training:
                normalized_input = (input_tensor - batch_mean) / torch.sqrt(batch_var + 1e-5)
                batch_output = module.weight * normalized_input + module.bias
                return batch_output
            else:
                eval_normalized_input = (input_tensor - running_mean) / torch.sqrt(running_var + 1e-5)
                eval_output = module.weight * eval_normalized_input + module.bias
                return eval_output

        def get_q_values(obs, action, training, bn_stats, critic_net):
            h = critic_net.trunk(obs)
            h_action = torch.cat([h, action], dim=-1)
            h1 = h_action
            h2 = h_action

            if self.critic_norm == 2:
                with torch.no_grad():
                    for i in range(critic_net.Q1.len):
                        h1 = critic_net.Q1.linears[i](h1)
                        critic_net.Q1.bns[i].track_running_stats = False
                        h1 = custom_bn_forward(critic_net.Q1.bns[i], h1, str(i), training, num="q1",
                                               bn_stats=bn_stats)
                        critic_net.Q1.bns[i].track_running_stats = True
                        h1 = critic_net.Q1.activations[i](h1)

                    q1 = critic_net.Q1.outlayer(h1)

                with torch.no_grad():
                    for i in range(critic_net.Q2.len):
                        h2 = critic_net.Q2.linears[i](h2)
                        critic_net.Q2.bns[i].track_running_stats = False
                        h2 = custom_bn_forward(critic_net.Q2.bns[i], h2, str(i), training, num="q2",
                                               bn_stats=bn_stats)
                        critic_net.Q2.bns[i].track_running_stats = True
                        h2 = critic_net.Q2.activations[i](h2)

                    q2 = critic_net.Q2.outlayer(h2)
            else:
                with torch.no_grad():
                    q1 = critic_net.Q1(h1)
                    q2 = critic_net.Q2(h2)
            return q1, q2

        if len(self.bn_stats) == 0 and self.critic_norm == 2:
            return 0.0, 0.0
        ori_obs = data["ori_obs"]
        action = data["ori_action"]
        mc_return = data["mc_return"]
        norm_cof = np.abs(np.mean(mc_return.detach().cpu().numpy())) + 1e-3

        dataset = TensorDataset(ori_obs, action, mc_return)
        dataloader = DataLoader(dataset, batch_size=self.mc_batch_size, shuffle=True)

        all_bias = []
        for batch_ori_obs, batch_action, batch_mc_return in dataloader:
            batch_ori_obs = self.aug(batch_ori_obs)
            with torch.no_grad():
                batch_obs = self.encoder(batch_ori_obs)

            Q1, Q2 = get_q_values(batch_obs, batch_action, training, self.bn_stats, self.critic)
            batch_q_prediction = torch.min(Q1, Q2).detach().cpu().numpy()

            batch_mc_return = batch_mc_return.cpu().numpy()
            batch_bias = batch_q_prediction - batch_mc_return
            relative_bias = batch_bias / norm_cof
            all_bias.append(relative_bias)

        normalized_bias = np.concatenate(all_bias, axis=0)
        bias_mean = np.mean(normalized_bias)
        bias_std = np.std(normalized_bias)

        return bias_mean, bias_std