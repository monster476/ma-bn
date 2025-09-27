import math
import pdb

import torch
import torch.nn as nn
from normalization.cnsn import CrossNorm, SelfNorm, CNSN
import numpy as np
import utils

class GELU(nn.Module):

    def forward(self, x):
      return torch.sigmoid(1.702 * x) * x


def make_layers_custom(cfg, norm_func, pos, beta, crop, cnsn_type):
    """Create a single layer."""
    layers = []
    in_channels = 3
    pos = int(pos)
    print('pos in [conv, norm, relu]: {}'.format(pos))
    assert pos in [1, 2, 3]
    assert cnsn_type in ['sn', 'cn', 'cnsn']

    for v in cfg:
        if v == 'Md':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.5)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=8)]
        elif v == 'NIN':
            conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=1)
            tmp_layers = [conv2d, norm_func(in_channels), GELU()]

            if 'cn' in cnsn_type:
                print('using CrossNorm with crop: {}'.format(crop))
                crossnorm = CrossNorm(crop=crop, beta=beta)
            else:
              crossnorm = None

            if 'sn' in cnsn_type:
              print('using SelfNorm')
              selfnorm = SelfNorm(in_channels)
            else:
              selfnorm = None

            cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

            tmp_layers.insert(pos, cnsn)

            layers += tmp_layers
        elif v == 'nopad':
            conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
            tmp_layers = [conv2d, norm_func(in_channels), GELU()]

            if 'cn' in cnsn_type:
              print('using CrossNorm with crop: {}'.format(crop))
              crossnorm = CrossNorm(crop=crop, beta=beta)
            else:
              crossnorm = None

            if 'sn' in cnsn_type:
              print('using SelfNorm')
              selfnorm = SelfNorm(in_channels)
            else:
              selfnorm = None

            cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

            tmp_layers.insert(pos, cnsn)

            layers += tmp_layers
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            tmp_layers = [conv2d, norm_func(v), GELU()]

            if 'cn' in cnsn_type:
              print('using CrossNorm with crop: {}'.format(crop))
              crossnorm = CrossNorm(crop=crop, beta=beta)
            else:
              crossnorm = None

            if 'sn' in cnsn_type:
              print('using SelfNorm')
              selfnorm = SelfNorm(v)
            else:
              selfnorm = None

            cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

            tmp_layers.insert(pos, cnsn)

            layers += tmp_layers
            in_channels = v

    return nn.Sequential(*layers)



class EncoderCNSN(nn.Module):
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