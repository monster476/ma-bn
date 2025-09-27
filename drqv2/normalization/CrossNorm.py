import torch
import torch.nn as nn

class CrossNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.ones(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def cross_norm(self, x, gamma, beta, moving_mean, moving_var, eps, momentum, cn_alpha=0.99):
        if self.training:
            assert len(x.shape) in (2, 4), "CrossNorm.input x.shape is nor 2 and 4"
            batch_size = (int)(x.shape[0] / 2)
            batches = torch.split(x, batch_size, dim=0)
            off = batches[0]
            on = batches[1]
            if len(x.shape) == 2:
                mean_on = on.mean(dim=0)
                mean_off = off.mean(dim=0)
                mean = cn_alpha * mean_off + (1 - cn_alpha) * mean_on
                var = ((on - mean) ** 2 + (off - mean) ** 2) / (2 * batch_size - 1)
                var = var.mean(dim=0)
                bn_var = ((x - mean) ** 2).mean(dim=0)
            else:
                mean_on = on.mean(dim=(0, 2, 3), keepdim=True)
                mean_off = off.mean(dim=(0, 2, 3), keepdim=True)
                mean = cn_alpha * mean_off + (1 - cn_alpha) * mean_on
                var = ((on - mean) ** 2 + (off - mean) ** 2) / (2 * batch_size - 1)
            x_hat = (x - mean) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        else:
            x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
        y = gamma * x_hat + beta
        return y, moving_mean.data, moving_var.data

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = self.cross_norm(x, self.gamma, self.beta,\
                    self.moving_mean, self.moving_var, eps=1e-5, momentum=0.9)
        return y

