import random
import numpy as np
import torch
import torch.nn as nn
from config import args  # 新增导入语句

class AvgMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = 0
        self.avg = 0.

    def update(self, val, n=1):
        assert n > 0
        self.val += val
        self.n += n
        self.avg = self.val / self.n

    def get(self):
        return self.avg


class BestMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = -1

    def update(self, val, n):
        assert n > self.n
        if val > self.val:
            self.val = val
            self.n = n

    def get(self):
        return self.val, self.n


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def add_gaussian_noise(tensor, mean, std):
    return torch.randn(tensor.size()) * std + mean


def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_beta_schedule():
    if args.beta_schedule == 'linear':
        return torch.linspace(args.beta_start, args.beta_end, args.diffusion_steps)
    elif args.beta_schedule == 'cosine':
        steps = args.diffusion_steps
        # 修正：使用 steps+1 个点生成正确长度的 betas
        x = torch.linspace(0, steps + 1, steps + 1)  # 关键修改点
        alpha_cumprod = torch.cos((x / (steps + 1)) * torch.pi / 2) ** 2
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    # 其他调度或默认情况
    return torch.linspace(args.beta_start, args.beta_end, args.diffusion_steps)