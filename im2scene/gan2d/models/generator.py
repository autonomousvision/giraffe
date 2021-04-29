import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from im2scene.layers import ResnetBlock

'''
ACKNOWLEDGEMENT: This code is largely adopted from:
https://github.com/LMescheder/GAN_stability
'''


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class Generator(nn.Module):
    def __init__(self, device, z_dim, prior_dist, size=64, nfilter=16,
                 nfilter_max=512, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        self.device = device
        self.z_dim = z_dim
        self.prior_dist = prior_dist

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        self.fc = nn.Linear(z_dim, self.nf0*s0*s0)

        blocks = []
        for i in range(nlayers):
            nf0 = min(nf * 2**(nlayers-i), nf_max)
            nf1 = min(nf * 2**(nlayers-i-1), nf_max)
            blocks += [
                ResnetBlock(nf0, nf1),
                nn.Upsample(scale_factor=2)
            ]

        blocks += [
            ResnetBlock(nf, nf),
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def sample_z(self, to_device=True):
        z = self.prior_dist()
        if to_device:
            z = z.to(self.device)
        return z

    def forward(self, z):
        if z is None:
            z = self.prior_dist().to(self.device)
        batch_size = z.size(0)
        out = self.fc(z)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(actvn(out))
        out = torch.tanh(out)

        return out
