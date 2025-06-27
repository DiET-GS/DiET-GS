
import torch
from torch import nn as nn
from tqdm import tqdm

class CRF(nn.Module):
    def __init__(self):
        super(CRF, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.init_identity()

    def init_identity(self):
        batch_size = 64
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)
        pbar = tqdm(range(3000), desc='CRF init identity')
        gen = torch.Generator(device="cpu").manual_seed(42)
        for _ in pbar:
            x = torch.rand(batch_size, 3, generator=gen)

            ori_shape = x.shape
            x_in = x.reshape(-1, 1)
            res_x = self.linear(x_in) * 0.1
            x_out = torch.sigmoid(res_x + x_in)
            y = x_out.reshape(ori_shape)

            loss = torch.mean((y - x) ** 2)
            pbar.set_postfix({'loss': loss.item()})

            optim.zero_grad()
            loss.backward()
            optim.step()
        optim.zero_grad()

    def forward(self, x, skip_learn=False):
        if not skip_learn:
            x = x.permute(1, 2, 0)
            ori_shape = x.shape
            x_in = x.reshape(-1, 1)
            res_x = self.linear(x_in) * 0.1
            x_out = torch.sigmoid(res_x + x_in)
            return x_out.reshape(ori_shape).permute(2, 0, 1)
        else:
            return x
