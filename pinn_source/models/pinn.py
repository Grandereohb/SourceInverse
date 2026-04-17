import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        self.bg_net = self._make_bg_mlp(hidden, depth)
        self.plume_net = self._make_mlp(hidden, depth)

        # Learnable physical parameters
        self.rawD = nn.Parameter(torch.tensor(0.0))
        self.logQ = nn.Parameter(torch.tensor(0.0))  # baseline source strength
        self.raw_bg_scale = nn.Parameter(torch.tensor(-2.0))
        self.raw_source_bias = nn.Parameter(torch.tensor(-2.0))
        self.xs = nn.Parameter(torch.tensor(0.0))
        self.ys = nn.Parameter(torch.tensor(0.0))

        # Time-dependent source modulation: Q(t) = exp(logQ + q_net(t))
        self.q_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        # Start from near-constant source and let training learn temporal variation.
        nn.init.zeros_(self.q_net[-1].weight)
        nn.init.zeros_(self.q_net[-1].bias)

    @staticmethod
    def _make_mlp(hidden, depth):
        layers = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_bg_mlp(hidden, depth):
        layers = [nn.Linear(1, hidden // 2), nn.Tanh()]
        for _ in range(max(depth - 2, 0)):
            layers.append(nn.Linear(hidden // 2, hidden // 2))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden // 2, 1))
        return nn.Sequential(*layers)

    def background(self, t):
        if t.dim() == 1:
            t = t.view(-1, 1)
        bg = torch.tanh(self.bg_net(t))
        return torch.nn.functional.softplus(self.raw_bg_scale) * bg

    def plume_strength(self, xyt):
        return torch.nn.functional.softplus(self.plume_net(xyt))

    def source_bias(self):
        return torch.nn.functional.softplus(self.raw_source_bias)

    def forward(self, xyt):
        return self.background(xyt[:, 2:3]) + self.plume_strength(xyt)

    def D(self):
        return torch.nn.functional.softplus(self.rawD) + 1e-6

    def Q(self, t=None):
        if t is None:
            return torch.exp(self.logQ)
        if t.dim() == 1:
            t = t.view(-1, 1)
        return torch.exp(self.logQ + self.q_net(t))
