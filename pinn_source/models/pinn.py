import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, hidden=64, depth=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(3, hidden))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

        # Learnable physical parameters
        self.logD = nn.Parameter(torch.tensor(0.0))  # D = exp(logD)
        self.logQ = nn.Parameter(torch.tensor(0.0))  # baseline source strength
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

    def forward(self, xyt):
        return self.net(xyt)

    def D(self):
        return torch.exp(self.logD)

    def Q(self, t=None):
        if t is None:
            return torch.exp(self.logQ)
        if t.dim() == 1:
            t = t.view(-1, 1)
        return torch.exp(self.logQ + self.q_net(t))
