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
        self.logQ = nn.Parameter(torch.tensor(0.0))  # Q = exp(logQ)
        self.xs = nn.Parameter(torch.tensor(0.0))
        self.ys = nn.Parameter(torch.tensor(0.0))

    def forward(self, xyt):
        return self.net(xyt)

    def D(self):
        return torch.exp(self.logD)

    def Q(self):
        return torch.exp(self.logQ)
