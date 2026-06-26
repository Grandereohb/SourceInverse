import torch
import torch.nn as nn

from config import PLUME_MAX, PLUME_RAW_SHIFT


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

        self.q_mode = "neural"
        self.logQ_segments = None
        self.logQ_time = None
        self.q_min = None
        self.q_max = None
        self.register_buffer("q_segment_breaks", torch.empty(0))
        self.register_buffer("q_regularization_times", torch.empty(0))
        self.register_buffer("q_time_grid", torch.empty(0))
        self.register_buffer("transport_times", torch.empty(0))
        self.register_buffer("transport_u", torch.empty(0))
        self.register_buffer("transport_v", torch.empty(0))

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
        if PLUME_MAX is None:
            return torch.nn.functional.softplus(self.plume_net(xyt))
        raw = self.plume_net(xyt)
        return float(PLUME_MAX) * torch.sigmoid(raw - float(PLUME_RAW_SHIFT))

    def source_bias(self):
        return torch.nn.functional.softplus(self.raw_source_bias)

    def forward(self, xyt):
        return self.background(xyt[:, 2:3]) + self.plume_strength(xyt)

    def D(self):
        return torch.nn.functional.softplus(self.rawD) + 1e-6

    def source_xy(self, t=None):
        if t is not None:
            if t.dim() == 1:
                t = t.view(-1, 1)
            return self.xs.expand_as(t), self.ys.expand_as(t)
        return self.xs, self.ys

    def Q(self, t=None):
        if t is None:
            return torch.exp(self.logQ)
        if t.dim() == 1:
            t = t.view(-1, 1)
        if self.q_mode == "piecewise" and self.logQ_segments is not None:
            t_flat = t.reshape(-1).contiguous()
            segment_ids = torch.bucketize(t_flat, self.q_segment_breaks)
            logq = self.logQ + self.logQ_segments[segment_ids].view(-1, 1)
            q = torch.exp(logq)
        elif self.q_mode == "smooth_time" and self.logQ_time is not None:
            logq = self.logQ + self._interpolate_logq_time(t)
            q = torch.exp(logq)
        else:
            q = torch.exp(self.logQ + self.q_net(t))
        if self.q_min is not None or self.q_max is not None:
            q = torch.clamp(q, min=self.q_min, max=self.q_max)
        return q

    def _interpolate_logq_time(self, t):
        t_grid = self.q_time_grid.to(device=t.device, dtype=t.dtype).view(-1)
        logq_grid = self.logQ_time.to(device=t.device, dtype=t.dtype).view(-1)
        if t_grid.numel() == 0:
            return torch.zeros_like(t)
        if t_grid.numel() == 1:
            return logq_grid[0].expand_as(t)

        t_flat = t.reshape(-1).contiguous()
        idx_hi = torch.bucketize(t_flat, t_grid)
        idx_hi = torch.clamp(idx_hi, min=1, max=t_grid.numel() - 1)
        idx_lo = idx_hi - 1
        t_lo = t_grid[idx_lo]
        t_hi = t_grid[idx_hi]
        q_lo = logq_grid[idx_lo]
        q_hi = logq_grid[idx_hi]
        alpha = (t_flat - t_lo) / torch.clamp(t_hi - t_lo, min=1e-6)
        alpha = torch.clamp(alpha, min=0.0, max=1.0)
        return (q_lo + alpha * (q_hi - q_lo)).view_as(t)

    def configure_piecewise_q(self, n_segments, segment_breaks=None):
        n_segments = max(1, int(n_segments))
        self.q_mode = "piecewise"
        self.logQ_segments = nn.Parameter(torch.zeros(n_segments))
        self.logQ_time = None
        if segment_breaks is None:
            segment_breaks = torch.linspace(0.0, 1.0, steps=n_segments + 1)[1:-1]
        segment_breaks = torch.as_tensor(segment_breaks, dtype=torch.float32)
        self.q_segment_breaks = segment_breaks

    def configure_neural_q(self, t_values=None):
        self.q_mode = "neural"
        self.logQ_segments = None
        self.logQ_time = None
        if t_values is None:
            q_times = torch.empty(0)
        else:
            q_times = torch.as_tensor(t_values, dtype=torch.float32).view(-1, 1)
        self.q_regularization_times = q_times

    def configure_smooth_time_q(self, t_values):
        q_times = torch.as_tensor(t_values, dtype=torch.float32).view(-1)
        if q_times.numel() == 0:
            q_times = torch.linspace(0.0, 1.0, steps=1)
        self.q_mode = "smooth_time"
        self.logQ_segments = None
        self.logQ_time = nn.Parameter(torch.zeros(q_times.numel()))
        self.q_time_grid = q_times
        self.q_regularization_times = q_times.view(-1, 1)

    def configure_transport_history(self, t_values, u_values, v_values):
        self.transport_times = torch.as_tensor(t_values, dtype=torch.float32).view(-1)
        self.transport_u = torch.as_tensor(u_values, dtype=torch.float32).view(-1)
        self.transport_v = torch.as_tensor(v_values, dtype=torch.float32).view(-1)

    def set_q_bounds(self, q_min=None, q_max=None):
        self.q_min = float(q_min) if q_min is not None else None
        self.q_max = float(q_max) if q_max is not None else None

    def q_regularization(self):
        if self.q_mode == "smooth_time" and self.logQ_time is not None:
            if self.logQ_time.numel() > 1:
                first_diff = self.logQ_time[1:] - self.logQ_time[:-1]
                smooth = torch.mean(first_diff**2)
                if self.logQ_time.numel() > 2:
                    second_diff = (
                        self.logQ_time[2:]
                        - 2.0 * self.logQ_time[1:-1]
                        + self.logQ_time[:-2]
                    )
                    smooth = smooth + torch.mean(second_diff**2)
            else:
                smooth = self.logQ_time.sum() * 0.0
            l2 = torch.mean(self.logQ_time**2)
            return smooth, l2

        if self.q_mode == "neural":
            if self.q_regularization_times.numel() == 0:
                zero = self.logQ * 0.0
                return zero, zero
            t_grid = self.q_regularization_times.to(
                device=self.logQ.device,
                dtype=self.logQ.dtype,
            )
            q_delta = self.q_net(t_grid).view(-1)
            if q_delta.numel() > 1:
                first_diff = q_delta[1:] - q_delta[:-1]
                smooth = torch.mean(first_diff**2)
                if q_delta.numel() > 2:
                    second_diff = q_delta[2:] - 2.0 * q_delta[1:-1] + q_delta[:-2]
                    smooth = smooth + torch.mean(second_diff**2)
            else:
                smooth = q_delta.sum() * 0.0
            l2 = torch.mean(q_delta**2)
            return smooth, l2

        if self.q_mode != "piecewise" or self.logQ_segments is None:
            zero = self.logQ * 0.0
            return zero, zero
        if self.logQ_segments.numel() > 1:
            smooth = torch.mean((self.logQ_segments[1:] - self.logQ_segments[:-1]) ** 2)
        else:
            smooth = self.logQ_segments.sum() * 0.0
        l2 = torch.mean(self.logQ_segments**2)
        return smooth, l2
