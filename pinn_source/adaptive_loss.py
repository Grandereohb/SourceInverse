import torch
import torch.nn as nn


class AdaptiveLossWeights(nn.Module):
    """Stable bounded adaptive weights for multi-term loss balancing."""

    def __init__(
        self,
        n_terms: int = 3,
        init_log_vars=None,
        min_precisions=None,
        max_precisions=None,
        reg_lambda: float = 1e-3,
    ):
        super().__init__()
        if init_log_vars is None:
            init_log_vars = [0.0] * n_terms
        if len(init_log_vars) != n_terms:
            raise ValueError("init_log_vars length must match n_terms")

        if min_precisions is None:
            min_precisions = [0.0] * n_terms
        if max_precisions is None:
            max_precisions = [1e6] * n_terms
        if len(min_precisions) != n_terms or len(max_precisions) != n_terms:
            raise ValueError("precision bounds length must match n_terms")

        self.raw_params = nn.Parameter(torch.tensor(init_log_vars, dtype=torch.float32))
        self.register_buffer(
            "min_precisions", torch.tensor(min_precisions, dtype=torch.float32)
        )
        self.register_buffer(
            "max_precisions", torch.tensor(max_precisions, dtype=torch.float32)
        )
        self.register_buffer("ref_losses", torch.ones(n_terms, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.eps = 1e-8
        self.reg_lambda = reg_lambda

    def _bounded_precision(self):
        # Map raw params -> [min, max] smoothly.
        s = torch.sigmoid(self.raw_params)
        return self.min_precisions + (self.max_precisions - self.min_precisions) * s

    def forward(self, losses):
        if len(losses) != self.raw_params.numel():
            raise ValueError("loss count must match number of adaptive weights")

        # Initialize references once and keep fixed to avoid drifting objective scale.
        loss_vec = torch.stack([li.detach() for li in losses])
        if not bool(self.initialized.item()):
            self.ref_losses.copy_(loss_vec.clamp_min(self.eps))
            self.initialized.fill_(True)

        weights = self._bounded_precision()

        total = 0.0
        for i, loss_i in enumerate(losses):
            scaled_loss = loss_i / (self.ref_losses[i] + self.eps)
            total = total + weights[i] * scaled_loss

        # Mild regularization keeps weights from saturating at extremes.
        weight_center = 0.5 * (self.min_precisions + self.max_precisions)
        reg = torch.mean((weights - weight_center) ** 2)
        total = total + self.reg_lambda * reg

        return total, [w.detach() for w in weights]
