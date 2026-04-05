import torch
from torch import nn


class RoPEEmbedding(nn.Module):
    """
    x: (B, H, L, D)
    """
    def __init__(self, head_dim, max_seq_len, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires head_dim to be even."

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)  # (L, D/2)

        self.register_buffer("cos_cached", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, None, :, :], persistent=False)

    def forward(self, x):
        # x: (B, H, L, D)
        B, H, L, D = x.shape
        cos = self.cos_cached[:, :, :L, :]   # (1, 1, L, D/2)
        sin = self.sin_cached[:, :, :L, :]   # (1, 1, L, D/2)

        x_even = x[..., 0::2]   # (B, H, L, D/2)
        x_odd  = x[..., 1::2]   # (B, H, L, D/2)

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        x_out = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return x_out
