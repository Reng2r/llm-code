import math
import torch
import torch.nn as nn


class MLA(nn.Module):
    """
    核心思路：
    1. Q 做低秩压缩再恢复
    2. K/V 先联合压缩到 latent，再分别恢复
    3. RoPE 不直接加在压缩后的 K_content 上，而是单独走一支 decoupled RoPE
    4. 最终 attention 用的是:
         q = [q_content, q_rope]
         k = [k_content, k_rope]
         v = v_content
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_head_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert hidden_size > 0
        assert num_heads > 0
        assert head_dim > 0
        assert q_lora_rank > 0
        assert kv_lora_rank > 0
        assert rope_head_dim > 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim                  # 内容部分维度
        self.rope_head_dim = rope_head_dim        # decoupled RoPE 维度
        self.total_qk_dim = head_dim + rope_head_dim

        # ---- Q: 先压缩，再恢复 ----
        self.q_down = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_up = nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)

        # ---- Q 的 RoPE 分支：每个 head 都有一份 ----
        self.q_rope_proj = nn.Linear(
            q_lora_rank, num_heads * rope_head_dim, bias=False
        )

        # ---- KV: 先联合压缩到一个 latent ----
        self.kv_down = nn.Linear(hidden_size, kv_lora_rank, bias=False)

        # 从同一个 latent 恢复 K_content / V
        self.k_up = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)

        # ---- K 的 RoPE 分支：共享一份，再广播到所有 head ----
        self.k_rope_proj = nn.Linear(hidden_size, rope_head_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        把最后一维拆成两半，做 RoPE 里的旋转:
        [x1, x2] -> [-x2, x1]
        """
        d = x.size(-1)
        assert d % 2 == 0, "RoPE dimension must be even."
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        对 x 的最后一维做 RoPE
        支持:
          x: [B, L, H, D] 或 [B, L, D]
        """
        d = x.size(-1)
        assert d % 2 == 0, "rope_head_dim must be even for RoPE."

        seq_len = x.size(1)
        device = x.device
        dtype = x.dtype

        # 经典 RoPE 频率
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, d, 2, device=device, dtype=dtype) / d)
        )  # [D/2]

        pos = torch.arange(seq_len, device=device, dtype=dtype)  # [L]
        freqs = torch.outer(pos, inv_freq)  # [L, D/2]

        cos = torch.cos(freqs)  # [L, D/2]
        sin = torch.sin(freqs)  # [L, D/2]

        # 扩展到 [L, D]
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        # 适配维度
        if x.dim() == 4:       # [B, L, H, D]
            cos = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]
            sin = sin.unsqueeze(0).unsqueeze(2)
        elif x.dim() == 3:     # [B, L, D]
            cos = cos.unsqueeze(0)               # [1, L, D]
            sin = sin.unsqueeze(0)
        else:
            raise ValueError("Unsupported shape for RoPE.")

        return x * cos + self.rotate_half(x) * sin

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        x: [B, L, hidden_size]
        attention_mask:
            可选，建议传可加型 mask，shape 可为:
            - [B, 1, 1, L]
            - [B, 1, L, L]
            mask 里被屏蔽位置应为一个很小的负数，比如 -1e9
        causal:
            是否使用因果 mask
        """
        B, L, _ = x.shape

        # =========================
        # 1. Q 路径
        # =========================
        q_latent = self.q_down(x)   # [B, L, q_lora_rank]

        q_content = self.q_up(q_latent)  # [B, L, H * head_dim]
        q_content = q_content.view(B, L, self.num_heads, self.head_dim)

        q_rope = self.q_rope_proj(q_latent)  # [B, L, H * rope_head_dim]
        q_rope = q_rope.view(B, L, self.num_heads, self.rope_head_dim)
        q_rope = self.apply_rope(q_rope)

        # =========================
        # 2. KV 路径（联合压缩）
        # =========================
        kv_latent = self.kv_down(x)  # [B, L, kv_lora_rank]

        k_content = self.k_up(kv_latent)  # [B, L, H * head_dim]
        k_content = k_content.view(B, L, self.num_heads, self.head_dim)

        v = self.v_up(kv_latent)  # [B, L, H * head_dim]
        v = v.view(B, L, self.num_heads, self.head_dim)

        # K 的 decoupled RoPE 分支：共享一份，再广播到所有 head
        k_rope = self.k_rope_proj(x)       # [B, L, rope_head_dim]
        k_rope = self.apply_rope(k_rope)   # [B, L, rope_head_dim]
        k_rope = k_rope.unsqueeze(2).expand(
            B, L, self.num_heads, self.rope_head_dim
        )  # [B, L, H, rope_head_dim]

        # =========================
        # 3. 拼接成最终 q / k
        # =========================
        q = torch.cat([q_content, q_rope], dim=-1)  # [B, L, H, head_dim + rope_head_dim]
        k = torch.cat([k_content, k_rope], dim=-1)  # [B, L, H, head_dim + rope_head_dim]

        # 转成 attention 常见形状 [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # =========================
        # 4. 计算 attention score
        # =========================
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.total_qk_dim)   # [B, H, L, L]

        # causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # 外部 attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # =========================
        # 5. 聚合 value
        # =========================
        out = torch.matmul(attn, v)   # [B, H, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)

        # 输出投影
        out = self.o_proj(out)        # [B, L, hidden_size]
        return out
