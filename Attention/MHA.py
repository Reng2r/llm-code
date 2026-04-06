import torch
from torch import nn
from xxx import RoPEEmbedding   # 假设外部已实现


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_seq_len, dropout=0.1, is_causal=False):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_causal = is_causal

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, padding_mask=None):
        """
        x: (B, L, hidden_dim)
        padding_mask: (B, L), True 表示该位置要被屏蔽
        """
        B, L, _ = x.shape

        # 1. Q K V 投影
        q = self.q_proj(x)  # (B, L, hidden_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 分头 -> (B, H, L, D)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE 加到 Q 和 K 上
        q = self.rope(q)
        k = self.rope(k)

        # 4. 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # 5. causal mask
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            )  # (L, L)
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                torch.finfo(attn_scores.dtype).min
            )

        # 6. padding mask
        if padding_mask is not None:
            padding_mask = padding_mask[:, None, None, :].bool()  # (B, 1, 1, L)
            attn_scores = attn_scores.masked_fill(
                padding_mask,
                torch.finfo(attn_scores.dtype).min
            )

        # 7. softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 8. 加权求和
        out = torch.matmul(attn_probs, v)  # (B, H, L, D)

        # 9. 合并多头
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_dim)

        # 10. 输出投影
        out = self.o_proj(out)

        return out, attn_probs
