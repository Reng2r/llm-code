import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Q 仍然是多头，所以输出 hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # K、V 只有一组，所以输出 head_dim
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=False)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, causal=False):
        """
        x: [B, L, hidden_size]
        attention_mask:
            - 可以是 [B, L]，1 表示有效，0 表示 padding
            - 也可以是 [B, 1, 1, L] 这种可广播形式
        causal: 是否使用因果 mask
        """
        B, L, _ = x.shape

        # 1) 线性映射得到 Q, K, V
        q = self.q_proj(x)   # [B, L, hidden_size]
        k = self.k_proj(x)   # [B, L, head_dim]
        v = self.v_proj(x)   # [B, L, head_dim]

        # 2) reshape 成 attention 需要的形状
        # Q: [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V: 只有 1 个 KV head
        # [B, 1, L, head_dim]
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        # 3) 计算注意力分数
        # q: [B, h, L, d]
        # k.transpose(-2, -1): [B, 1, d, L]
        # 广播后 scores: [B, h, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4) causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            )  # [L, L]
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # 5) 外部 attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # [B, L] -> [B, 1, 1, L]
                attention_mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # 6) softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7) attention 输出
        # attn_weights: [B, h, L, L]
        # v:            [B, 1, L, d]
        # 广播后 out:   [B, h, L, d]
        out = torch.matmul(attn_weights, v)

        # 8) 拼回去
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)

        # 9) 输出投影
        out = self.out_proj(out)

        return out
