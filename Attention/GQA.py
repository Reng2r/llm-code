import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads, dropout=0.1):
        super().__init__()

        assert hidden_size % num_q_heads == 0, "hidden_size 必须能整除 num_q_heads"
        assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须能被 num_kv_heads 整除"

        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_q_heads
        self.group_size = num_q_heads // num_kv_heads

        # Q 头多，K/V 头少
        self.q_proj = nn.Linear(hidden_size, num_q_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None, is_causal=True):
        """
        x: [B, L, hidden_size]
        attention_mask:
            - None
            - [B, L]，其中 1 表示有效 token，0 表示 padding
        """
        B, L, _ = x.shape

        # 1) 线性映射得到 Q, K, V
        # q: [B, L, num_q_heads * head_dim]
        # k/v: [B, L, num_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) 拆成多头
        # q: [B, num_q_heads, L, head_dim]
        # k/v: [B, num_kv_heads, L, head_dim]
        q = q.view(B, L, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3) GQA 的关键：
        #    每个 kv head 服务一组 query heads
        #    把 k/v 从 num_kv_heads 扩展到 num_q_heads
        # 例如 q_heads=8, kv_heads=2
        # 则每个 kv head 要复制 4 次，最终 k/v 头数也变成 8
        k = k.repeat_interleave(self.group_size, dim=1)  # [B, num_q_heads, L, head_dim]
        v = v.repeat_interleave(self.group_size, dim=1)  # [B, num_q_heads, L, head_dim]

        # 4) 计算 attention score
        # [B, num_q_heads, L, head_dim] @ [B, num_q_heads, head_dim, L]
        # -> [B, num_q_heads, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 5) causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x.device, dtype=torch.bool),
                diagonal=1
            )  # 上三角为 True，表示要 mask
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # 6) padding mask
        # attention_mask: [B, L]，1=有效，0=padding
        if attention_mask is not None:
            # 变成 [B, 1, 1, L]，对 key 位置做 mask
            key_padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        # 7) softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 8) 加权求和
        # [B, num_q_heads, L, L] @ [B, num_q_heads, L, head_dim]
        # -> [B, num_q_heads, L, head_dim]
        out = torch.matmul(attn_weights, v)

        # 9) 拼回去
        # [B, L, num_q_heads, head_dim] -> [B, L, hidden_size]
        out = out.transpose(1, 2).contiguous().view(B, L, self.hidden_size)

        # 10) 输出投影
        out = self.out_proj(out)

        return out
