import math
import torch

def self_attention(x, W_q, W_k, W_v, mask=None):
    """
    x: [B, L, D]
    W_q: [D, D_k]
    W_k: [D, D_k]
    W_v: [D, D_v]
    mask: [B, L, L] or [1, L, L]
          1 表示可见，0 表示不可见
    return:
        out:  [B, L, D_v]
        attn: [B, L, L]
    """
    # 1. 线性映射得到 Q, K, V
    Q = x @ W_q   # [B, L, D_k]
    K = x @ W_k   # [B, L, D_k]
    V = x @ W_v   # [B, L, D_v]

    # 2. 计算 attention score
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))  # [B, L, L]

    # 3. 加 mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 4. softmax 得到 attention 权重
    attn = torch.softmax(scores, dim=-1)  # [B, L, L]

    # 5. 加权求和
    out = attn @ V  # [B, L, D_v]

    return out, attn
