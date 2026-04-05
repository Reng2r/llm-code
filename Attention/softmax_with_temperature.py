import torch

def stable_softmax_with_temperature(logits, dim=-1, temperature=1.0):
    """
    logits: 任意形状张量，例如 [B, V] 或 [B, H, L, L]
    dim:    softmax 所在维度
    temperature: 温度参数，必须 > 0
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    # 1) 温度缩放
    x = logits / temperature

    # 2) 数值稳定：减去当前维度最大值，避免 exp 溢出
    x = x - x.max(dim=dim, keepdim=True).values

    # 3) softmax
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
