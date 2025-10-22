import torch

DEBUG = False  # 包级别调试开关，便于统一控制

def log(msg: str):
    if DEBUG:
        print(msg, flush=True)

@torch.no_grad()
def pack_batch_tensors(tensor_list, pad_value, dtype=None):
    """将不同长度的 (T, D) 或 (T,) 张量打包成 (B, T_max, D)/(B, T_max)。"""
    lengths = [t.shape[0] for t in tensor_list]
    T_max = max(lengths)
    B = len(tensor_list)

    if tensor_list[0].dim() == 2:
        D = tensor_list[0].shape[1]
        out = torch.zeros((B, T_max, D),
                          dtype=dtype or tensor_list[0].dtype,
                          device=tensor_list[0].device)
        if pad_value != 0:
            out.fill_(pad_value)
        for i, t in enumerate(tensor_list):
            out[i, :t.shape[0]] = t
    else:
        out = torch.full((B, T_max),
                         pad_value,
                         dtype=dtype or tensor_list[0].dtype,
                         device=tensor_list[0].device)
        for i, t in enumerate(tensor_list):
            out[i, :t.shape[0]] = t
    return out, lengths