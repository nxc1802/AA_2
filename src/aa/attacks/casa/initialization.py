import torch


def apply_spatial_nms(
    gain_flat: torch.Tensor,
    H: int,
    W: int,
    k: int,
    radius: int = 1,
) -> torch.Tensor:
    """
    Selects top-k pixel indices using spatial non-maximum suppression
    to disperse initial coalition members across diverse receptive fields.
    """
    B, HW = gain_flat.shape
    gain_2d = gain_flat.view(B, H, W).clone()
    selected = []
    for _ in range(k):
        flat = gain_2d.view(B, HW)
        best_idx = flat.argmax(dim=1)
        selected.append(best_idx)
        bh = best_idx // W
        bw = best_idx % W
        for b in range(B):
            h_min = max(0, bh[b].item() - radius)
            h_max = min(H, bh[b].item() + radius + 1)
            w_min = max(0, bw[b].item() - radius)
            w_max = min(W, bw[b].item() + radius + 1)
            gain_2d[b, h_min:h_max, w_min:w_max] = 0.0
    return torch.stack(selected, dim=1)


CORNERS = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
], dtype=torch.float32)
