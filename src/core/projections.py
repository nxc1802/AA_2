import torch

def compute_spatial_l0(delta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Computes spatial L0 norm per sample in batch.
    A pixel (h, w) is considered modified if any channel has absolute change > eps.
    
    Args:
        delta: Tensor of shape (B, C, H, W)
        eps: Threshold float for considering pixel modified
    Returns:
        Tensor of shape (B,) containing number of modified pixels per sample.
    """
    # max over channels -> (B, H, W)
    channel_max = delta.abs().max(dim=1).values
    # count modified spatial locations
    l0_per_sample = (channel_max > eps).flatten(1).sum(dim=1)
    return l0_per_sample


def exact_spatial_topk_mask(score: torch.Tensor, k: int) -> torch.Tensor:
    """
    Constructs a binary spatial mask selecting EXACTLY k pixels per sample
    using top-K indices. Prevents budget overflow caused by tie-breaking in thresholding.
    
    Args:
        score: Tensor of shape (B, 1, H, W) or (B, H, W) representing pixel importance.
        k: Number of spatial pixels to select per sample.
    Returns:
        Boolean Tensor of shape (B, 1, H, W) where exactly k values per sample are True.
    """
    if score.dim() == 4:
        score_spatial = score.squeeze(1) # (B, H, W)
    else:
        score_spatial = score # (B, H, W)
        
    B, H, W = score_spatial.shape
    num_pixels = H * W
    k_bounded = min(max(1, k), num_pixels)
    
    flat_score = score_spatial.flatten(1) # (B, H*W)
    indices = flat_score.topk(k_bounded, dim=1).indices # (B, k)
    
    flat_mask = torch.zeros_like(flat_score, dtype=torch.bool)
    flat_mask.scatter_(1, indices, True)
    
    mask = flat_mask.view(B, 1, H, W)
    return mask


def project_l0(delta: torch.Tensor, k: int) -> torch.Tensor:
    """
    Projects perturbation delta onto the spatial L0-ball of radius k.
    Keeps perturbation values at top-k spatial locations with largest L2 norm across channels,
    zeroing out all other spatial locations.
    
    Args:
        delta: Tensor of shape (B, C, H, W)
        k: Maximum allowed modified spatial pixels
    Returns:
        Projected delta Tensor of shape (B, C, H, W)
    """
    # Compute L2 norm across channels for each spatial pixel (B, 1, H, W)
    spatial_mag = torch.norm(delta, p=2, dim=1, keepdim=True)
    mask = exact_spatial_topk_mask(spatial_mag, k)
    return delta * mask.to(delta.dtype)
