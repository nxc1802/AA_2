import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Support:
    """Manages spatial coordinate support indices S ⊆ {0, ..., H*W - 1}."""
    mask: torch.Tensor  # Boolean tensor of shape (B, 1, H, W)
    max_k: int

    @classmethod
    def from_mask(cls, mask: torch.Tensor, max_k: int) -> "Support":
        return cls(mask=mask.bool(), max_k=max_k)

    @classmethod
    def from_indices(cls, indices: torch.Tensor, B: int, H: int, W: int, max_k: int) -> "Support":
        device = indices.device
        mask_flat = torch.zeros(B, H * W, dtype=torch.bool, device=device)
        mask_flat.scatter_(1, indices, True)
        return cls(mask=mask_flat.view(B, 1, H, W), max_k=max_k)

    def size(self) -> torch.Tensor:
        """Returns active pixel count per sample (B,)."""
        B = self.mask.shape[0]
        return self.mask.view(B, -1).sum(dim=1)

    def to_spatial_mask(self) -> torch.Tensor:
        """Returns (B, 1, H, W) boolean mask."""
        return self.mask

    def to_flat_mask(self) -> torch.Tensor:
        """Returns (B, H*W) boolean mask."""
        B = self.mask.shape[0]
        return self.mask.view(B, -1)

    def get_indices(self) -> list:
        """Returns list of 1D tensors containing active 1D spatial indices for each sample in batch."""
        B = self.mask.shape[0]
        flat = self.mask.view(B, -1)
        return [torch.nonzero(flat[b], as_tuple=False).squeeze(-1) for b in range(B)]

    def swap(self, remove_idx: torch.Tensor, add_idx: torch.Tensor) -> "Support":
        """
        Swaps out remove_idx and swaps in add_idx for each sample in the batch.
        remove_idx: 1D tensor of shape (B,) with indices to remove
        add_idx: 1D tensor of shape (B,) with indices to add
        """
        B, _, H, W = self.mask.shape
        flat = self.mask.view(B, -1).clone()
        b_idx = torch.arange(B, device=self.mask.device)
        flat[b_idx, remove_idx] = False
        flat[b_idx, add_idx] = True
        return Support(mask=flat.view(B, 1, H, W), max_k=self.max_k)

    def add(self, add_idx: torch.Tensor) -> "Support":
        """Adds indices (B,) into the support mask."""
        B, _, H, W = self.mask.shape
        flat = self.mask.view(B, -1).clone()
        b_idx = torch.arange(B, device=self.mask.device)
        flat[b_idx, add_idx] = True
        return Support(mask=flat.view(B, 1, H, W), max_k=self.max_k)

    def remove(self, remove_idx: torch.Tensor) -> "Support":
        """Removes indices (B,) from the support mask."""
        B, _, H, W = self.mask.shape
        flat = self.mask.view(B, -1).clone()
        b_idx = torch.arange(B, device=self.mask.device)
        flat[b_idx, remove_idx] = False
        return Support(mask=flat.view(B, 1, H, W), max_k=self.max_k)

    def contains(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Checks if given indices (B,) are currently in the support.
        Returns boolean tensor of shape (B,).
        """
        B = self.mask.shape[0]
        flat = self.mask.view(B, -1)
        b_idx = torch.arange(B, device=self.mask.device)
        return flat[b_idx, indices]
