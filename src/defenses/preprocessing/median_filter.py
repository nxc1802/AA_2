import torch
import torch.nn.functional as F

class MedianFilterDefense:
    """Applies Median Filtering pre-processing as a defense against sparse perturbations."""
    def __init__(self, kernel_size: int = 3):
        self.kernel_size = kernel_size

    def defend(self, images: torch.Tensor) -> torch.Tensor:
        # Pad images
        pad = self.kernel_size // 2
        padded = F.pad(images, (pad, pad, pad, pad), mode='reflect')
        
        # Unfold spatial dimensions into local sliding patches
        B, C, H, W = images.shape
        patches = padded.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # patches shape: (B, C, H, W, kernel_size, kernel_size)
        
        flat_patches = patches.contiguous().view(B, C, H, W, -1)
        defended = flat_patches.median(dim=-1).values
        return torch.clamp(defended, 0.0, 1.0)
