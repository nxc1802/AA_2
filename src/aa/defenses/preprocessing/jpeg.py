from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
import numpy as np
from PIL import Image
import torch
from aa.defenses.base import BaseDefense


def _compress_single_jpeg(img_np: np.ndarray, quality: int) -> np.ndarray:
    pil_img = Image.fromarray(img_np)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    reconstructed = Image.open(buf).convert("RGB")
    return np.array(reconstructed).astype(np.float32) / 255.0


def jpeg_compression(x: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """Applies JPEG compression to tensor x (B, C, H, W) in range [0, 1].
    Parallelized across available CPU threads via ThreadPoolExecutor.
    """
    device = x.device
    np_imgs = (x.detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).astype(np.uint8)
    B = len(np_imgs)

    if B <= 1:
        defended_np = [_compress_single_jpeg(np_imgs[0], quality)]
    else:
        num_workers = min(B, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            defended_np = list(executor.map(lambda img: _compress_single_jpeg(img, quality), np_imgs))

    defended_tensor = torch.from_numpy(np.stack(defended_np)).permute(0, 3, 1, 2).contiguous().to(device)
    return defended_tensor


class JPEGDefense(BaseDefense):
    is_differentiable = False

    def __init__(self, quality: int = 75):
        self.quality = quality

    def defend(self, x: torch.Tensor) -> torch.Tensor:
        return jpeg_compression(x, quality=self.quality)
